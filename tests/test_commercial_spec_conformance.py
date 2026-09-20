# -*- coding: utf-8 -*-
"""
tests/test_commercial_spec_conformance.py — 商用验收 [第 1 组：规格/承诺兑现]

设计原则（与「全绿式」测试相反）：
1) 每条用例的判定标准来自**外部规格**（README 承诺、utils 分片规格注释、
   API client 文档注释、requirements 声明），不来自当前实现行为——
   即先写「应当是什么」，再让实现来对；
2) 允许失败：失败即缺陷证据，不做「为了过而放宽阈值」；
3) 只读真实语料 / 只写 tmp_path。

规格来源标注：每条 docstring 中的【规格来源】。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import _nr_commercial as H

ROOT = H.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(H.TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(H.TESTS_DIR))

import utils  # noqa: E402
import novel_context  # noqa: E402

PROD = pytest.mark.skipif(
    not H.PROD_CORPUS.exists(), reason="生产语料不存在，跳过"
)


# ==================== 公共：真实语料只切片一次 ====================

@pytest.fixture(scope="module")
def prod_chunks():
    """用真实生产语料（316 万字）跑一次线上切片，全模块共享。"""
    text = H.PROD_CORPUS.read_text(encoding="utf-8")
    spec = utils.resolve_split_spec(None, None)
    from chunking import build_chunks
    result = build_chunks(text, book_id="prod", book_title="绝世主宰", spec=spec)
    H.record_metric("prod.corpus_chars", len(text), "字符", "data/绝世主宰/cleaned/cleaned.txt")
    H.record_metric("prod.chunk_count", len(result["records"]), "块")
    H.record_metric("prod.parent_count", len(result["parents"]), "父块")
    H.record_metric("prod.chapter_count", len(result["chapters"]), "章")
    return result


def _overlap_of(a: str, b: str, cap: int = 800) -> int:
    """相邻块真实重叠长度 = a 的后缀 ∩ b 的前缀 的最长公共长度。"""
    for k in range(min(len(a), len(b), cap), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


# ============================================================
# 一、切片规格（规格来源：utils.py 分片规格注释 / README 切片章节）
# ============================================================

@PROD
class TestChunkingSpec:
    """切片规格验收：块长窗口 / 硬上限 / 重叠窗口 / 切点边界。"""

    def test_hard_limit_never_exceeded(self, prod_chunks):
        """【规格来源】utils.py：「距上次切分点超过 MAX_CHUNK_CHARS(672 字)
        仍无一级标点 → 依次降级二级/三级/四级硬切兜底」，即 672 为硬上限。

        【商用预期】硬上限是下游嵌入模型窗口保护线，越界即可能被静默截断。
        【判定标准】所有块长 <= 672。
        """
        lens = sorted(len(r["text"]) for r in prod_chunks["records"])
        over = [l for l in lens if l > 672]
        H.record_metric("spec.chunk_over_hard_limit_count", len(over), "块",
                        f"最大块长 {lens[-1]}；越界块数 {len(over)}/{len(lens)}")
        assert not over, (
            f"存在 {len(over)} 个块超过硬上限 672 字（最大 {lens[-1]} 字，"
            f"超出 {lens[-1] - 672} 字）"
        )

    def test_block_length_window_coverage(self, prod_chunks):
        """【规格来源】README/utils：目标 400 token≈560 字，硬上限 480 token≈672 字。

        【判定标准】落在 [min_chars, max_chars] = [210, 672] 的块占比 >= 90%
        （低于 min 的仅允许出现在章尾）。
        """
        lens = [len(r["text"]) for r in prod_chunks["records"]]
        inside = sum(1 for l in lens if 210 <= l <= 672)
        ratio = inside / len(lens)
        H.record_metric("spec.chunk_in_window_ratio", round(ratio * 100, 2), "%",
                        f"{inside}/{len(lens)} 块落在 [210,672]")
        assert ratio >= 0.90, f"窗口内块占比仅 {ratio:.2%}，低于 90% 验收线"

    def test_overlap_window_upper_bound(self, prod_chunks):
        """【规格来源】utils.py：「重叠 = max(50, 前块×12.5%)，上限
        OVERLAP_MAX_CHARS(100 字)」+ README「重叠 10%-15%（50-100 字）」。

        【判定标准】同一章内相邻块的重叠 <= 100 字。
        """
        recs = prod_chunks["records"]
        bad, checked, max_ov = 0, 0, 0
        for i in range(len(recs) - 1):
            if recs[i]["chapter_id"] != recs[i + 1]["chapter_id"]:
                continue
            ov = _overlap_of(recs[i]["text"], recs[i + 1]["text"])
            checked += 1
            max_ov = max(max_ov, ov)
            if ov > 100:
                bad += 1
        H.record_metric("spec.overlap_over_100_ratio",
                        round(bad / max(checked, 1) * 100, 2), "%",
                        f"{bad}/{checked} 对相邻块重叠 >100，最大 {max_ov}")
        assert bad == 0, (
            f"{bad}/{checked} 对同章相邻块重叠超出规格上限 100 字（最大 {max_ov} 字）"
        )

    def test_overlap_lower_bound(self, prod_chunks):
        """【规格来源】同上：重叠下限 50 字（保证跨块语义连续）。

        【判定标准】同章相邻块重叠 >= 50 字，不得出现 0 重叠（除硬切兜底）。
        """
        recs = prod_chunks["records"]
        zero = 0
        checked = 0
        for i in range(len(recs) - 1):
            if recs[i]["chapter_id"] != recs[i + 1]["chapter_id"]:
                continue
            checked += 1
            if _overlap_of(recs[i]["text"], recs[i + 1]["text"]) == 0:
                zero += 1
        H.record_metric("spec.overlap_zero_pairs", zero, "对",
                        f"同章相邻对共 {checked}")
        assert zero == 0, f"{zero}/{checked} 对相邻块零重叠（下限 50 字未生效）"

    def test_no_punctuation_text_uses_fallback(self):
        """【规格来源】utils.py：无一级标点 → 降级二级/三级/四级硬切兜底。

        【判定标准】3000 字无标点文本必须被切成多块且每块 <= 672 字。
        """
        text = "无标点文本" * 600  # 3000 字
        chunks = utils.split_text_recursive(text, min_chars=210, max_chars=672)
        lens = [len(c) for c in chunks]
        H.record_metric("spec.hardcut.chunk_count", len(chunks), "块",
                        f"3000 字无标点 → {len(chunks)} 块，最大 {max(lens)}")
        assert len(chunks) >= 2, "无标点长文本未触发硬切兜底"
        assert max(lens) <= 672, f"硬切块长 {max(lens)} 超过 672"

    def test_tiny_text_single_chunk(self):
        """【规格来源】utils.py：min_chars 为最优先硬约束，不足 min 不切。

        【判定标准】100 字短文本 → 单块，不产生碎片。
        """
        chunks = utils.split_text_recursive("啊" * 100, min_chars=210, max_chars=672)
        assert len(chunks) == 1, f"短文本被切成 {len(chunks)} 块（预期 1 块）"

    def test_split_spec_artifact_matches_actual_chunks(self, tmp_path, monkeypatch):
        """【规格来源】step2：split_spec.json 落盘供「检索端对齐 + 运维/测试对账」。

        【判定标准】建库后 split_spec.json 记录的 chunk_count / chunk_length_max
        必须与实际 metadata.json 一致（对账可信）。
        """
        ok, out, _ = H.build_tiny_index(
            tmp_path, "第一段。" * 400, monkeypatch, dim=16, name="spec_vdb")
        assert ok is True, "替身模型建库应成功"
        spec = json.loads((out / "split_spec.json").read_text(encoding="utf-8"))
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        real_max = max(len(m["text"]) for m in meta)
        H.record_metric("spec.split_spec_max_vs_real", f"{spec['chunk_length_max']}/{real_max}",
                        "", "split_spec 与实际块长最大值")
        assert spec["chunk_count"] == len(meta), "split_spec 块数与实际不一致"
        assert spec["chunk_length_max"] == real_max, "split_spec 最大块长与实际不一致"


# ============================================================
# 二、检索规格（规格来源：README 检索章节 R1-R5 + 模块常量）
# ============================================================

class _StubModel:
    """返回固定查询向量的替身（用于构造确定的相似度排序）。"""

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        return self.vec


class _Spy:
    """记录调用的方法包装。"""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.fn(*a, **k)


class TestRetrievalSpec:
    """检索规格验收：召回池宽度 / RRF k / top_k 语义 / 章节多样性 / 降级可观测。"""

    def test_recall_pool_width_is_topk_x5(self, tmp_path):
        """【规格来源】README R1：「每路各取 top_k×5 候选」。

        【判定标准】向量通道与关键词通道被调用的候选数都等于 top_k*5。
        """
        r = H.make_retriever(H.write_library(
            tmp_path / "vdb", ["张三走在路上。" * 5, "李四在河边。" * 5]),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        spy_v, spy_k = _Spy(r._vector_retrieve), _Spy(r._keyword_retrieve)
        r._vector_retrieve, r._keyword_retrieve = spy_v, spy_k
        r.retrieve("张三", top_k=2)
        H.record_metric("spec.pool_width_calls",
                        f"vec={spy_v.calls} kw={spy_k.calls}", "", "top_k=2 → 期望 10")
        assert all(c[0][1] == 10 for c in spy_v.calls), f"向量通道池宽不符: {spy_v.calls}"
        assert all(c[0][1] == 10 for c in spy_k.calls), f"关键词通道池宽不符: {spy_k.calls}"

    def test_rrf_fusion_k_is_60(self, tmp_path):
        """【规格来源】README R2：「RRF k=60」。

        【判定标准】融合调用时 k 必须为 60。
        """
        r = H.make_retriever(H.write_library(
            tmp_path / "vdb", ["张三走在路上。" * 5]),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        spy = _Spy(r._rrf_fusion_multi)
        r._rrf_fusion_multi = spy
        r.retrieve("张三", top_k=1)
        ks = [c[1].get("k") for c in spy.calls]
        assert ks and all(k == 60 for k in ks), f"RRF k 取值不符: {ks}"

    def test_rerank_input_width_is_topk_x3(self, tmp_path):
        """【规格来源】README R3：「重排输入截断到 top_k×3」。

        【判定标准】CrossEncoder 收到的 pair 数 <= top_k*3。
        """
        captured = {}

        class _FakeReranker:
            def predict(self, pairs, **kw):
                captured["n"] = len(pairs)
                return np.arange(len(pairs), 0, -1).astype(np.float32)

        texts = [f"第{i}块正文。" * 8 for i in range(40)]
        r = H.make_retriever(
            H.write_library(tmp_path / "vdb", texts),
            embedding_model=H.FakeEmbeddingModel(dim=16),
            reranker_model=_FakeReranker())
        r.retrieve("正文", top_k=2, enable_rerank=True)
        H.record_metric("spec.rerank_input_pairs", captured.get("n"), "条",
                        "top_k=2 → 上限 6")
        assert captured.get("n") is not None, "重排模型未被调用"
        assert captured["n"] <= 6, f"重排输入 {captured['n']} 条，超过 top_k×3=6"

    def test_topk_hits_span_distinct_chapters(self, tmp_path):
        """【规格来源】README R3：「按章节聚合（每章保留最高分）再排序截断」。

        【判定标准】top_k=3 时前 3 条命中必须来自 3 个不同章节。
        """
        dim = 8
        # 6 章 × 1 块：首维递减保证相似度严格递减且全部高于 0.1 阈值
        vecs = np.zeros((6, dim), dtype=np.float32)
        for i in range(6):
            vecs[i, 0] = 1.0 - 0.1 * i
            vecs[i, i + 1] = 1.0
        meta = [{"text": f"章节{i}内容" + "正文" * 20, "chapter": f"第{i}章",
                 "chunk_id": i} for i in range(6)]
        r = H.make_retriever(
            H.write_library(tmp_path / "vdb", texts=[m["text"] for m in meta],
                            vectors=vecs, metadata=meta),
            embedding_model=_StubModel([1, 0, 0, 0, 0, 0, 0, 0]))
        hits = r.retrieve("完全不同的问题XYZ", top_k=3)
        chaps = [(h.get("chapter") or h.get("file") or "") for h in hits
                 if not h.get("is_parent") and not h.get("is_neighbor")]
        H.record_metric("spec.topk3_chapters", "|".join(chaps), "", "期望 3 个互异章节")
        assert len(hits) >= 3, f"命中数 {len(hits)} 少于 top_k=3（high-score 候选不足）"
        assert len(set(chaps[:3])) == 3, f"top_k=3 命中章节重复: {chaps}"

    def test_low_similarity_filter_is_observable(self, tmp_path):
        """【规格来源】rag_retriever：相似度阈值 0.1 + 「降级必须可观测」。

        【判定标准】全部候选低于阈值时必须给出可观测状态提示，不得静默返回空。
        """
        vecs = np.zeros((3, 8), dtype=np.float32)
        vecs[:, 0] = 1.0          # 库向量都指向 e0
        query = np.zeros(8, dtype=np.float32)
        query[1] = 1.0            # 查询正交 → 余弦 0
        r = H.make_retriever(
            H.write_library(tmp_path / "vdb", ["甲" * 50, "乙" * 50, "丙" * 50],
                            vectors=vecs),
            embedding_model=_StubModel(query))
        hits = r.retrieve("不存在的词QQQ", top_k=3)
        status = r.last_vector_status
        H.record_metric("spec.low_score_status", status or "<空>", "", "低于阈值时的状态提示")
        assert r.last_fusion_mode.startswith("仅关键词"), f"融合模式异常: {r.last_fusion_mode}"
        assert status, "阈值过滤后未给出任何可观测提示（静默降级）"

    def test_keyword_fallback_when_no_embedding_model(self, tmp_path):
        """【规格来源】README R4：「缺嵌入模型自动降级关键词检索」。

        【判定标准】无嵌入模型时仍能返回命中，且融合模式与状态提示可见。
        """
        r = H.make_retriever(H.write_library(
            tmp_path / "vdb", ["叶星觉醒血脉。", "路人甲说话。"],
            chapters=["第一章", "第二章"]), embedding_model=None)
        hits = r.retrieve("叶星", top_k=2)
        H.record_metric("spec.fallback_hits", len(hits), "条",
                        f"fusion_mode={r.last_fusion_mode}")
        assert hits, "无嵌入模型时关键词兜底未返回任何命中"
        assert "关键词" in r.last_fusion_mode
        assert r.last_vector_status, "降级为关键词模式但无状态提示"

    def test_rerank_requested_but_unavailable_is_warned(self, tmp_path, capsys):
        """【规格来源】README R5 + 商用可观测性要求：用户显式请求重排时，
        若重排模型不可用，必须显式告知（不得静默降级成「未精排」）。

        【判定标准】enable_rerank=True 且无重排模型 → stdout/状态字段出现
        明确的重排不可用提示。
        """
        r = H.make_retriever(H.write_library(
            tmp_path / "vdb", ["叶星觉醒血脉。" * 3]), embedding_model=None,
            reranker_model=None)
        r.retrieve("叶星", top_k=2, enable_rerank=True)
        out = capsys.readouterr().out
        H.record_metric("spec.rerank_unavailable_notice",
                        "有" if ("重排" in out and any(w in out for w in ("不可用", "跳过", "未加载"))) else "无",
                        "", "显式请求重排但模型缺失时的提示")
        assert "重排" in out and any(w in out for w in ("不可用", "跳过", "未加载")), (
            "用户显式请求重排但模型不可用，未给出任何提示（静默 no-op）"
        )

    def test_empty_query_returns_empty_without_crash(self, tmp_path):
        """【规格来源】商用健壮性：空查询应可预期返回空结果。

        【判定标准】"" / 纯空白 / 纯符号查询不抛异常，返回 list。
        """
        r = H.make_retriever(H.write_library(
            tmp_path / "vdb", ["叶星觉醒血脉。"]),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        for q in ["", "   ", "\n\t", "!!!???"]:
            got = r.retrieve(q, top_k=3)
            assert isinstance(got, list), f"查询 {q!r} 未返回 list"

    def test_nonpositive_topk_must_not_return_results(self, tmp_path):
        """【规格来源】top_k 语义：返回条数上限，非正数无意义。

        【商用预期】应返回空列表（或抛参数错误），不得靠 Python 负索引
        返回「除末尾 N 条外的全部」这种未定义行为。
        【判定标准】top_k=0 与 top_k=-1 均不得返回非空结果。
        """
        texts = [f"第{i}章内容叶星出现。" for i in range(12)]
        r = H.make_retriever(H.write_library(tmp_path / "vdb", texts),
                             embedding_model=H.FakeEmbeddingModel(dim=16))
        zero = r.retrieve("叶星", top_k=0)
        neg = r.retrieve("叶星", top_k=-1)
        H.record_metric("spec.topk0_len", len(zero), "条")
        H.record_metric("spec.topk_neg1_len", len(neg), "条", "库容量 12 条")
        assert len(zero) == 0, f"top_k=0 返回了 {len(zero)} 条"
        assert len(neg) == 0, f"top_k=-1 返回了 {len(neg)} 条（负值未校验）"

    def test_huge_topk_bounded_by_library_size(self, tmp_path):
        """【规格来源】容量边界：top_k 不得超出库容量。

        【判定标准】top_k=1e6 时结果数 <= 库中文档数，且不抛异常。
        """
        texts = [f"第{i}块叶星。" for i in range(30)]
        r = H.make_retriever(H.write_library(tmp_path / "vdb", texts),
                             embedding_model=H.FakeEmbeddingModel(dim=16))
        hits = r.retrieve("叶星", top_k=10 ** 6)
        H.record_metric("spec.topk_huge_len", len(hits), "条", "库容量 30 条")
        assert len(hits) <= 30


# ============================================================
# 三、文档 / 声明一致性（承诺 → 实现）
# ============================================================

class TestDeclaredPromiseConsistency:
    """README / docstring / requirements 的外部承诺必须与实现一致。"""

    def test_readme_test_count_matches_reality(self):
        """【规格来源】README 声称的测试用例数量。

        【判定标准】README 数字 == 实际 collect 到的用例数（文档可信度）。
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"(\d+)\s*项测试", readme)
        assert m, "README 未声明测试用例数量"
        declared = int(m.group(1))
        done = H.run_cli(["-m", "pytest", "tests/", "-q", "--collect-only"])
        last = [l for l in done.stdout.strip().splitlines() if "tests collected" in l
                or "test collected" in l]
        assert last, f"无法解析 collect 结果: {done.stdout[-300:]}"
        actual = int(re.search(r"(\d+)\s+tests? collected", last[-1]).group(1))
        H.record_metric("doc.readme_test_count", f"{declared} vs {actual}", "",
                        "README 声明 vs 实际收集")
        assert declared == actual, (
            f"README 声明 {declared} 项测试，实际收集 {actual} 项（文档已失真）"
        )

    def test_epub_dependencies_declared_as_extras(self):
        """【规格来源】epub 依赖（ebooklib / beautifulsoup4）为 extras 轻量安装，
        不作为 requirements.txt 的强制依赖；README 安装说明须给出 extras 安装入口。

        【判定标准】①requirements-extras.txt 中两包为生效行；
        ②requirements.txt 中两包不得为生效行（仅允许注释形式提示）；
        ③README 提到 extras 安装命令。
        """
        def active_lines(path):
            lines = [l.strip() for l in path.read_text(
                encoding="utf-8", errors="ignore").splitlines()]
            return [l.split("==")[0].split(">=")[0].strip().lower()
                    for l in lines if l and not l.startswith("#")]

        extras_path = ROOT / "requirements-extras.txt"
        assert extras_path.is_file(), (
            "缺少 requirements-extras.txt（epub 可选依赖无处声明）"
        )
        extras_active = active_lines(extras_path)
        base_active = active_lines(ROOT / "requirements.txt")
        H.record_metric("doc.epub_deps_extras", "|".join(extras_active),
                        "|".join(base_active),
                        "extras 生效依赖 vs requirements.txt 生效依赖")

        for pkg in ("ebooklib", "beautifulsoup4"):
            assert pkg in extras_active, (
                f"requirements-extras.txt 未声明 {pkg}（epub 支持无法按指引安装）"
            )
            assert pkg not in base_active, (
                f"{pkg} 仍被列为 requirements.txt 强制依赖"
                f"（epub 依赖应为 extras 轻量安装）"
            )

        readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore")
        assert "requirements-extras.txt" in readme, (
            "README 未给出 extras 安装入口（用户无法按指引装上 epub 支持）"
        )

    def test_gitignore_covers_secret_files(self):
        """【规格来源】安全基线：含真实 API Key 的 config.json / .env 不得入库。

        【判定标准】.gitignore 命中这两类文件，且 git 未跟踪任何实例。
        """
        content = (ROOT / ".gitignore").read_text(encoding="utf-8", errors="ignore")
        names = [l.strip() for l in content.splitlines()]
        for target in ("config.json", ".env"):
            assert target in names, f".gitignore 未包含 {target}"
        sp = H.run_cli(["ls-files", "config.json", ".env"])
        tracked = [l for l in sp.stdout.splitlines() if l.strip()]
        assert not tracked, f"密钥文件已被 git 跟踪: {tracked}"

    def test_no_test_residue_inside_repo(self):
        """【规格来源】测试隔离基线：测试不得在仓库内留下残留目录/文件。

        【判定标准】models/ 下不存在测试使用的伪模型目录。
        """
        residue = [p.name for p in (ROOT / "models").glob("*")
                   if "NONEXISTENT" in p.name or "FAKE" in p.name.upper()]
        assert not residue, f"仓库 models/ 下存在测试残留: {residue}"

    def test_overlap_docstring_matches_constant(self):
        """【规格来源】utils.compute_overlap_chars docstring 与 OVERLAP_RATIO 常量
        必须表述同一比例。

        【判定标准】docstring 中出现的比例数字与 OVERLAP_RATIO 一致。
        """
        import inspect
        doc = inspect.getdoc(utils.compute_overlap_chars) or ""
        pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", doc)
        ratio_pct = utils.OVERLAP_RATIO * 100
        H.record_metric("doc.overlap_ratio_declared",
                        f"docstring={pcts} 常量={ratio_pct:g}%", "",
                        "重叠比例声明一致性")
        assert pcts, "docstring 未声明重叠比例"
        assert any(abs(float(p) - ratio_pct) < 1e-6 for p in pcts), (
            f"docstring 声明比例 {pcts}% 与常量 {ratio_pct:g}% 不一致"
        )

    def test_apiclient_documented_defaults_match_code(self):
        """【规格来源】api_client.APIClient docstring 声明的默认值
        （「默认 3 次重试」「默认 120 秒超时」）。

        【判定标准】docstring 声明值与类常量一致（否则运维按文档配置会误判）。
        """
        import inspect
        from api_client import APIClient
        doc = inspect.getdoc(APIClient.__init__) or ""
        retry_doc = re.search(r"最大重试次数（默认\s*(\d+)", doc)
        timeout_doc = re.search(r"超时秒数（默认\s*(\d+)", doc)
        H.record_metric("doc.apiclient_defaults",
                        f"doc=({retry_doc and retry_doc.group(1)},{timeout_doc and timeout_doc.group(1)}) "
                        f"code=({APIClient.DEFAULT_MAX_RETRIES},{APIClient.DEFAULT_TIMEOUT})", "",
                        "重试/超时默认值")
        assert retry_doc and int(retry_doc.group(1)) == APIClient.DEFAULT_MAX_RETRIES, (
            f"重试次数文档 {retry_doc and retry_doc.group(1)} != 实际 {APIClient.DEFAULT_MAX_RETRIES}"
        )
        assert timeout_doc and int(timeout_doc.group(1)) == APIClient.DEFAULT_TIMEOUT, (
            f"超时文档 {timeout_doc and timeout_doc.group(1)} != 实际 {APIClient.DEFAULT_TIMEOUT}"
        )

    def test_readme_promise_ask_without_api_key_verifies_retrieval(self):
        """【规格来源】README 步骤 4：「验证问答（不需要 API Key 也能验证检索链路）」。

        【判定标准】在无任何模型/Key 的配置下执行 ask，检索链路必须被真实走到
        （检索函数被调用），而不是在检索前直接退出。

        隔离手段：子进程内替换 novel_rag._get_config，不触碰真实 config.json。
        """
        script = (
            "import sys, types\n"
            f"sys.path.insert(0, r'{ROOT}')\n"
            "import novel_rag, rag_retriever\n"
            "orig = rag_retriever.RAGRetriever.retrieve\n"
            "def spy(self, *a, **k):\n"
            "    print('__RETRIEVE_CALLED__')\n"
            "    return orig(self, *a, **k)\n"
            "rag_retriever.RAGRetriever.retrieve = spy\n"
            "class FakeCfg:\n"
            "    def get(self, k, default=None): return default\n"
            "    def get_active_model(self): return None\n"
            "    def get_prompt_template(self): return '{novel_name}\\n{context}\\n{question}'\n"
            "novel_rag._get_config = lambda: FakeCfg()\n"
            "args = types.SimpleNamespace(name='demo', question='叶星是谁', api_key=None,\n"
            "                             model_id=None, rerank=None, thinking=False)\n"
            "try:\n"
            "    novel_rag.cmd_ask(args)\n"
            "except SystemExit as e:\n"
            "    print('__EXIT__', e.code)\n"
        )
        sp = H.run_pyscript(script, timeout=300)
        out = sp.stdout + sp.stderr
        H.record_metric("doc.ask_without_key_retrieval_called",
                        "是" if "__RETRIEVE_CALLED__" in out else "否", "",
                        "README 步骤 4 承诺可验证检索链路")
        assert "__RETRIEVE_CALLED__" in out, (
            "无 API Key 时 ask 在检索前即退出，README 承诺的「不需要 API Key 也能"
            f"验证检索链路」未兑现。实际输出尾部：{out[-400:]}"
        )


# ============================================================
# 四、内置评测证据强度（README 引用的指标必须可复现且有区分度）
# ============================================================

class TestEvalEvidence:
    """README 引用的检索指标：可复现性 + 数值一致性 + 评测集区分度。"""

    def test_demo_eval_reproducible(self):
        """【规格来源】README：`python eval_retrieval.py --name demo --top-k 3`
        「输出示例：QA 总数: 12 / Recall@3 = 12/12 = 100.0%」。

        【判定标准】按文档命令实跑可复现 Recall@3 >= 90%，且 QA 总数为 12。
        """
        sp = H.run_cli(["eval_retrieval.py", "--name", "demo", "--top-k", "3"],
                       timeout=600)
        out = sp.stdout + sp.stderr
        m = re.search(r"Recall@3\s*=\s*\d+/\d+\s*=\s*([\d.]+)%", out)
        assert m, f"未解析到 Recall@3（输出尾部）：{out[-500:]}"
        recall = float(m.group(1))
        total = re.search(r"QA 总数:\s*(\d+)", out)
        H.record_metric("eval.demo_recall_at_3", recall, "%", "官方脚本实跑 (--name demo)")
        H.record_metric("eval.demo_qa_total", int(total.group(1)) if total else None, "题")
        assert total and int(total.group(1)) == 12, "demo QA 题数与文档不符"
        assert recall >= 90.0, f"demo Recall@3 仅 {recall}%"

    def test_demo_mrr_matches_readme_claim(self):
        """【规格来源】README 声明的 demo MRR 指标（须与实跑一致）。

        【判定标准】README 引用的 MRR 必须与实跑值一致（文档数值可信度）。
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"MRR\s*=\s*([\d.]+)", readme)
        assert m, "README 未声明 MRR"
        claimed = float(m.group(1))
        sp = H.run_cli(["eval_retrieval.py", "--name", "demo", "--top-k", "3"],
                       timeout=600)
        out = sp.stdout + sp.stderr
        real_m = re.search(r"MRR\s*=\s*([\d.]+)", out)
        assert real_m, f"未解析到 MRR（输出尾部）：{out[-400:]}"
        real = float(real_m.group(1))
        H.record_metric("eval.demo_mrr_claim_vs_real", f"{claimed} vs {real}", "",
                        "README 声明 vs 实跑")
        assert abs(claimed - real) < 0.001, (
            f"README 声明 MRR={claimed}，实跑 MRR={real}（文档数值已失真）"
        )

    def test_default_eval_warns_on_project_qa_mismatch(self):
        """【规格来源】商用可观测性：评估脚本默认「最近使用项目 + 内置 demo QA」，
        两者可能不是同一语料；此时指标毫无意义，必须给出提示而非静默输出 0%。

        【判定标准】默认无参调用且 Recall@3 = 0% 时，输出中必须包含项目/QA
        不匹配或「请用 --name 指定对应项目」之类的提示。
        """
        sp = H.run_cli(["eval_retrieval.py"], timeout=900)
        out = sp.stdout + sp.stderr
        m = re.search(r"Recall@3\s*=\s*\d+/\d+\s*=\s*([\d.]+)%", out)
        assert m, f"未解析到 Recall@3（输出尾部）：{out[-500:]}"
        recall = float(m.group(1))
        H.record_metric("eval.default_recall_at_3", recall, "%",
                        "默认调用（最近使用项目 + 内置 demo QA）")
        if recall < 50.0:
            warned = any(w in out for w in ("不匹配", "不适用", "请指定 --name", "语料不一致"))
            assert warned, (
                f"默认评估拿 demo QA 去评非 demo 项目，Recall@3 仅 {recall}%，"
                "脚本未给出任何「项目与 QA 集不匹配」提示，容易误判为检索能力为零"
            )

    def test_demo_eval_has_discrimination_power(self):
        """【规格来源】评测方法论：评测集随机基线应显著低于 100%，否则指标无意义
        （top_k 随机命中率 = top_k / 库容量）。

        【判定标准】demo 库块数 >= 30，使 Recall@3 的随机基线 < 10%。
        """
        meta = json.loads((H.DEMO_VDB / "metadata.json").read_text(encoding="utf-8"))
        n = len(meta)
        baseline = 3 / n * 100
        H.record_metric("eval.demo_corpus_chunks", n, "块",
                        f"Recall@3 随机基线 = {baseline:.1f}%")
        assert n >= 30, (
            f"demo 评测库仅 {n} 块，Recall@3 随机基线高达 {baseline:.0f}%，"
            "该指标不具区分度（无法支撑 README 的 100% 结论）"
        )
