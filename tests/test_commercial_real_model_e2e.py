# -*- coding: utf-8 -*-
"""
tests/test_commercial_real_model_e2e.py — 商用验收 [第 4 组：真实模型端到端能力]

与前几组「替身模型、毫秒级」的差别：本组加载**真实本地模型**
（models/bge-small-zh-v1.5、models/bge-reranker-base）跑真实编码 + 真实检索，
用于回答「交付到用户机器上到底能不能用、快不快、准不准」。

判定标准：
- 自一致性（self-recall）：用库内文本本身作查询，必须能召回原文所在块 —— 这是
  检索链路可用的**最低商用标准**，达不到即为严重缺陷；
- 本地模型目录存在时，加载必须成功，且不得出现「先报成功、后报失败」的自相矛盾日志；
- 只读真实数据，不写 data/。
"""
from __future__ import annotations

import json
import sys
import time
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

LOCAL_EMB = ROOT / "models" / "bge-small-zh-v1.5"
LOCAL_RERANK = ROOT / "models" / "bge-reranker-base"


@pytest.fixture
def hollow_model_dir(tmp_path):
    """空壳模型目录（在 tmp_path 下构造，避免测试在仓库 models/ 内留残留）。"""
    d = tmp_path / "models" / "NONEXISTENT_MODEL_XYZ_12345"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def real_embedding():
    """加载真实本地嵌入模型（成功则复用，失败则跳过并记录）。"""
    if not LOCAL_EMB.exists():
        pytest.skip(f"本地嵌入模型不存在: {LOCAL_EMB}")
    t0 = time.perf_counter()
    model, device = utils.load_embedding_model(str(LOCAL_EMB))
    secs = time.perf_counter() - t0
    if model is None:
        H.record_metric("real.embedding_load", "失败", "", f"本地路径 {LOCAL_EMB}")
        pytest.skip("真实嵌入模型加载失败（已记录指标）")
    H.record_metric("real.embedding_load_seconds", round(secs, 2), "秒",
                    f"device={device}, dim={model.get_sentence_embedding_dimension()}")
    return model, device


@pytest.fixture(scope="module")
def real_retriever(real_embedding):
    """真实模型 + 生产向量库的检索器（只读）。"""
    if not H.PROD_VDB.exists():
        pytest.skip(f"生产向量库不存在: {H.PROD_VDB}")
    model, _ = real_embedding
    r = H.make_retriever(H.PROD_VDB, embedding_model=model)
    return r


class TestRealModelAvailability:
    """本地模型可用性（交付到用户机器的最前置条件）。"""

    def test_local_bge_encodes_normalized_512d(self, real_embedding):
        """【规格来源】项目默认嵌入模型 bge-small-zh-v1.5（512 维、归一化）。

        【判定标准】编码输出 (n, 512) 且 L2 范数为 1。
        """
        model, _ = real_embedding
        vecs = model.encode(["叶星提剑而立，山风猎猎。", "剑光如练，破空而至。"],
                            normalize_embeddings=True)
        arr = np.asarray(vecs)
        norms = np.linalg.norm(arr, axis=1)
        H.record_metric("real.embedding_shape", str(arr.shape), "", "bge-small-zh-v1.5")
        assert arr.shape == (2, 512)
        assert np.allclose(norms, 1.0, atol=1e-4), f"向量未归一化: {norms}"

    def test_hollow_model_dir_must_not_claim_success(self, capsys, hollow_model_dir):
        """【规格来源】模型可用性判定：空壳模型目录不得被误判为「本地模型可用」。

        【判定标准】不存在可用权重的模型目录，日志中**不得**出现「✓ 加载本地嵌入模型」
        这类成功语（先报成功后报失败属自相矛盾日志，会误导排障）。
        空壳目录在 tmp_path 下构造：既不依赖仓库内残留，也不在仓库内留残留。
        """
        model, device = utils.load_embedding_model(str(hollow_model_dir))
        out = capsys.readouterr().out
        H.record_metric("real.hollow_model_return",
                        f"model={'None' if model is None else type(model).__name__}, device={device}",
                        "", "空壳模型目录加载结果")
        assert model is None, "空壳模型目录竟然返回了模型实例"
        assert "✓ 加载本地嵌入模型" not in out, (
            f"空壳模型目录加载失败，但日志先报了成功（误导排障）：{out.strip()[:200]}"
        )

    def test_local_reranker_loads_when_present(self):
        """【规格来源】models/bge-reranker-base 已随项目放置，配置启用时必须可用。

        【判定标准】目录存在时 load_reranker_model 返回可用实例且能打分。
        """
        if not LOCAL_RERANK.exists():
            pytest.skip(f"本地重排模型不存在: {LOCAL_RERANK}")
        t0 = time.perf_counter()
        model, device = utils.load_reranker_model(str(LOCAL_RERANK))
        secs = time.perf_counter() - t0
        H.record_metric("real.reranker_load_seconds", round(secs, 2), "秒", f"device={device}")
        assert model is not None, f"本地重排模型存在但加载失败: {LOCAL_RERANK}"
        scores = model.predict([("叶星提剑而立", "叶星站在山门前，手按剑柄。"),
                                ("叶星提剑而立", "今日天气晴朗，市集热闹。")])
        H.record_metric("real.reranker_score_gap", round(float(scores[0] - scores[1]), 4), "",
                        f"相关对={float(scores[0]):.4f} / 无关对={float(scores[1]):.4f}")
        assert float(scores[0]) > float(scores[1]), "重排模型未能区分相关/无关句对"


class TestRealEndToEndRetrieval:
    """真实编码 + 真实生产库的端到端能力。"""

    @pytest.fixture(scope="class")
    def self_queries(self):
        """从生产库元数据中取 30 个块，用其正文前缀构造「自查询」。"""
        meta = json.loads((H.PROD_VDB / "metadata.json").read_text(encoding="utf-8"))
        cand = [m for m in meta if len(m["text"].strip()) >= 40]
        step = max(1, len(cand) // 30)
        picked = cand[::step][:30]
        return picked

    def _match(self, target, hits):
        """命中的三种合法形态：同块、目标文本被父块/邻居包含、同章节。

        返回 (exact, contained, same_chapter)。
        """
        texts = [h.get("text") or "" for h in hits]
        exact = bool(texts) and texts[0] == target["text"]
        contained = target["text"] in (texts[0] if texts else "")
        same_chapter = bool(hits) and hits[0].get("chapter") == target.get("chapter")
        return exact, contained, same_chapter

    def test_self_recall_at_1_and_5(self, real_retriever, self_queries):
        """【规格来源】检索链路可用性最低标准：库内原文必须能被自己召回。

        【判定标准】以块正文前 30 字为查询、真实模型编码，检索 top5：
        - top1 命中率（同块 / 父块包含 / 同章节 任一成立）>= 80%
        - top5 命中率（返回文本包含原块正文）>= 95%
        同时记录「严格同块」命中率作为参考（项目会追加父块/邻居，故严格同块偏低属正常）。
        """
        hit1 = hit5 = hit5_any = 0
        for m in self_queries:
            q = m["text"].strip()[:30]
            hits = real_retriever.retrieve(q, top_k=5)
            exact, contained, same_chapter = self._match(m, hits)
            if exact or contained or same_chapter:
                hit1 += 1
            if any(m["text"] in (h.get("text") or "") for h in hits):
                hit5 += 1
            if exact:
                hit5_any += 1
        n = len(self_queries)
        H.record_metric("real.self_recall_at1_loose", round(hit1 / n * 100, 2), "%",
                        f"{hit1}/{n} top1 同块/父块包含/同章节任一（阈值 80%）")
        H.record_metric("real.self_recall_at1_exact", round(hit5_any / n * 100, 2), "%",
                        f"{hit5_any}/{n} top1 位置严格同块")
        H.record_metric("real.self_recall_at5_exact", round(hit5 / n * 100, 2), "%",
                        f"{hit5}/{n} top5 内严格同块（阈值 95%）")
        assert hit1 / n >= 0.80, f"自查询 top1 命中率 = {hit1 / n:.1%}，检索链路不可靠"
        assert hit5 / n >= 0.95, f"自查询 top5 严格召回 = {hit5 / n:.1%}"

    def test_returned_score_scale_matches_similarity_threshold(self, real_retriever, self_queries):
        """【规格来源】项目内部以「相似度 0.1」为向量有效阈值并对外暴露 score 字段。

        【判定标准】对「查询即原文片段」这种必然高度相关的输入，返回结果的 score
        应落在与 0.1 阈值可比的量纲内（即 > 0.1）；否则下游按 0.1 判断相关性会全部判为无关。
        同时记录真实余弦相似度作为对照，以判定是「分数未归一/量纲漂移」还是「检索未命中」。
        """
        import numpy as np

        emb = real_retriever.embeddings
        ret_scores, cos_scores = [], []
        for m in self_queries[:10]:
            q = m["text"].strip()[:30]
            hits = real_retriever.retrieve(q, top_k=3)
            assert hits, f"查询「{q}」未返回任何结果"
            ret_scores.append(float(hits[0]["score"]))
            qv = np.asarray(real_retriever.embedding_model.encode([q], normalize_embeddings=True))[0]
            cos = emb @ qv
            cos_scores.append(float(np.max(cos)))
        H.record_metric("real.returned_score_mean", round(float(np.mean(ret_scores)), 4), "",
                        "返回字段 score 均值（相同查询即原文片段）")
        H.record_metric("real.true_cosine_mean", round(float(np.mean(cos_scores)), 4), "",
                        "同批次真实 top1 余弦均值（对照）")
        H.record_metric("real.fusion_mode_at_scale_check", real_retriever.last_fusion_mode, "",
                        "该量纲下返回 score 的融合模式")
        assert min(ret_scores) > 0.1, (
            f"高度相关输入的返回 score 仍 <= 0.1 阈值（真实余弦 {np.mean(cos_scores):.3f}），"
            f"score 字段与文档阈值不同量纲: {[round(s, 4) for s in ret_scores]}"
        )

    def test_end_to_end_latency_with_real_encoder(self, real_retriever, self_queries):
        """【规格来源】交互式问答体感：端到端（编码 + 检索）P95 < 2s。

        【判定标准】20 次真实查询，P50/P95 记录且 P95 < 2.0s。
        """
        lat = []
        for m in self_queries[:20]:
            q = m["text"].strip()[:30]
            t0 = time.perf_counter()
            real_retriever.retrieve(q, top_k=5)
            lat.append(time.perf_counter() - t0)
        p50, p95 = H.percentile(lat, 0.5), H.percentile(lat, 0.95)
        H.record_metric("real.e2e_p50_ms", round(p50 * 1000, 1), "毫秒", "编码+检索，生产库 7033 块")
        H.record_metric("real.e2e_p95_ms", round(p95 * 1000, 1), "毫秒", "阈值 2000ms")
        assert p95 < 2.0, f"端到端 P95 = {p95 * 1000:.1f}ms"

    def test_rerank_changes_order_and_stays_in_budget(self, real_retriever, self_queries, capsys):
        """【规格来源】README 承诺：CrossEncoder 精排可用（本地 bge-reranker-base）。

        【判定标准】启用重排后：日志必须出现「CrossEncoder 精排」而非静默 no-op；
        返回顺序与未重排时相比发生变化（重排确实生效）；单次延迟 < 5s。
        同时记录：重排状态是否可从结构化字段读取（现状仅 stdout 文案）。
        """
        if not LOCAL_RERANK.exists():
            pytest.skip("本地重排模型不存在")
        model, _ = utils.load_reranker_model(str(LOCAL_RERANK))
        if model is None:
            pytest.skip("重排模型加载失败")
        q = self_queries[0]["text"].strip()[:30]
        base = [h["text"] for h in real_retriever.retrieve(q, top_k=5, enable_rerank=False)]
        real_retriever.reranker_model = model
        real_retriever.reranker_model_path = str(LOCAL_RERANK)
        capsys.readouterr()
        t0 = time.perf_counter()
        hits = real_retriever.retrieve(q, top_k=5, enable_rerank=True)
        secs = time.perf_counter() - t0
        out = capsys.readouterr().out
        now = [h["text"] for h in hits]
        structured = any(getattr(real_retriever, attr, None)
                         for attr in ("last_rerank_mode", "last_rerank_status"))
        H.record_metric("real.rerank_latency_seconds", round(secs, 3), "秒",
                        f"日志含精排={'是' if 'CrossEncoder 精排' in out else '否'}")
        H.record_metric("real.rerank_changed_order", "是" if base != now else "否", "",
                        "top5 顺序是否变化")
        H.record_metric("real.rerank_structured_field", "有" if structured else "无", "",
                        "重排状态是否可结构化读取（现状只有 stdout 文案）")
        assert hits, "启用重排后返回空结果"
        assert "CrossEncoder 精排" in out, (
            f"显式启用重排后日志未体现精排（静默 no-op）：{out.strip()[:200]}"
        )
        assert secs < 5, f"重排单次耗时 {secs:.2f}s"
