# -*- coding: utf-8 -*-
"""
tests/test_commercial_negative_recovery.py — 商用验收 [第 2 组：负向输入与错误回退]

设计原则：
1) 负向优先：先构造「不该沉默」的异常输入，观察是否被检测、是否可观测；
2) 判定标准来自规格/商用预期（明确的失败信号、可机读的状态、不静默吞错），
   不来自当前实现的选择；失败即缺陷证据；
3) 全部离线（本地 HTTP 桩 + tmp_path），只读真实项目数据。
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

import _nr_commercial as H

ROOT = H.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(H.TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(H.TESTS_DIR))

import api_client  # noqa: E402
import step2_split_embed as step2  # noqa: E402
import utils  # noqa: E402
from api_client import APIClient, APIClientError  # noqa: E402


# ==================== 本地 HTTP 桩 ====================

class _CountingHandler(BaseHTTPRequestHandler):
    """按预设状态码返回、并记录请求次数。"""

    status_code = 500
    body = {"error": "stub"}
    hits = 0

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        type(self).hits += 1
        payload = json.dumps(self.body).encode("utf-8")
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # 静音
        pass


@pytest.fixture
def http_stub():
    """启动本地 HTTP 桩，用法：stub.status = 401 / 500 / …，stub.hits。"""
    class _Stub:
        def __init__(self):
            self._server = None
            self._thread = None
            self.port = 0

        def start(self, status_code: int, body=None):
            handler = type("H", (_CountingHandler,), {
                "status_code": status_code,
                "body": body or {"error": f"stub-{status_code}"},
                "hits": 0,
            })
            self._server = HTTPServer(("127.0.0.1", 0), handler)
            self.port = self._server.server_address[1]
            self._handler = handler
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            return self

        @property
        def hits(self):
            return self._handler.hits

        @property
        def url(self):
            return f"http://127.0.0.1:{self.port}/v1/chat/completions"

        def stop(self):
            if self._server:
                self._server.shutdown()
                self._server.server_close()

    stub = _Stub()
    yield stub
    stub.stop()


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """把退避 sleep 置零，避免测试等待。"""
    monkeypatch.setattr(api_client.time, "sleep", lambda *a, **k: None)


# ============================================================
# 一、API 错误回退矩阵
# ============================================================

class TestApiErrorRecovery:
    """API 层：可重试 / 不可重试 / 耗尽后的失败信号。"""

    def test_4xx_is_not_retried(self, http_stub):
        """【规格来源】api_client 文档注释：「4xx 不重试，直接抛 APIClientError」。

        【判定标准】401 只请求 1 次，抛 APIClientError。
        """
        http_stub.start(401)
        c = APIClient(http_stub.url, "sk-test", "m", max_retries=2, timeout=5)
        with pytest.raises(APIClientError):
            c.call_api("hi")
        H.record_metric("recover.http_401_requests", http_stub.hits, "次", "期望 1")
        assert http_stub.hits == 1, f"401 被重试了 {http_stub.hits} 次"

    def test_5xx_is_retried_then_raises(self, http_stub):
        """【规格来源】同上：「网络/429/5xx 指数退避重试」。

        【判定标准】5xx 请求次数 == max_retries + 1，最终抛错。
        """
        http_stub.start(500)
        c = APIClient(http_stub.url, "sk-test", "m", max_retries=2, timeout=5)
        with pytest.raises(APIClientError):
            c.call_api("hi")
        H.record_metric("recover.http_500_requests", http_stub.hits, "次", "max_retries=2 → 期望 3")
        assert http_stub.hits == 3, f"5xx 请求 {http_stub.hits} 次，与 max_retries+1=3 不符"

    def test_429_is_retried(self, http_stub):
        """【规格来源】同上：429 属可重试（限流）。

        【判定标准】429 请求次数 == max_retries + 1。
        """
        http_stub.start(429)
        c = APIClient(http_stub.url, "sk-test", "m", max_retries=2, timeout=5)
        with pytest.raises(APIClientError):
            c.call_api("hi")
        H.record_metric("recover.http_429_requests", http_stub.hits, "次", "期望 3")
        assert http_stub.hits == 3

    def test_unreachable_host_fails_with_clear_error(self):
        """【规格来源】商用预期：网络不可达必须在有限时间内抛出可识别错误。

        【判定标准】连接被拒时抛 APIClientError（而非裸 ConnectionError）。
        """
        c = APIClient("http://127.0.0.1:9/v1/chat/completions", "sk-test", "m",
                      max_retries=1, timeout=2)
        with pytest.raises(APIClientError) as ei:
            c.call_api("hi")
        H.record_metric("recover.unreachable_error_type", type(ei.value).__name__, "",
                        str(ei.value)[:80])

    def test_timeout_is_honored_and_reported(self):
        """【规格来源】商用预期：超时必须真实生效。

        【判定标准】对不可路由地址（10.255.255.1）设置 2s 超时，
        整体耗时 <= 15s（含 1 次重试），且抛 APIClientError。
        """
        import time as _t
        c = APIClient("http://10.255.255.1:81/v1/chat/completions", "sk-test", "m",
                      max_retries=1, timeout=2)
        t0 = _t.perf_counter()
        with pytest.raises(APIClientError):
            c.call_api("hi")
        elapsed = _t.perf_counter() - t0
        H.record_metric("recover.timeout_elapsed", round(elapsed, 2), "秒", "timeout=2, retries=1")
        assert elapsed <= 15, f"超时回退耗时 {elapsed:.1f}s，远超设定超时"

    def test_empty_choice_raises_not_silent(self, http_stub):
        """【规格来源】_extract_reply 文档：空内容抛 APIClientError（不返回空串）。

        【判定标准】HTTP 200 但 choices 为空 → 抛错。
        """
        http_stub.start(200, {"choices": []})
        c = APIClient(http_stub.url, "sk-test", "m", max_retries=0, timeout=5)
        with pytest.raises(APIClientError):
            c.call_api("hi")

    def test_test_connection_never_raises(self, http_stub):
        """【规格来源】test_connection 文档：返回 (bool, msg)，不抛异常。

        【判定标准】4xx 与网络失败两种情形均返回 (False, 非空说明)。
        """
        http_stub.start(401)
        c = APIClient(http_stub.url, "sk-test", "m", max_retries=0, timeout=5)
        ok, msg = c.test_connection()
        assert ok is False and msg, f"401 下 test_connection 返回 {(ok, msg)}"
        c2 = APIClient("http://127.0.0.1:9/v1/chat/completions", "sk-test", "m",
                       max_retries=0, timeout=2)
        ok2, msg2 = c2.test_connection()
        assert ok2 is False and msg2, f"不可达下 test_connection 返回 {(ok2, msg2)}"


# ============================================================
# 二、向量库损坏 / 缺失（不得静默降级）
# ============================================================

class TestIndexIntegrityRecovery:
    """损坏与缺失的向量库：必须可检测、可观测。"""

    def test_missing_index_is_visible_to_user(self, tmp_path, capsys):
        """【规格来源】商用预期：索引未构建时用户必须能看懂原因。

        【判定标准】stdout 出现明确提示（路径不存在 / 无法检索），且返回空结果。
        """
        r = H.make_retriever(tmp_path / "not_built",
                             embedding_model=H.FakeEmbeddingModel(dim=16))
        hits = r.retrieve("叶星", top_k=3)
        out = capsys.readouterr().out
        H.record_metric("recover.missing_index_user_notice",
                        "有" if "不存在" in out else "无", "",
                        out.strip().replace("\n", " / ")[:120])
        assert hits == []
        assert "不存在" in out or "未加载文档" in out, "索引缺失时用户看不到任何原因"

    def test_missing_index_sets_machine_readable_status(self, tmp_path):
        """【规格来源】项目自身「降级必须可观测」设计（last_vector_status 字段）。

        【判定标准】向量库目录不存在时，检索器必须通过结构化状态暴露「索引缺失」，
        而不是仅打印日志后返回空列表——否则调用方无法区分「索引没建」与
        「确实没有相关内容」。
        """
        r = H.make_retriever(tmp_path / "not_built",
                             embedding_model=H.FakeEmbeddingModel(dim=16))
        r.retrieve("叶星", top_k=3)
        status = (r.last_vector_status or "") + (r.last_fusion_mode or "")
        H.record_metric("recover.missing_index_status", status or "<空>", "",
                        "向量库目录不存在时的结构化状态")
        assert any(w in status for w in ("缺失", "不存在", "未构建", "无向量库")), (
            f"向量库缺失时无结构化失败信号（status={status!r}），调用方无法机读区分"
        )

    def test_dimension_mismatch_between_npy_and_metadata(self, tmp_path, capsys):
        """【规格来源】索引自洽性：embeddings.npy 行数必须等于 metadata 条数。

        【判定标准】行数与元数据条数不一致时必须给出可观测告警，
        且不得静默丢弃 / 错配向量。
        """
        vd = tmp_path / "bad"
        vd.mkdir()
        np.save(vd / "embeddings.npy", np.random.rand(5, 8).astype(np.float32))
        (vd / "metadata.json").write_text(json.dumps(
            [{"text": "叶星", "chapter": "第1章"}, {"text": "李四", "chapter": "第2章"}]),
            encoding="utf-8")
        r = H.make_retriever(vd, embedding_model=H.FakeEmbeddingModel(dim=8))
        r.retrieve("叶星", top_k=2)
        out = capsys.readouterr().out
        shape = None if r.embeddings is None else r.embeddings.shape
        H.record_metric("recover.count_mismatch_result_shape", str(shape), "",
                        f"npy 5 行 vs metadata 2 条；用户提示={'有' if '不一致' in out else '无'}")
        assert any(w in out for w in ("不一致", "不匹配", "错位", "跳过")), (
            f"npy 行数与 metadata 条数不一致（5 vs 2）却无任何告警，"
            f"多余向量被静默丢弃（实际矩阵形状 {shape}）"
        )

    def test_corrupt_metadata_json_is_reported(self, tmp_path, capsys):
        """【规格来源】异常元数据回退：坏 JSON 必须被识别且不可静默。

        【判定标准】用户可见明确失败提示 + 结构化状态非空。
        """
        vd = tmp_path / "corrupt"
        vd.mkdir()
        np.save(vd / "embeddings.npy", np.random.rand(2, 8).astype(np.float32))
        (vd / "metadata.json").write_text("{ 这不是合法 JSON", encoding="utf-8")
        r = H.make_retriever(vd, embedding_model=H.FakeEmbeddingModel(dim=8))
        hits = r.retrieve("叶星", top_k=2)
        out = capsys.readouterr().out
        status = (r.last_vector_status or "") + (r.last_fusion_mode or "")
        H.record_metric("recover.corrupt_metadata_user_notice",
                        "有" if "失败" in out else "无", "",
                        out.strip().replace("\n", " / ")[:120])
        assert hits == []
        assert any(w in out for w in ("失败", "损坏", "解析")), (
            f"metadata.json 损坏时用户看不到失败原因：{out[:200]}"
        )
        assert status, "metadata.json 损坏但结构化状态为空（调用方无法机读）"

    def test_nan_vectors_do_not_break_retrieval(self, tmp_path):
        """【规格来源】数值稳健性：库内向量含 NaN/Inf 时不得抛异常或返回 NaN 分数。

        【判定标准】检索正常返回，且所有 score 为有限数。
        """
        vecs = np.random.rand(4, 8).astype(np.float32)
        vecs[1, 0] = np.nan
        vecs[2, 1] = np.inf
        r = H.make_retriever(
            H.write_library(tmp_path / "nanvdb",
                            ["叶星走在路上。" * 3, "李四看着河水。" * 3,
                             "王五在山上。" * 3, "赵六在城中。" * 3], vectors=vecs),
            embedding_model=H.FakeEmbeddingModel(dim=8))
        hits = r.retrieve("叶星", top_k=4)
        bad = [h["score"] for h in hits if not np.isfinite(h["score"])]
        H.record_metric("recover.nan_scores_returned", len(bad), "条",
                        f"库内含 NaN/Inf 行的检索返回 {len(hits)} 条")
        assert not bad, f"返回了非有限相似度: {bad}"

    def test_scores_are_monotonic_non_increasing(self, tmp_path):
        """【规格来源】排序契约：相似度阈值 0.1 之上的候选必须按分值降序输出。

        【判定标准】score 序列单调不增。
        """
        fake = H.FakeEmbeddingModel(dim=16)
        texts = [f"第{i}块叶星在山门修炼。" * 4 for i in range(20)]
        r = H.make_retriever(
            H.write_library(tmp_path / "mono", texts, vectors=fake.encode(texts, normalize_embeddings=True)),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        hits = r.retrieve("叶星山门修炼", top_k=8)
        scores = [h["score"] for h in hits]
        H.record_metric("recover.score_monotonic", "是" if all(
            scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)) else "否", "",
            f"top8 分值序列={[round(s, 4) for s in scores]}")
        assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)), \
            f"分值未单调下降: {scores}"


# ============================================================
# 三、输入边界
# ============================================================

class TestInputBoundaryRecovery:
    """极端输入不得导致崩溃；失败必须显式。"""

    @pytest.mark.parametrize("bad", ["", " ", "\n\n", "。", "啊"])
    def test_degenerate_chunking_inputs(self, bad):
        """【规格来源】切片健壮性：空/极短/纯空白输入不得崩溃。

        【判定标准】不抛异常，返回 list（允许为空）。
        """
        chunks = utils.split_text_recursive(bad, min_chars=210, max_chars=672)
        assert isinstance(chunks, list)

    def test_pathological_long_text_without_punctuation(self):
        """【规格来源】切片硬上限：10 万字符无标点不得死循环/超时。

        【判定标准】5s 内完成，块数 > 100，每块 <= 672。
        """
        text = "啊" * 100_000
        chunks, secs = H.timed(utils.split_text_recursive, text,
                               min_chars=210, max_chars=672)
        H.record_metric("capacity.no_punct_100k_seconds", round(secs, 3), "秒",
                        f"10 万字符 → {len(chunks)} 块")
        assert secs < 5, f"10 万字符硬切耗时 {secs:.2f}s"
        assert len(chunks) > 100
        assert max(len(c) for c in chunks) <= 672

    def test_none_query_is_explicit_failure(self, tmp_path):
        """【规格来源】参数契约：None 查询属调用方错误，应显式报错（TypeError/ValueError）
        或返回空；不得半途崩溃于无关位置。

        【判定标准】retrieve(None) 抛 TypeError/ValueError 或返回 []。
        """
        r = H.make_retriever(H.write_library(tmp_path / "vdb", ["叶星。"]),
                             embedding_model=H.FakeEmbeddingModel(dim=16))
        try:
            got = r.retrieve(None, top_k=2)
        except (TypeError, ValueError):
            return
        assert got == [], f"retrieve(None) 返回 {got!r}（既非空也非显式错误）"

    def test_whitespace_only_corpus(self, tmp_path, monkeypatch):
        """【规格来源】建库健壮性：纯空白语料不得产出「假成功」。

        【判定标准】要么返回失败，要么返回 True 且产出的块数为 0 且元数据齐全。
        """
        ok, out, _ = H.build_tiny_index(tmp_path, "   \n\t  ", monkeypatch, name="ws")
        meta_path = out / "metadata.json"
        H.record_metric("recover.whitespace_corpus_ok", ok, "", "纯空白语料建库返回")
        if ok:
            assert meta_path.exists(), "返回成功但未落盘 metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert len(meta) == 0, f"纯空白语料产出了 {len(meta)} 个块"


# ============================================================
# 四、配置与路径安全
# ============================================================

class TestConfigAndPathSafety:
    """配置损坏回退 + 路径穿越防护。"""

    @pytest.mark.parametrize("name,content", [("empty.json", ""), ("broken.json", "{oops")])
    def test_corrupt_config_json_falls_back_to_defaults(self, tmp_path, name, content):
        """【规格来源】ConfigManager.load 文档契约：「config.json 不存在 → 返回默认配置」，
        对空文件 / 非法 JSON 同理。

        【判定标准】load() 返回 dict 且含检索必需键（top_k / models）。
        """
        from config_manager import ConfigManager
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        cm = ConfigManager.__new__(ConfigManager)
        cm.config_file = p
        cfg = cm.load()
        assert isinstance(cfg, dict), f"{name} 加载结果不是 dict: {type(cfg).__name__}"
        assert "top_k" in cfg and "models" in cfg, f"{name} 未回退到默认配置键"

    def test_config_list_top_level_is_rejected_clearly(self, tmp_path, monkeypatch):
        """【规格来源】配置契约：顶层必须是对象（load 的返回类型标注为 dict）。

        【判定标准】顶层为数组时 load() 必须返回 dict（规范化或回退默认），
        不得把 list 原样返回给调用方。
        """
        from config_manager import ConfigManager
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        cm = ConfigManager.__new__(ConfigManager)
        cm.config_file = p
        cfg = cm.load()
        H.record_metric("recover.config_list_top_level_type", type(cfg).__name__, "",
                        "顶层为数组时 load() 的返回类型")
        assert isinstance(cfg, dict), (
            f"顶层为数组时 load() 返回 {type(cfg).__name__}，与 dict 契约不符"
        )

    def test_config_list_top_level_does_not_crash_with_api_key(self, tmp_path, monkeypatch):
        """【规格来源】异常配置下的可用性：手改坏配置不得导致启动崩溃。

        【判定标准】顶层为数组 + 存在 DEEPSEEK_API_KEY 时 load() 依然返回 dict。
        """
        from config_manager import ConfigManager
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")
        p = tmp_path / "list_key.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        cm = ConfigManager.__new__(ConfigManager)
        cm.config_file = p
        try:
            cfg = cm.load()
        except Exception as e:  # noqa: BLE001
            H.record_metric("recover.config_list_with_key_exception", type(e).__name__, "",
                            str(e)[:80])
            raise AssertionError(
                f"顶层为数组 + 有环境变量 API Key 时 load() 直接抛 {type(e).__name__}: {e}"
            ) from e
        assert isinstance(cfg, dict)

    def test_project_name_traversal_does_not_escape_data_dir(self, tmp_path, monkeypatch):
        """【规格来源】安全基线：项目名不得逃逸 data/ 目录（路径穿越防护）。

        【判定标准】带 `../` 的项目名必须被拒绝或规范化到 data/ 内（POC：不得在
        data/ 之外创建目录）。
        """
        import utils as u
        p = u.get_project_dir("..\\escape_poc")
        resolved = Path(p).resolve()
        data_root = (ROOT / "data").resolve()
        H.record_metric("recover.traversal_project_dir", str(resolved), "",
                        f"data 根目录 = {data_root}")
        assert data_root in resolved.parents or resolved == data_root, (
            f"项目名含 ../ 时生成的目录逃逸出 data/：{resolved}"
        )

    def test_demo_qa_is_consistent_with_demo_corpus(self):
        """【规格来源】内置 demo 自洽性：QA 的「预期章节」必须存在于 demo 语料。

        【判定标准】每条 demo QA 的预期章节都能在 demo 向量库章节集中找到。
        """
        from novel_rag import DEMO_QA
        meta = json.loads((H.DEMO_VDB / "metadata.json").read_text(encoding="utf-8"))
        chapters = {(m.get("chapter") or m.get("file") or "") for m in meta}
        missing = [q["chapter"] for q in DEMO_QA
                   if not any(c[:4] == q["chapter"][:4] for c in chapters)]
        H.record_metric("recover.demo_qa_chapters_missing", len(missing), "条",
                        f"demo 章节集 {sorted(chapters)}")
        assert not missing, f"demo QA 引用了不存在的章节: {missing}"


# ============================================================
# 五、建库契约（P0）
# ============================================================

class TestBuildIndexContract:
    """建库返回值的可信度：True 必须意味着「向量库真的可用」。"""

    def test_true_return_implies_embeddings_written(self, tmp_path, monkeypatch):
        """【规格来源】历史缺陷（建库只写 metadata.json、状态假 ready）。

        【判定标准】build_vector_index 返回 True 时，embeddings.npy 必须存在且
        行数 == metadata 条数。
        """
        ok, out, _ = H.build_tiny_index(tmp_path, "叶星觉醒血脉。" * 50, monkeypatch,
                                        name="contract")
        assert ok is True
        npy = out / "embeddings.npy"
        assert npy.exists(), "返回 True 但未落盘 embeddings.npy（契约不可信）"
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        mat = np.load(npy)
        H.record_metric("build.contract_shape", f"{mat.shape} vs meta={len(meta)}", "",
                        "建库成功契约")
        assert mat.shape[0] == len(meta), "向量行数与元数据条数不一致"

    def test_build_failure_returns_false_when_model_missing(self, tmp_path, monkeypatch):
        """【规格来源】历史缺陷（缺 torch 时仍返回 True → 状态假 ready）。

        【判定标准】嵌入模型加载失败时必须返回 False（或抛错），不得返回 True。
        """
        import step2_split_embed as s2
        monkeypatch.setattr(s2, "load_embedding_model",
                            lambda *a, **k: (None, None))
        out = tmp_path / "failvdb"
        ok = s2.build_vector_index(text="叶星觉醒血脉。" * 30, output_dir=str(out),
                                   embedding_model_path="nonexistent-model",
                                   book_id="b", book_title="测试书")
        H.record_metric("build.model_missing_return", ok, "", "模型加载失败时返回值")
        assert ok is False, f"嵌入模型加载失败，建库仍返回 {ok!r}（会污染项目状态）"

    def test_incremental_rebuild_reuses_vectors(self, tmp_path, monkeypatch):
        """【规格来源】README：增量重建「未变章节向量复用」。

        【判定标准】文本不变时二次建库的嵌入编码次数显著下降
        （复用 >= 50% 块）。
        """
        text = "".join(f"第{i}章 叶星在山门修炼剑法，日夜不辍。\n" for i in range(20)) * 30
        fake = H.install_fake_embedding(monkeypatch, dim=16)
        out = tmp_path / "inc"
        assert step2.build_vector_index(text=text, output_dir=str(out),
                                        embedding_model_path="fake", book_id="b",
                                        book_title="测试书") is True
        first = sum(fake.encode_calls)
        fake.encode_calls.clear()
        assert step2.build_vector_index(text=text, output_dir=str(out),
                                        embedding_model_path="fake", book_id="b",
                                        book_title="测试书") is True
        second = sum(fake.encode_calls)
        H.record_metric("build.incremental_encode_calls", f"{first} -> {second}", "次",
                        "文本未变时的二次建库编码量")
        assert second < first * 0.5, (
            f"文本未变二次建库仍需编码 {second} 条（首次 {first} 条），增量复用未生效"
        )
