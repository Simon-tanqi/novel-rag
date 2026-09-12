# -*- coding: utf-8 -*-
"""
检索前查询改写（query rewrite）测试

覆盖任务要求的三条验证路径：
1. 元词触发改写：问「主角叫什么名字？」→ 触发一次 LLM 调用 → 得到实体名 + 改写查询
2. 非元词不触发：问「陈曦第一次遇见苏瑶在哪一章？」→ 零额外 LLM 调用
3. API 失败降级：网络 / 解析异常 → 静默回退原查询，不抛错、不影响问答链路

另含：
- 缺陷回归：单查询（原查询）召回不到主角登场片段 → 改写后可通过多路 RRF 召回
- retrieve_multi 与 retrieve 在「无改写查询」时结果等价（不破坏原检索行为）
- 多路 RRF 融合的分数累加语义
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# 无 GUI 依赖的环境下给 main / settings_window 的导入兜底
# （与 tests/test_hybrid_rerank.py 同款做法，纯逻辑测试不需要真实 GUI）
try:
    import customtkinter  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("customtkinter", MagicMock())
    sys.modules.setdefault("tkinter", MagicMock())
    sys.modules.setdefault("tkinter.messagebox", MagicMock())
    sys.modules.setdefault("tkinter.filedialog", MagicMock())

from api_client import APIClient, APIClientError
from query_rewriter import (
    RewriteResult,
    build_rewrite_prompt,
    detect_meta_terms,
    has_meta_terms,
    parse_rewrite_response,
    rewrite_query,
)
from rag_retriever import RAGRetriever

META_Q = "主角叫什么名字？"
REWRITTEN_Q = "陈曦的姓名"
REWRITE_JSON = '{"entities": ["陈曦"], "queries": ["陈曦的姓名", "陈曦是谁"]}'


# ===================== 测试替身 =====================


class FakeAPIClient:
    """记录调用参数的假 APIClient"""

    def __init__(self, reply: str = "", error: Exception = None):
        self.reply = reply
        self.error = error
        self.calls = []

    def call_api(self, message, enable_thinking=False, max_retries=None,
                 temperature=0.1, max_tokens=None):
        self.calls.append({
            "message": message,
            "enable_thinking": enable_thinking,
            "max_retries": max_retries,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.error is not None:
            raise self.error
        return self.reply


class FakeConfig:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def get_enable_query_rewrite(self) -> bool:
        return self.enabled


class FakeStatusLabel:
    def __init__(self):
        self.texts = []

    def configure(self, **kwargs):
        self.texts.append(kwargs.get("text", ""))


def _make_app(config=None, project_name="测试小说"):
    """构造一个不初始化 GUI 的 NovelRAGApp，仅用于测试 _maybe_rewrite_query"""
    import main as main_mod

    app = main_mod.NovelRAGApp.__new__(main_mod.NovelRAGApp)
    app.config_manager = config or FakeConfig()
    app.current_project = {"name": project_name} if project_name else None
    app.status_label = FakeStatusLabel()
    app.after = lambda delay=0, func=None: func() if func else None
    return main_mod, app


# ===================== 1. 元词识别 =====================


@pytest.mark.parametrize("question", [
    "主角叫什么名字？",
    "男主角是谁",
    "女主的身世是什么",
    "主人公最后怎么样了",
    "男一号和女一号是什么关系",
    "主角团都有谁",
])
def test_meta_terms_detected(question):
    assert has_meta_terms(question) is True
    assert detect_meta_terms(question)


@pytest.mark.parametrize("question", [
    "陈曦第一次遇见苏瑶是在哪一章？",
    "他和师父的对话内容是什么",
    "第三章讲了什么",
    "这本书大概多少字",
    "",
])
def test_non_meta_terms_not_detected(question):
    assert has_meta_terms(question) is False
    assert detect_meta_terms(question) == []


def test_meta_terms_longest_first_no_duplicate():
    """「主角团」命中后不再重复计「主角」"""
    assert detect_meta_terms("主角团都有谁") == ["主角团"]


# ===================== 2. 改写调用（路由触发 + 降级） =====================


def test_rewrite_query_triggers_on_meta_term():
    client = FakeAPIClient(reply=REWRITE_JSON)
    result = rewrite_query(META_Q, client, novel_name="测试小说")

    assert result.triggered is True
    assert result.succeeded is True
    assert result.entities == ["陈曦"]
    assert result.queries == ["陈曦的姓名", "陈曦是谁"]
    assert result.extra_queries == ["陈曦的姓名", "陈曦是谁"]
    assert result.summary == "查询改写：主角 → 陈曦"

    # 短调用：temperature=0 / max_tokens=150 / 不重试 / 不带思考
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 150
    assert call["max_retries"] == 0
    assert call["enable_thinking"] is False


def test_rewrite_query_skips_llm_for_non_meta_term():
    """非元词问题：零额外 LLM 调用"""
    client = FakeAPIClient(reply=REWRITE_JSON)
    result = rewrite_query("陈曦第一次遇见苏瑶是在哪一章？", client)

    assert result.triggered is False
    assert result.succeeded is False
    assert result.extra_queries == []
    assert result.summary == ""
    assert client.calls == []


def test_rewrite_query_degrades_on_api_error():
    """网络异常 → 静默降级为原查询（不抛错）"""
    client = FakeAPIClient(error=APIClientError("模拟网络错误"))
    result = rewrite_query(META_Q, client, max_retries=0)

    assert result.triggered is True
    assert result.succeeded is False
    assert result.extra_queries == []
    assert result.error
    assert client.calls  # 确实尝试过调用


@pytest.mark.parametrize("error", [
    TimeoutError("timeout"),
    ConnectionError("connection reset"),
    TypeError("call_api() got an unexpected keyword argument 'max_tokens'"),
])
def test_rewrite_query_degrades_on_unexpected_exception(error):
    client = FakeAPIClient(error=error)
    result = rewrite_query(META_Q, client)
    assert result.succeeded is False
    assert result.extra_queries == []


def test_rewrite_query_degrades_on_bad_reply():
    """模型返回非 JSON → 解析失败 → 降级"""
    client = FakeAPIClient(reply="抱歉，我无法确定主角的名字。")
    result = rewrite_query(META_Q, client)
    assert result.triggered is True
    assert result.succeeded is False
    assert result.extra_queries == []
    assert "JSON" in result.error


def test_rewrite_query_without_api_client():
    result = rewrite_query(META_Q, None)
    assert result.succeeded is False
    assert result.extra_queries == []


def test_rewrite_prompt_contains_question_and_meta_terms():
    prompt = build_rewrite_prompt(META_Q, ["主角"], novel_name="斗罗大陆")
    assert META_Q in prompt
    assert "主角" in prompt
    assert "斗罗大陆" in prompt


# ===================== 3. 响应解析 =====================


def test_parse_tolerates_code_fence_and_extra_text():
    raw = ('好的，结果如下：\n```json\n'
           '{"entities": ["陈曦"], "queries": ["陈曦的姓名"]}\n```\n以上。')
    result = parse_rewrite_response(raw, ["主角"], original_query=META_Q)
    assert result.succeeded is True
    assert result.entities == ["陈曦"]
    assert result.queries == ["陈曦的姓名"]


def test_parse_filters_meta_words_in_result():
    """改写结果里仍含元词（等于没改写）→ 视为失败，回退原查询"""
    raw = '{"entities": ["主角"], "queries": ["主角的姓名"]}'
    result = parse_rewrite_response(raw, ["主角"], original_query=META_Q)
    assert result.succeeded is False
    assert result.extra_queries == []


def test_parse_filters_duplicate_of_original_query():
    raw = '{"entities": [], "queries": ["主角叫什么名字？"]}'
    result = parse_rewrite_response(raw, ["主角"], original_query=META_Q)
    assert result.succeeded is False


def test_parse_invalid_json():
    result = parse_rewrite_response("不是 JSON", ["主角"], original_query=META_Q)
    assert result.succeeded is False
    assert result.error


def test_rewrite_result_to_dict():
    result = RewriteResult(triggered=True, entities=["陈曦"], queries=["陈曦是谁"],
                           meta_terms=["主角"])
    data = result.to_dict()
    assert data["succeeded"] is True
    assert data["summary"] == "查询改写：主角 → 陈曦"
    assert data["entities"] == ["陈曦"]


# ===================== 4. APIClient 参数透传 =====================


def test_api_client_request_body_carries_rewrite_params():
    client = APIClient("http://127.0.0.1:1/v1/chat/completions", "key", "deepseek-chat")

    body = client._build_request("hi", False, temperature=0.0, max_tokens=150)
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 150
    assert "thinking" not in body

    # 默认行为不变（原有问答链路零影响）
    default_body = client._build_request("hi", False)
    assert default_body["temperature"] == 0.1
    assert "max_tokens" not in default_body


# ===================== 5. 多路检索（retrieve_multi） =====================


def _doc(cid, chapter, text="", score=0.5):
    return {
        "chunk_id": cid,
        "text": text or f"片段{cid}",
        "score": score,
        "chapter": chapter,
        "file": "f",
    }


def _make_retriever(docs, embeddings=None, embedding_model=None,
                    reranker_model=None, vec_ready=True):
    """绕过 __init__ 直接构造 RAGRetriever（不读磁盘 / 不加载模型）

    vec_ready=True 时伪造「嵌入模型 + 二维向量库」，让向量通道被判定为就绪；
    召回结果由 _install_fake_channels 注入，不涉及真实编码。
    """
    retriever = RAGRetriever.__new__(RAGRetriever)
    if vec_ready:
        if embedding_model is None:
            embedding_model = MagicMock()
        if embeddings is None:
            embeddings = np.zeros((max(len(docs), 1), 4), dtype=np.float32)
    retriever.vector_path = "/fake"
    retriever.reranker_model_path = ""
    retriever.embedding_model_path = ""
    retriever.vector_file = None
    retriever.metadata_file = None
    retriever.documents = docs
    retriever.texts = [d["text"] for d in docs]
    retriever.embeddings = embeddings
    retriever.reranker_model = reranker_model
    retriever.embedding_model = embedding_model
    retriever._embedding_loaded = True
    retriever._reranker_loaded = True
    retriever.last_vector_status = ""
    retriever.last_fusion_mode = ""
    retriever.last_retrieval_queries = []
    return retriever


def _install_fake_channels(retriever, per_query):
    """按查询串返回预设的召回结果：{query: (vec_list, kw_list)}"""
    def fake_vec(query, k):
        vec, _ = per_query.get(query, ([], []))
        return list(vec)

    def fake_kw(query, k):
        _, kw = per_query.get(query, ([], []))
        return list(kw)

    retriever._vector_retrieve = fake_vec
    retriever._keyword_retrieve = fake_kw


def test_merge_queries_dedupe_and_limit():
    assert RAGRetriever._merge_queries("q", None) == ["q"]
    assert RAGRetriever._merge_queries("q", []) == ["q"]
    merged = RAGRetriever._merge_queries(
        "q", ["q", " a ", "", None, 123, "b", "c", "d", "e"]
    )
    assert merged == ["q", "a", "b", "c"]  # 去重 + 去空 + 限流（默认最多 3 条改写）


def test_rrf_fusion_multi_accumulates_over_lists():
    d1 = _doc(1, "第一章")
    d2 = _doc(2, "第二章")
    fused = RAGRetriever._rrf_fusion_multi([[d1, d2], [d2]], k=60)
    # d2 = 1/62 + 1/61 > d1 = 1/61
    assert fused[0]["chunk_id"] == 2
    assert len(fused) == 2


def test_rrf_fusion_two_way_matches_multi():
    d1, d2 = _doc(1, "第一章"), _doc(2, "第二章")
    retriever = _make_retriever([d1, d2])
    two_way = retriever._rrf_fusion([d1], [d2], k=60)
    multi = RAGRetriever._rrf_fusion_multi([[d1], [d2]], k=60)
    assert [d["chunk_id"] for d in two_way] == [d["chunk_id"] for d in multi]


def test_single_query_cannot_recall_protagonist():
    """缺陷复现：只含元词的原查询排不进 top_k，召回的是无关配角对白"""
    docs = [
        _doc(9, "第九章", "陈山与林悦的对话", 0.9),
        _doc(8, "第八章", "无关配角对白", 0.8),
    ]
    retriever = _make_retriever(docs)
    _install_fake_channels(retriever, {
        META_Q: ([_doc(9, "第九章", "陈山与林悦的对话", 0.9),
                  _doc(8, "第八章", "无关配角对白", 0.8)], []),
    })

    hits = retriever.retrieve(META_Q, top_k=1)
    assert hits[0]["chunk_id"] == 9  # 召回的是无关片段 → 触发拒答


def test_multi_query_recalls_protagonist_after_rewrite():
    """改写后：主角登场片段（第一章）被多路 RRF 提升到第 1 位"""
    docs = [
        _doc(9, "第九章", "陈山与林悦的对话", 0.9),
        _doc(8, "第八章", "无关配角对白", 0.8),
        _doc(1, "第一章", "陈曦首次登场", 0.7),
        _doc(2, "第二章", "陈曦的背景介绍", 0.6),
    ]
    retriever = _make_retriever(docs)
    _install_fake_channels(retriever, {
        META_Q: ([_doc(9, "第九章", "陈山与林悦的对话", 0.9),
                  _doc(8, "第八章", "无关配角对白", 0.8)], []),
        REWRITTEN_Q: ([_doc(1, "第一章", "陈曦首次登场", 0.7),
                       _doc(2, "第二章", "陈曦的背景介绍", 0.6)],
                      [_doc(1, "第一章", "陈曦首次登场", 12.0)]),
    })

    hits = retriever.retrieve_multi(
        META_Q, extra_queries=[REWRITTEN_Q], top_k=1
    )
    assert hits[0]["chunk_id"] == 1
    assert retriever.last_retrieval_queries == [META_Q, REWRITTEN_Q]
    assert "多查询×2" in retriever.last_fusion_mode
    assert retriever.last_vector_status == ""


def test_retrieve_multi_without_extra_queries_equals_retrieve():
    """无改写查询时，多查询入口必须与原有单查询行为完全一致"""
    docs = [
        _doc(1, "第一章", "关键词命中A", 0.9),
        _doc(2, "第二章", "关键词命中A", 0.8),
        _doc(3, "第三章", "关键词命中A", 0.7),
    ]
    retriever = _make_retriever(docs)
    _install_fake_channels(retriever, {
        "苹果": ([_doc(1, "第一章", "关键词命中A", 0.9)],
                 [_doc(1, "第一章", "关键词命中A", 9.0)]),
    })

    single = retriever.retrieve("苹果", top_k=2)
    single_mode = retriever.last_fusion_mode
    multi = retriever.retrieve_multi("苹果", extra_queries=None, top_k=2)

    assert [h["chunk_id"] for h in single] == [h["chunk_id"] for h in multi]
    assert single_mode == "RRF（向量+关键词）"
    assert retriever.last_fusion_mode == "RRF（向量+关键词）"
    assert retriever.last_retrieval_queries == ["苹果"]


def test_retrieve_multi_keyword_only_fallback_keeps_status_semantics():
    """向量通道异常（未产出结果）时，仍走仅关键词模式并保留状态提示语义"""
    docs = [_doc(1, "第一章", "陈曦登场", 0.5)]
    retriever = _make_retriever(docs, vec_ready=False)
    _install_fake_channels(retriever, {
        META_Q: ([], [_doc(1, "第一章", "陈曦登场", 5.0)]),
    })

    hits = retriever.retrieve_multi(META_Q, extra_queries=[REWRITTEN_Q], top_k=1)
    assert hits[0]["chunk_id"] == 1
    assert "仅关键词" in retriever.last_fusion_mode
    assert "嵌入模型不可用" in retriever.last_vector_status


# ===================== 6. main.py 检索链路接入 =====================


def test_main_triggers_rewrite_on_meta_term(monkeypatch):
    main_mod, app = _make_app()
    client = FakeAPIClient(reply=REWRITE_JSON)
    monkeypatch.setattr(main_mod, "APIClient", lambda *a, **k: client)

    extra, note = app._maybe_rewrite_query(META_Q, "http://api", "key", "deepseek-chat")

    assert extra == ["陈曦的姓名", "陈曦是谁"]
    assert note == "查询改写：主角 → 陈曦"
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["max_tokens"] == 150


def test_main_skips_rewrite_for_non_meta_term(monkeypatch):
    main_mod, app = _make_app()
    client = FakeAPIClient(reply=REWRITE_JSON)
    monkeypatch.setattr(main_mod, "APIClient", lambda *a, **k: client)

    extra, note = app._maybe_rewrite_query(
        "陈曦第一次遇见苏瑶是在哪一章？", "http://api", "key", "deepseek-chat"
    )

    assert extra == []
    assert note == ""
    assert client.calls == []  # 零额外 LLM 调用


def test_main_skips_rewrite_when_switch_disabled(monkeypatch):
    main_mod, app = _make_app(config=FakeConfig(enabled=False))
    client = FakeAPIClient(reply=REWRITE_JSON)
    monkeypatch.setattr(main_mod, "APIClient", lambda *a, **k: client)

    extra, note = app._maybe_rewrite_query(META_Q, "http://api", "key", "deepseek-chat")

    assert (extra, note) == ([], "")
    assert client.calls == []


def test_main_falls_back_when_rewrite_api_fails(monkeypatch):
    main_mod, app = _make_app()
    client = FakeAPIClient(error=APIClientError("模拟网络异常"))
    monkeypatch.setattr(main_mod, "APIClient", lambda *a, **k: client)

    extra, note = app._maybe_rewrite_query(META_Q, "http://api", "key", "deepseek-chat")

    # 静默降级：不提示、不打断，等价于原查询检索
    assert extra == []
    assert note == ""
    assert app.status_label.texts  # 只更新过「正在改写查询…」状态


def test_main_falls_back_when_rewrite_result_unusable(monkeypatch):
    main_mod, app = _make_app()
    client = FakeAPIClient(reply='{"entities": [], "queries": []}')
    monkeypatch.setattr(main_mod, "APIClient", lambda *a, **k: client)

    extra, note = app._maybe_rewrite_query(META_Q, "http://api", "key", "deepseek-chat")

    assert (extra, note) == ([], "")


def test_main_survives_client_construction_failure(monkeypatch):
    """连 APIClient 构造都失败时也不能抛错"""
    main_mod, app = _make_app()

    def _boom(*args, **kwargs):
        raise RuntimeError("构造失败")

    monkeypatch.setattr(main_mod, "APIClient", _boom)

    extra, note = app._maybe_rewrite_query(META_Q, "http://api", "key", "deepseek-chat")
    assert (extra, note) == ([], "")
