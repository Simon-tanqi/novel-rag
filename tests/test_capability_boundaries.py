# -*- coding: utf-8 -*-
"""
tests/test_capability_boundaries.py — 能力边界测试（离线、快速、不依赖真实模型）

覆盖 7 类真实能力边界（均为「项目实际会遇到、但此前无测试覆盖」的场景）：

A. 空库 / 极小库        —— 空目录、仅有 metadata.json 无 embeddings.npy、单块库
B. 缺本地模型           —— 无嵌入模型建库、缺 torch 依赖自检、无重排模型
C. 无 API Key           —— CLI 前置校验退出、客户端构造不崩
D. 超长 / 超短文本      —— 空串、1 字、无标点超长文、极小文本建库
E. 增量重建             —— 全量复用、单章改动只重编码该章、维度变化全量重编码
F. 异常 metadata        —— JSON 损坏、非 list、缺 text 字段、向量/元数据行数不匹配
G. 资源缺失             —— 重排模型缺失时 enable_rerank 退化为无操作

设计原则：
1) 全部离线：嵌入模型用确定性替身（字符哈希袋向量），不加载 bge、不联网；
2) 断言「行为契约」而非实现细节：不崩、可降级、状态可观测、产物形状自洽；
3) 只读或写入 pytest tmp_path，绝不动项目 data/ 与真实向量库。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils  # noqa: E402
import step2_split_embed as step2  # noqa: E402
from rag_retriever import RAGRetriever  # noqa: E402


# ==================== 公共替身与工具 ====================

class FakeEmbeddingModel:
    """确定性嵌入替身：字符哈希袋 → dim 维向量（同文本必同向量）

    兼容 step2 的真实调用签名：
        encode(list[str], normalize_embeddings=True, show_progress_bar=False)
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encode_calls: list = []      # 记录每次 encode 的文本条数（用于验证复用）

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for ch in text:
            v[ord(ch) % self.dim] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    def encode(self, texts, normalize_embeddings=False,
               show_progress_bar=False, **kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        self.encode_calls.append(len(texts))
        arr = np.stack([self._vec(t) for t in texts]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr[0] if single else arr

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def _make_retriever(vector_path, embedding_model=None, reranker_model=None):
    """绕过 __init__ 的重模型加载，直接构造检索器（保证测试离线且快）"""
    r = RAGRetriever.__new__(RAGRetriever)
    r.vector_path = str(vector_path)
    r.reranker_model_path = None
    r.embedding_model_path = None
    r.vector_file = None
    r.metadata_file = None
    r.documents = []
    r.parent_chunks = {}
    r.embeddings = None
    r.texts = []
    r.reranker_model = reranker_model
    r.embedding_model = embedding_model
    r._embedding_loaded = True
    r._reranker_loaded = True
    r.last_vector_status = ""
    r.last_fusion_mode = ""
    r.last_retrieval_queries = []
    r._last_vector_error = ""
    r.load_documents()
    return r


def _write_library(vd: Path, texts: list, vectors: np.ndarray = None,
                   metadata: list = None, extra: dict = None):
    """在 tmp 目录构造一个最小向量库（默认写入标准 embeddings.npy + metadata.json）"""
    vd.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = [
            {"text": t, "chapter": f"第{i + 1}章", "chunk_id": i}
            for i, t in enumerate(texts)
        ]
    if vectors is not None:
        np.save(vd / "embeddings.npy", vectors.astype(np.float32))
    (vd / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    for name, obj in (extra or {}).items():
        (vd / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return vd


def _chapter_text(n_chapters: int = 3, paras: int = 4, para_chars: int = 120) -> str:
    """构造带章节结构的测试原文（章节标题可被 chunking 识别）"""
    lines = []
    for c in range(1, n_chapters + 1):
        lines.append(f"第{c}章 试炼{c}")
        for p in range(paras):
            body = ("叶星挥剑而前，风雷激荡，山石崩裂。" * (para_chars // 16 + 1))[:para_chars]
            lines.append(f"　　第{c}章第{p}段落：{body}")
        lines.append("")
    return "\n".join(lines)


# ==================== A. 空库 / 极小库 ====================

def test_empty_dir_no_crash(tmp_path):
    """空向量库目录：不抛异常，退化为 0 文档，检索返回空列表"""
    vd = tmp_path / "vector_db"
    vd.mkdir()
    r = _make_retriever(vd, embedding_model=FakeEmbeddingModel())

    assert r.documents == []
    assert r.embeddings is None
    assert r.retrieve("叶星是谁") == []


def test_metadata_only_library_falls_back_to_keyword(tmp_path):
    """只有 metadata.json、无 embeddings.npy（用户截图场景）：可观测降级为关键词模式"""
    vd = _write_library(tmp_path / "vector_db", ["叶星是主角", "苏浅是女主"])
    r = _make_retriever(vd, embedding_model=FakeEmbeddingModel())

    assert r.embeddings is None
    results = r.retrieve("叶星")
    assert "向量库不可用" in r.last_vector_status
    assert "仅关键词" in r.last_fusion_mode
    assert any("叶星" in item["text"] for item in results)


def test_single_chunk_library_vector_hit(tmp_path):
    """极小库（1 块）：向量召回可用，且分数为归一化余弦（≈1.0）"""
    text = "叶星是《绝世主宰》的主角，出身微末。"
    model = FakeEmbeddingModel()
    vec = model.encode([text], normalize_embeddings=True)
    vd = _write_library(tmp_path / "vector_db", [text], vectors=vec)
    r = _make_retriever(vd, embedding_model=model)

    assert len(r.documents) == 1
    assert r.embeddings.shape == (1, 16)

    # 向量直召回：同文本 → L2 归一化向量点积即余弦 ≈ 1.0
    vec_hits = r._vector_retrieve(text, 3)
    assert vec_hits and vec_hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert "叶星" in vec_hits[0]["text"]

    # 融合检索返回的 score 是 RRF 名次分（非余弦），只断言命中与状态可观测
    fused = r.retrieve(text, top_k=3)
    assert any("叶星" in x["text"] for x in fused)
    assert r.last_vector_status == ""


# ==================== B. 缺失资源：嵌入模型 / torch / 重排模型 ====================

def test_build_with_unavailable_embedding_model_returns_false(tmp_path, monkeypatch):
    """显式指定嵌入模型但加载不到 → 建库必须失败（返回 False），不得产出假就绪索引。

    【规格来源】2026-09-15 规格修订（用户拍板）：建库失败即返回失败。
    本用例原为 test_build_without_embedding_model_writes_metadata_only，断言
    「模型不可用时返回 True、只落 metadata.json」，属规格修订前的旧行为，
    现按新规格重写（不放宽断言、不删除覆盖面）：
    - 旧行为的问题：返回 True 会让上游把项目状态标成 ready，检索端拿到
      无向量的半成品索引，用户看到「已建库」却检索不到内容。
    - 新行为：返回 False，且不落盘 metadata.json / embeddings.npy，并回收
      本次创建的空目录（不留幽灵目录）。
    纯关键词检索模式（把 embedding_model_path 置空）仍属合法路径，
    由 tests/test_chunking_spec.py 与 tests/test_split_spec_wiring.py 覆盖。
    """
    monkeypatch.setattr(step2, "load_embedding_model", lambda *a, **k: (None, "cpu"))
    src = tmp_path / "novel.txt"
    src.write_text(_chapter_text(n_chapters=2), encoding="utf-8")
    out = tmp_path / "vector_db"

    ok = step2.build_vector_index_from_file(
        str(src), output_dir=str(out), embedding_model_path="models/not-exist",
        book_id="b1", book_title="测试书")

    assert ok is False, "嵌入模型不可用时建库必须返回 False（否则项目状态假 ready）"
    assert not (out / "embeddings.npy").exists(), "模型不可用时不应生成 embeddings.npy"
    assert not (out / "metadata.json").exists(), (
        "模型不可用时不应产出 metadata.json（半成品索引会被误判为已建库）"
    )
    assert not out.exists(), "建库失败后不应残留本次创建的空目录"


def test_keyword_only_mode_still_succeeds_when_model_path_blank(tmp_path, monkeypatch):
    """embedding_model_path 置空 = 显式选择纯关键词模式，建库仍应成功（只落 metadata）。

    与上一条测试配对，明确区分两种语义：置空是「用户主动选择离线」，
    指定了模型却加载不到才是「失败」。
    """
    monkeypatch.setattr(step2, "load_embedding_model", lambda *a, **k: (None, "cpu"))
    src = tmp_path / "novel.txt"
    src.write_text(_chapter_text(n_chapters=2), encoding="utf-8")
    out = tmp_path / "vdb_kw"

    ok = step2.build_vector_index_from_file(
        str(src), output_dir=str(out), embedding_model_path="",
        book_id="b1", book_title="测试书")

    assert ok is True
    assert (out / "metadata.json").is_file()
    assert not (out / "embeddings.npy").exists()


def test_check_vector_dependencies_reports_environment():
    """依赖自检返回结构化结果：键齐全，ok 等价于 torch 与 sentence_transformers 同时可用"""
    dep = utils.check_vector_dependencies()
    assert set(dep) >= {"torch", "sentence_transformers", "ok", "python"}
    assert dep["python"], "应回显当前解释器路径（便于定位误用系统 Python）"
    assert dep["ok"] is (dep["torch"] and dep["sentence_transformers"])


def test_check_vector_dependencies_detects_missing_torch(monkeypatch):
    """依赖自检：torch 不可用时应报 ok=False 且告警文案点名缺失项"""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch 将抛 ImportError
    dep = utils.check_vector_dependencies()
    assert dep["ok"] is False
    assert dep["torch"] is False
    hint = utils.format_vector_dependency_hint(dep)
    assert "torch" in hint and "embeddings.npy" in hint


def test_reranker_missing_enable_rerank_is_noop(tmp_path):
    """无重排模型时 enable_rerank=True 不报错，结果与 False 一致"""
    text = "叶星在试炼中突破。"
    model = FakeEmbeddingModel()
    vec = model.encode([text], normalize_embeddings=True)
    vd = _write_library(tmp_path / "vector_db", [text], vectors=vec)
    r = _make_retriever(vd, embedding_model=model, reranker_model=None)

    a = r.retrieve(text, top_k=3, enable_rerank=False)
    b = r.retrieve(text, top_k=3, enable_rerank=True)
    assert [x["text"] for x in a] == [x["text"] for x in b]


# ==================== C. 无 API Key ====================

def test_get_api_params_exits_without_key(tmp_path, monkeypatch):
    """无 API Key / 无模型时，CLI 参数解析应显式退出（exit code 1）而非带着空密钥请求"""
    from config_manager import ConfigManager
    import novel_rag

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "models": [{"name": "m1", "api_url": "https://api.example.com",
                    "api_key": "", "model_id": "demo"}],
    }, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        novel_rag._get_api_params(ConfigManager(str(cfg_file)), None, None)
    assert exc.value.code == 1


def test_api_client_constructs_without_key_and_normalizes_url():
    """无 Key 时客户端构造不崩（校验由调用方前置完成），URL 归一化稳定"""
    from api_client import APIClient

    c = APIClient("https://api.deepseek.com", "", "deepseek-chat")
    assert c.api_key == ""
    assert c.api_url == "https://api.deepseek.com/v1/chat/completions"
    # 已带完整路径时不重复拼接
    c2 = APIClient("https://api.deepseek.com/v1/chat/completions", "k", "m")
    assert c2.api_url == "https://api.deepseek.com/v1/chat/completions"


# ==================== D. 超长 / 超短文本 ====================

def test_split_empty_and_very_short_text():
    """空串 → 空列表；1 字 → 单块且内容不丢"""
    assert utils.split_text_recursive("") == []
    assert utils.split_text_recursive("叶") == ["叶"]


def test_split_long_text_without_punctuation_respects_hard_limit():
    """无标点超长文：块长不超硬上限，且字符总量不低于原文（允许重叠）"""
    text = "叶" * 5000
    chunks = utils.split_text_recursive(
        text, max_chars=utils.MAX_CHUNK_CHARS, overlap_chars=50)

    assert len(chunks) >= 5000 // utils.MAX_CHUNK_CHARS
    assert all(len(c) <= utils.MAX_CHUNK_CHARS for c in chunks)
    assert sum(len(c) for c in chunks) >= len(text)
    assert set("".join(chunks)) == {"叶"}, "无标点文本不应引入任何新字符"

def test_build_tiny_text_single_chunk(tmp_path, monkeypatch):
    """极小文本建库：退化为兜底单块，metadata 与向量行数自洽（1×dim）"""
    fake = FakeEmbeddingModel()
    monkeypatch.setattr(step2, "load_embedding_model", lambda *a, **k: (fake, "cpu"))
    src = tmp_path / "tiny.txt"
    src.write_text("叶星。", encoding="utf-8")
    out = tmp_path / "vector_db"

    assert step2.build_vector_index_from_file(
        str(src), output_dir=str(out), book_id="t", book_title="极小书") is True

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    emb = np.load(out / "embeddings.npy")
    assert len(meta) == 1 and emb.shape == (1, fake.dim)


# ==================== E. 增量重建 ====================

def _build(tmp_path, text, fake, monkeypatch, out, **kw):
    monkeypatch.setattr(step2, "load_embedding_model", lambda *a, **k: (fake, "cpu"))
    src = tmp_path / "novel.txt"
    src.write_text(text, encoding="utf-8")
    assert step2.build_vector_index_from_file(
        str(src), output_dir=str(out), book_id="b", book_title="测试书", **kw) is True
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def test_incremental_rebuild_reuses_all_chunks(tmp_path, monkeypatch):
    """原文未变再次建库：全部复用旧向量，不再调用 encode（增量重建零编码）"""
    fake = FakeEmbeddingModel()
    text = _chapter_text(n_chapters=3)
    out = tmp_path / "vector_db"

    meta1 = _build(tmp_path, text, fake, monkeypatch, out)
    calls_after_first = len(fake.encode_calls)
    emb1 = np.load(out / "embeddings.npy")

    meta2 = _build(tmp_path, text, fake, monkeypatch, out)
    emb2 = np.load(out / "embeddings.npy")

    assert all(m["reused"] for m in meta2), "未改动原文应全部命中章节指纹复用"
    assert len(fake.encode_calls) == calls_after_first, "全量复用时不应再编码"
    assert np.array_equal(emb1, emb2)
    assert len(meta1) == len(meta2)


def test_chapter_change_reencodes_only_changed_chapter(tmp_path, monkeypatch):
    """只改一章正文：其余章复用旧向量，仅改动章重编码，库行数与元数据保持一致"""
    fake = FakeEmbeddingModel()
    base = _chapter_text(n_chapters=3)
    # 正文改写（章节指纹按正文计算，标题改动不改变指纹，故此处改正文）
    changed = base.replace("第3章第0段落：", "第3章第0段落（改写）：")
    assert changed != base
    out = tmp_path / "vector_db"

    _build(tmp_path, base, fake, monkeypatch, out)
    calls_after_first = len(fake.encode_calls)
    meta2 = _build(tmp_path, changed, fake, monkeypatch, out)

    reused = [m for m in meta2 if m["reused"]]
    assert 0 < len(reused) < len(meta2), "应部分复用、部分重编码"
    assert {m["chapter_id"] for m in reused} == {"c1", "c2"}, "仅未改动的章可复用"
    assert len(fake.encode_calls) > calls_after_first, "改动章必须重新编码"
    assert np.load(out / "embeddings.npy").shape[0] == len(meta2)


def test_chapter_title_change_does_not_invalidate_fingerprint(tmp_path, monkeypatch):
    """边界：章节标题不在正文指纹内（标题仅存 metadata.chapter_title）→ 改标题全部复用

    推论（已实测）：仅修改章节标题不会触发该章重编码，也不会让标题文字进入块文本，
    因此「按标题文字检索」无法命中该章的块内容——检索依赖正文与 chapter_title 字段。
    """
    fake = FakeEmbeddingModel()
    base = _chapter_text(n_chapters=2)
    out = tmp_path / "vector_db"

    _build(tmp_path, base, fake, monkeypatch, out)
    calls_after_first = len(fake.encode_calls)

    retitled = base.replace("第2章 试炼2", "第2章 全新的标题")
    meta2 = _build(tmp_path, retitled, fake, monkeypatch, out)

    assert all(m["reused"] for m in meta2)
    assert len(fake.encode_calls) == calls_after_first
    assert any(m["chapter_title"] == "第2章 全新的标题" for m in meta2)
    assert not any("全新的标题" in m["text"] for m in meta2), "标题不应混入块文本"


def test_vector_dim_mismatch_triggers_full_reencode(tmp_path, monkeypatch):
    """旧库向量维度与新模型不符：放弃复用，全量重编码为新维度"""
    out = tmp_path / "vector_db"
    text = _chapter_text(n_chapters=2)

    old = FakeEmbeddingModel(dim=16)
    _build(tmp_path, text, old, monkeypatch, out)
    assert np.load(out / "embeddings.npy").shape[1] == 16

    new = FakeEmbeddingModel(dim=8)
    meta2 = _build(tmp_path, text, new, monkeypatch, out)

    assert np.load(out / "embeddings.npy").shape == (len(meta2), 8)
    assert all(m["reused"] is False for m in meta2), "维度不一致时不应复用旧向量"


def test_rebuild_with_different_spec_keeps_shape_consistent(tmp_path, monkeypatch):
    """换切分规格重建：库仍可用，向量行数与元数据条数始终自洽"""
    fake = FakeEmbeddingModel()
    text = _chapter_text(n_chapters=3, paras=6)
    out = tmp_path / "vector_db"

    _build(tmp_path, text, fake, monkeypatch, out, chunk_size=672, overlap=50)
    meta2 = _build(tmp_path, text, fake, monkeypatch, out, chunk_size=400, overlap=80)

    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (len(meta2), fake.dim)
    assert len(meta2) > 0


# ==================== F. 异常 metadata ====================

def test_corrupt_metadata_json_degrades(tmp_path):
    """metadata.json 内容损坏：不抛异常，向量置空并降级为关键词模式"""
    vd = tmp_path / "vector_db"
    vd.mkdir()
    np.save(vd / "embeddings.npy", np.ones((3, 8), dtype=np.float32))
    (vd / "metadata.json").write_text("{ this is not json", encoding="utf-8")

    r = _make_retriever(vd, embedding_model=FakeEmbeddingModel())
    assert r.embeddings is None
    assert r.documents == []
    assert r.retrieve("任意问题") == []


def test_metadata_not_a_list_degrades(tmp_path):
    """metadata.json 是 dict 而非 list：不抛异常，按空库处理"""
    vd = tmp_path / "vector_db"
    vd.mkdir()
    np.save(vd / "embeddings.npy", np.ones((2, 8), dtype=np.float32))
    (vd / "metadata.json").write_text(
        json.dumps({"text": "叶星", "chapter": "第一章"}), encoding="utf-8")

    r = _make_retriever(vd, embedding_model=FakeEmbeddingModel())
    assert r.documents == []
    assert r.retrieve("叶星") == []


def test_metadata_missing_text_field_is_safe(tmp_path):
    """metadata 缺 text 字段 / text 为 None：不崩，检索可正常返回，text 归一化为空串"""
    meta = [
        {"chapter": "第一章", "chunk_id": 0},                 # 无 text
        {"text": None, "chapter": "第一章", "chunk_id": 1},   # text 为 None
        {"text": "叶星登场", "chapter": "第一章", "chunk_id": 2},
    ]
    model = FakeEmbeddingModel()
    vecs = np.stack([model.encode(["x"], normalize_embeddings=True)[0]] * 3)
    vd = _write_library(tmp_path / "vector_db", [], vectors=vecs, metadata=meta)
    r = _make_retriever(vd, embedding_model=model)

    assert len(r.documents) == 3
    assert r.documents[0]["text"] == "" and r.documents[1]["text"] == ""
    results = r.retrieve("叶星登场", top_k=3)
    assert all(isinstance(x.get("text"), str) for x in results)


def test_vector_row_count_mismatch_truncates(tmp_path):
    """向量行数与元数据条数不一致：按较小者截断对齐，检索不越界"""
    meta = [{"text": f"片段{i}", "chapter": "第一章", "chunk_id": i} for i in range(3)]
    vectors = np.ones((10, 8), dtype=np.float32)
    vd = _write_library(tmp_path / "vector_db", [], vectors=vectors, metadata=meta)
    r = _make_retriever(vd, embedding_model=FakeEmbeddingModel(dim=8))

    assert len(r.documents) == 3
    assert r.embeddings.shape == (3, 8)
    results = r.retrieve("片段0", top_k=5)
    assert all(x["chunk_id"] < 3 for x in results if "chunk_id" in x)
