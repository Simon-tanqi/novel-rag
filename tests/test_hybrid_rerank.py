# -*- coding: utf-8 -*-
"""
验证 RAGRetriever 的混合召回 (RRF) + CrossEncoder 精排 行为

测试目标：
- RRF 融合：向量 + 关键词的排序 → RRF 分数合并
- 章节聚合重排 + 精排 都启用时，最终结果按精排分数排序
- enable_rerank=False 时，不调 CrossEncoder
- CrossEncoder 失败时优雅降级
"""
import sys
from unittest.mock import MagicMock

# 顶部 stub 掉 main.py 的重型 GUI 依赖
class _FakeCTk:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return MagicMock()
    def after(self, *args, **kwargs): pass
    def mainloop(self, *args, **kwargs): pass
_fake_ctk_module = MagicMock()
_fake_ctk_module.CTk = _FakeCTk
sys.modules['customtkinter'] = _fake_ctk_module
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()

import numpy as np
import pytest
from rag_retriever import RAGRetriever


def _make_retriever_with_docs(docs, embeddings=None, embedding_model=None,
                              reranker_model=None):
    """直接构造 RAGRetriever 绕过 load_documents/_load_*_model"""
    # 不调 __init__（避免 load_documents 走文件 I/O）
    retriever = RAGRetriever.__new__(RAGRetriever)
    retriever.vector_path = "/fake"
    retriever.reranker_model_path = "/fake/reranker"
    retriever.embedding_model_path = "/fake/embedding"
    retriever.vector_file = None
    retriever.metadata_file = None
    retriever.documents = docs
    retriever.texts = [d['text'] for d in docs]
    retriever.embeddings = embeddings
    retriever.reranker_model = reranker_model
    retriever.embedding_model = embedding_model
    retriever._embedding_loaded = True
    retriever._reranker_loaded = True
    return retriever


# ============== RRF 融合测试 ==============

def test_rrf_fusion_merges_overlap_and_distinct():
    """RRF 融合：重叠文档分数累加，独家文档只取单路分数"""
    retriever = _make_retriever_with_docs([])

    vec_results = [
        {'chunk_id': 1, 'text': 'doc1', 'score': 0.9, 'chapter': 'ch1', 'file': 'f'},
        {'chunk_id': 2, 'text': 'doc2', 'score': 0.8, 'chapter': 'ch1', 'file': 'f'},
        {'chunk_id': 3, 'text': 'doc3', 'score': 0.7, 'chapter': 'ch2', 'file': 'f'},
    ]
    kw_results = [
        {'chunk_id': 2, 'text': 'doc2', 'score': 5.0, 'chapter': 'ch1', 'file': 'f'},  # 重叠
        {'chunk_id': 4, 'text': 'doc4', 'score': 3.0, 'chapter': 'ch3', 'file': 'f'},  # 独家
    ]

    fused = retriever._rrf_fusion(vec_results, kw_results, k=60)

    # RRF 分数 (k=60)：
    #   doc1: 1/(60+1) = 0.01639  (仅向量)
    #   doc2: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  (重叠累加)
    #   doc3: 1/(60+3) = 0.01587  (仅向量)
    #   doc4: 1/(60+2) = 0.01613  (仅关键词)
    # 排序：doc2 > doc4 ≈ doc1 > doc3
    assert len(fused) == 4
    assert fused[0]['chunk_id'] == 2, f"重排 doc2 应排第一，实际: {fused[0]['chunk_id']}"
    # doc4 和 doc1 分数很接近，可能并列第一以外
    top_scores = {d['chunk_id']: d['score'] for d in fused}
    assert top_scores[2] > top_scores[1]  # doc2 分数 > doc1
    assert top_scores[2] > top_scores[3]  # doc2 分数 > doc3


def test_rrf_fusion_only_vector():
    """只有向量召回时，RRF 不崩溃"""
    retriever = _make_retriever_with_docs([])
    vec_results = [
        {'chunk_id': 1, 'text': 'doc1', 'score': 0.9, 'chapter': 'ch1', 'file': 'f'},
    ]
    fused = retriever._rrf_fusion(vec_results, [], k=60)
    assert len(fused) == 1
    assert fused[0]['chunk_id'] == 1


def test_rrf_fusion_empty_inputs():
    """两边都空时返回空列表"""
    retriever = _make_retriever_with_docs([])
    assert retriever._rrf_fusion([], [], k=60) == []


# ============== CrossEncoder 精排测试 ==============

def test_rerank_reorders_by_cross_encoder_score():
    """rerank 候选按 CrossEncoder 分数降序排"""
    docs = [
        {'chunk_id': 1, 'text': '关于三国鼎立的描述', 'score': 0.5, 'chapter': 'ch1', 'file': 'f'},
        {'chunk_id': 2, 'text': '完全无关的其他内容', 'score': 0.5, 'chapter': 'ch1', 'file': 'f'},
        {'chunk_id': 3, 'text': '三国演义中曹操用兵', 'score': 0.5, 'chapter': 'ch1', 'file': 'f'},
    ]
    # mock CrossEncoder：让 doc3 分数最高，doc2 最低
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.3, 0.1, 0.9])

    retriever = _make_retriever_with_docs(docs, reranker_model=mock_ce)
    reranked = retriever._rerank("三国曹操", docs)

    assert len(reranked) == 3
    assert reranked[0]['chunk_id'] == 3, "最高分 0.9 应该是 doc3"
    assert reranked[0]['score'] == 0.9
    assert reranked[2]['chunk_id'] == 2, "最低分 0.1 应该是 doc2"


def test_rerank_falls_back_on_failure():
    """CrossEncoder 抛错时，rerank 返回原顺序"""
    docs = [
        {'chunk_id': 1, 'text': 'a', 'score': 0.1, 'chapter': 'c', 'file': 'f'},
        {'chunk_id': 2, 'text': 'b', 'score': 0.2, 'chapter': 'c', 'file': 'f'},
    ]
    mock_ce = MagicMock()
    mock_ce.predict.side_effect = RuntimeError("模拟 GPU 失败")

    retriever = _make_retriever_with_docs(docs, reranker_model=mock_ce)
    result = retriever._rerank("query", docs)

    # 失败时返回原列表
    assert result == docs


# ============== retrieve() 集成测试 ==============

def test_retrieve_hybrid_no_rerank():
    """混合召回 + 不精排：流程跑通，返回 top_k 结果"""
    docs = [
        {'chunk_id': i, 'text': f'文档{i}关于三国', 'score': 0.0,
         'chapter': f'第{i}章', 'file': 'f'}
        for i in range(1, 11)
    ]
    # mock embedding model 让向量召回能跑
    mock_emb = MagicMock()
    mock_emb.encode.return_value = np.ones(4) / 2  # 4 维，1/2 范数

    embeddings = np.eye(10, 4, dtype=np.float32)  # 10 个正交向量
    retriever = _make_retriever_with_docs(docs, embeddings=embeddings, embedding_model=mock_emb)

    results = retriever.retrieve("三国", top_k=3, enable_rerank=False)
    assert len(results) > 0
    assert len(results) <= 3


def test_retrieve_with_rerank_uses_cross_encoder():
    """启用精排时，调用 CrossEncoder.predict"""
    docs = [
        {'chunk_id': 1, 'text': 'doc1 苹果', 'score': 0.5, 'chapter': 'c1', 'file': 'f'},
        {'chunk_id': 2, 'text': 'doc2 苹果', 'score': 0.6, 'chapter': 'c1', 'file': 'f'},
        {'chunk_id': 3, 'text': 'doc3 苹果', 'score': 0.7, 'chapter': 'c2', 'file': 'f'},
    ]
    mock_emb = MagicMock()
    mock_emb.encode.return_value = np.array([0.5, 0.5], dtype=np.float32)
    embeddings = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.5, 0.9, 0.1])

    retriever = _make_retriever_with_docs(
        docs, embeddings=embeddings,
        embedding_model=mock_emb, reranker_model=mock_ce,
    )
    results = retriever.retrieve("苹果", top_k=2, enable_rerank=True)

    # 精排后 doc2 分数最高（0.9），应该是第一位
    assert results[0]['chunk_id'] == 2
    # CrossEncoder 被调用了
    assert mock_ce.predict.called


def test_retrieve_disables_rerank_when_model_unavailable():
    """reranker_model 未加载时，enable_rerank=True 也跳过精排"""
    docs = [
        {'chunk_id': 1, 'text': '苹果红', 'score': 0.5, 'chapter': 'c', 'file': 'f'},
        {'chunk_id': 2, 'text': '苹果绿', 'score': 0.6, 'chapter': 'c', 'file': 'f'},
    ]
    mock_emb = MagicMock()
    mock_emb.encode.return_value = np.array([0.5, 0.5], dtype=np.float32)
    embeddings = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    retriever = _make_retriever_with_docs(
        docs, embeddings=embeddings,
        embedding_model=mock_emb, reranker_model=None,  # 关键：没模型
    )
    # 不应崩溃
    results = retriever.retrieve("苹果", top_k=2, enable_rerank=True)
    assert len(results) > 0


def test_retrieve_keyword_only_path():
    """没有 embedding model 时，只走关键词检索"""
    docs = [
        {'chunk_id': 1, 'text': '三国演义', 'score': 0.0, 'chapter': 'c1', 'file': 'f'},
        {'chunk_id': 2, 'text': '其他内容', 'score': 0.0, 'chapter': 'c2', 'file': 'f'},
    ]
    retriever = _make_retriever_with_docs(
        docs, embeddings=None, embedding_model=None
    )
    results = retriever.retrieve("三国", top_k=2, enable_rerank=False)
    # 应该召回"三国演义"
    assert len(results) > 0
    assert results[0]['chunk_id'] == 1
