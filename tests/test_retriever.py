# -*- coding: utf-8 -*-
"""测试 RAGRetriever 关键词检索回退路径（无嵌入模型时）。"""
import json

import numpy as np
import pytest

from rag_retriever import RAGRetriever


@pytest.fixture()
def vector_db_dir(tmp_path):
    """构造一个最小可用的向量库目录：dummy embeddings + metadata"""
    # 2 条文档、8 维向量（内容与向量无关，因为走关键词回退）
    embeddings = np.zeros((2, 8), dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", embeddings)

    metadata = [
        {"text": "叶凡在荒古禁地中得到了机缘。", "chapter": "第一章 荒古禁地", "chunk_id": 1},
        {"text": "九龙拉棺从天而降，震惊世人。", "chapter": "第二章 九龙拉棺", "chunk_id": 2},
    ]
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


class TestKeywordRetrieval:
    def test_keyword_fallback_finds_hit(self, vector_db_dir):
        """无嵌入模型时应走关键词检索并命中相关片段"""
        retriever = RAGRetriever(str(vector_db_dir))
        hits = retriever.retrieve("荒古禁地有什么机缘", top_k=2)
        assert len(hits) >= 1
        assert "荒古禁地" in hits[0]["text"]

    def test_documents_loaded(self, vector_db_dir):
        retriever = RAGRetriever(str(vector_db_dir))
        assert retriever.documents  # 非空
        assert len(retriever.documents) == 2

    def test_irrelevant_query_no_hit(self, vector_db_dir):
        retriever = RAGRetriever(str(vector_db_dir))
        hits = retriever.retrieve("完全不相关的问题XYZ", top_k=2)
        assert hits == []
