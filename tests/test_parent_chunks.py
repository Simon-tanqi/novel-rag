# -*- coding: utf-8 -*-
"""分层切片与父子块召回测试。

覆盖：
1. BoundaryDetector 四级边界探测；
2. build_parent_chunks：父块不跨章节、字符数达标、子块不被遗漏；
3. RAGRetriever.expand_context：命中子块 → 父块 + 邻居，结果无重复；
4. step2_split_embed 落盘：metadata 新增字段 + parent_chunks.json。
"""
import json

import numpy as np
import pytest

from utils import (
    BoundaryDetector,
    build_parent_chunks,
    compute_overlap_chars,
    MAX_CHUNK_CHARS,
    PARENT_MIN_CHARS,
    PARENT_MAX_CHARS,
    PARENT_GROUP_MAX,
    OVERLAP_MIN_CHARS,
)
from rag_retriever import RAGRetriever


# ===================== 1. 边界探测 =====================

class TestBoundaryDetector:
    def test_find_primary_returns_cut_after_punct(self):
        text = "他没有说话。然后转身离开。"
        cut = BoundaryDetector.find_primary(text)
        assert cut == text.index("。") + 1
        assert text[:cut].endswith("。")

    def test_find_primary_consumes_closing_quote(self):
        text = "他喊道：“走！”然后离开。"
        cut = BoundaryDetector.find_primary(text)
        # 闭合引号并入前一块，不留在下一块开头
        assert text[:cut].endswith("！”")
        assert text[cut] == "然"

    def test_find_primary_no_hit(self):
        assert BoundaryDetector.find_primary("没有句末标点的长句") == -1

    def test_find_secondary(self):
        text = "甲；乙"
        assert BoundaryDetector.find_secondary(text) == 2
        assert BoundaryDetector.find_secondary("没有从句标点") == -1

    def test_find_tertiary(self):
        text = "甲 乙"
        assert BoundaryDetector.find_tertiary(text) == 2
        assert BoundaryDetector.find_tertiary("没有空白字符") == -1

    def test_split_oversized_degrades_to_secondary(self):
        """超长文本优先按二级边界切，且不丢字符、不超上限"""
        text = "甲" * 300 + "；" + "乙" * 300 + "；" + "丙" * 300
        pieces = BoundaryDetector.split_oversized(text, 400, min_scan=200)
        assert len(pieces) >= 2
        assert all(len(p) <= 400 for p in pieces)
        assert "".join(pieces) == text  # 无丢失、无重复

    def test_split_oversized_falls_back_to_hard_cut(self):
        """无任何可切点时走四级硬切，仍不丢字符"""
        text = "甲" * 1000
        pieces = BoundaryDetector.split_oversized(text, 400, min_scan=200)
        assert len(pieces) == 3
        assert all(len(p) <= 400 for p in pieces)
        assert "".join(pieces) == text

    def test_compute_overlap_chars(self):
        assert compute_overlap_chars(200) == OVERLAP_MIN_CHARS      # 12.5% < 50 → 取 50
        assert compute_overlap_chars(1000) == 125                   # 12.5% > 50
        assert compute_overlap_chars(0) == OVERLAP_MIN_CHARS


# ===================== 2. 父块构造 =====================

class TestBuildParentChunks:
    @staticmethod
    def _corpus():
        """两章、每章 6 个子块，每块 300 字符（聚合后为 900 字符/父块）"""
        chunks, chapters = [], []
        for name in ["第一章 起始", "第二章 转折"]:
            for _ in range(6):
                chunks.append("甲" * 299 + "。")
                chapters.append(name)
        return chunks, chapters

    def test_parent_not_cross_chapter(self):
        chunks, chapters = self._corpus()
        parents = build_parent_chunks(chunks, chapters)
        assert parents
        for parent in parents:
            chs = {chapters[i] for i in parent["child_indices"]}
            assert len(chs) == 1, "父块不得跨章节"
            assert parent["chapter"] in chs

    def test_parent_length_within_range(self):
        chunks, chapters = self._corpus()
        parents = build_parent_chunks(chunks, chapters)
        for parent in parents:
            assert PARENT_MIN_CHARS <= parent["chunk_length"] <= PARENT_MAX_CHARS

    def test_parent_group_size_between_two_and_four(self):
        chunks, chapters = self._corpus()
        parents = build_parent_chunks(chunks, chapters)
        for parent in parents:
            assert 2 <= parent["child_count"] <= PARENT_GROUP_MAX

    def test_every_child_belongs_to_exactly_one_parent(self):
        chunks, chapters = self._corpus()
        parents = build_parent_chunks(chunks, chapters)
        covered = [i for p in parents for i in p["child_indices"]]
        assert sorted(covered) == list(range(len(chunks)))  # 全覆盖、不重复

    def test_parent_text_is_child_concat(self):
        chunks, chapters = self._corpus()
        parents = build_parent_chunks(chunks, chapters)
        for parent in parents:
            assert parent["text"] == "".join(chunks[i] for i in parent["child_indices"])
            assert parent["child_count"] == len(parent["child_indices"])

    def test_parent_ids_unique(self):
        chunks, chapters = self._corpus()
        parents = build_parent_chunks(chunks, chapters)
        ids = [p["parent_id"] for p in parents]
        assert len(ids) == len(set(ids))

    def test_empty_input(self):
        assert build_parent_chunks([], []) == []

    def test_single_chapter_single_chunk(self):
        parents = build_parent_chunks(["短文本。"], ["第一章"])
        assert len(parents) == 1
        assert parents[0]["child_count"] == 1
        assert parents[0]["text"] == "短文本。"


# ===================== 3. 召回扩展 =====================

@pytest.fixture()
def layered_db(tmp_path):
    """带分层字段的向量库：2 章 4 子块 2 父块"""
    embeddings = np.zeros((4, 8), dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", embeddings)

    metadata = [
        {"text": "子块一", "chapter": "第一章", "chunk_id": 1, "parent_id": "p1",
         "prev_chunk_id": None, "next_chunk_id": 2},
        {"text": "子块二", "chapter": "第一章", "chunk_id": 2, "parent_id": "p1",
         "prev_chunk_id": 1, "next_chunk_id": 3},
        {"text": "子块三", "chapter": "第一章", "chunk_id": 3, "parent_id": "p1",
         "prev_chunk_id": 2, "next_chunk_id": None},
        {"text": "子块四", "chapter": "第二章", "chunk_id": 4, "parent_id": "p2",
         "prev_chunk_id": None, "next_chunk_id": None},
    ]
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )

    parents = [
        {"parent_id": "p1", "text": "父块一（子块一二三）", "chapter": "第一章",
         "child_indices": [0, 1, 2], "child_chunk_ids": [1, 2, 3], "child_count": 3},
        {"parent_id": "p2", "text": "父块二（子块四）", "chapter": "第二章",
         "child_indices": [3], "child_chunk_ids": [4], "child_count": 1},
    ]
    (tmp_path / "parent_chunks.json").write_text(
        json.dumps(parents, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


class TestExpandContext:
    def test_expand_appends_parent_and_neighbors(self, layered_db):
        retriever = RAGRetriever(str(layered_db))
        hits = retriever.retrieve("子块二", top_k=1)
        assert hits
        assert hits[0]["text"] == "子块二"  # 命中子块仍排在最前

        texts = [h["text"] for h in hits]
        assert "父块一（子块一二三）" in texts          # 父块被追加
        assert "子块一" in texts and "子块三" in texts  # 前后邻居被追加

        marks = {(h["text"], h.get("is_parent"), h.get("is_neighbor")) for h in hits}
        assert ("父块一（子块一二三）", True, False) in marks
        assert ("子块一", False, True) in marks

    def test_expand_no_duplicates(self, layered_db):
        retriever = RAGRetriever(str(layered_db))
        bare = [
            {"text": "子块二", "chapter": "第一章", "chunk_id": 2, "parent_id": "p1",
             "prev_chunk_id": 1, "next_chunk_id": 3, "score": 2.0},
            {"text": "子块三", "chapter": "第一章", "chunk_id": 3, "parent_id": "p1",
             "prev_chunk_id": 2, "next_chunk_id": None, "score": 1.0},
        ]
        expanded = retriever.expand_context(bare, neighbor_count=1)

        keys = [str(e.get("chunk_id")) for e in expanded]
        assert len(keys) == len(set(keys)), "扩展结果不得含重复 chunk_id"
        texts = [e["text"] for e in expanded]
        assert len(texts) == len(set(texts)), "扩展结果不得含重复文本"

    def test_expand_empty_input(self, layered_db):
        retriever = RAGRetriever(str(layered_db))
        assert retriever.expand_context([], neighbor_count=1) == []

    def test_expand_neighbor_count_zero_only_parent(self, layered_db):
        retriever = RAGRetriever(str(layered_db))
        bare = [{"text": "子块二", "chapter": "第一章", "chunk_id": 2, "parent_id": "p1",
                 "prev_chunk_id": 1, "next_chunk_id": 3, "score": 2.0}]
        expanded = retriever.expand_context(bare, neighbor_count=0)
        texts = [e["text"] for e in expanded]
        assert texts == ["子块二", "父块一（子块一二三）"]

    def test_expand_degrades_on_legacy_db(self, tmp_path):
        """老向量库（无 parent_id / 无 parent_chunks.json）→ 恒等返回"""
        np.save(tmp_path / "embeddings.npy", np.zeros((1, 8), dtype=np.float32))
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"text": "荒古禁地中得到了机缘。", "chapter": "第一章",
                         "chunk_id": 1}], ensure_ascii=False), encoding="utf-8"
        )
        retriever = RAGRetriever(str(tmp_path))
        assert retriever.parent_chunks == {}
        hits = retriever.retrieve("荒古禁地机缘", top_k=1)
        assert len(hits) == 1
        assert hits[0]["text"] == "荒古禁地中得到了机缘。"


# ===================== 4. 建库落盘 =====================

class TestPipelineLayering:
    @staticmethod
    def _novel_text():
        # 两章正文，每章 840 字符（保证切出多个子块、可聚合成父块）
        body_a = "叶凡踏入门中，看见一条长河缓缓流过。" * 42
        body_b = "庞博惊呼出声，却无人应答他的呼喊。" * 42
        return f"第一章 起始\n{body_a}\n第二章 转折\n{body_b}"

    def test_build_vector_index_writes_parent_layer(self, tmp_path):
        import step2_split_embed as s2

        out_dir = tmp_path / "vector_db"
        ok = s2.build_vector_index(
            self._novel_text(), output_dir=str(out_dir), embedding_model_path=None
        )
        assert ok is True

        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata
        for item in metadata:
            assert "parent_id" in item
            assert "prev_chunk_id" in item
            assert "next_chunk_id" in item
            # 原有字段不丢
            for legacy in ["text", "chapter", "chunk_id", "chunk_length", "coref_prefix"]:
                assert legacy in item

        parent_file = out_dir / "parent_chunks.json"
        assert parent_file.is_file()
        parents = json.loads(parent_file.read_text(encoding="utf-8"))
        assert parents
        for parent in parents:
            assert parent["child_count"] >= 1
            assert parent["chapter"] in parent["text"] or parent["text"]

        # 父块覆盖全部子块且不跨章节
        covered = [i for p in parents for i in p["child_indices"]]
        assert sorted(covered) == list(range(len(metadata)))
        chapter_of = [m["chapter"] for m in metadata]
        for parent in parents:
            assert len({chapter_of[i] for i in parent["child_indices"]}) == 1

    def test_metadata_parent_id_links_to_parent_file(self, tmp_path):
        import step2_split_embed as s2

        out_dir = tmp_path / "vector_db2"
        assert s2.build_vector_index(
            self._novel_text(), output_dir=str(out_dir), embedding_model_path=None
        )
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        parents = json.loads((out_dir / "parent_chunks.json").read_text(encoding="utf-8"))
        parent_ids = {p["parent_id"] for p in parents}

        for item in metadata:
            assert item["parent_id"] in parent_ids, "子块的 parent_id 必须在父块表中可查"
            if item["prev_chunk_id"] is not None:
                assert item["prev_chunk_id"] == item["chunk_id"] - 1
            if item["next_chunk_id"] is not None:
                assert item["next_chunk_id"] == item["chunk_id"] + 1

    def test_chunk_ids_unchanged(self, tmp_path):
        """chunk_id 生成规则与改造前一致：从 1 开始的连续整数"""
        import step2_split_embed as s2

        out_dir = tmp_path / "vector_db3"
        assert s2.build_vector_index(
            self._novel_text(), output_dir=str(out_dir), embedding_model_path=None
        )
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        assert [m["chunk_id"] for m in metadata] == list(range(1, len(metadata) + 1))
        assert all(m["chunk_length"] <= MAX_CHUNK_CHARS for m in metadata)
