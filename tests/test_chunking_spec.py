# -*- coding: utf-8 -*-
"""规格切片测试：递归/标点优先切片

规格：
1) 文本不足 200 字符不切分；
2) 超过 200 字符后寻找句号/感叹号/冒号/问号，命中即切片；
3) 距上次切分点超过 512 字符仍无上述标点则强制切片；
4) 相邻片段设置 50 字符重叠。
"""
import json

from utils import (
    split_text_recursive,
    split_text_by_chapters,
    resolve_split_params,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    CHUNK_BOUNDARY_PUNCT,
)


# ===================== 规格常量与参数映射 =====================

class TestSpecConstants:
    def test_spec_values(self):
        assert (MIN_CHUNK_CHARS, MAX_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS) == (200, 512, 50)

    def test_boundary_punct_set(self):
        """切片标点集合：句号、感叹号、冒号、问号"""
        assert set(CHUNK_BOUNDARY_PUNCT) == {"。", "！", "：", "？"}


class TestResolveSplitParams:
    def test_chunk_size_in_spec_range_kept(self):
        assert resolve_split_params(500, 50) == (200, 500, 50)
        assert resolve_split_params(512, 50) == (200, 512, 50)

    def test_chunk_size_out_of_range_falls_back_to_spec(self):
        assert resolve_split_params(800, 50) == (200, 512, 50)
        assert resolve_split_params(100, 50) == (200, 512, 50)
        assert resolve_split_params(None, None) == (200, 512, 50)

    def test_overlap_is_char_count(self):
        assert resolve_split_params(500, 0)[2] == 0
        assert resolve_split_params(500, -3)[2] == 0
        assert resolve_split_params(500, 80)[2] == 80


# ===================== 规则 1：不足 min_chars 不切 =====================

class TestMinCharsNoSplit:
    def test_short_text_kept_as_single_chunk(self):
        text = "短文本没有标点" * 10  # 80 字符
        assert split_text_recursive(text, 200, 512, 50) == [text]

    def test_exactly_min_chars_not_split(self):
        text = "啊" * 100 + "。" + "呀" * 99  # 恰好 200 字符
        assert split_text_recursive(text, 200, 512, 50) == [text]

    def test_one_char_over_min_starts_scanning(self):
        text = "啊" * 100 + "。" + "呀" * 99 + "？"  # 201 字符
        chunks = split_text_recursive(text, 200, 512, 50)
        assert chunks == [text]
        assert chunks[0].endswith("？")


# ===================== 规则 2：超过 min_chars 遇标点即切 =====================

class TestPunctBoundaryCut:
    def test_cut_at_first_punct_after_min(self):
        text = "啊" * 199 + "。" + "呀" * 199 + "？" + "嘿" * 100
        chunks = split_text_recursive(text, 200, 512, 50)
        assert len(chunks) == 2
        # 第 200 字符之后的第一个标点是 index=399 的「？」，在其后切分
        assert chunks[0] == text[:400]
        assert chunks[0].endswith("？")
        assert len(chunks[0]) == 400

    def test_each_boundary_punct_triggers_cut(self):
        for punct in CHUNK_BOUNDARY_PUNCT:
            text = "啊" * 210 + punct + "呀" * 100
            chunks = split_text_recursive(text, 200, 512, 50)
            assert chunks[0] == text[:211], f"标点 {punct} 未触发切分"
            assert chunks[0].endswith(punct)

    def test_secondary_punct_triggers_fallback_cut(self):
        """逗号/分号/顿号属二级边界：一级标点缺失时降级切分（原实现整段硬切）"""
        text = "啊" * 205 + "，" + "呀" * 5 + "；" + "嘿" * 5 + "、" + "哈" * 100
        chunks = split_text_recursive(text, 200, 512, 50)
        assert len(chunks) >= 2, "二级边界未触发降级切分"
        assert chunks[0].endswith("，"), "未在首个二级标点后切分"
        assert chunks[-1].endswith("哈"), "尾块未覆盖原文末尾"
        assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


# ===================== 规则 3：超过 max_chars 无标点强制切 =====================

class TestHardLimit:
    def test_force_split_at_512_without_punct(self):
        text = "啊" * 1300  # 全文无标点
        chunks = split_text_recursive(text, 200, 512, 50)
        assert [len(c) for c in chunks] == [512, 512, 376]
        assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)

    def test_force_split_then_continue(self):
        """强制切分后，后续文本继续按标点规则切分"""
        text = "啊" * 600 + "呀" * 100 + "。" + "嘿" * 200
        chunks = split_text_recursive(text, 200, 512, 50)
        # 第一块：前 512 字符内无标点 → 512 处强制切
        assert len(chunks[0]) == 512
        assert chunks[0] == text[:512]
        # 第二块：切分点后的标点（index=700 的「。」）被命中，在其后切
        assert chunks[1] == text[462:701]
        assert chunks[1].endswith("。")
        assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


# ===================== 规则 4：50 字符重叠 =====================

class TestOverlap:
    def _corpus(self):
        return ("甲" * 250 + "。") * 10  # 每单元 251 字符，全含标点

    def test_adjacent_chunks_overlap_50_chars(self):
        text = self._corpus()
        chunks = split_text_recursive(text, 200, 512, 50)
        assert len(chunks) >= 5
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur.startswith(prev[-DEFAULT_OVERLAP_CHARS:]), "相邻片段未保持 50 字符重叠"

    def test_zero_overlap_is_contiguous(self):
        """overlap=0 时片段应无缝拼接（旧实现在 overlap=0 时仍重叠 1 句，已修复）"""
        text = self._corpus()
        chunks = split_text_recursive(text, 200, 512, 0)
        assert "".join(chunks) == text

    def test_overlap_not_larger_than_chunk(self):
        """重叠不小于块长时应退化为不重叠，且不得死循环"""
        text = "啊" * 300
        chunks = split_text_recursive(text, 200, 512, 999)
        assert chunks and "".join(chunks) == text


# ===================== 原文保真（不改写字符） =====================

class TestVerbatim:
    def test_chunks_are_verbatim_slices(self):
        text = "句子一！句子二？句子三：句子四。" * 40
        chunks = split_text_recursive(text, 200, 512, 50)
        assert len(chunks) > 2
        for c in chunks:
            assert c in text, "切片必须是原文的连续片段（不得改写标点）"
        joined = "".join(chunks)
        assert "！" in joined and "？" in joined and "：" in joined

    def test_chapter_split_verbatim_and_labeled(self):
        text = (
            "第一章 起始\n"
            + "叶凡踏入门中，看见一条长河。" * 30
            + "\n第二章 转折\n"
            + "庞博惊呼，却无人应答！" * 30
        )
        chunks, chapters = split_text_by_chapters(text)
        assert set(chapters) == {"第一章 起始", "第二章 转折"}
        for c in chunks:
            assert c in text
            assert len(c) <= MAX_CHUNK_CHARS

    def test_empty_text(self):
        assert split_text_recursive("", 200, 512, 50) == []


# ===================== 管线集成（不落向量，仅元数据） =====================

class TestPipelineIntegration:
    def test_build_vector_index_uses_spec(self, tmp_path, monkeypatch):
        import step2_split_embed as s2

        # 强制走关键词模式，避免加载/下载嵌入模型
        monkeypatch.setattr(s2, "load_embedding_model", lambda spec: (None, "cpu"))

        text = (
            "第一章 起始\n"
            + "叶凡踏入门中，看见一条长河。" * 30
            + "\n第二章 转折\n"
            + "庞博惊呼，却无人应答！" * 30
        )
        out = tmp_path / "vector_db"
        assert s2.build_vector_index(text, output_dir=str(out)) is True

        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert meta
        for m in meta:
            assert len(m["text"]) <= MAX_CHUNK_CHARS
            assert m["text"] in text
        # 章节标题作为元数据保留
        assert {m["chapter"] for m in meta} == {"第一章 起始", "第二章 转折"}
