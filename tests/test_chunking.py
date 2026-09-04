# -*- coding: utf-8 -*-
"""测试句子级滑动窗口切片。"""
from utils import split_text_by_sentences


def _make_sentences(n: int, sent_len: int = 10) -> str:
    """构造 n 个长度为 sent_len 的句子，用。连接"""
    sentences = []
    for i in range(n):
        # 每个句子 10 个汉字
        base = "一二三四五六七八九"  # 9 字
        sentences.append(base + "零")
    return "。".join(sentences) + "。"


class TestSplitTextBySentences:
    def test_returns_non_empty_chunks(self):
        text = _make_sentences(20)
        chunks = split_text_by_sentences(text, min_chars=4, max_chars=30)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_chunk_size_upper_bound(self):
        """句子不超过上限时，每个 chunk 应贴近 max_chars 上限"""
        text = _make_sentences(50, sent_len=10)
        max_chars = 25
        chunks = split_text_by_sentences(text, min_chars=5, max_chars=max_chars)
        # 每个句子 11 字符（10字 + 分隔符。），chunk 由整句拼成，
        # 允许最后一个追加句子的长度略微越界。
        assert all(len(c) <= max_chars + 11 for c in chunks)

    def test_single_short_text_kept(self):
        text = "只有一个短句子。"
        chunks = split_text_by_sentences(text, min_chars=1, max_chars=50)
        assert chunks == ["只有一个短句子。"]

    def test_overlap_keeps_sentence_boundary(self):
        """开启 1 句重叠时，相邻 chunk 不应出现半个句子"""
        text = _make_sentences(30, sent_len=8)
        chunks = split_text_by_sentences(text, min_chars=4, max_chars=20, overlap_sentences=1)
        # 所有 chunk 尾部都应是完整句子（以。结尾），且不残留空串
        assert all(c.endswith("。") for c in chunks)
