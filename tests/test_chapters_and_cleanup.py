# -*- coding: utf-8 -*-
"""测试章节级切片、清洗防误杀、向量库归一化检索。"""
import json

import numpy as np
import pytest

from step1_clean import clean_text, _remove_ads
from utils import split_text_by_chapters, split_text_by_sentences
from rag_retriever import RAGRetriever


# ===================== 章节级切片 =====================

class TestSplitTextByChapters:
    def _novel(self):
        """三段式网文样本：两个真实章节 + 无标题尾部"""
        return (
            "第一章 荒古禁地\n"
            "叶凡踏入荒古禁地，九条龙尸拉着一口青铜巨棺横空而过。\n"
            "他捡到一枚古玉，上面刻着神秘的纹路。\n"
            "\n"
            "第二章 九龙拉棺\n"
            "九龙拉棺从天而降，震惊了所有人。\n"
            "庞博惊呼出声，脸色煞白。\n"
            "\n"
            "一些散落的段落。\n"
        )

    def test_chunks_carry_chapter_titles(self):
        text = self._novel()
        chunks, chapters = split_text_by_chapters(text, min_chars=5, max_chars=120)
        assert len(chunks) >= 2
        # 第一章的块应归属「第一章 荒古禁地」
        first_chapter_chunks = [c for c, ch in zip(chunks, chapters)
                                if ch == "第一章 荒古禁地"]
        assert first_chapter_chunks
        assert any("青铜巨棺" in c for c in first_chapter_chunks)
        # 正文中不应残留章节标题行本身
        assert not any("第一章 荒古禁地" in c for c in chunks)

    def test_title_not_leaked_into_body(self):
        """chunk 不应包含标题行文本（标题作为元数据，不进正文）"""
        text = "第1章 起始\n这是正文第一句话，讲了一个故事的开端。\n第2章 发展\n正文第二段内容继续展开情节。\n"
        chunks, chapters = split_text_by_chapters(text, min_chars=5, max_chars=500)
        assert chapters == ["第1章 起始", "第2章 发展"]
        assert all("第" not in c or c.startswith("这是") or c.startswith("正文") for c in chunks)
        assert len(chunks) == 2

    def test_no_chapter_falls_back_to_plain(self):
        text = "没有章节标记的一段话。这里讲了一个故事。"
        chunks, chapters = split_text_by_chapters(text, min_chars=5, max_chars=50)
        assert chapters == ["正文"] * len(chunks)
        assert chunks

    def test_long_sentence_gets_hard_split(self):
        """单句超过上限时应硬切，避免产生超长块"""
        long_text = "第一章 标题\n" + "啊" * 300 + "。结束句。"
        chunks, _ = split_text_by_chapters(long_text, min_chars=5, max_chars=100)
        # 300 字硬切为 3 块（每块 100），加结束句一块
        assert len(chunks) >= 3
        assert all(len(c) <= 100 + 10 for c in chunks[:-1])  # 允许最后一个略超

    def test_legacy_split_signature_still_works(self):
        """旧 split_text_by_sentences 接口保持可用（不抛异常、返回非空）"""
        text = "句子一。句子二。句子三！"
        chunks = split_text_by_sentences(text, min_chars=1, max_chars=100)
        assert chunks and all(c.strip() for c in chunks)


# ===================== 清洗防误杀 =====================

class TestAdsRemovalNoFalsePositive:
    def test_body_line_with_free_word_kept(self):
        """正文含「免费」等泛词不应被整行删除"""
        text = "他免费教村里的孩子识字，从不收一文钱。"
        cleaned = _remove_ads(text)
        assert "他免费教村里的孩子识字" in cleaned

    def test_body_line_with_download_word_kept(self):
        text = "老者说：心法下载于天地之间，非人力可为。"
        cleaned = _remove_ads(text)
        assert "心法下载于天地之间" in cleaned

    def test_watermark_line_removed(self):
        text = "正文内容。\n本文由看小说到网提供，最快更新请收藏本站。\n更多精彩。"
        cleaned = _remove_ads(text)
        assert "看小说到网" not in cleaned
        assert "正文内容。" in cleaned

    def test_trailing_bracket_watermark_stripped(self):
        """行尾括号水印应被剔除，但正文保留"""
        text = "他默默练功，气息渐稳。（本章未完，请点击下一页继续阅读）"
        cleaned = _remove_ads(text)
        assert "本章未完" not in cleaned
        assert "他默默练功" in cleaned

    def test_custom_word_short_line_removed(self):
        text = "正文。\nxx小说网提醒您：请支持正版。\n后文继续。"
        cleaned = _remove_ads(text, custom_words=["xx小说网"])
        assert "xx小说网" not in cleaned

    def test_full_pipeline_keeps_body(self):
        """完整清洗流程不应误伤含泛词的正文"""
        text = "李强下载了剑谱，老者免费传他口诀，并说正版武学要心正。\n他从此勤学不辍。"
        cleaned = clean_text(text, rules=["去除广告"])
        assert "李强下载了剑谱" in cleaned
        assert "他从此勤学不辍" in cleaned


# ===================== 归一化向量检索 =====================

class TestRetrieverNormalizedDot:
    def test_non_normalized_vectors_still_retrieve(self, tmp_path):
        """未归一化向量库（旧数据）加载后应被归一化，点积检索仍可用"""
        # 构造两段文档，向量故意不归一化（模长不同）
        vec_a = np.array([[3.0, 4.0]], dtype=np.float32)   # 模长 5
        vec_b = np.array([[0.0, 2.0]], dtype=np.float32)   # 模长 2
        np.save(tmp_path / "embeddings.npy", np.vstack([vec_a, vec_b]))

        metadata = [
            {"text": "青云剑诀第一式：云起。", "chapter": "第一章 青云剑诀", "chunk_id": 1},
            {"text": "村里井水甘甜。", "chapter": "第五章 井水之争", "chunk_id": 2},
        ]
        (tmp_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        class _FakeEmbedder:
            """伪嵌入模型：encode 返回与文档语义对齐的向量"""
            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                return np.array([
                    np.array([4.0, 3.0]) if "剑诀" in t or "云起" in t
                    else np.array([0.0, 5.0])
                    for t in texts
                ], dtype=np.float32)

        retriever = RAGRetriever(str(tmp_path), embedding_model_path="")
        retriever.embedding_model = _FakeEmbedder()
        # 手工触发归一化（正常流程在 load 时已做）
        retriever._align()

        hits = retriever.retrieve("青云剑诀的云起怎么练", top_k=1)
        assert hits and "青云剑诀" in hits[0]["chapter"]


# ===================== 清洗保留章节结构 =====================

class TestCleanKeepsChapterStructure:
    """清洗管线（含合并段落）不应破坏章节标题行——标题是切片的结构锚点"""

    def test_merge_paragraphs_keeps_title_lines(self):
        text = (
            "第一章 荒古禁地\n"
            "叶凡踏入禁地，九条龙尸横空而过。\n"
            "他捡到一枚古玉，上面刻着纹路。\n"
            "\n"
            "第二章 九龙拉棺\n"
            "青铜巨棺从天而降，震惊所有人。\n"
        )
        cleaned = clean_text(text, rules=["合并段落"])
        # 章节标题行必须独立成行（清洗后仍可作为切片锚点）
        assert "第一章 荒古禁地\n" in cleaned or "第一章 荒古禁地" in cleaned.split("\n")
        assert any("第二章 九龙拉棺" in ln for ln in cleaned.split("\n"))

    def test_full_pipeline_chapters_still_split(self):
        """全默认规则清洗后，章节切片仍能识别标题（防合并段落吞标题回归）"""
        text = (
            "第一章 荒古禁地\n"
            "叶凡踏入荒古禁地，九条龙尸拉着一口青铜巨棺横空而过。\n"
            "他捡到一枚古玉，上面刻着神秘的纹路。\n"
            "\n"
            "第二章 九龙拉棺\n"
            "九龙拉棺从天而降，震惊了所有人。\n"
            "\n"
        )
        cleaned = clean_text(text)
        chunks, chapters = split_text_by_chapters(cleaned, min_chars=5, max_chars=500)
        assert "第一章 荒古禁地" in chapters
        assert "第二章 九龙拉棺" in chapters
        assert any("龙尸" in c for c in chunks)
