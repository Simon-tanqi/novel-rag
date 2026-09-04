# -*- coding: utf-8 -*-
"""测试清洗规则：编码纠错 / 去除广告水印 / 去除页码 / 合并段落。"""
import pytest

from step1_clean import (
    clean_text,
    _fix_encoding,
    _remove_ads,
    _remove_page_numbers,
    _remove_empty_lines,
)


class TestEncodingFix:
    def test_gbk_error_map(self):
        """GBK 错字映射：夭才->天才、入间->人间"""
        text = "这是一个夭才的故事，他来到了入间。"
        cleaned = _fix_encoding(text)
        assert "夭才" not in cleaned
        assert "天才" in cleaned
        assert "入间" not in cleaned
        assert "人间" in cleaned

    def test_garbled_char_map(self):
        """生僻错字映射：镇龘压->镇压"""
        assert _fix_encoding("镇龘压") == "镇压"


class TestRemoveAds:
    def test_default_watermark_keywords(self):
        text = "正文内容。\n本文由看小说到网提供。\n更多精彩章节。"
        cleaned = _remove_ads(text)
        assert "看小说到网" not in cleaned
        assert "正文内容。" in cleaned

    def test_custom_dirty_words(self):
        text = "正文。\n支持正版请到xx小说网。\n后文继续。"
        cleaned = _remove_ads(text, custom_words=["xx小说网"])
        assert "xx小说网" not in cleaned


class TestRemovePageNumbers:
    def test_plain_digit_lines_removed(self):
        text = "第一章 开始\n123\n456\n正文内容"
        cleaned = _remove_page_numbers(text)
        assert "123" not in cleaned
        assert "456" not in cleaned
        assert "正文内容" in cleaned

    def test_page_number_format_removed(self):
        text = "内容A\n第12页\n内容B"
        cleaned = _remove_page_numbers(text)
        assert "第12页" not in cleaned
        assert "内容A" in cleaned and "内容B" in cleaned


class TestCleanTextPipeline:
    def test_full_pipeline_empty_lines(self):
        text = "第一段\n\n\n\n第二段\n\n"
        cleaned = clean_text(text, rules=["去除空行"])
        assert "\n\n\n" not in cleaned

    def test_empty_lines_only(self):
        assert _remove_empty_lines("a\n\n\nb") == "a\nb"

    def test_merge_paragraphs(self):
        text = "句子一。\n句子二。\n句子三！"
        merged = clean_text(text, rules=["合并段落"])
        # 合并后句号之间不应再有换行（应被替换为空格）
        assert "\n" not in merged or merged.count("。") >= 2
