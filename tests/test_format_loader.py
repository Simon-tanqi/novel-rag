"""test_format_loader.py — format_loader 模块测试

测试 epub 格式加载和格式判断逻辑。
不依赖真实 epub 文件，用 ebooklib 动态构造测试用 epub。
"""

import os
import tempfile

import pytest

try:  # 可选依赖：未安装时仅跳过 epub 相关用例
    import ebooklib  # noqa: F401
    _HAS_EBOOKLIB = True
except ImportError:
    _HAS_EBOOKLIB = False

from format_loader import (
    SUPPORTED_EXTENSIONS,
    is_supported_format,
    load_raw_text,
)


# ─── is_supported_format ───────────────────────────────────────

class TestIsSupportedFormat:
    def test_txt_supported(self):
        assert is_supported_format("novel.txt") is True

    def test_epub_supported(self):
        assert is_supported_format("novel.epub") is True

    def test_pdf_not_supported(self):
        assert is_supported_format("novel.pdf") is False

    def test_mobi_not_supported(self):
        assert is_supported_format("novel.mobi") is False

    def test_no_extension(self):
        assert is_supported_format("novel") is False

    def test_case_insensitive(self):
        assert is_supported_format("novel.EPUB") is True
        assert is_supported_format("novel.Txt") is True


# ─── load_raw_text — txt fallback ──────────────────────────────

class TestLoadTxt:
    def test_txt_read(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("你好世界\n第二行", encoding="utf-8")
        text = load_raw_text(str(txt_file))
        assert "你好世界" in text
        assert "第二行" in text


# ─── load_raw_text — epub ──────────────────────────────────────

def _make_test_epub(path):
    """用 ebooklib 构造一个最小 epub 测试文件（2 章节）。"""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-id-001")
    book.set_title("测试小说")
    book.set_language("zh")

    c1 = epub.EpubHtml(title="第一章", file_name="chap1.xhtml", lang="zh")
    c1.content = (
        "<html><body>"
        "<h1>第一章 山门初开</h1>"
        "<p>少年沈青背着药箱下山。</p>"
        "<p>山路蜿蜒，云雾缭绕。</p>"
        "</body></html>"
    )

    c2 = epub.EpubHtml(title="第二章", file_name="chap2.xhtml", lang="zh")
    c2.content = (
        "<html><body>"
        "<h1>第二章 剑庐第一课</h1>"
        "<p>灰衣老者端坐石上。</p>"
        "<p>「心要静。」老者说。</p>"
        "</body></html>"
    )

    book.add_item(c1)
    book.add_item(c2)

    # 必须添加 NCX 和 Nav，否则 read_epub 时 spine 加载会报错
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = ["nav", c1, c2]

    epub.write_epub(path, book, {})


class TestLoadEpub:
    @pytest.mark.skipif(not _HAS_EBOOKLIB, reason="可选依赖 ebooklib 未安装")
    def test_epub_extract_text(self, tmp_path):
        epub_path = str(tmp_path / "test.epub")
        _make_test_epub(epub_path)

        text = load_raw_text(epub_path)

        # 应包含两章的标题和内容
        assert "山门初开" in text
        assert "沈青" in text
        assert "剑庐第一课" in text
        assert "灰衣老者" in text
        assert "心要静" in text

    @pytest.mark.skipif(not _HAS_EBOOKLIB, reason="可选依赖 ebooklib 未安装")
    def test_epub_strips_html_tags(self, tmp_path):
        epub_path = str(tmp_path / "test.epub")
        _make_test_epub(epub_path)

        text = load_raw_text(epub_path)
        # 不应残留 HTML 标签
        assert "<html>" not in text
        assert "<p>" not in text
        assert "<h1>" not in text

    @pytest.mark.skipif(not _HAS_EBOOKLIB, reason="可选依赖 ebooklib 未安装")
    def test_epub_chapter_separation(self, tmp_path):
        epub_path = str(tmp_path / "test.epub")
        _make_test_epub(epub_path)

        text = load_raw_text(epub_path)
        # 两章之间应有空行分隔
        assert "山门初开" in text
        assert "剑庐第一课" in text
        # 第一章内容和第二章内容不应在同一行
        lines = text.split("\n")
        chapter1_line = [i for i, l in enumerate(lines) if "山门初开" in l]
        chapter2_line = [i for i, l in enumerate(lines) if "剑庐第一课" in l]
        assert chapter1_line and chapter2_line
        assert chapter2_line[0] > chapter1_line[0]

    def test_epub_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            load_raw_text("/nonexistent/path/book.epub")

    def test_unsupported_format_raises(self, tmp_path):
        fake_file = tmp_path / "book.xyz"
        fake_file.write_text("dummy")
        with pytest.raises(ValueError, match="不支持"):
            load_raw_text(str(fake_file))
