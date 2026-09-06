"""format_loader.py — 多格式文件加载器

检测文件后缀，将非 txt 格式（目前支持 epub）提取为纯文本。
txt 文件仍走 step1_clean 原有的编码检测 + open 读取流程，不经过本模块。

用法：
    from format_loader import is_supported_format, load_raw_text

    if is_supported_format(path):
        text = load_raw_text(path)   # -> str
"""

import os

__all__ = ["is_supported_format", "load_raw_text", "SUPPORTED_EXTENSIONS"]

# 已支持的扩展名（小写含点号）
SUPPORTED_EXTENSIONS = {".txt", ".epub"}


def is_supported_format(file_path: str) -> bool:
    """判断文件后缀是否被支持（txt / epub）。"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def load_raw_text(file_path: str) -> str:
    """从文件提取纯文本。

    - .txt  → 直接读取（编码由调用方处理，这里仅做 fallback UTF-8）
    - .epub → 用 ebooklib 提取章节文本

    返回纯文本字符串。

    对于 epub，依赖 ebooklib 和 beautifulsoup4（懒加载）。
    未安装时抛出 ImportError 并给出安装提示。
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        # txt 走 step1_clean 的编码检测流程，这里只是 fallback
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".epub":
        return _load_epub(file_path)

    raise ValueError(f"不支持的文件格式: {ext}（仅支持 {', '.join(SUPPORTED_EXTENSIONS)}）")


def _load_epub(file_path: str) -> str:
    """从 epub 文件提取纯文本。

    epub 本质是 zip 包，内部按 spine 顺序排列 XHTML 文档。
    使用 ebooklib 读取，BeautifulSoup 解析 HTML 提取纯文本。

    章节之间用双换行分隔，保留章节标题（<h1>/<h2> 等的文本）。
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "读取 epub 文件需要安装 ebooklib 和 beautifulsoup4：\n"
            "  pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(file_path)

    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content()
        soup = BeautifulSoup(html_content, "html.parser")

        # 提取纯文本，保留段落换行
        # 先把 block 元素转成换行
        for block_tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6"]):
            block_tag.append("\n")

        text = soup.get_text()
        # 清理多余空白但保留段落结构
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]  # 去空行

        if lines:
            chapters.append("\n".join(lines))

    return "\n\n".join(chapters)
