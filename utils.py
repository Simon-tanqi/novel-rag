"""
utils.py — 精简公共工具函数
仅包含必要的文本处理和路径工具函数，不依赖 jieba/faiss/bm25 等重依赖
"""
import re
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional


# ===================== 路径工具 =====================
def get_root_dir() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.resolve()


def get_data_dir() -> Path:
    """获取数据存储目录"""
    data_dir = get_root_dir() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_chat_logs_dir() -> Path:
    """获取聊天日志根目录"""
    logs_dir = get_root_dir() / "chat_logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_project_dir(project_name: str) -> Path:
    """获取项目专属目录"""
    project_dir = get_data_dir() / project_name
    project_dir.mkdir(exist_ok=True)
    return project_dir


def get_vector_db_path(project_name: str) -> str:
    """获取项目向量库路径"""
    vector_db = get_project_dir(project_name) / "vector_db"
    vector_db.mkdir(exist_ok=True)
    return str(vector_db)


def to_relative_path(abs_path: str) -> str:
    """将绝对路径转换为相对路径"""
    try:
        root = str(get_root_dir())
        if abs_path.startswith(root):
            return os.path.relpath(abs_path, root)
        return abs_path
    except Exception:
        return abs_path


def to_absolute_path(rel_path: str) -> str:
    """将相对路径转换为绝对路径"""
    try:
        if os.path.isabs(rel_path):
            return rel_path
        return str(get_root_dir() / rel_path)
    except Exception:
        return rel_path


# ===================== 文本处理 =====================
def split_text_by_sentences(
    text: str,
    min_chars: int = 100,
    max_chars: int = 512,
    overlap_sentences: int = 1
) -> List[str]:
    """
    按句子切分文本，使用滑动窗口策略

    Args:
        text: 输入文本
        min_chars: 最小块字符数
        max_chars: 最大块字符数
        overlap_sentences: 重叠句子数

    Returns:
        切分后的文本块列表
    """
    sentences = re.split(r'[。！？；\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_chars and current_chunk:
            chunks.append('。'.join(current_chunk) + '。')
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap
            current_len = sum(len(s) for s in overlap)
        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunk_text = '。'.join(current_chunk) + '。'
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)
        elif len(chunk_text) > 0:
            chunks.append(chunk_text)

    return chunks


def extract_chapter_title(text: str, title_pattern: str = r"^第[^\n]+") -> Optional[str]:
    """
    从章节文本中提取标题

    Args:
        text: 章节文本
        title_pattern: 标题匹配正则

    Returns:
        章节标题或None
    """
    for line in text.split('\n')[:5]:
        stripped = line.strip()
        if stripped and re.match(title_pattern, stripped):
            return stripped
    return None


def generate_project_id() -> str:
    """生成项目ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"project_{timestamp}"


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """格式化时间戳"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")
