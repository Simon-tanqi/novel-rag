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


def load_env_file(env_path=None) -> None:
    """加载项目根目录 .env 文件（若存在）到 os.environ。

    支持 KEY=VALUE 行与 # 注释，值不解析引号/变量展开，够用即可；
    仅在键未设置时写入（真实环境变量优先）。供 CLI/GUI/评估入口
    在 import 后最早处调用一次。
    """
    path = Path(env_path) if env_path else (get_root_dir() / ".env")
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


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


# ===================== 嵌入模型 =====================

# 开箱即用的默认嵌入模型（HuggingFace 模型 ID）。
# 不配置时自动使用该模型做语义检索；首次使用会自动下载并缓存。
# 中文小说场景推荐 bge-small-zh-v1.5（约 95MB，CPU 可跑）。
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def resolve_embedding_model(spec: Optional[str]):
    """
    解析嵌入模型配置：兼容「本地目录」与「HuggingFace 模型 ID」两种写法。

    返回:
        ('local', path)  本地模型目录（已存在）
        ('hub', name)    HuggingFace 模型 ID（首次使用自动下载并缓存）
        None             未配置 / 本地目录不存在 / 明显无效

    设计说明：
    - 空值 → None（调用方降级为关键词检索，不阻断流程）；
    - 以盘符、./、.. 开头或包含反斜杠 → 视为本地路径，不存在时返回 None 并提示；
    - 其余（如 BAAI/bge-small-zh-v1.5）→ 视为 HF 模型 ID，交给
      sentence-transformers 在线下载缓存。
    """
    if not spec:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    looks_like_path = (
        re.match(r'^[A-Za-z]:[\\/]', spec) is not None
        or spec.startswith(('.', os.sep))
        or os.sep in spec
        or os.path.isdir(spec)
        or os.path.exists(spec)
    )

    if looks_like_path:
        if os.path.isdir(spec):
            return ('local', spec)
        print(f"⚠ 本地模型目录不存在（将使用关键词检索）: {spec}")
        return None

    return ('hub', spec)


_GPU_AVAILABLE: Optional[bool] = None
_GPU_DEVICE: Optional[str] = None


def get_compute_device() -> str:
    """
    返回最适合的推理设备：GPU（CUDA）> MPS > CPU。
    结果缓存，全局只查一次。
    """
    global _GPU_AVAILABLE, _GPU_DEVICE
    if _GPU_DEVICE is not None:
        return _GPU_DEVICE

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            _GPU_AVAILABLE = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            _GPU_AVAILABLE = True
        else:
            device = "cpu"
            _GPU_AVAILABLE = False
    except Exception:
        device = "cpu"
        _GPU_AVAILABLE = False

    _GPU_DEVICE = device
    return device


def load_embedding_model(spec: Optional[str]):
    """
    根据配置加载 sentence-transformers 嵌入模型（懒加载，自动选择 GPU/CPU）。

    Args:
        spec: 本地模型目录 或 HF 模型 ID（见 resolve_embedding_model）

    Returns:
        (模型实例, 设备字符串) 或 (None, "cpu")（失败时）
    """
    resolved = resolve_embedding_model(spec)
    if resolved is None:
        return None, "cpu"
    kind, target = resolved
    device = get_compute_device()
    try:
        from sentence_transformers import SentenceTransformer
        if kind == 'local':
            print(f"✓ 加载本地嵌入模型: {target}  [设备: {device}]")
            model = SentenceTransformer(target, local_files_only=True, device=device)
        else:
            print(
                f"⏳ 首次使用将下载嵌入模型 {target} "
                f"（之后自动缓存到 ~/.cache/huggingface）"
            )
            model = SentenceTransformer(target, device=device)
        model.max_seq_length = 512
        return model, device
    except ImportError:
        print("⚠ 未安装 sentence_transformers，将使用关键词检索模式")
    except Exception as e:
        print(f"⚠ 加载嵌入模型失败: {e}（将使用关键词检索模式）")
    return None, "cpu"


# ===================== 文本处理 =====================
# 章节标题行匹配（中文网文常见格式）
CHAPTER_PATTERNS = [
    r'^\s*第[一二三四五六七八九十百千万零〇两\d]+[章节回卷部集篇话]\s*[^\n]{0,60}',
    r'^\s*(?:序章|楔子|引子|番外|后记|尾声|完本感言)[^\n]{0,60}',
]
CHAPTER_RE = re.compile("|".join(f"(?:{p})" for p in CHAPTER_PATTERNS))


def _is_chapter_line(line: str) -> bool:
    """判断一行是否像章节标题（行首匹配，且行内无句读）"""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    # 标题行不应包含句末标点（避免把正文行误判为标题）
    if re.search(r'[。！？；…]', stripped):
        return False
    return bool(CHAPTER_RE.match(stripped))


def _find_chapter_boundaries(text: str):
    """
    定位章节边界，返回 [(start, end, title), ...]。

    - 把含章节标题的连续行（标题行 + 紧邻的装饰行）归入同一章节头；
    - 相邻标题行间隔 ≤3 行视为同一章节头（如“第X章 标题”下一行紧跟“===”）；
    - 最后一个标题到文本末尾为最后一章。
    """
    lines = text.split('\n')
    title_indices = [i for i, line in enumerate(lines) if _is_chapter_line(line)]
    if not title_indices:
        return []

    # 合并“相邻”标题行（间隔 ≤ 1 行视为同属一个章节头：
    # 只合并紧挨着的装饰行，如“第X章 标题”下一行紧跟“======”；
    # 间隔更大的独立标题行不合并，避免把 1 行正文误判为章节头）
    groups = []
    cur = [title_indices[0]]
    for idx in title_indices[1:]:
        if idx - cur[-1] <= 1:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)

    boundaries = []
    for g, group in enumerate(groups):
        title = lines[group[0]].strip()
        # 若标题行下还有紧邻的装饰行（如“======”），把其首行并入标题
        first = group[0]
        if len(group) > 1 and lines[group[1]].strip() and not _is_chapter_line(lines[group[1]]):
            decor = lines[group[1]].strip()
            if decor and not re.search(r'[。！？；]', decor) and len(decor) <= 20:
                title = f"{title} {decor}" if decor else title
        start = first
        end = groups[g + 1][0] if g + 1 < len(groups) else len(lines)
        boundaries.append((start, end, title))
    return boundaries


def split_text_by_sentences(
    text: str,
    min_chars: int = 100,
    max_chars: int = 512,
    overlap_sentences: int = 1
) -> List[str]:
    """
    按句子切分文本，使用滑动窗口策略（不含章节标题，兼容旧签名）

    Args:
        text: 输入文本
        min_chars: 最小块字符数
        max_chars: 最大块字符数
        overlap_sentences: 重叠句子数

    Returns:
        切分后的文本块列表
    """
    return _chunk_by_sentences(text, min_chars, max_chars, overlap_sentences)


def _chunk_by_sentences(
    text: str,
    min_chars: int,
    max_chars: int,
    overlap_sentences: int,
    max_chunk_length: Optional[int] = None
) -> List[str]:
    """
    将一段连续正文按句子切分为文本块。

    - 单句超过 max_chars 时按 max_chunk_length（默认 max_chars）硬切，避免超长块；
    - 其余逻辑与旧 split_text_by_sentences 一致（句子边界不截断）。
    """
    sentences = re.split(r'[。！？；\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_len = 0
    hard_limit = max_chunk_length or max_chars

    for sent in sentences:
        sent_len = len(sent)
        if sent_len > hard_limit:
            # 单句超长：先收尾当前块，再按字符硬切该句
            if current_chunk:
                chunks.append('。'.join(current_chunk) + '。')
                current_chunk = []
                current_len = 0
            for start in range(0, sent_len, hard_limit):
                chunks.append(sent[start:start + hard_limit])
            continue
        if current_len + sent_len > max_chars and current_chunk:
            chunks.append('。'.join(current_chunk) + '。')
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap
            current_len = sum(len(s) for s in overlap)
        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunk_text = '。'.join(current_chunk) + '。'
        chunks.append(chunk_text)

    return chunks


def split_text_by_chapters(
    text: str,
    min_chars: int = 100,
    max_chars: int = 512,
    overlap_sentences: int = 1
):
    """
    按章节边界切片：每个章节独立切块，chunk 携带所属章节标题。

    Args:
        text: 输入全文
        min_chars: 最小块字符数
        max_chars: 最大块字符数
        overlap_sentences: 重叠句子数（句）

    Returns:
        (chunks, chapter_of_chunk):
            chunks: List[str] 文本块（不含章节标题行）
            chapter_of_chunk: List[str] 每个块所属的章节标题（无标题时为默认名）

    无章节标题（未识别到）时退化为整篇正文切片，章节名统一为“正文”。
    """
    boundaries = _find_chapter_boundaries(text)
    if not boundaries:
        chunks = _chunk_by_sentences(text, min_chars, max_chars, overlap_sentences)
        return chunks, ["正文"] * len(chunks)

    chunks, chapter_of_chunk = [], []
    for start, end, title in boundaries:
        # 章节正文 = 标题行之后、下一章之前（跳过与标题同组的装饰行）
        body_lines = []
        started = False
        for line in text.split('\n')[start + 1:end]:
            if not started and not line.strip():
                continue
            started = True
            body_lines.append(line)
        body = '\n'.join(body_lines).strip()
        if not body:
            continue
        for chunk in _chunk_by_sentences(body, min_chars, max_chars,
                                         overlap_sentences):
            chunks.append(chunk)
            chapter_of_chunk.append(title)

    # 末章之后若有未归属正文（异常情况兜底）
    last_end = boundaries[-1][1]
    tail = '\n'.join(text.split('\n')[last_end:]).strip()
    if tail:
        for chunk in _chunk_by_sentences(tail, min_chars, max_chars,
                                         overlap_sentences):
            chunks.append(chunk)
            chapter_of_chunk.append("正文")

    if not chunks:
        chunks = _chunk_by_sentences(text, min_chars, max_chars, overlap_sentences)
        chapter_of_chunk = ["正文"] * len(chunks)
    return chunks, chapter_of_chunk


def extract_chapter_title(text: str, title_pattern: str = r"^第[^\n]+") -> Optional[str]:
    """
    从章节文本中提取标题（兼容旧接口）

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
