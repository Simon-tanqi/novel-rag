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
# 不配置时自动使用该模型做语义检索。
# 也可填本地模型目录（如 models/bge-small-zh-v1.5）实现纯离线加载。
# 首次使用 HF 模型 ID 时，会自动下载到项目根 models/ 目录，无需联网到 ~/.cache。
# 中文小说场景推荐 bge-small-zh-v1.5（约 95MB，GPU/CPU 均可跑）。
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def resolve_embedding_model(spec: Optional[str]):
    """
    智能识别模型来源：本地目录优先，HF 模型 ID 兜底。

    查找顺序（找到即返回）：
    1. 空 → None（调用方降级关键词检索，不阻断流程）
    2. 绝对路径且目录存在 → ('local', abs)
    3. 相对路径：先按 cwd 解析，再按项目根解析（兼容 GUI 启动时 cwd 不在项目根）
    4. 形如 org/name 的 HF ID，且 <root>/models/<basename>/ 存在 → ('local', <root>/models/<basename>)
    5. 都失败 → ('hub', spec)（由 load_embedding_model 自动下载到 <root>/models/<basename>/）

    Returns:
        ('local', path)  本地目录（已验证存在）
        ('hub', hf_id)   HuggingFace 模型 ID（需下载）
        None             未配置 / 无效
    """
    if not spec:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    # 1) 当作本地路径尝试
    candidates: list = []
    if os.path.isabs(spec):
        candidates.append(spec)
    else:
        # 相对路径：先按 cwd 解析（兼容 import 时的 cwd）
        try:
            candidates.append(os.path.abspath(spec))
        except Exception:
            pass
        # 再按项目根解析（关键：GUI 启动时 cwd 不在项目根的情况）
        try:
            candidates.append(str(get_root_dir() / spec))
        except Exception:
            pass

    for p in candidates:
        if os.path.isdir(p):
            return ('local', p)

    # 2) 形如 org/name 的 HF ID：检查项目根/models/<basename>/
    if re.match(r'^[\w.-]+/[\w.-]+$', spec):
        try:
            root = get_root_dir()
            default_local = root / "models" / spec.split('/')[-1]
            if default_local.is_dir():
                return ('local', str(default_local))
        except Exception:
            pass

    # 3) 兜底：HF 模型 ID（由 load_embedding_model 负责下载到项目根/models/）
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


def _try_download_to_local_models(model_id: str) -> Optional[str]:
    """
    下载 HF 模型到项目根 models/<basename>/ 目录。
    成功返回本地绝对路径，失败返回 None。

    优先尝试用户设置的 HF_ENDPOINT（国内 hf-mirror），
    下载的 .safetensors / .bin 等核心权重保留 README/TXT 等冗余文件不下载。
    """
    try:
        from huggingface_hub import snapshot_download
        root = get_root_dir()
        local_dir = root / "models" / model_id.split('/')[-1]
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(root / ".hf_cache"),
            ignore_patterns=["*.md", "*.txt", "*.py"],
        )
        return str(local_dir)
    except Exception as e:
        print(f"  ⚠ 自动下载失败：{e}")
        return None


def load_embedding_model(spec: Optional[str]):
    """
    根据配置加载 sentence-transformers 嵌入模型（懒加载，自动选择 GPU/CPU）。

    加载顺序：
    1. 本地路径直接加载（local_files_only=True，离线可用）
    2. HF 模型 ID → 先自动下载到项目根 models/<basename>/ 再本地加载
    3. 下载失败 → 兜底走 sentence-transformers 默认缓存（~/.cache/huggingface）

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

    model = None
    try:
        from sentence_transformers import SentenceTransformer
        if kind == 'local':
            print(f"✓ 加载本地嵌入模型: {target}  [设备: {device}]")
            model = SentenceTransformer(target, local_files_only=True, device=device)
        else:
            # kind == 'hub'：先自动下载到项目根 models/<basename>/
            print(f"⏳ 首次使用将下载嵌入模型 {target}")
            print(f"   → 保存到项目根 models/{model_id_to_dirname(target)}/ 目录")
            local_dir = _try_download_to_local_models(target)
            if local_dir and os.path.isdir(local_dir):
                print(f"✓ 已下载到本地，加载中: {local_dir}  [设备: {device}]")
                model = SentenceTransformer(local_dir, local_files_only=True, device=device)
            else:
                # 兜底：HF 默认缓存
                print(f"⏳ 本地下载失败，尝试 HF 默认缓存: {target}")
                model = SentenceTransformer(target, device=device)
    except ImportError:
        print("⚠ 未安装 sentence_transformers，将使用关键词检索模式")
        return None, "cpu"
    except Exception as e:
        print(f"⚠ 加载嵌入模型失败: {e}（将使用关键词检索模式）")
        return None, "cpu"

    if model is not None:
        model.max_seq_length = 512
        return model, device
    return None, "cpu"


def load_reranker_model(spec: Optional[str]):
    """
    根据配置加载 sentence-transformers CrossEncoder 重排模型（懒加载，自动选择 GPU/CPU）。

    加载顺序与嵌入模型一致：
    1. 本地路径直接加载（local_files_only=True，离线可用）
    2. HF 模型 ID → 先自动下载到项目根 models/<basename>/ 再本地加载
    3. 下载失败 → 兜底走 HF 默认缓存

    适用于 BAAI/bge-reranker-base / BAAI/bge-reranker-v2-m3 等 CrossEncoder 类模型。

    Args:
        spec: 本地模型目录 或 HF 模型 ID（见 resolve_embedding_model）

    Returns:
        (CrossEncoder 实例, 设备字符串) 或 (None, "cpu")（失败时）
    """
    resolved = resolve_embedding_model(spec)
    if resolved is None:
        return None, "cpu"
    kind, target = resolved
    device = get_compute_device()

    model = None
    try:
        from sentence_transformers import CrossEncoder
        if kind == 'local':
            print(f"✓ 加载本地重排模型: {target}  [设备: {device}]")
            model = CrossEncoder(target, max_length=512, device=device)
        else:
            # hub：先下载到项目根 models/<basename>/
            print(f"⏳ 首次使用将下载重排模型 {target}")
            local_dir = _try_download_to_local_models(target)
            if local_dir and os.path.isdir(local_dir):
                print(f"✓ 已下载到本地，加载中: {local_dir}  [设备: {device}]")
                model = CrossEncoder(local_dir, max_length=512, device=device)
            else:
                # 兑底：HF 默认缓存
                print(f"⏳ 本地下载失败，尝试 HF 默认缓存: {target}")
                model = CrossEncoder(target, max_length=512, device=device)
    except ImportError:
        print("⚠ 未安装 sentence_transformers，重排不可用")
        return None, "cpu"
    except Exception as e:
        print(f"⚠ 加载重排模型失败: {e}")
        return None, "cpu"

    if model is not None:
        return model, str(model.device)
    return None, "cpu"

    try:
        model.max_seq_length = 512
    except Exception:
        pass
    return model, device


def model_id_to_dirname(model_id: str) -> str:
    """HF 模型 ID → 本地目录名（org/name → name）"""
    return model_id.split('/')[-1] if '/' in model_id else model_id


# ===================== 文本处理 =====================
# 章节标题行匹配（中文网文常见格式）
CHAPTER_PATTERNS = [
    r'^\s*第[一二三四五六七八九十百千万零〇两\d]+[章节回卷部集篇话]\s*[^\n]{0,60}',
    r'^\s*(?:序章|楔子|引子|番外|后记|尾声|完本感言)[^\n]{0,60}',
]
CHAPTER_RE = re.compile("|".join(f"(?:{p})" for p in CHAPTER_PATTERNS))


# 章节标题最大长度（超长行一律判为正文）
MAX_TITLE_CHARS = 32

# 中文数字字符与位值（章节序号解析用）
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def is_chapter_title(line: str) -> bool:
    """章节标题判定的唯一权威实现（step1_clean / novel_context 均转调此处）。

    判定条件：
    1. 行长 ≤ MAX_TITLE_CHARS(32)：标题行不会很长；
    2. 行首匹配「第X章 / 序章 / 楔子 / 番外 …」（见 CHAPTER_PATTERNS）；
    3. 行内不含句号「。」。

    注意：网文标题常以「！」「？」结尾，也可能用逗号分组
    （如「第147章 赵家，雷家！」），故显式放行 ！？，、；。
    旧实现把这些标点一律当句末标点排除，会吞掉绝大多数标题行
    （《绝世主宰》1029 章中 1013 章被判为正文），是结构坍塌的根因。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_TITLE_CHARS:
        return False
    if "。" in stripped:
        return False
    return bool(CHAPTER_RE.match(stripped))


def _is_chapter_line(line: str) -> bool:
    """（兼容保留）章节标题判定，转调 is_chapter_title。"""
    return is_chapter_title(line)


def _cn_to_int(raw: str) -> Optional[int]:
    """中文数字转整数（支持 十/百/千/万 组合，如「一千零二十八」→ 1028）。"""
    total = 0    # 万级累计
    section = 0  # 当前小节累计
    num = 0      # 待入位的末位数字
    for ch in raw:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            section += (num if num else 1) * _CN_UNITS[ch]
            num = 0
        elif ch == "万":
            section = (section + num) * 10000
            total += section
            section = 0
            num = 0
        else:
            return None
    return total + section + num


def parse_chapter_number(title: str) -> Optional[int]:
    """解析章节标题中的序号；无序号标题（序章/楔子/番外）返回 None。"""
    m = re.match(r'^\s*第\s*([零〇一二三四五六七八九十百千万两\d]+)\s*[章节回卷部集篇话]', title)
    if not m:
        return None
    raw = m.group(1)
    return int(raw) if raw.isdigit() else _cn_to_int(raw)


def _find_chapter_boundaries(text: str):
    """
    定位章节边界，返回 [(start, end, title), ...]。

    - 把含章节标题的连续行（标题行 + 紧邻的装饰行）归入同一章节头；
    - 相邻标题行间隔 ≤1 行视为同一章节头（如“第X章 标题”下一行紧跟“===”）；
      间隔 ≥2 行的独立标题行不合并（避免把 1 行正文误并进章节头），
      该阈值由下方「idx - cur[-1] <= 1」唯一实现，文档与代码保持同口径；
    - 标题行下方紧邻的短装饰行（≤20 字且不含句末标点）并入标题文本；
    - 最后一个标题到文本末尾为最后一章。
    """
    lines = text.split('\n')
    candidates = [i for i, line in enumerate(lines) if _is_chapter_line(line)]
    # 章节序号连续性兜底：正文中引用章节的句子（如「他在第三章提过」）也可能
    # 命中标题模式，但序号通常「回退」。按出现顺序扫描，剔除序号逆序的候选；
    # 无序号标题（序章/楔子/番外）不参与校验，也不会被剔除。
    title_indices = []
    last_num = 0
    for i in candidates:
        num = parse_chapter_number(lines[i])
        if num is not None:
            if num < last_num:
                continue
            last_num = num
        title_indices.append(i)
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


# ===================== 分片规格（递归切片） =====================
# 规格（优先级从高到低，split_text_recursive 唯一实现）：
# 0) MIN_CHUNK_CHARS 是最优先硬约束：距上次切分点不足 200 字符 →
#    即使遇到句末标点也不切分，剩余整体作为尾块；
# 1) 达到 200 字符后进入可切区间：在 (上次切分点, 上次切分点+200] 之后、
#    512 字符之前，遇到一级标点 。！：？ → 在其后切分；给定 target_chars
#    时优先挑最接近 400 字符的标点，避免块长全部极化到上限；
# 2) 距上次切分点超过 MAX_CHUNK_CHARS(512) 仍无一级标点 → 依次降级
#    二级（；：，、）/ 三级（空白）/ 四级硬切兜底；
# 3) 相邻片段重叠 = max(DEFAULT_OVERLAP_CHARS, 前块长度 × OVERLAP_RATIO)。
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 512
DEFAULT_OVERLAP_CHARS = 50
# 触发切分的一级标点集合（句号、感叹号、冒号、问号）
CHUNK_BOUNDARY_PUNCT = ("。", "！", "：", "？")


def resolve_split_params(chunk_size=None, overlap=None):
    """把配置中的 chunk_size / overlap 映射为规格切片参数。

    - 硬上限 max_chars：chunk_size 仅当落在 [MIN_CHUNK_CHARS, MAX_CHUNK_CHARS]
      区间内时生效，否则回落规格值 MAX_CHUNK_CHARS（避免超出模型 512 上下文）；
    - 下限 min_chars：规格值 MIN_CHUNK_CHARS（不足不切分）；
    - 重叠 overlap_chars：按字符透传（默认 50），<=0 表示不重叠。

    Returns:
        (min_chars, max_chars, overlap_chars)
    """
    try:
        cs = int(chunk_size) if chunk_size is not None else 0
    except (TypeError, ValueError):
        cs = 0
    max_chars = cs if MIN_CHUNK_CHARS < cs <= MAX_CHUNK_CHARS else MAX_CHUNK_CHARS
    min_chars = min(MIN_CHUNK_CHARS, max_chars)
    try:
        ov = int(overlap) if overlap is not None else DEFAULT_OVERLAP_CHARS
    except (TypeError, ValueError):
        ov = DEFAULT_OVERLAP_CHARS
    return min_chars, max_chars, max(0, ov)


def resolve_split_spec(chunk_size=None, overlap=None,
                       target_chars=None, overlap_ratio=None) -> dict:
    """分层切片完整规格：切片参数的唯一出口（建库 / 检索 / CLI / GUI 均经此）。

    Args:
        chunk_size: 硬上限覆盖值；仅当落在 [MIN_CHUNK_CHARS, MAX_CHUNK_CHARS]
            区间内才生效，否则回落 MAX_CHUNK_CHARS（避免超出模型 512 上下文）。
        overlap: 重叠下限（字符）覆盖值；None → DEFAULT_OVERLAP_CHARS。
        target_chars: 目标块长覆盖值；None → TARGET_CHUNK_CHARS(400)。
            取 min(max(min_chars, 目标值), max_chars)，保证落在 [min, max] 内。
        overlap_ratio: 重叠比例覆盖值；None → OVERLAP_RATIO(0.15)，
            取值裁剪到 [0, 0.5]，避免重叠吞掉整块。

    Returns:
        {
            "min_chars": 200,        # 最小切分距离（硬优先级：不足不切分）
            "max_chars": 512,        # 硬上限
            "overlap_chars": 50,     # 重叠下限
            "target_chars": 400,     # 目标块长（未接线前长期失效，现由本函数统一发放）
            "overlap_ratio": 0.15,   # 重叠比例
        }

    说明：min_chars 是**最优先约束**——距上次切分点不足 min_chars 时，
    即使遇到句末标点也不切分；只有落在 [min_chars, max_chars] 区间内的
    一级标点（。！：？）才触发切分，超过 max_chars 无标点则硬切兜底。
    """
    min_chars, max_chars, overlap_chars = resolve_split_params(chunk_size, overlap)

    # 目标长度：显式入参优先（CLI --target-chars），否则用规格常量；始终裁剪进 [min, max]
    try:
        tgt = int(target_chars) if target_chars is not None else TARGET_CHUNK_CHARS
    except (TypeError, ValueError):
        tgt = TARGET_CHUNK_CHARS
    target_chars = min(max(min_chars, tgt), max_chars)

    # 重叠比例：显式入参优先（CLI --overlap-ratio），裁剪到 [0, 0.5]
    try:
        ratio = float(overlap_ratio) if overlap_ratio is not None else OVERLAP_RATIO
    except (TypeError, ValueError):
        ratio = OVERLAP_RATIO
    ratio = min(max(0.0, ratio), 0.5)

    return {
        "min_chars": min_chars,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "target_chars": target_chars,
        "overlap_ratio": ratio,
    }


def split_text_recursive(
    text: str,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    target_chars: Optional[int] = None,
    overlap_ratio: Optional[float] = None
) -> List[str]:
    """按规格切片：标点优先 + 硬上限 + 字符重叠。

    规则：
    1. 距上次切分点不足 min_chars → 不切分，剩余整体作为尾块；
    2. 超过 min_chars 后寻找 。！：？ → 命中即在其后切分；
       指定 target_chars 时，改为在窗口内挑「最接近目标长度」的标点，
       避免所有块长都极化到 max_chars；
    3. 一级标点未命中 → 依次降级二级（；：，、）→ 三级（空白）→ 四级硬切；
    4. 相邻片段重叠：给定 overlap_ratio 时取
       max(overlap_chars, 前块长度 × overlap_ratio)，否则固定 overlap_chars。

    切分只做位置裁剪、不改写任何字符（标点/换行原样保留），
    因此 chunk 文本与原文一致，可安全用于引用溯源。

    Args:
        text: 输入文本
        min_chars: 最小切分距离（不足不切分）
        max_chars: 无标点时的强制切分距离（硬上限）
        overlap_chars: 相邻片段重叠字符数（重叠下限）
        target_chars: 目标块长（给定时按目标择优切分点，None 保持原行为）
        overlap_ratio: 重叠比例（给定时按前块长度动态计算重叠）

    Returns:
        文本块列表
    """
    if not text:
        return []
    min_chars = max(1, int(min_chars))
    max_chars = max(min_chars, int(max_chars))
    overlap_chars = max(0, int(overlap_chars))
    if overlap_chars >= max_chars:
        overlap_chars = 0  # 重叠不得吞掉整块，避免原地重复切分

    punct = set(CHUNK_BOUNDARY_PUNCT)
    target = int(target_chars) if target_chars else 0
    if target:
        target = min(max(target, min_chars), max_chars)
    n = len(text)
    chunks: List[str] = []
    start = 0
    last_cut = 0  # 上一次切分点（绝对索引）：防止回退重叠后重复命中同一标点

    while start < n:
        if n - start <= min_chars:
            # 不足 min_chars：不再切分，剩余作为尾块
            tail = text[start:]
            if tail.strip():
                chunks.append(tail)
            break

        window_end = min(n, start + max_chars)
        scan_from = max(start + min_chars, last_cut)
        cut = -1
        if target:
            # 目标长度优先：窗口内挑最接近 target 的一级标点（同级边界择优）。
            # 用 str.find/rfind 在窗口内定位（C 级实现），避免逐字符 Python 循环。
            aim = min(max(start + target, scan_from), window_end)
            right, left = -1, -1
            for p in punct:
                i = text.find(p, aim, window_end)
                if i != -1 and (right == -1 or i < right):
                    right = i
                j = text.rfind(p, scan_from, aim)
                if j != -1 and j > left:
                    left = j
            if right != -1 and (left == -1 or (right - aim) <= (aim - left)):
                cut = right + 1
            elif left != -1:
                cut = left + 1
        else:
            cut = -1
            for p in punct:
                i = text.find(p, scan_from, window_end)
                if i != -1 and (cut == -1 or i + 1 < cut):
                    cut = i + 1
        if cut < 0:
            # 一级边界未命中 → 二级（；：，、）→ 三级（空白）→ 四级硬切
            # 降级查找必须限定在 [scan_from, window_end)，否则每块退化为全量扫描
            cut = BoundaryDetector.find_secondary(text, scan_from, window_end)
            if cut < 0:
                cut = BoundaryDetector.find_tertiary(text, scan_from, window_end)
            if cut < 0:
                cut = window_end
        if cut <= start:
            cut = window_end

        chunk = text[start:cut]
        if chunk.strip():
            chunks.append(chunk)
        if cut >= n:
            break
        last_cut = cut
        ov = overlap_chars
        if overlap_ratio:
            ov = max(ov, compute_overlap_chars(cut - start, overlap_chars, overlap_ratio))
            ov = min(ov, max_chars - 1)
        next_start = cut - ov
        start = next_start if next_start > start else start + 1
    return chunks


def split_text_by_sentences(
    text: str,
    min_chars: int = 100,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_sentences: int = 1
) -> List[str]:
    """（旧接口，兼容保留）标点切片 + 句子级重叠。

    切片主逻辑已统一为规格切片 split_text_recursive（overlap 以字符计），
    本接口仅额外按“句”追加重叠，供历史调用方使用。
    新代码请直接调用 split_text_recursive。

    Args:
        text: 输入文本
        min_chars: 最小切分距离（不足不切分）
        max_chars: 无标点时的强制切分距离
        overlap_sentences: 重叠句子数（0/负数表示不重叠）

    Returns:
        切分后的文本块列表
    """
    chunks = split_text_recursive(text, min_chars, max_chars, overlap_chars=0)
    if not overlap_sentences or overlap_sentences <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        prev_sents = [s for s in re.split(r'(?<=[。！？：…])', prev) if s.strip()]
        overlap = ''.join(prev_sents[-overlap_sentences:])
        out.append(overlap + cur)
    return out


def _chunk_by_sentences(
    text: str,
    min_chars: int,
    max_chars: int,
    overlap_sentences: int = 1,
    max_chunk_length: Optional[int] = None
) -> List[str]:
    """（旧私有入口，兼容保留）转调 split_text_by_sentences。

    max_chunk_length 已废弃：无标点超长段落由规格硬上限 max_chars 兜底。

    在旧有「一级边界（。！：？）→ 四级硬切」之上补齐降级链：
    仍超 max_chars 的单片依次尝试二级边界（；：，、）→ 三级边界（空白/换行）
    → 四级硬切，避免超长无句末标点的段落整段塞进向量库。
    签名保持不变；切片口径仅在「单片超限」这一支路上被细化。
    """
    chunks = split_text_by_sentences(text, min_chars, max_chars, overlap_sentences)
    out: List[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            out.extend(BoundaryDetector.split_oversized(chunk, max_chars, min_scan=min_chars))
        else:
            out.append(chunk)
    return out


def split_text_by_chapters(
    text: str,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    target_chars: Optional[int] = None,
    overlap_ratio: Optional[float] = None
):
    """
    按章节边界切片：每个章节独立切块，chunk 携带所属章节标题。

    章内切块走规格切片 split_text_recursive（标点优先 + 硬上限 + 字符重叠），
    章节标题仅作为元数据、不进入正文。

    Args:
        text: 输入全文
        min_chars: 最小切分距离（不足不切分）
        max_chars: 无标点时的强制切分距离
        overlap_chars: 相邻片段重叠字符数（重叠下限）
        target_chars: 目标块长（透传 split_text_recursive）
        overlap_ratio: 重叠比例（透传 split_text_recursive）

    Returns:
        (chunks, chapter_of_chunk):
            chunks: List[str] 文本块（不含章节标题行）
            chapter_of_chunk: List[str] 每个块所属的章节标题（无标题时为默认名）

    无章节标题（未识别到）时退化为整篇正文切片，章节名统一为“正文”。
    """
    boundaries = _find_chapter_boundaries(text)
    if not boundaries:
        chunks = split_text_recursive(text, min_chars, max_chars, overlap_chars,
                                      target_chars, overlap_ratio)
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
        for chunk in split_text_recursive(body, min_chars, max_chars, overlap_chars,
                                          target_chars, overlap_ratio):
            chunks.append(chunk)
            chapter_of_chunk.append(title)

    # 末章之后若有未归属正文（异常情况兜底）
    last_end = boundaries[-1][1]
    tail = '\n'.join(text.split('\n')[last_end:]).strip()
    if tail:
        for chunk in split_text_recursive(tail, min_chars, max_chars, overlap_chars,
                                          target_chars, overlap_ratio):
            chunks.append(chunk)
            chapter_of_chunk.append("正文")

    if not chunks:
        chunks = split_text_recursive(text, min_chars, max_chars, overlap_chars,
                                      target_chars, overlap_ratio)
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


# ===================== 分层切片：边界探测 + 父块构造 =====================
# 目标：把「切片粒度 = 召回粒度」解耦为「子块检索 + 父块召回 + 邻居扩展」。
# 本段全部为新增（新增常量 / 类 / 函数），不改动上方任何既有函数的签名与行为：
# 子块仍由 split_text_recursive / split_text_by_chapters 产出，
# 新增的只是「父块聚合」与「边界降级探测」这两层能力。

TARGET_CHUNK_CHARS = 400   # 子块目标长度（字符）
OVERLAP_MIN_CHARS = 50     # 子块重叠下限（字符）
OVERLAP_RATIO = 0.15       # 子块重叠比例（前块长度的 15%）
PARENT_MIN_CHARS = 800     # 父块长度下限（字符）
PARENT_MAX_CHARS = 1500    # 父块长度上限（字符）
PARENT_GROUP_MIN = 2       # 父块最少聚合子块数
PARENT_GROUP_MAX = 4       # 父块最多聚合子块数


class BoundaryDetector:
    """文段边界探测器（四级降级策略）。

    一级边界：句末标点 。！？!?… 及其后紧随的闭合引号 ”"』」
    二级边界：从句标点 ；;：:，,、
    三级边界：空白字符（\\s、\\n）
    四级边界：以上均无命中时按 max_chars 硬切（由调用方兜底）

    find_* 系列统一返回「切分点下标」（即边界之后的位置），无命中返回 -1。
    """

    PRIMARY_PUNCT = ("。", "！", "？", "!", "?", "…")
    CLOSING_QUOTES = ("”", '"', "』", "」")
    SECONDARY_PUNCT = ("；", ";", "：", ":", "，", ",", "、")
    _WHITESPACE = re.compile(r"\s")  # 三级边界预编译（避免每块重复编译）

    @classmethod
    def find_primary(cls, text: str, start: int = 0) -> int:
        """一级边界：第一个句末标点（连同其后闭合引号）之后的切分点；无命中返回 -1。"""
        if not text:
            return -1
        for i in range(max(0, start), len(text)):
            if text[i] in cls.PRIMARY_PUNCT:
                cut = i + 1
                while cut < len(text) and text[cut] in cls.CLOSING_QUOTES:
                    cut += 1
                return cut
        return -1

    @classmethod
    def find_secondary(cls, text: str, start: int = 0, limit: Optional[int] = None) -> int:
        """二级边界：第一个从句标点之后的切分点；无命中返回 -1。

        limit 给定时只在 [start, limit) 内查找（默认 None = 扫描到文末）。
        切片主链路必须传 limit，否则每块都会退化为全量扫描（O(n²)）。
        """
        if not text:
            return -1
        begin = max(0, start)
        end = len(text) if limit is None else max(0, min(int(limit), len(text)))
        best = -1
        for p in cls.SECONDARY_PUNCT:
            i = text.find(p, begin, end)
            if i != -1 and (best == -1 or i < best):
                best = i
        return best + 1 if best != -1 else -1

    @classmethod
    def find_tertiary(cls, text: str, start: int = 0, limit: Optional[int] = None) -> int:
        """三级边界：第一段空白（含换行）之后的切分点；无命中返回 -1。

        limit 语义同 find_secondary。
        """
        if not text:
            return -1
        begin = max(0, start)
        end = len(text) if limit is None else max(0, min(int(limit), len(text)))
        m = cls._WHITESPACE.search(text, begin, end)
        if not m:
            return -1
        cut = m.end()
        while cut < len(text) and text[cut].isspace():
            cut += 1
        return cut

    @classmethod
    def split_oversized(cls, text: str, max_chars: int, min_scan: int = 1) -> List[str]:
        """超长文本降级切分：二级 → 三级 → 四级（硬切）。

        仅在单段/单句长度超过 max_chars 时使用；切分只做位置裁剪、
        不改写任何字符，因此每片仍是原文的连续片段。

        Args:
            text: 待切分文本
            max_chars: 单片硬上限（字符）
            min_scan: 扫描起点相对片首的最小偏移（避免切出过碎的前片）

        Returns:
            文本片列表（按原文顺序、无重复、无空隙）
        """
        if not text:
            return []
        max_chars = max(1, int(max_chars))
        min_scan = max(1, min(int(min_scan), max_chars))
        if len(text) <= max_chars:
            return [text]

        pieces: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            if n - start <= max_chars:
                tail = text[start:]
                if tail.strip():
                    pieces.append(tail)
                break

            window_end = start + max_chars
            scan_from = min(start + min_scan, window_end - 1)
            cut = BoundaryDetector.find_secondary(text, scan_from, window_end)
            if cut < 0:
                cut = BoundaryDetector.find_tertiary(text, scan_from, window_end)
            if cut < 0:
                cut = window_end  # 四级：无任何可切点 → 硬切
            if cut <= start:
                cut = window_end

            piece = text[start:cut]
            if piece.strip():
                pieces.append(piece)
            start = cut
        return pieces


def compute_overlap_chars(
    prev_chunk_len: int,
    overlap_min: int = OVERLAP_MIN_CHARS,
    overlap_ratio: float = OVERLAP_RATIO
) -> int:
    """分层切片的相邻子块重叠长度：max(overlap_min, 前块长度 × overlap_ratio)。

    即规格中的 `max(50, 前块 × 15%)`。供需要按「句首对齐」重建重叠的
    调用方使用（既有 split_text_recursive 的字符重叠口径保持不变）。
    """
    try:
        prev_len = max(0, int(prev_chunk_len))
    except (TypeError, ValueError):
        prev_len = 0
    try:
        ratio = float(overlap_ratio)
    except (TypeError, ValueError):
        ratio = OVERLAP_RATIO
    base = max(0, int(overlap_min))
    return max(base, int(prev_len * max(0.0, ratio)))


def build_parent_chunks(
    chunks: List[str],
    chapter_of_chunk: List[str],
    parent_min: int = PARENT_MIN_CHARS,
    parent_max: int = PARENT_MAX_CHARS
) -> List[dict]:
    """把同章节内连续的 2-4 个子块聚合为父块（父块召回层）。

    规则：
    - 章节是硬边界：父块只由同一章节的连续子块聚合，绝不跨章节；
    - 聚合顺序为原文顺序，父块内部子块连续无空洞；
    - 每组 2-4 个子块（PARENT_GROUP_MIN/MAX）；
    - 字符数目标 [parent_min, parent_max]：贪心累加至达到 parent_min 即止，
      累加下一块会超出 parent_max 时提前收束；
    - 章节尾部残余不足 parent_min 时，若并入前一组仍不超 parent_max 且
      组内不超 PARENT_GROUP_MAX，则并入前一组（避免产出过碎的父块）。

    Args:
        chunks: 子块文本列表（split_text_recursive / split_text_by_chapters 的产物）
        chapter_of_chunk: 与 chunks 等长的章节标题列表
        parent_min: 父块长度下限（字符）
        parent_max: 父块长度上限（字符）

    Returns:
        父块列表，每项为：
            {
                "parent_id": "p1",              # 父块唯一标识（字符串）
                "text": "...",                  # 子块原文顺序拼接
                "chapter": "第一章 ...",         # 所属章节（父块不跨章节）
                "child_indices": [0, 1, 2],     # 覆盖的子块下标（0-based）
                "child_count": 3,               # 覆盖的子块数量
                "chunk_length": 900,            # 父块字符数
            }
    """
    if not chunks:
        return []

    parent_min = max(1, int(parent_min))
    parent_max = max(parent_min, int(parent_max))

    chapters = list(chapter_of_chunk or [])
    if len(chapters) < len(chunks):
        chapters.extend(["正文"] * (len(chunks) - len(chapters)))

    # 1) 先按章节切成「连续区间」——章节是硬边界，父块不跨章节
    runs: List[tuple] = []  # [(chapter, [子块下标, ...]), ...]
    for i in range(len(chunks)):
        chap = chapters[i]
        if runs and runs[-1][0] == chap:
            runs[-1][1].append(i)
        else:
            runs.append((chap, [i]))

    parents: List[dict] = []
    for chapter, idx_list in runs:
        lens = [len(chunks[i]) for i in idx_list]
        n = len(idx_list)

        # 2) 章内贪心聚合：连续 2-4 个子块，字符数贴近 [parent_min, parent_max]
        groups: List[List[int]] = []
        i = 0
        while i < n:
            members: List[int] = []
            total = 0
            while i + len(members) < n and len(members) < PARENT_GROUP_MAX:
                j = i + len(members)
                if members and total + lens[j] > parent_max:
                    break
                members.append(j)
                total += lens[j]
                if total >= parent_min and len(members) >= PARENT_GROUP_MIN:
                    break
            if not members:
                members = [i]
            groups.append(members)
            i += len(members)

        # 3) 章尾残余单薄组：并入前一组（不超 parent_max 且不超块数上限）
        if len(groups) >= 2:
            last = groups[-1]
            prev = groups[-2]
            if len(prev) + len(last) <= PARENT_GROUP_MAX and \
                    sum(lens[k] for k in prev) + sum(lens[k] for k in last) <= parent_max:
                groups[-2] = prev + last
                groups.pop()

        # 4) 物化父块
        for members in groups:
            child_indices = [idx_list[k] for k in members]
            parent_text = "".join(chunks[k] for k in child_indices)
            parents.append({
                "parent_id": f"p{len(parents) + 1}",
                "text": parent_text,
                "chapter": chapter,
                "child_indices": child_indices,
                "child_count": len(child_indices),
                "chunk_length": len(parent_text),
            })
    return parents
