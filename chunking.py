"""
chunking.py — 结构预处理 + 场景分层 + token 切片引擎（新切片逻辑的唯一实现）

对应需求（2026-09 切片逻辑改造）：
1. 结构预处理：统一换行、识别「第X章/卷/序章/番外」标题、剔除章首章尾空行、
   连续空行归一为场景边界；
2. 章 = 一级元数据（而非唯一 chunk）：章内按空行/场景切，场景内按段落与句子聚合；
3. 以 token 为准分块：目标 ≈ 嵌入模型上限的 70%-80%（中文约 300-500 token /
   400-700 字），硬上限约 480 token，最小 200-250 字，低于下限则合并相邻段落；
4. 切分优先级：空行/段落边界 > 句末标点（。！？……；）> 逗号/冒号 > 硬切，
   并在目标长度附近**前后搜索最佳切点**（而非从 min 起找第一个标点），
   切点需通过引号/括号闭合检查；
5. 相邻块重叠 10%-15%（约 50-100 字），在句边界对齐；
6. 每个 chunk 前缀 = 书名 + 章节标题 + 场景摘要；元数据记录 book_id / chapter_id /
   chapter_title / scene_id / chunk_index / 起止位置 / token 数 / prev / next；
7. 章节变更只重切该章：每章落 chapter_hash 指纹，供 step2 增量复用（未变章
   直接复用旧向量与旧元数据）。

本模块只做「文本 → 块记录」，不依赖嵌入模型；向量化与落盘由 step2_split_embed 编排。
token 估算为启发式（中文 0.72 token/字），窗口搜索在字符坐标系完成，
换算比例由 utils.resolve_split_spec 的 chars_per_token 统一发放。
"""
import hashlib
import re
from typing import Dict, List, Optional

from utils import (
    CHARS_PER_TOKEN,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    OVERLAP_MAX_CHARS,
    OVERLAP_RATIO,
    find_chapter_boundaries,
    resolve_split_spec,
)

# ===================== token 估算 =====================
# 中文小说场景的经验换算：1 汉字 ≈ 0.72 token，1 英文词 ≈ 1.3 token，
# 其余字符（标点/数字/空白）≈ 0.35 token。用于把「token 规格」映射到字符窗口，
# 并记录每个 chunk 的实际 token 数（供调参与对账）。
_HAN_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
_LATIN_WORD_RE = re.compile(r'[A-Za-z]+')


def estimate_tokens(text: str) -> int:
    """估算文本 token 数（启发式，中文为主场景足够稳定）。"""
    if not text:
        return 0
    han = len(_HAN_RE.findall(text))
    words = _LATIN_WORD_RE.findall(text)
    latin_chars = sum(len(w) for w in words)
    other = max(0, len(text) - han - latin_chars)
    return int(round(han * 0.72 + len(words) * 1.3 + other * 0.35))


# ===================== 结构预处理 =====================
# 归一后：段落之间恰好 1 个空行（场景边界），章首/章尾无空行。
_INDENT_RE = re.compile(r'[ \t\u3000]+')


def normalize_newlines(text: str) -> str:
    """统一换行与行内空白：\\r\\n / \\r / U+2028 / U+2029 → \\n；全角空格 → 半角。"""
    if not text:
        return ""
    text = (text.replace('\r\n', '\n').replace('\r', '\n')
                .replace('\u2028', '\n').replace('\u2029', '\n')
                .replace('\u0085', '\n').replace('\u3000', ' '))
    return _INDENT_RE.sub(' ', text)


def _normalize_blank_lines(lines: List[str]) -> List[str]:
    """连续空行归一为 1 个空行（= 场景边界）；剔除多余行尾空白。"""
    out: List[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if out and not blank:
                out.append("")
            blank = True
        else:
            out.append(line.rstrip())
            blank = False
    return out


def _split_scene_lines(lines: List[str]) -> List[List[str]]:
    """按空行把章内行序列切成场景（每个场景是若干非空行）。"""
    scenes, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line)
        elif cur:
            scenes.append(cur)
            cur = []
    if cur:
        scenes.append(cur)
    return scenes


def _chapter_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]


def scene_summary(scene_text: str, max_chars: int = 40) -> str:
    """场景摘要：取场景首句（截断），供 chunk 前缀与元数据使用。"""
    body = re.sub(r'\s+', ' ', (scene_text or "").strip())
    if not body:
        return ""
    parts = re.split(r'(?<=[。！？…；])', body, maxsplit=1)
    head = parts[0].strip() if parts and parts[0].strip() else body
    return head[:max_chars]


def parse_structure(text: str) -> Dict:
    """结构预处理：统一换行 → 识别标题 → 剔章首尾空行 → 空行归一为场景边界。

    Returns:
        {
          "text": 归一化后的全文（章之间以空行相连）,
          "chapters": [
             {
               "chapter_id": "c1",
               "chapter_title": "第1章 ...",
               "hash": "ab12...",          # 章节正文指纹（增量重切依据）
               "text": 章正文（场景以空行分隔）,
               "start": 0, "end": 512,     # 归一化全文中的字符区间
               "scenes": [
                   {"scene_id": "c1_s1", "text": ..., "start": 0, "end": 120,
                    "summary": "..."},
               ],
             }, ...
          ],
        }
    """
    normalized = normalize_newlines(text or "")
    lines = normalized.split('\n')
    boundaries = find_chapter_boundaries(normalized)

    raw_chapters: List[Dict] = []
    if boundaries:
        # 首个标题之前的内容（书名页/前言）单列一章，避免被丢弃
        if boundaries[0][0] > 0:
            raw_chapters.append({"title": "正文", "lines": lines[:boundaries[0][0]]})
        for start, end, title in boundaries:
            raw_chapters.append({"title": title, "lines": lines[start + 1:end]})
    else:
        raw_chapters.append({"title": "正文", "lines": lines})

    chapters: List[Dict] = []
    cursor = 0  # 归一化全文坐标
    for raw in raw_chapters:
        body_lines = _normalize_blank_lines(raw["lines"])
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        scene_texts: List[str] = []
        for scene_lines in _split_scene_lines(body_lines):
            st = "\n".join(scene_lines).strip()
            if st:
                scene_texts.append(st)
        if not scene_texts:
            continue

        chapter_id = f"c{len(chapters) + 1}"
        chapter_text = "\n\n".join(scene_texts)

        scenes: List[Dict] = []
        pos = 0
        for i, st in enumerate(scene_texts, 1):
            scenes.append({
                "scene_id": f"{chapter_id}_s{i}",
                "text": st,
                "start": pos,
                "end": pos + len(st),
                "summary": scene_summary(st),
            })
            pos += len(st) + 2  # 场景间空行 "\n\n"

        chapters.append({
            "chapter_id": chapter_id,
            "chapter_title": raw["title"],
            "hash": _chapter_hash(chapter_text),
            "text": chapter_text,
            "start": cursor,
            "end": cursor + len(chapter_text),
            "scenes": scenes,
        })
        cursor += len(chapter_text) + 2

    return {"text": normalized, "chapters": chapters}


# ===================== 切点搜索 =====================
# 层级优先级（高 → 低）：空行/段落边界 > 句末标点（含闭合引号）> 逗号/冒号 > 硬切
_SENTENCE_END_RE = re.compile(r'[。！？…；!?;]+[”’」』）】]*')
_COMMA_RE = re.compile(r'[，、：,:]+')
_NEWLINE_RE = re.compile(r'\n+')
_CLOSING_CHARS = "”’」』\"')）】》"
_QUOTE_PAIRS = (("“", "”"), ("「", "」"), ("『", "』"), ("‘", "’"),
                ("（", "）"), ("《", "》"), ("【", "】"), ("(", ")"))


def is_balanced(segment: str) -> bool:
    """引号/括号闭合检查：各类成对符号左右数量必须一致。"""
    for open_ch, close_ch in _QUOTE_PAIRS:
        if segment.count(open_ch) != segment.count(close_ch):
            return False
    return True


def _newline_candidates(text: str, scan_from: int, window_end: int) -> List[int]:
    """一级候选：换行（段落/空行边界）之后的首个非空白字符位置。"""
    cands: List[int] = []
    for m in _NEWLINE_RE.finditer(text, scan_from, window_end):
        pos = m.end()
        while pos < window_end and text[pos] in ' \t':
            pos += 1
        if pos > scan_from:
            cands.append(pos)
    return cands


def _regex_candidates(pattern: re.Pattern, text: str, scan_from: int, window_end: int) -> List[int]:
    """二/三级候选：标点（连同尾随闭合引号）之后的位置。"""
    return [m.end() for m in pattern.finditer(text, scan_from, window_end)
            if m.end() > scan_from]


def _pick_candidate(cands: List[int], aim: int, text: str, start: int,
                    check_balance: bool = True, depth: int = 8) -> int:
    """在候选点中挑「离目标长度最近」者（左右同时搜索），并做闭合检查。"""
    if not cands:
        return -1
    ordered = sorted(cands, key=lambda p: (abs(p - aim), p))
    for pos in ordered[:depth]:
        if not check_balance or is_balanced(text[start:pos]):
            return pos
    return -1


def find_best_cut(text: str, start: int, spec: Dict) -> int:
    """在 [start+min_chars, start+max_chars] 窗口内按优先级搜索最佳切点。

    与旧实现的关键差异：**以目标长度为中心前后搜索**（旧实现从 min_chars 起
    取第一个标点，导致块长全部偏短），且每个候选点需通过引号/括号闭合检查。
    """
    n = len(text)
    min_chars = spec["min_chars"]
    max_chars = spec["max_chars"]
    target_chars = spec["target_chars"]

    window_end = min(n, start + max_chars)
    scan_from = start + min_chars
    if window_end - start <= min_chars:
        return window_end
    aim = min(max(start + target_chars, scan_from), window_end)

    # 1) 空行/段落边界
    cut = _pick_candidate(_newline_candidates(text, scan_from, window_end), aim, text, start)
    if cut > start:
        return cut

    # 2) 句末标点（。！？……；）
    cut = _pick_candidate(_regex_candidates(_SENTENCE_END_RE, text, scan_from, window_end),
                          aim, text, start)
    if cut > start:
        return cut

    # 3) 逗号/冒号
    cut = _pick_candidate(_regex_candidates(_COMMA_RE, text, scan_from, window_end),
                          aim, text, start)
    if cut > start:
        return cut

    # 4) 硬切：落在目标长度；若切在引号内则后移到最近闭合字符之后（≤30 字）
    cut = aim if aim > start else window_end
    if not is_balanced(text[start:cut]):
        moved = -1
        for ch in _CLOSING_CHARS:
            i = text.find(ch, cut, min(window_end + 30, n))
            if i != -1 and (moved == -1 or i < moved):
                moved = i
        if moved != -1 and moved + 1 > start:
            cut = moved + 1
    return max(cut, min(window_end, start + 1))


def overlap_start(text: str, cut: int, start: int, spec: Dict) -> int:
    """相邻块重叠起点：[cut - 重叠长度, cut)，并在句边界对齐。

    重叠长度 = clamp(前块长度 × ratio, overlap_min_chars, overlap_max_chars)
    （10%-15% / 50-100 字）；对齐时把起点后移到最近句末标点之后，
    保证重叠部分从完整句子开始。
    """
    prev_len = cut - start
    ov = int(round(prev_len * spec.get("overlap_ratio", OVERLAP_RATIO)))
    ov = max(spec.get("overlap_chars", 50), ov)
    ov = min(ov, spec.get("overlap_max_chars", OVERLAP_MAX_CHARS))
    pos = cut - ov
    if pos <= start:
        pos = start

    aligned = -1
    for ch in "。！？…":
        i = text.rfind(ch, start, pos)
        if i != -1 and i > aligned:
            aligned = i
    if aligned != -1:
        cand = aligned + 1
        while cand < cut and text[cand] in _CLOSING_CHARS:
            cand += 1
        # 对齐后重叠不得超出上限太多，否则保留字符对齐
        if cand > start and (cut - cand) <= int(ov * 1.5) + 20:
            pos = cand
    return pos


def _merge_short_tail(blocks: List[Dict], text: str, spec: Dict) -> List[Dict]:
    """尾块低于最小长度时并入前一块（低于 200-250 字则合并相邻段落）。"""
    if len(blocks) < 2:
        return blocks
    min_chars = spec["min_chars"]
    max_chars = spec["max_chars"]
    tail = blocks[-1]
    prev = blocks[-2]
    if len(tail["text"]) >= min_chars:
        return blocks
    merged_len = len(prev["text"]) + len(tail["text"])
    if merged_len > max_chars:
        return blocks
    prev["text"] = text[prev["start"]:tail["end"]]
    prev["end"] = tail["end"]
    blocks.pop()
    return blocks


def split_text_tokenwise(text: str, spec: Optional[Dict] = None) -> List[Dict]:
    """按 token 规格切分单段文本（章正文），返回 [{text, start, end}]。

    规则：不足 min_chars 不切；否则在 [min, max] 窗口内按
    「空行/段落 > 句末标点 > 逗号/冒号 > 硬切」搜索最接近目标长度的切点；
    相邻块重叠 10%-15%（50-100 字）并与句边界对齐。
    """
    spec = spec or resolve_split_spec()
    n = len(text or "")
    if n == 0 or not text.strip():
        return []

    blocks: List[Dict] = []
    start = 0
    while start < n:
        if n - start <= spec["min_chars"]:
            tail = text[start:]
            if tail.strip():
                blocks.append({"text": tail.strip(), "start": start, "end": n})
            break

        cut = find_best_cut(text, start, spec)
        if cut <= start:
            cut = min(n, start + spec["max_chars"])
        seg = text[start:cut]
        if seg.strip():
            blocks.append({"text": seg.strip(), "start": start, "end": cut})
        if cut >= n:
            break
        start = overlap_start(text, cut, start, spec)

    return _merge_short_tail(blocks, text, spec)


# ===================== 块记录装配 =====================
def _locate_scene(scenes: List[Dict], pos: int) -> Dict:
    """定位块起点所属场景（落在场景间隙时归入前一个场景）。"""
    found = scenes[0] if scenes else {}
    for scene in scenes:
        if scene["start"] <= pos:
            found = scene
        else:
            break
    return found


def build_chunk_prefix(book_title: str, chapter_title: str, scene: Dict) -> str:
    """chunk 前缀：书名 + 章节标题 + 场景摘要。"""
    parts = []
    if book_title:
        parts.append(f"《{book_title}》")
    if chapter_title:
        parts.append(chapter_title)
    summary = (scene or {}).get("summary", "")
    head = " ".join(parts)
    if summary:
        head = f"{head}｜{summary}" if head else summary
    return f"{head}\n" if head else ""


def build_chunks(
    text: str,
    book_id: str = "",
    book_title: str = "",
    spec: Optional[Dict] = None,
) -> Dict:
    """完整切片：结构预处理 → 章内场景分层切块 → 前缀与元数据装配。

    Returns:
        {
          "book_id": ..., "book_title": ..., "spec": {...},
          "chapters": [{chapter_id, chapter_title, hash, chunk_count, scene_count,
                        first_chunk_id, last_chunk_id}],
          "records": [ {text, prefix, embed_text, book_id, book_title,
                        chapter_id, chapter_title, chapter(兼容), scene_id,
                        scene_summary, chunk_index(章内), global_index,
                        chunk_id(兼容:全局 1-based), chunk_length, token_count,
                        total_token_count, start, end, chapter_hash,
                        prev_chunk_id, next_chunk_id, parent_id, coref_prefix} ],
          "parents": [...],
        }
    """
    spec = spec or resolve_split_spec()
    structure = parse_structure(text)

    records: List[Dict] = []
    chapter_metas: List[Dict] = []

    for chapter in structure["chapters"]:
        blocks = split_text_tokenwise(chapter["text"], spec)
        if not blocks:
            continue
        chunk_count = len(blocks)
        for i, block in enumerate(blocks, 1):
            scene = _locate_scene(chapter["scenes"], block["start"])
            body = block["text"]
            prefix = build_chunk_prefix(book_title, chapter["chapter_title"], scene)
            global_index = len(records)
            records.append({
                "text": body,
                "prefix": prefix,
                "embed_text": prefix + body,
                "book_id": book_id,
                "book_title": book_title,
                "chapter_id": chapter["chapter_id"],
                "chapter_title": chapter["chapter_title"],
                "chapter": chapter["chapter_title"],  # 兼容旧字段
                "scene_id": scene.get("scene_id", ""),
                "scene_summary": scene.get("summary", ""),
                "chunk_index": i,
                "global_index": global_index,
                "chunk_id": global_index + 1,          # 兼容旧字段（全局 1-based）
                "chunk_length": len(body),
                "token_count": estimate_tokens(body),
                "total_token_count": estimate_tokens(prefix + body),
                "start": chapter["start"] + block["start"],
                "end": chapter["start"] + block["end"],
                "chapter_hash": chapter["hash"],
                "prev_chunk_id": global_index if i > 1 else None,
                "next_chunk_id": (global_index + 2) if i < chunk_count else None,
                "parent_id": None,
                "coref_prefix": "",
            })
        chapter_metas.append({
            "chapter_id": chapter["chapter_id"],
            "chapter_title": chapter["chapter_title"],
            "hash": chapter["hash"],
            "chunk_count": chunk_count,
            "scene_count": len(chapter["scenes"]),
            "first_chunk_id": records[-chunk_count]["chunk_id"],
            "last_chunk_id": records[-1]["chunk_id"],
        })

    # 父块：同章连续 2-4 个子块聚合（章节为硬边界，不跨章）
    from utils import build_parent_chunks
    parents = build_parent_chunks(
        [r["text"] for r in records],
        [r["chapter_title"] for r in records],
    )
    for parent in parents:
        for child_idx in parent.get("child_indices", []):
            if 0 <= child_idx < len(records):
                records[child_idx]["parent_id"] = parent["parent_id"]
        parent["chapter_id"] = records[parent["child_indices"][0]]["chapter_id"] \
            if parent.get("child_indices") else ""
        parent["child_chunk_ids"] = [records[i]["chunk_id"]
                                     for i in parent.get("child_indices", [])]

    return {
        "book_id": book_id,
        "book_title": book_title,
        "spec": spec,
        "chapters": chapter_metas,
        "records": records,
        "parents": parents,
    }


__all__ = [
    "estimate_tokens",
    "normalize_newlines",
    "scene_summary",
    "parse_structure",
    "is_balanced",
    "find_best_cut",
    "overlap_start",
    "split_text_tokenwise",
    "build_chunk_prefix",
    "build_chunks",
    "CHARS_PER_TOKEN",
    "MIN_CHUNK_CHARS",
    "MAX_CHUNK_CHARS",
]
