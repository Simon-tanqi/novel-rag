# -*- coding: utf-8 -*-
"""novel_context.py — 小说场景检索增强：指代消解前缀 + 章节聚合重排

解决两类小说检索特有问题：

1) 指代消解（Coreference / 主语注入）
   原文切片后，「他/她/他们」等指代与其先行词可能被切到不同 chunk，
   导致问「林雷的剑叫什么」时检索不到只写「他拔出那柄剑」的片段。
   方案：切片阶段对每个 chunk 开头注入「该段最近一次出现的主角名」，
   向量化与检索都基于注入后的文本，使指代片段可被先行词召回。

2) 章节聚合重排（Chapter-level aggregation）
   小说是跨段连续叙事，同一事件横跨多个 chunk，相邻章节语义高度相似，
   纯向量 top-k 容易「一章内重复命中、漏掉真正关键的另一章」。
   方案：宽松召回 → 按 chapter 聚合取每章最高分 → 章节间按分数排序，
   保证 top_k 结果覆盖多个相关章节（多样性）。
"""
import re
from typing import Dict, List, Optional

# ===================== 主角表 =====================
# 示例主角表，覆盖本项目两部小说的主要角色/别称；
# 更通用的做法是 ingest 时从原文自动挖掘（见 build_character_table）。
DEFAULT_CHARACTERS = [
    "叶凡", "庞博", "姬紫月", "黑皇", "姜太虚", "狠人", "无始",
    "林雷", "迪莉娅", "贝贝", "德斯黎", "霍格", "林蒙", "盘龙戒指",
]

# 行首指代词：行首出现时说明主语指代前文
COREFERENCE_PREFIXES = ("他", "她", "它", "自己")

# 主角表中不适合做「主语注入」的实体（物品/地名等只作召回辅助）
_NON_SUBJECT_NAMES = {"盘龙戒指"}

# 跨 chunk 参考窗口大小（行数）：指代距离通常不远
_LOOKBACK_LINES = 12


def build_character_table(text: str, known: Optional[List[str]] = None) -> List[str]:
    """构建主角表：已知名单 + 按词频从原文补充 2-4 字人名候选

    启发式：统计 2-4 字连续中文片段词频，取高频且非停用词的候选合并。
    实际工程可用 jieba + 词性过滤 + 人物共现，这里保持零依赖。
    """
    characters = list(known or DEFAULT_CHARACTERS)

    # 人名通常 2-3 字：对每个纯中文 run 滑窗统计 2-gram/3-gram 词频
    # （不能用贪婪定长切割，会把「叶凡与庞博同行」切碎导致人名统计不到）
    freq: Dict[str, int] = {}
    for run in re.findall(r'[\u4e00-\u9fff]{2,}', text):
        for start in range(len(run)):
            for size in (2, 3):
                if start + size <= len(run):
                    w = run[start:start + size]
                    freq[w] = freq.get(w, 0) + 1

    stop = {"我们", "你们", "他们", "自己", "一个", "什么", "没有",
            "已经", "可以", "知道", "时候", "现在", "就是",
            "忽然", "突然", "如何", "这样", "那个", "这个"}
    # 以常见单字动词/虚词结尾的多半是短语（同行/又说/点头），非人名
    suffix_noise = "说笑道行走看着过是有了在也又就都很去来上下和与"

    candidates = [
        w for w, c in freq.items()
        if c >= 3 and w not in stop and w[-1] not in suffix_noise
    ]
    top = sorted(candidates, key=lambda w: -freq[w])[:30]
    for w in top:
        if w not in characters:
            characters.append(w)
    return characters


def _looks_like_quote_line(line: str) -> bool:
    """对话行（以引号结尾的短行）：其主语是「说话人」，不适合做内容主语参考"""
    return bool(re.search(r'[」』”"\']\s*$', line)) and len(line) < 80


def _is_chapter_title_line(line: str) -> bool:
    """轻量章节标题判定（与 utils 一致，避免循环依赖）"""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    if re.search(r'[。！？；…]', stripped):
        return False
    return bool(re.match(
        r'^\s*(?:第[一二三四五六七八九十百千万零〇两\d]+[章节回卷部集篇话]\s*[^\n]{0,60}|序章|楔子|引子|番外|后记|尾声|完本感言)',
        stripped
    ))


def _first_name_in_line(line: str, characters: List[str]) -> Optional[str]:
    """行内第一个出现的主角名（叙述句主语通常在句首附近）"""
    for name in characters:
        if name in line:
            return name
    return None


def _name_events_in_line(line: str, characters: List[str]) -> List[str]:
    """行内主语候选：仅取句首位置（前 1/3 或行首前 12 字内）出现的名字

    名字出现在句首附近时最可能是叙述主语（「林雷拜入剑庐」「霍格传他」），
    出现在句中/句尾时多为宾语（「师从霍格」「父亲霍格」），不作主语候选。
    对话行（引号结尾）由调用方提前跳过。
    """
    head_len = max(10, len(line) // 3)
    head = line[:head_len]
    found = []
    for name in characters:
        idx = head.find(name)
        if idx >= 0:
            found.append((idx, name))
    found.sort(key=lambda t: t[0])
    return [name for _, name in found]


def inject_coref_prefix(
    chunks: List[str],
    chapter_titles: List[str],
    characters: Optional[List[str]] = None,
    max_prefix: int = 1,
) -> List[str]:
    """为疑似「以指代开头」的 chunk 注入主语前缀（原文保留，仅检索/向量化用）

    两遍处理：
    - 第一遍：找 chunk 内第一个「行首为指代词」的行，若能从参考窗口
      （前文 + 本 chunk 该行之前）解析出主角名 → 记录注入；
    - 第二遍：把本 chunk 中所有「显式出现主角名」的行并入参考窗口，
      供后续 chunk 解析（含 break 之后的行，保证跨 chunk 连续性）。

    注入格式：『【前文主语：叶凡】\n<原文>』

    Args:
        chunks: 原始 chunk 列表（不含章节标题）
        chapter_titles: 与 chunks 等长的章节标题列表
        characters: 主角表（默认 DEFAULT_CHARACTERS）
        max_prefix: 每个 chunk 最多注入几处前缀（当前实现只注入 chunk 开头）

    Returns:
        与 chunks 等长的注入后列表；未命中指代的 chunk 原样返回
    """
    if characters is None:
        characters = DEFAULT_CHARACTERS
    subject_names = [c for c in characters if c not in _NON_SUBJECT_NAMES]

    result: List[str] = []
    refs: List[str] = []  # 主角事件栈（旧→新，仅存主角名），跨 chunk 连续
    prev_chapter: Optional[str] = None

    for chunk, chapter in zip(chunks, chapter_titles):
        # 章节切换时：新章大概率换场景，主语继承上一章结尾主角
        # （若上一章结尾已由 refs 尾部记录，直接延续即可；事件栈天然覆盖）
        lines = chunk.split("\n")

        # ---- 第一遍：定位首个可解析的指代句 ----
        inject_name: Optional[str] = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(COREFERENCE_PREFIXES):
                # 该行本身是否点名（如「他叫叶凡」）优先
                inline = _first_name_in_line(stripped, subject_names)
                if inline:
                    inject_name = inline
                    break
                # 回溯参考栈：最近一个主角名事件即最可能主语
                if refs:
                    inject_name = refs[-1]
                    break

        # ---- 第二遍：把本 chunk 内显式点名事件并入参考栈 ----
        local_events: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _looks_like_quote_line(stripped):
                continue  # 对话行：主语是说话人，事件不可靠
            local_events.extend(_name_events_in_line(stripped, subject_names))
        refs.extend(local_events)
        refs = refs[-_LOOKBACK_LINES:]
        prev_chapter = chapter

        # ---- 输出 ----
        if inject_name:
            result.append(f"【前文主语：{inject_name}】\n{chunk}")
        else:
            result.append(chunk)

    return result


def chapter_aggregate_rerank(
    hits: List[Dict],
    top_k: int,
) -> List[Dict]:
    """章节聚合重排：top_k 结果覆盖多个章节（多样性）

    Args:
        hits: 宽松召回的命中列表（含 score, chapter），通常为 top_k*N
        top_k: 目标返回条数

    Returns:
        每章最高分条目按章节分数降序，截断到 top_k 条；
        若章节信息缺失（未知），保持原有相对顺序拼在末尾。
    """
    if not hits:
        return []

    # 1) 按章节聚合：每章保留最高分条目
    best_per_chapter: Dict[str, Dict] = {}
    no_chapter: List[Dict] = []
    for h in hits:
        chap = h.get("chapter") or "未知"
        if chap == "未知":
            no_chapter.append(h)
            continue
        cur = best_per_chapter.get(chap)
        if cur is None or h.get("score", 0) > cur.get("score", 0):
            best_per_chapter[chap] = h

    # 2) 章节间按最高分排序
    ordered = sorted(best_per_chapter.values(),
                     key=lambda h: h.get("score", 0), reverse=True)

    # 3) 截断；「未知」章节条目兜底拼接（数量极少）
    top = ordered[:top_k]
    if len(top) < top_k and no_chapter:
        top.extend(no_chapter[: top_k - len(top)])
    return top
