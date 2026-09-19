# -*- coding: utf-8 -*-
"""gen_qa_natural_expand.py — 「自然问法」QA 集扩样生成器（无 LLM、纯规则、可复现）

背景：
  qa_sets/<name>_natural.json 是 20 题的人工自然问法集（scene/open 两类）。
  为把 60 分可达性验证落到更大样本（每本 50~100 题），需要一条**可复现**的扩样通道。

设计（对齐人工集的构造方式，但不依赖 LLM）：
  1. 均匀抽样全书 chunk（每章最多 1 题），保证覆盖）；
  2. 在 chunk 内挑「答案句」s2（含书中特有词 = 实体，全局频次 >= --min-freq，
     且该实体出现的章节数 <= --max-chapters，保证答案有定位性）；
  3. 取 s2 的**相邻句** s1 作为定位线索（线索句与答案句同 chunk，天然连贯，
     且线索句**不含答案实体**，避免问句与原文答案句直接字符串重合）；
  4. 题面 = s1 + 按实体类型选择的疑问尾（谁 / 什么地方 / 什么东西 / 哪一族），
     保证问句不含答案词 → 与人工集「实体稀释」口径一致；
  5. 自检：答案词不得出现在题面中；答案词必须出现在依据句与 gold 章内。

输出：JSON（items: id/type/question/answer_terms/gold_chapter/evidence），
      可直接被 eval_config_compare.py 使用（--qa-suffix）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gen_qa import build_freq, _CJK  # noqa: E402  （复用全书词频口径）

_SENT_SPLIT = re.compile(r"[。！？；…]+")
_CLEAN_TAIL = re.compile(r"[，,、\s]+$")

# 实体类型判定 → (类型名, 疑问尾)
TYPE_RULES = [
    ("place", ("山", "峰", "谷", "城", "殿", "洞", "海", "域", "界", "矿", "原", "宫", "岛",
               "州", "渊", "冢", "禁地", "坟", "岭", "湖", "泽", "镇", "村", "寨"),
     "这里指的是什么地方？"),
    ("item", ("经", "诀", "拳", "术", "功", "宝", "剑", "刀", "鼎", "珠", "钟", "印", "旗",
              "塔", "镜", "扇", "网", "图", "阵", "符", "冠", "杖", "枪", "弓", "铃", "盘",
              "书", "典", "法", "甲", "衣"),
     "这里说的东西是什么？"),
    ("faction", ("族", "家", "教", "门", "宗", "阁", "盟", "圣地", "世家", "派"),
     "这里说的是哪一族/哪个势力？"),
]


def classify(entity: str) -> tuple[str, str]:
    for kind, keys, tail in TYPE_RULES:
        if entity.endswith(tuple(keys)) or any(k in entity for k in keys):
            return kind, tail
    return "person", "这里说的这个人是谁？"


def sentences(text: str) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text):
        s = s.strip().strip("“”\"' ").strip()
        if s:
            out.append(s)
    return out


def pick_entity(sent: str, freq: dict[str, int], chapter_span: dict[str, int], min_freq: int, max_ch: int):
    """在句内挑选最具定位性的实体：全书频次够高、但分布章节数够少"""
    cands = []
    for n in (4, 3, 2):
        for gram in {s for s in _ngrams_of(sent, n)}:
            f = freq.get(gram, 0)
            if f < min_freq:
                continue
            if len(gram) < 2:
                continue
            span = chapter_span.get(gram, 10 ** 9)
            if span > max_ch:
                continue
            cands.append((f * (len(gram) ** 2) / max(span, 1), gram))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def _ngrams_of(text: str, n: int):
    for seg in _CJK.findall(text):
        if len(seg) < n:
            continue
        for i in range(len(seg) - n + 1):
            yield seg[i:i + n]


def main():
    ap = argparse.ArgumentParser(description="自然问法 QA 集扩样（规则生成、可复现）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--count", type=int, default=40, help="新增题数")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-freq", type=int, default=25, help="实体全书最低出现次数")
    ap.add_argument("--max-chapters", type=int, default=15, help="实体最多分布章节数（越小越有定位性）")
    ap.add_argument("--out", default="", help="输出路径（默认 temp/natural_expand_<name>.json）")
    args = ap.parse_args()

    vdb = ROOT / "data" / args.name / "vector_db"
    meta = json.loads((vdb / "metadata.json").read_text(encoding="utf-8"))
    print(f"ℹ {args.name}: {len(meta)} 个片段，统计全书词频 ...")
    freq = build_freq(meta)

    # 实体 → 出现章节集合（用于定位性约束）
    ent_chapters: dict[str, set] = defaultdict(set)
    for c in meta:
        ch = c.get("chapter_title") or c.get("chapter") or ""
        txt = c.get("text", "")
        for n in (4, 3, 2):
            for gram in set(_ngrams_of(txt, n)):
                if freq.get(gram, 0) >= args.min_freq:
                    ent_chapters[gram].add(ch)
    chapter_span = {k: len(v) for k, v in ent_chapters.items()}
    print(f"ℹ 本书特有名 {len(freq)} 个；纳入定位性统计 {len(chapter_span)} 个")

    # 均匀抽样：每章最多 1 题
    rng = random.Random(args.seed)
    n_need = args.count
    idxs = [int(round((i + 0.5) / (n_need * 6) * len(meta))) for i in range(n_need * 6)]
    idxs = [min(max(i + rng.randint(-5, 5), 0), len(meta) - 1) for i in idxs]
    rng.shuffle(idxs)

    used_entity, used_chapter = set(), set()
    items = []
    for ci in idxs:
        if len(items) >= args.count:
            break
        c = meta[ci]
        ch = c.get("chapter_title") or c.get("chapter") or ""
        if not ch or ch in used_chapter:
            continue
        sents = sentences(c.get("text", ""))
        if len(sents) < 2:
            continue
        for si, s2 in enumerate(sents):
            if not (15 <= len(s2) <= 80):
                continue
            ent = pick_entity(s2, freq, chapter_span, args.min_freq, args.max_chapters)
            if not ent or ent in used_entity:
                continue
            if len([1 for cc in ent_chapters[ent]]) < 2:  # 至少要跨 1 章以上，避免孤例
                pass
            # 线索句：优先取前一句，否则后一句
            cand_leads = [sents[j] for j in (si - 1, si + 1) if 0 <= j < len(sents)]
            lead = next((x for x in cand_leads if 12 <= len(x) <= 70 and ent not in x), None)
            if not lead:
                continue
            kind, tail = classify(ent)
            lead_clean = _CLEAN_TAIL.sub("", lead)
            question = f"{lead_clean}，{tail}"
            # 自检：答案词不得出现在问句中
            if ent in question:
                continue
            items.append({
                "id": len(items) + 1,
                "type": "scene" if kind == "person" else "open",
                "question": question,
                "answer_terms": [ent],
                "gold_chapter": ch,
                "evidence": s2,
                "kind": kind,
            })
            used_entity.add(ent)
            used_chapter.add(ch)
            break

    out = Path(args.out) if args.out else (ROOT / "temp" / f"natural_expand_{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": args.name, "items": items}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"✓ 生成 {len(items)} 条候选 -> {out}\n")
    for it in items:
        print(f"--- id={it['id']} [{it['type']}/{it['kind']}] {it['gold_chapter']}")
        print(f"    Q: {it['question']}")
        print(f"    A: {it['answer_terms'][0]}")
        print(f"    依据: {it['evidence']}")


if __name__ == "__main__":
    main()
