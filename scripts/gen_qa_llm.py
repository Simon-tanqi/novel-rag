# -*- coding: utf-8 -*-
"""llm_gen_qa.py — 用项目自身 LLM 通道批量生成「自然问法」QA（扩样用，可复现）

流程：均匀抽章 → 把该章片段交给 LLM 出题 → 规则核验（答案必须是原文连续子串、
答案词不得出现在问句、答案需有定位性）→ 通过者入库。

用法：
  python llm_gen_qa.py --name zhetian --count 40            # 正式生成
  python llm_gen_qa.py --name zhetian --count 3 --dry-run   # 只试跑，不落盘
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api_client import APIClient, APIClientError  # noqa: E402
from config_manager import ConfigManager  # noqa: E402

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_JSON = re.compile(r"\{.*\}", re.S)
_lock = threading.Lock()

PROMPT = """你是小说检索测试集的出题专家。下面给出《{book}》其中一章的原文片段（来自 OCR，可能有少量错字）。

请生成 1 道「自然问法」检索测试题，要求：
1. 答案(answer) 必须是片段中明确出现的一个专名实体：人名 / 地名 / 法宝功法名 / 势力种族名 / 称号，长度 2~8 个字，必须逐字出现在片段原文中；不要选「叶凡」「庞博」这类主角名。
2. 问题(question) 要像读者提问的自然口吻，包含足以定位到这一章的场景/情节线索（例：「叶凡在荒古禁地中与旧识女子交手时使出、记载于成仙路上自古无人练成的绝世拳法叫什么？」）。
3. 问题中绝对不出现答案词本身，也不出现答案中任意连续 2 个及以上的字。
4. 问题不要直接照抄原文整句，需改写为提问句式；长度 20~90 字。
5. 只输出 JSON，格式：{{"question": "...", "answer": "...", "evidence": "片段中包含答案的完整原句"}}

原文片段（章节：{chapter}）：
{text}
"""


def load_llm():
    cfg = ConfigManager()
    models = cfg.get("models") or []
    m = next((x for x in models if not x.get("is_local")), models[0])
    return APIClient(m["api_url"], m["api_key"], m["model_id"]), m["name"]


def ask_one(client, book, chapter, text, idx):
    msg = PROMPT.format(book=book, chapter=chapter, text=text[:1400])
    try:
        resp = client.call_api(msg, enable_thinking=False, temperature=0.8, max_tokens=400)
    except APIClientError as e:
        return {"id": idx, "error": f"API: {e.message}"}
    except Exception as e:  # noqa: BLE001
        return {"id": idx, "error": f"{type(e).__name__}: {e}"}
    m = _JSON.search(resp or "")
    if not m:
        return {"id": idx, "error": "无 JSON 输出", "raw": (resp or "")[:200]}
    try:
        obj = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        return {"id": idx, "error": f"JSON 解析失败 {e}", "raw": m.group(0)[:200]}
    obj["id"] = idx
    obj["chapter"] = chapter
    obj["chunk_text"] = text
    return obj


def verify(obj, span_fn):
    """核验：答案须为原文连续子串、不出现在问句中、且具备定位性"""
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    ev = (obj.get("evidence") or "").strip()
    text = obj.get("chunk_text") or ""
    bad = []
    if not q or not a:
        return False, ["缺字段"]
    if not (18 <= len(q) <= 100):
        bad.append(f"问句长度异常({len(q)})")
    if a not in text and a not in ev:
        bad.append("答案不在原文中")
    if a in q:
        bad.append("答案出现在问句中")
    for gram in [a[i:i + 3] for i in range(len(a) - 2)]:
        if gram in q:
            bad.append(f"答案片段泄漏({gram})")
            break
    span = span_fn(a)
    if span > 30:
        bad.append(f"答案定位性差(分布{span}章)")
    if a in ("叶凡", "庞博", "李七夜", "遮天", "凡人"):
        bad.append("答案为高频主角名")
    return (len(bad) == 0), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--book", default="")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    vdb = ROOT / "data" / args.name / "vector_db"
    meta = json.loads((vdb / "metadata.json").read_text(encoding="utf-8"))
    book = args.book or args.name
    print(f"ℹ {args.name}: {len(meta)} 片段")

    # 实体 → 章节分布（核验定位性用；按需计算并缓存）
    text_by_ch = {}
    for c in meta:
        ch = c.get("chapter_title") or c.get("chapter") or ""
        t = c.get("text", "")
        if ch and (ch not in text_by_ch or len(t) > len(text_by_ch[ch])):
            text_by_ch[ch] = t
    _span_cache = {}

    def span_fn(word):
        if word in _span_cache:
            return _span_cache[word]
        n = sum(1 for t in text_by_ch.values() if word in t)
        _span_cache[word] = n
        return n

    rng = random.Random(args.seed)
    chapters = list(text_by_ch.keys())
    chapters = [c for c in chapters if len(text_by_ch[c]) >= 600]
    rng.shuffle(chapters)
    picks = chapters[: max(args.count * 3, 30)]

    client, model = load_llm()
    print(f"ℹ LLM: {model} | 候选章节 {len(picks)} | 目标 {args.count} 题")

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask_one, client, book, ch, text_by_ch[ch], i): ch
                for i, ch in enumerate(picks, 1)}
        for fu in futs:
            r = fu.result()
            with _lock:
                if "error" in r:
                    errors.append(r)
                else:
                    results.append(r)
    print(f"ℹ 原始返回 {len(results)} 条，错误 {len(errors)} 条")
    for e in errors[:5]:
        print("   err:", e.get("chapter", ""), e["error"])

    passed, rejected = [], []
    for r in results:
        ok, bad = verify(r, span_fn)
        (passed if ok else rejected).append((r, bad))
    print(f"ℹ 核验通过 {len(passed)} 条，拒绝 {len(rejected)} 条")
    for r, bad in rejected[:8]:
        print(f"   reject[{r['id']}] {bad} | Q={r.get('question','')[:50]} | A={r.get('answer','')}")

    passed = passed[: args.count]
    # 去重：同一章只留 1 题、同一答案词只留 1 题
    seen_ch, seen_a, dedup = set(), set(), []
    for r, _ in passed:
        ch, a = r["chapter"], (r.get("answer") or "").strip()
        if ch in seen_ch or a in seen_a:
            continue
        seen_ch.add(ch); seen_a.add(a); dedup.append(r)
    print(f"ℹ 去重后 {len(dedup)} 题（同章/同答案各留 1）")
    passed = [(r, []) for r in dedup]
    items = []
    for i, (r, _) in enumerate(passed, 1):
        items.append({
            "id": i, "type": "scene", "question": r["question"].strip(),
            "answer_terms": [r["answer"].strip()], "gold_chapter": r["chapter"],
            "evidence": (r.get("evidence") or "").strip(),
        })
    out = Path(args.out) if args.out else (ROOT / "temp" / f"llm_qa_{args.name}.json")
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"name": args.name, "book": book, "items": items},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✓ 已写入 {out}")
    for it in items:
        print(f"{it['id']:>3} | {it['gold_chapter']} | A={it['answer_terms'][0]}")
        print(f"      Q: {it['question']}")


if __name__ == "__main__":
    main()
