"""
eval_recall_anchor.py — 无 LLM 召回基线评测（锚定式 / 章节式 / 关键词式）

与 eval_retrieval.py 的区别：
  eval_retrieval.py         — 仅按「预期章节」前缀判定命中（demo 内建 QA 适用）
  eval_recall_anchor.py     — 三种判定模式，适配生成 QA 与人工 QA：
      chapter  命中片段.chapter 与预期章节前 4 字前缀相同
      anchor   gold answer 逐字出现在命中片段中（或顺序覆盖 >= 85%），
               适合 gen_qa.py 生成的「原文锚定」QA
      keyword  gold answer 中抽取 >=2 个书中特有词命中片段，
               适合人工总结型 QA（答案非逐字原文）
      auto     先看 anchor 再看 keyword，任一命中即算命中（推荐默认）

输出：
  1) 控制台汇总表：Recall@k / MRR@5
  2) eval_out/<name>_recall_<mode>.json：逐题明细（含命中片段、判定结果），供复现审计

用法：
    python scripts/eval_recall_anchor.py --name zhetian --qa qa_sets/zhetian.json --top-k 1 3 5
    python scripts/eval_recall_anchor.py --name demo --top-k 3 --mode chapter   # demo 兼容模式
    python scripts/eval_recall_anchor.py --name zhetian --qa qa_sets/zhetian_human.json --mode keyword
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config_manager import ConfigManager  # noqa: E402
from project_manager import ProjectManager  # noqa: E402
from rag_retriever import RAGRetriever  # noqa: E402
from utils import DEFAULT_EMBEDDING_MODEL, load_env_file  # noqa: E402

load_env_file()

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_STOP = set("""一个 什么 怎么 自己 他们 我们 你们 这个 那个 因为 所以 但是 如果 已经
之后 然后 突然 直接 同时 开始 起来 下来 过去 回来 出来 上去 看到 看见 听到 说道
问道 笑道 一声 一眼 一步 手中 身上 脸上 眼中 心里 整个 如今 知道 时候 现在 就是
没有 不是 只是 还是 但是 不过 却 都 也 又 很 太 更 最 再 可 并 而 与 及 或 等
好 像 仿佛 似乎 应该 可能 大概 顿时 立刻 马上 终于 竟然 居然 难道 如何 为何 这时
那里 这里 前面 后面 旁边 上方 下方 虚空 天地 整个 无数 十分 非常 特别 一般 有些
同时 期间 其中 之中 之上 之下 之间 左右 上下 前后 内外 天地间 一瞬间 下一刻""".split())


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _ngrams(text: str, n: int):
    for seg in _CJK.findall(text):
        if len(seg) < n:
            continue
        for i in range(len(seg) - n + 1):
            yield seg[i : i + n]


def _seq_coverage(answer_norm: str, text_norm: str) -> float:
    """answer 中按顺序在 text 中可匹配的字符占比（贪心）"""
    if not answer_norm:
        return 0.0
    i = 0
    for ch in text_norm:
        if i < len(answer_norm) and ch == answer_norm[i]:
            i += 1
    return i / len(answer_norm)


def _chapter_hit(chapter: str, expected: str) -> bool:
    if not chapter or not expected:
        return False
    if chapter == expected:
        return True
    if len(expected) < 4:
        return False
    return chapter[:4] == expected[:4]


def build_retriever(project: dict, cfg: ConfigManager) -> RAGRetriever:
    vector_path = project.get("vector_db_path", "")
    embedding_spec = (
        cfg.get("embedding_model_path", "")
        or __import__("os").environ.get("EMBEDDING_MODEL", "").strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    return RAGRetriever.get_or_create(
        vector_path,
        reranker_model_path=cfg.get("reranker_model_path", ""),
        embedding_model_path=embedding_spec,
        vector_file=project.get("vector_file"),
        metadata_file=project.get("metadata_file"),
    )


def _project_terms(chunks) -> dict[str, int]:
    """项目中书名级特征词表（全局 >=30 次）"""
    counter: Counter[str] = Counter()
    for c in chunks:
        for gram in _ngrams(c.get("text", ""), 2):
            counter[gram] += 1
        for gram in _ngrams(c.get("text", ""), 3):
            counter[gram] += 1
    return {k: v for k, v in counter.items() if v >= 30 and k not in _STOP}


def _keyword_hit(answer: str, text_norm: str, freq: dict[str, int]) -> bool:
    cands = []
    for gram in _ngrams(answer, 2):
        if gram in freq and gram not in cands:
            cands.append(gram)
    for gram in _ngrams(answer, 3):
        if gram in freq and gram not in cands:
            cands.append(gram)
    cands = sorted(cands, key=lambda g: freq[g], reverse=True)[:3]
    if not cands:
        return False
    hit = sum(1 for g in cands if g in text_norm)
    return hit >= 2


def main():
    parser = argparse.ArgumentParser(description="无 LLM 召回基线评测")
    parser.add_argument("--name", required=True)
    parser.add_argument("--qa", default="", help="QA 文件（默认内置 demo QA）")
    parser.add_argument("--top-k", default="1 3 5")
    parser.add_argument("--mode", default="auto", choices=["auto", "chapter", "anchor", "keyword"])
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 题（0=全部）")
    args = parser.parse_args()

    cfg = ConfigManager()
    pm = ProjectManager()
    project = pm.get_project(args.name)
    if not project:
        print(f"✗ 项目不存在: {args.name}")
        sys.exit(1)
    retriever = build_retriever(project, cfg)

    if args.qa:
        qa = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    else:
        from novel_rag import DEMO_QA
        qa = DEMO_QA
    if args.limit:
        qa = qa[: args.limit]
    print(f"ℹ 评测项目: {args.name} | QA 数: {len(qa)} | 模式: {args.mode}")

    # keyword 模式需要项目特征词表
    freq = None
    meta_file = Path(project["vector_db_path"]) / "metadata.json"
    if meta_file.exists():
        freq = _project_terms(json.loads(meta_file.read_text(encoding="utf-8")))

    top_ks = [int(x) for x in args.top_k.split()]
    max_k = max(top_ks)
    stats = {k: {"chapter": 0, "anchor": 0, "keyword": 0, "any": 0} for k in top_ks}
    mrr_ch = 0.0
    mrr_any = 0.0
    details = []

    for i, item in enumerate(qa, 1):
        q = item.get("question", "")
        expected = item.get("chapter", "")
        answer = item.get("answer", "")
        if not q or not answer:
            print(f"⚠ 第 {i} 条缺 question/answer，跳过")
            continue
        hits = retriever.retrieve(q, top_k=max_k)
        ans_norm = _norm(answer)
        per = {"index": i, "question": q, "chapter": expected, "answer": answer,
               "hits": [{"chapter": h.get("chapter", ""), "score": round(float(h.get("score", 0)), 4),
                          "text": (h.get("text", "") or "")[:120]} for h in hits]}
        ok_ch_first = 0
        ok_any_first = 0
        for k in top_ks:
            for r, h in enumerate(hits[:k], 1):
                ch_ok = _chapter_hit(h.get("chapter", ""), expected)
                text_norm = _norm(h.get("text", ""))
                anchor_ok = _seq_coverage(ans_norm, text_norm) >= 0.85
                kw_ok = _keyword_hit(answer, text_norm, freq) if freq else False
                if args.mode == "chapter":
                    any_ok = ch_ok
                elif args.mode == "anchor":
                    any_ok = anchor_ok
                elif args.mode == "keyword":
                    any_ok = kw_ok
                else:
                    any_ok = anchor_ok or kw_ok
                if ch_ok:
                    stats[k]["chapter"] += 1
                    ok_ch_first = ok_ch_first or r
                if anchor_ok:
                    stats[k]["anchor"] += 1
                if kw_ok:
                    stats[k]["keyword"] += 1
                if any_ok:
                    stats[k]["any"] += 1
                    ok_any_first = ok_any_first or r
        if ok_ch_first:
            mrr_ch += 1.0 / ok_ch_first
        if ok_any_first:
            mrr_any += 1.0 / ok_any_first
        per["hit_rank_chapter"] = ok_ch_first
        per["hit_rank_any"] = ok_any_first
        details.append(per)

    n = len(qa)
    print("\n" + "=" * 70)
    print(f"无 LLM 召回基线 | 项目: {args.name} | QA: {n} 题 | 模式: {args.mode}")
    print("=" * 70)
    print(f"{'K':<5}{'章节命中 Recall@k':<18}{'锚定命中 Recall@k':<18}{'关键词命中 Recall@k':<18}{'综合命中 Recall@k':<18}")
    for k in top_ks:
        print(f"{k:<5}{stats[k]['chapter']/n:<18.3f}{stats[k]['anchor']/n:<18.3f}{stats[k]['keyword']/n:<18.3f}{stats[k]['any']/n:<18.3f}")
    print("-" * 70)
    print(f"MRR@5（章节）: {mrr_ch/n:.3f}   MRR@5（综合）: {mrr_any/n:.3f}")
    print("=" * 70)

    out_dir = ROOT / "eval_out"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.name}_recall_{args.mode}.json"
    out_file.write_text(json.dumps({"name": args.name, "mode": args.mode,
                                    "qa_file": args.qa or "demo",
                                    "top_k": top_ks, "n": n,
                                    "stats": {str(k): {kk: vv / n for kk, vv in v.items()} for k, v in stats.items()},
                                    "mrr_chapter": mrr_ch / n, "mrr_any": mrr_any / n,
                                    "details": details},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 明细已保存: {out_file}")


if __name__ == "__main__":
    main()
