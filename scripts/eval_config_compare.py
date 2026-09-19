# -*- coding: utf-8 -*-
"""eval_config_compare.py — 自然问法 QA 集上的检索配置对照评测（无需 API Key）

用途：在**不依赖 LLM**的前提下，量化不同检索配置对「目标章节是否被召回」的影响，
用于确定线上默认检索配置、以及验证「60 分可达性」的前置条件（召回率 ≥ 60%）。

配置：
  baseline_top5         top_k=5  , 无精排, 单查询
  top50                 top_k=50 , 无精排, 单查询
  top100                top_k=100, 无精排, 单查询
  rerank_top5           top_k=5  , CrossEncoder 精排
  rerank_top20          top_k=20 , CrossEncoder 精排
  entity_top5           top_k=5  , 无精排, 实体增强（问句内专名单独作附加查询）
  entity_rerank_top20   top_k=20 , 精排 + 实体增强

判据：
  chapter(chunk) : 前 K 条命中里任一条 chapter 章号 == gold 章号
  chapter(dedup) : 前 K 条按章去重后的章节序列中含 gold 章号（主口径，防同章多 chunk 刷分）
  MRR            : 首个命中位置倒数

用法：
  python scripts/eval_config_compare.py                       # 默认跑 zhetian + jszz
  python scripts/eval_config_compare.py --books zhetian
  python scripts/eval_config_compare.py --qa-dir qa_sets --out-dir eval_results
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config_manager import ConfigManager          # noqa: E402
from project_manager import ProjectManager        # noqa: E402
from rag_retriever import RAGRetriever            # noqa: E402
try:
    from utils import load_env_file  # noqa: E402
    load_env_file()
except Exception as _e:  # pragma: no cover
    print("env load skip:", _e)

CONFIGS = [
    ("baseline_top5",       dict(top_k=5,   rerank=False, entity=False)),
    ("top50",               dict(top_k=50,  rerank=False, entity=False)),
    ("top100",              dict(top_k=100, rerank=False, entity=False)),
    ("rerank_top5",         dict(top_k=5,   rerank=True,  entity=False)),
    ("rerank_top20",        dict(top_k=20,  rerank=True,  entity=False)),
    ("entity_top5",         dict(top_k=5,   rerank=False, entity=True)),
    ("entity_rerank_top20", dict(top_k=20,  rerank=True,  entity=True)),
]

CN = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_STOP = set("""一个 什么 怎么 自己 他们 我们 你们 这个 那个 因为 所以 但是 如果 已经 之后
然后 突然 直接 同时 开始 起来 下来 过去 回来 出来 上去 看到 看见 听到 说道 问道 笑道
一声 一眼 一步 手中 身上 脸上 眼中 心里 整个 如今 知道 时候 现在 就是 没有 不是 只是
还是 但是 不过 却 都 也 又 很 太 更 最 再 可 并 而 与 及 或 等 好 像 仿佛 似乎 应该
可能 大概 顿时 立刻 马上 终于 竟然 居然 难道 如何 为何 这时 那里 这里 前面 后面 旁边
上方 下方 虚空 天地 无数 十分 非常 特别 一般 有些 期间 其中 之中 之上 之下 之间 左右
上下 前后 内外 天地间 一瞬间 下一刻""".split())


def cn2int(s: str):
    if s.isdigit():
        return int(s)
    total = section = num = 0
    for ch in s:
        if ch in CN:
            num = CN[ch]
        elif ch == "十":
            section += (num or 1) * 10; num = 0
        elif ch == "百":
            section += (num or 1) * 100; num = 0
        elif ch == "千":
            section += (num or 1) * 1000; num = 0
        elif ch == "万":
            total = (total + section + num) * 10000; section = num = 0
    return total + section + num


def ch_no(title: str):
    m = re.search(r"第\s*([0-9]+)\s*章", title or "")
    if m:
        return int(m.group(1))
    m = re.search(r"第\s*([零一二三四五六七八九十百千万]+)\s*章", title or "")
    return cn2int(m.group(1)) if m else None


def project_terms(meta):
    """书名级特征词表（>=30 次的 2/3 gram），用于实体增强抽取"""
    counter = Counter()
    for c in meta:
        txt = c.get("text", "")
        for seg in _CJK.findall(txt):
            for n in (2, 3):
                for i in range(len(seg) - n + 1):
                    counter[seg[i:i + n]] += 1
    return {k: v for k, v in counter.items() if v >= 30 and k not in _STOP}


def extract_entities(question, freq, top=3):
    """从问句中抽取书籍专名（最长匹配优先）"""
    found = []
    for seg in _CJK.findall(question):
        i, l = 0, len(seg)
        while i < l:
            matched = ""
            for n in (4, 3, 2):
                if i + n <= l and seg[i:i + n] in freq:
                    matched = seg[i:i + n]; break
            if matched:
                found.append(matched); i += len(matched)
            else:
                i += 1
    seen, out = set(), []
    for w in sorted(found, key=len, reverse=True):
        if w not in seen:
            seen.add(w); out.append(w)
    return out[:top]


def metrics_for(hits, gold_no, ks=(5, 10)):
    """返回 chunk 口径与 dedup 口径的 Recall@k 与 MRR"""
    chs = [ch_no(h.get("chapter", "")) for h in hits]
    res = {}
    for k in ks:
        topk = chs[:k]
        res[f"recall@{k}_chunk"] = any(c == gold_no for c in topk)
        dedup, seen = [], set()
        for c in topk:
            if c is not None and c not in seen:
                seen.add(c); dedup.append(c)
        res[f"recall@{k}_dedup"] = gold_no in dedup
    first = next((i + 1 for i, c in enumerate(chs) if c == gold_no), None)
    res["mrr_chunk"] = (1.0 / first) if first else 0.0
    dedup_all, seen = [], set()
    for c in chs:
        if c is not None and c not in seen:
            seen.add(c); dedup_all.append(c)
    d_first = next((i + 1 for i, c in enumerate(dedup_all) if c == gold_no), None)
    res["first_rank_dedup"] = d_first
    res["mrr_dedup"] = (1.0 / d_first) if d_first else 0.0
    return res


def main():
    ap = argparse.ArgumentParser(description="自然问法 QA 集上的检索配置对照评测")
    ap.add_argument("--books", default="zhetian,jszz", help="逗号分隔的项目名")
    ap.add_argument("--qa-dir", default="qa_sets", help="QA 集目录（默认 qa_sets）")
    ap.add_argument("--out-dir", default="eval_results", help="结果输出目录")
    ap.add_argument("--qa-suffix", default="_natural", help="QA 文件名后缀：<name><suffix>.json")
    ap.add_argument("--embedding-model", default="", help="覆盖嵌入模型路径")
    ap.add_argument("--reranker-model", default="", help="覆盖精排模型路径")
    args = ap.parse_args()

    books = [b.strip() for b in args.books.split(",") if b.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ConfigManager()
    pm = ProjectManager()
    print("HF offline:", os.environ.get("HF_HUB_OFFLINE"), "| endpoint:", os.environ.get("HF_ENDPOINT"))

    all_out = {}
    for name in books:
        project = pm.get_project(name) or pm.get_project_by_name(name)
        if not project:
            print(f"✗ 项目不存在: {name}（先 novel_rag.py ingest）")
            continue
        vdb = Path(project["vector_db_path"])
        meta = json.loads((vdb / "metadata.json").read_text(encoding="utf-8"))
        freq = project_terms(meta)
        qa_file = ROOT / args.qa_dir / f"{name}{args.qa_suffix}.json"
        qa = json.loads(qa_file.read_text(encoding="utf-8"))["items"]

        emb = args.embedding_model or cfg.get("embedding_model_path") or ""
        if emb and not Path(emb).exists():
            emb = str(ROOT / "models" / "bge-small-zh-v1.5")
        rrk = args.reranker_model or cfg.get("reranker_model_path") or str(ROOT / "models" / "bge-reranker-base")

        t0 = time.time()
        retriever = RAGRetriever.get_or_create(
            str(vdb), reranker_model_path=rrk, embedding_model_path=emb,
            vector_file=project.get("vector_file"), metadata_file=project.get("metadata_file"),
        )
        print(f"[{name}] retriever ready in {time.time()-t0:.1f}s | QA={len(qa)} | 特征词={len(freq)} | "
              f"reranker={'on' if retriever.reranker_model is not None else 'off'} | "
              f"embedding={'on' if retriever.embedding_model is not None else 'off'}")

        book_res = {}
        for cname, c in CONFIGS:
            t1 = time.time()
            per_q, agg = [], {"recall@5_chunk": 0, "recall@5_dedup": 0, "recall@10_chunk": 0,
                              "recall@10_dedup": 0, "mrr_chunk": 0.0, "mrr_dedup": 0.0}
            for it in qa:
                gold = ch_no(it["gold_chapter"])
                extra = extract_entities(it["question"], freq) if c["entity"] else None
                try:
                    hits = retriever.retrieve_multi(it["question"], extra_queries=extra,
                                                    top_k=c["top_k"], enable_rerank=c["rerank"])
                except Exception as e:
                    hits = []
                    print(f"  ⚠ {name}#{it['id']} {cname} 检索异常: {e!r}")
                m = metrics_for(hits, gold)
                per_q.append({"id": it["id"], "type": it["type"], "question": it["question"],
                              "gold_chapter": it["gold_chapter"], "gold_no": gold,
                              "extra_queries": extra,
                              "n_hits": len(hits),
                              "top_chapters": [h.get("chapter", "") for h in hits[:10]],
                              "hit_rank_chunk": next((i + 1 for i, h in enumerate(hits)
                                                      if ch_no(h.get("chapter", "")) == gold), None),
                              **m})
                for kk in agg:
                    agg[kk] += m[kk] if kk.startswith("mrr") else (1 if m[kk] else 0)
            n = len(qa)
            summary = {kk: v / n for kk, v in agg.items()}
            summary["seconds"] = round(time.time() - t1, 1)
            book_res[cname] = {"config": c, "summary": summary, "per_question": per_q}
            print(f"[{name}] {cname:<22} R@5dedup={summary['recall@5_dedup']:.3f} "
                  f"R@10dedup={summary['recall@10_dedup']:.3f} R@5chunk={summary['recall@5_chunk']:.3f} "
                  f"MRR_dedup={summary['mrr_dedup']:.3f} ({summary['seconds']}s)")
        all_out[name] = book_res
        (out_dir / f"config_compare_{name}.json").write_text(
            json.dumps(book_res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "config_compare_all.json").write_text(
        json.dumps(all_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("✓ 结果已保存:", out_dir)


if __name__ == "__main__":
    main()
