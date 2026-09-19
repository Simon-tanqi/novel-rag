"""
eval_llm.py — 真实 LLM 问答评测（需 API KEY，本轮验收暂缓执行）

目标：把「60% 及格线」落到可复现命令上。
  - 逐题走与 `novel_rag.py ask` 完全相同的链路（同一检索器 + 同一 prompt 模板 + 同一 API 客户端）；
  - 自动判分（keyword 重叠）与人工判分（--manual 输出 Markdown 供逐题打分）二选一；
  - 及格线：自动判分通过率 >= 60%（即 20 题中 >=12 题命中关键实体信息）为「跨书泛化达标」。

用法：
    # 自动判分（需要 config.json 或 DEEPSEEK_API_KEY）
    python scripts/eval_llm.py --name zhetian --qa qa_sets/zhetian.json --top-k 5

    # 人工判分：输出每题 LLM 答案 + 命中片段到 eval_out/<name>_llm_manual.md
    python scripts/eval_llm.py --name jszz --qa qa_sets/jszz.json --manual

    # 限制题数快速试跑
    python scripts/eval_llm.py --name zhetian --qa qa_sets/zhetian.json --limit 5

输出：
    eval_out/<name>_llm.json / eval_out/<name>_llm_manual.md
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
from utils import load_env_file  # noqa: E402

load_env_file()

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_STOP = set("""一个 什么 怎么 自己 他们 我们 你们 这个 那个 因为 所以 但是 如果 已经
之后 然后 突然 直接 同时 开始 起来 下来 过去 回来 出来 上去 看到 看见 听到 说道
问道 笑道 一声 一眼 一步 手中 身上 脸上 眼中 心里 整个 如今 知道 时候 现在 就是
没有 不是 只是 还是 但是 不过 却 都 也 又 很 太 更 最 再 可 并 而 与 及 或 等
好 像 仿佛 似乎 应该 可能 大概 顿时 立刻 马上 终于 竟然 居然 难道 如何 为何 这时
那里 这里 前面 后面 旁边 上方 下方 虚空 天地 整个 无数 十分 非常 特别 一般 有些
同时 期间 其中 之中 之上 之下 之间 左右 上下 前后 内外 天地间 一瞬间 下一刻""".split())


def _ngrams(text: str, n: int):
    for seg in _CJK.findall(text):
        if len(seg) < n:
            continue
        for i in range(len(seg) - n + 1):
            yield seg[i : i + n]


def _answer_terms(answer: str, freq: dict[str, int]) -> list[str]:
    """gold answer 中的书中特有词（按频次降序取前 5）"""
    cands = []
    for gram in _ngrams(answer, 2):
        if gram in freq and gram not in cands:
            cands.append(gram)
    for gram in _ngrams(answer, 3):
        if gram in freq and gram not in cands:
            cands.append(gram)
    return sorted(cands, key=lambda g: freq[g], reverse=True)[:5]


def main():
    parser = argparse.ArgumentParser(description="LLM 问答评测")
    parser.add_argument("--name", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manual", action="store_true", help="输出人工判分 Markdown")
    args = parser.parse_args()

    cfg = ConfigManager()
    pm = ProjectManager()
    project = pm.get_project(args.name)
    if not project:
        print(f"✗ 项目不存在: {args.name}")
        sys.exit(1)

    import novel_rag as nr

    api_url, api_key, model_name = nr._get_api_params(cfg, None, None)  # 无 Key 直接退出
    retriever = nr._prepare_retriever(cfg, project)
    prompt_template = cfg.get_prompt_template()
    enable_rerank = bool(cfg.get("enable_rerank", False))

    qa = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    if args.limit:
        qa = qa[: args.limit]

    # 项目特征词表
    meta_file = Path(project["vector_db_path"]) / "metadata.json"
    freq: dict[str, int] = {}
    if meta_file.exists():
        counter: Counter[str] = Counter()
        for c in json.loads(meta_file.read_text(encoding="utf-8")):
            for gram in _ngrams(c.get("text", ""), 2):
                counter[gram] += 1
            for gram in _ngrams(c.get("text", ""), 3):
                counter[gram] += 1
        freq = {k: v for k, v in counter.items() if v >= 30 and k not in _STOP}

    results = []
    passed = 0
    for i, item in enumerate(qa, 1):
        q = item.get("question", "")
        answer = item.get("answer", "")
        hits = retriever.retrieve(q, top_k=args.top_k, enable_rerank=enable_rerank)
        context = nr._build_context(hits)
        prompt = (
            nr._build_prompt(prompt_template, args.name, context, q)
            if prompt_template else q
        )
        reply = nr.APIClient(api_url, api_key, model_name).call_api(prompt)
        terms = _answer_terms(answer, freq)
        hit_terms = [t for t in terms if t in reply]
        ok = len(hit_terms) >= max(1, len(terms) * 3 // 5) if terms else (answer in reply)
        if ok:
            passed += 1
        results.append({
            "index": i, "question": q, "entity": item.get("entity", ""),
            "answer": answer, "llm_reply": reply, "pass": ok,
            "checked_terms": terms, "hit_terms": hit_terms,
            "sources": [{"chapter": h.get("chapter", ""), "score": round(float(h.get("score", 0)), 4)} for h in hits],
        })
        print(f"[{i}/{len(qa)}] {'PASS' if ok else 'FAIL'} {q[:40]}")

    rate = passed / len(qa)
    out_dir = ROOT / "eval_out"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"{args.name}_llm.json"
    out_json.write_text(json.dumps({"name": args.name, "pass_rate": rate,
                                    "pass": passed, "n": len(qa),
                                    "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"LLM 问答通过率: {passed}/{len(qa)} = {rate:.1%}  （及格线 60%）")
    print(f"明细: {out_json}")

    if args.manual:
        md = [f"# {args.name} LLM 问答人工判分（及格线 60%）\n"]
        for r in results:
            md.append(f"## Q{r['index']}. {r['question']}")
            md.append(f"- 预期要点: {r['answer']}")
            md.append(f"- LLM 答案: {r['llm_reply']}")
            md.append("- 命中片段:")
            for s in r["sources"]:
                md.append(f"  - {s['chapter']} (score={s['score']})")
            md.append("")
        out_md = out_dir / f"{args.name}_llm_manual.md"
        out_md.write_text("\n".join(md), encoding="utf-8")
        print(f"人工判分表: {out_md}")


if __name__ == "__main__":
    main()
