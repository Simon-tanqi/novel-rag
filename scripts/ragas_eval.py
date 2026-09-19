# -*- coding: utf-8 -*-
"""ragas_eval.py — RAGAS 系统评测（两阶段：数据集生成 / 指标打分）

设计：本脚本跨「两个解释器」运行，因此拆成互相独立的两组子命令，谁缺依赖谁报错。

  ┌ 阶段 A（项目主环境 .venv 运行，需要项目依赖）
  │   gen          检索 + 组装 contexts / ground_truth → 评测数据集 JSON（无需 API KEY）
  │   gen-answers  走与 `novel_rag.py ask` 相同的链路生成 answer（需要 API KEY）
  └ 阶段 B（RAGAS 独立环境 .ragas_venv 运行，需要 ragas）
      score        调 ragas.evaluate 产出 4 指标（需要 API KEY）
      dry-run      数据集体检 + ragas 指标对象构建自检（无需 API KEY，不发起任何 LLM 调用）

四指标口径（本项目的落地定义）：
  context_recall    参考答案被检索上下文「覆盖」的比例。参考 ground_truth 拆分为陈述句后，
                    逐句判断能否由 contexts 推出。衡量「该召回的是否都召回了」。
  context_precision 检索上下文中「与问题相关」的比例，按排名加权。衡量「召回的是否都有用」。
  faithfulness      生成的 answer 中有多少陈述能从 contexts 找到依据。衡量幻觉程度。
  answer_relevancy  由 answer 反推的问题与原问题的语义相似度（用本地 bge-small-zh 算）。
                    衡量「答非所问」。

中文小说场景的已知局限（详见 RAGAS_EVAL.md）：ground_truth 若直接取原文摘录会抬高
context_recall；faithfulness 对「拒绝回答/未找到」型答案为定义盲区；answer_relevancy
的判分 embedding 与英文模板混用对中文长答案偏保守。

用法：
    # 阶段 A（主环境）
    python scripts/ragas_eval.py gen --name zhetian --qa qa_sets/zhetian_natural.json
    python scripts/ragas_eval.py gen-answers --dataset ragas_out/zhetian_natural_dataset.json
    # 阶段 B（RAGAS 环境）
    .\\.ragas_venv\\Scripts\\python.exe scripts/ragas_eval.py dry-run --dataset ragas_out/zhetian_natural_dataset.json
    .\\.ragas_venv\\Scripts\\python.exe scripts/ragas_eval.py score --dataset ragas_out/zhetian_natural_dataset.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_METRICS = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]

# --------------------------------------------------------------------------- #
# 公共：数据集 IO / 校验
# --------------------------------------------------------------------------- #


def dataset_path(name: str, out_dir: str = "ragas_out") -> Path:
    return ROOT / out_dir / f"{name}_natural_dataset.json"


def load_dataset(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):  # 兼容裸数组
        data = {"name": Path(path).stem, "samples": data}
    data.setdefault("samples", [])
    return data


def _norm_ws(s: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", (s or "").replace("\r", "")).strip()


_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn2int(s: str) -> int:
    """中文数字 → int（支持 十/百/千/万，如「一千零九十六」）"""
    if s.isdigit():
        return int(s)
    total = section = num = 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
        elif ch == "万":
            total = (total + section + num) * 10000
            section = num = 0
    return total + section + num


def ch_no(title: str):
    """从章节标题提取章号；书名/章节标题字面格式可能不一致（阿拉伯 vs 中文数字），
    因此 gold 章节一律按「章号」比对，不用字符串精确匹配。"""
    m = re.search(r"第\s*([0-9]+)\s*章", title or "")
    if m:
        return int(m.group(1))
    m = re.search(r"第\s*([零一二三四五六七八九十百千万]+)\s*章", title or "")
    return _cn2int(m.group(1)) if m else None


def validate_samples(samples: list, require_answer: bool) -> list[str]:
    """返回问题清单（为空表示通过）"""
    problems = []
    for i, s in enumerate(samples, 1):
        if not s.get("question"):
            problems.append(f"#{i} 缺 question")
        if not isinstance(s.get("contexts"), list) or not s["contexts"]:
            problems.append(f"#{i} contexts 为空（检索未命中或未跑 gen）")
        if not s.get("ground_truth"):
            problems.append(f"#{i} 缺 ground_truth")
        if require_answer and not (s.get("answer") or "").strip():
            problems.append(f"#{i} 缺 answer（请先跑 gen-answers，或去掉 --require-answer）")
    return problems


# --------------------------------------------------------------------------- #
# 阶段 A-1：gen —— 检索 + 组装数据集
# --------------------------------------------------------------------------- #


def _ground_truth_of(item: dict, gold_text: str, max_gt_chars: int) -> tuple[str, str]:
    """返回 (ground_truth, 来源标记)

    优先级：evidence（人工核验的原文依据句）> answer + answer_terms > gold 章节原文
    """
    ev = _norm_ws(item.get("evidence", ""))
    if ev:
        return ev[:max_gt_chars], "evidence"
    ans = _norm_ws(item.get("answer", ""))
    terms = item.get("answer_terms") or []
    if ans and terms:
        return f"{ans}（要点：{'、'.join(terms)}）"[:max_gt_chars], "answer+terms"
    if gold_text:
        return _norm_ws(gold_text)[:max_gt_chars], "gold_chapter_text"
    return "", "missing"


def cmd_gen(args) -> None:
    import novel_rag as nr
    from config_manager import ConfigManager
    from project_manager import ProjectManager

    cfg = ConfigManager()
    pm = ProjectManager()
    project = pm.get_project(args.name) or pm.get_project_by_name(args.name)
    if not project:
        print(f"✗ 项目不存在: {args.name}（先用 novel_rag.py ingest 建库）")
        sys.exit(1)

    qa_file = Path(args.qa)
    if not qa_file.is_absolute():
        qa_file = ROOT / qa_file
    qa_raw = json.loads(qa_file.read_text(encoding="utf-8"))
    items = qa_raw["items"] if isinstance(qa_raw, dict) else qa_raw
    if args.limit:
        items = items[: args.limit]

    top_k = args.top_k or int(cfg.get("top_k", 5) or 5)
    retriever = nr._prepare_retriever(cfg, project)
    print(f"ℹ 项目={args.name} | QA={len(items)} | top_k={top_k} | rerank={args.enable_rerank}")

    # gold 章节原文（供 ground_truth 兜底）：按「章号」建索引，规避标题字面差异
    from collections import defaultdict

    meta_file = Path(project["vector_db_path"]) / "metadata.json"
    by_no: dict[int, list[str]] = defaultdict(list)
    if meta_file.exists():
        for c in json.loads(meta_file.read_text(encoding="utf-8")):
            no = ch_no(c.get("chapter", ""))
            if no is not None:
                by_no[no].append(c.get("text", ""))

    samples, misses = [], []
    for i, item in enumerate(items, 1):
        q = item["question"]
        gold_ch = item.get("gold_chapter", "")
        gold_no = ch_no(gold_ch)
        gold_text = "\n".join(by_no.get(gold_no, [])) if gold_no is not None else ""

        t0 = time.time()
        hits = retriever.retrieve(q, top_k=top_k, enable_rerank=args.enable_rerank)
        ctxs = [_norm_ws(h.get("text", ""))[: args.ctx_chars] for h in hits[: args.max_ctx]]
        ctxs = [c for c in ctxs if c]
        if not ctxs:
            misses.append(q)
        gt, gt_src = _ground_truth_of(item, gold_text, args.max_gt_chars)
        samples.append({
            "question": q,
            "contexts": ctxs,
            "ground_truth": gt,
            "answer": item.get("answer", "") or None,
            "metadata": {
                "qa_id": item.get("id", i),
                "type": item.get("type", ""),
                "gold_chapter": gold_ch,
                "answer_terms": item.get("answer_terms", []),
                "gt_source": gt_src,
                "retrieved_chapters": [h.get("chapter", "") for h in hits[: args.max_ctx]],
                "gold_in_ctx": any(ch_no(h.get("chapter", "")) == gold_no for h in hits[: args.max_ctx]),
                "retrieval_seconds": round(time.time() - t0, 3),
            },
        })

    out = {
        "name": args.name,
        "book": qa_raw.get("book", "") if isinstance(qa_raw, dict) else "",
        "source_qa": str(qa_file),
        "top_k": top_k,
        "enable_rerank": bool(args.enable_rerank),
        "max_ctx": args.max_ctx,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples": samples,
    }
    out_path = Path(args.out) if args.out else dataset_path(args.name)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    hit = sum(1 for s in samples if s["metadata"]["gold_in_ctx"])
    print(f"✓ 数据集已生成: {out_path}")
    print(f"  样本 {len(samples)} 条 | 检索命中 gold 章 {hit}/{len(samples)} "
          f"({hit / max(len(samples), 1):.1%}) | 空 contexts {len(misses)} 条")
    print("  下一步（主环境）: python scripts/ragas_eval.py gen-answers --dataset "
          f"{out_path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# 阶段 A-2：gen-answers —— 复用 ask 链路生成 answer
# --------------------------------------------------------------------------- #


def cmd_gen_answers(args) -> None:
    import novel_rag as nr
    from config_manager import ConfigManager
    from project_manager import ProjectManager

    cfg = ConfigManager()
    data = load_dataset(Path(args.dataset))
    samples = data["samples"]
    params = nr._get_api_params_optional(cfg, args.api_key, args.model)
    if not params:
        print("✗ 未检测到 API KEY。请设置 DEEPSEEK_API_KEY 或 config.json 后再跑 gen-answers。")
        sys.exit(2)
    api_url, api_key, model_name = params
    template = cfg.get_prompt_template()
    client = nr.APIClient(api_url, api_key, model_name)
    book = data.get("book") or data.get("name", "")

    todo = [s for s in samples if not (s.get("answer") or "").strip()]
    print(f"ℹ 待生成 answer: {len(todo)}/{len(samples)} 条 | model={model_name}")

    ok, fail = 0, 0
    for i, s in enumerate(todo, 1):
        hits = [{"chapter": data_ch, "text": txt}
                for data_ch, txt in zip(s["metadata"].get("retrieved_chapters", []), s["contexts"])]
        context = nr._build_context(hits)
        prompt = nr._build_prompt(template, book, context, s["question"]) if template else s["question"]
        try:
            s["answer"] = client.call_api(prompt)
            ok += 1
        except Exception as e:  # 单题失败不中断整批
            s["answer"] = None
            fail += 1
            print(f"  ⚠ 第 {i} 条生成失败: {e!r}")
        print(f"  [{i}/{len(todo)}] {s['question'][:34]}… → {(s.get('answer') or '')[:40]}")

    Path(args.dataset).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ answer 生成完成：成功 {ok} / 失败 {fail}；数据集已回写 {args.dataset}")


# --------------------------------------------------------------------------- #
# 阶段 B-1：dry-run —— 数据集体检 + ragas 构建自检（不调 LLM）
# --------------------------------------------------------------------------- #


def cmd_dry_run(args) -> None:
    data = load_dataset(Path(args.dataset))
    samples = data["samples"]
    print("=" * 72)
    print(f"数据集体检：{args.dataset}")
    print(f"  样本数 {len(samples)} | top_k={data.get('top_k')} | rerank={data.get('enable_rerank')} | "
          f"生成于 {data.get('created_at', '未知')}")
    lens_q = [len(s["question"]) for s in samples]
    lens_c = [sum(len(c) for c in s["contexts"]) for s in samples]
    lens_g = [len(s["ground_truth"]) for s in samples]
    if samples:
        print(f"  题目字数 min/avg/max = {min(lens_q)}/{sum(lens_q)//len(lens_q)}/{max(lens_q)}")
        print(f"  上下文总字数 min/avg/max = {min(lens_c)}/{sum(lens_c)//len(lens_c)}/{max(lens_c)}")
        print(f"  参考答案字数 min/avg/max = {min(lens_g)}/{sum(lens_g)//len(lens_g)}/{max(lens_g)}")
    from collections import Counter
    print("  ground_truth 来源分布:", dict(Counter(s["metadata"].get("gt_source", "?") for s in samples)))
    print("  题型分布:", dict(Counter(s["metadata"].get("type", "?") for s in samples)))
    print("  contexts 条数分布:", dict(Counter(len(s["contexts"]) for s in samples)))
    ans_ok = sum(1 for s in samples if (s.get("answer") or "").strip())
    print(f"  answer 已生成 {ans_ok}/{len(samples)}")

    problems = validate_samples(samples, require_answer=False)
    print("-" * 72)
    if problems:
        print(f"✗ 数据集存在 {len(problems)} 处问题：")
        for p in problems[:20]:
            print("   -", p)
    else:
        print("✓ 基础字段完整（question / contexts / ground_truth）")

    print("-" * 72)
    print("RAGAS 侧自检（不发起 LLM 调用）:")
    try:
        import ragas
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)
        from ragas.llms import LangchainLLMWrapper  # noqa: F401
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401
        print(f"  ✓ ragas {getattr(ragas, '__version__', '?')} 可用；指标对象可构建："
              f" {[m.name for m in (context_recall, context_precision, faithfulness, answer_relevancy)]}")
        try:
            from ragas import EvaluationDataset, SingleTurnSample
            s = samples[0]
            EvaluationDataset(samples=[SingleTurnSample(
                user_input=s["question"], retrieved_contexts=s["contexts"],
                response=s.get("answer") or "", reference=s["ground_truth"])])
            print("  ✓ EvaluationDataset / SingleTurnSample 构建通过")
        except Exception as e:
            print(f"  ⚠ Dataset 构建自检失败（不影响 score）: {e!r}")
    except ImportError as e:
        print(f"  ✗ ragas 未就绪：{e!r}")
        print("    请先建独立环境：powershell -ExecutionPolicy Bypass -File scripts/ragas_env_setup.ps1")
    print("=" * 72)
    if not args.require_answer:
        print("提示：score 需要每条的 answer 字段；无 KEY 时先做 dry-run，KEY 到位后跑 gen-answers。")


# --------------------------------------------------------------------------- #
# 阶段 B-2：score —— 调 ragas.evaluate
# --------------------------------------------------------------------------- #


def _openai_base_url(api_url: str) -> str:
    """把项目里的 .../v1/chat/completions 归一化成 langchain 需要的 base_url"""
    u = (api_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    if u.endswith("/v1"):
        return u
    return u + "/v1" if u else u


def _build_llm(api_url: str, api_key: str, model: str, temperature: float):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    base = _openai_base_url(api_url)
    print(f"  judge LLM: {model} @ {base}")
    return LangchainLLMWrapper(ChatOpenAI(
        model=model, api_key=api_key, base_url=base,
        temperature=temperature, timeout=120, max_retries=2,
    ))


def _build_embeddings(model_path_or_id: str):
    """本地 bge-small-zh → langchain Embeddings（不依赖 langchain-huggingface）"""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class _STEmbeddings(Embeddings):
        def __init__(self, spec: str):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(spec)
            self.dim = self.model.get_sentence_embedding_dimension()

        def embed_documents(self, texts):
            return [v.tolist() for v in self.model.encode(list(texts), normalize_embeddings=True)]

        def embed_query(self, text):
            return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    emb = _STEmbeddings(model_path_or_id)
    print(f"  judge embeddings: {model_path_or_id} (dim={emb.dim})")
    return LangchainEmbeddingsWrapper(emb)


def cmd_score(args) -> None:
    from config_manager import ConfigManager
    import novel_rag as nr

    data = load_dataset(Path(args.dataset))
    samples = data["samples"]
    problems = validate_samples(samples, require_answer=True)
    if problems:
        print("✗ 数据集不满足打分条件：")
        for p in problems[:10]:
            print("   -", p)
        print("  提示：无 KEY 时先跑 gen-answers。")
        sys.exit(2)

    cfg = ConfigManager()
    params = nr._get_api_params_optional(cfg, args.api_key, args.model)
    if not params:
        print("✗ 未检测到 API KEY（DEEPSEEK_API_KEY / config.json），无法运行 RAGAS。")
        sys.exit(2)
    api_url, api_key, model_name = params

    import ragas
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    name2metric = {
        "context_recall": context_recall,
        "context_precision": context_precision,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in wanted if m not in name2metric]
    if bad:
        print(f"✗ 未知指标: {bad}；可选 {list(name2metric)}")
        sys.exit(2)
    metrics = [name2metric[m] for m in wanted]

    ds = EvaluationDataset(samples=[SingleTurnSample(
        user_input=s["question"], retrieved_contexts=s["contexts"],
        response=s["answer"], reference=s["ground_truth"]) for s in samples])
    print(f"ℹ ragas {getattr(ragas, '__version__', '?')} | 样本 {len(samples)} | 指标 {wanted}")

    llm = _build_llm(api_url, api_key, model_name, args.temperature)
    embeddings = None
    if "answer_relevancy" in wanted:
        from utils import DEFAULT_EMBEDDING_MODEL
        emb_spec = args.embedding_model or str(ROOT / "models" / "bge-small-zh-v1.5")
        if not Path(emb_spec).exists() and not args.embedding_model:
            emb_spec = DEFAULT_EMBEDDING_MODEL
        embeddings = _build_embeddings(emb_spec)

    if args.adapt_prompts:
        try:
            from ragas.prompt import adapt_prompts  # type: ignore
            metrics = adapt_prompts(metrics, llm=llm, language="chinese")
            print("  ✓ 已把指标 prompt 适配为中文")
        except Exception as e:
            print(f"  ⚠ prompt 中文适配跳过（{e!r}），使用默认英文模板")

    result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=embeddings,
                      raise_exceptions=False, show_progress=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.dataset).stem
    out_dir = ROOT / (args.out_dir or "ragas_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_pandas()
    csv_path = out_dir / f"{stem}_ragas_{ts}.csv"
    json_path = out_dir / f"{stem}_ragas_{ts}.json"
    md_path = out_dir / f"{stem}_ragas_{ts}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    scores = {m: (float(df[m].mean()) if m in df.columns else None) for m in wanted}
    per_q = []
    for i, row in df.iterrows():
        per_q.append({
            "id": samples[i]["metadata"].get("qa_id", i + 1),
            "question": samples[i]["question"],
            "gold_chapter": samples[i]["metadata"].get("gold_chapter", ""),
            "gold_in_ctx": samples[i]["metadata"].get("gold_in_ctx"),
            **{m: (None if row.get(m) is None or str(row.get(m)) == "nan" else float(row.get(m)))
               for m in wanted},
        })
    payload = {
        "dataset": str(args.dataset), "run_at": ts, "model": model_name,
        "metrics": wanted, "mean_scores": scores, "n": len(samples),
        "per_question": per_q,
        "ragas_version": getattr(ragas, "__version__", "?"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# RAGAS 评测报告 — {stem}", "",
             f"- 运行时间：{ts} | 样本数：{len(samples)} | judge 模型：{model_name}",
             f"- ragas 版本：{ragas.__version__}", "", "## 平均分", "",
             "| 指标 | 平均分 |", "| --- | --- |"]
    for m in wanted:
        v = scores.get(m)
        lines.append(f"| {m} | {'—' if v is None else f'{v:.3f}'} |")
    lines += ["", "## 逐题明细", "",
              "| # | 题目 | 题类 | " + " | ".join(wanted) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in wanted) + " |"]
    for r in per_q:
        vals = " | ".join("—" if r.get(m) is None else f"{r[m]:.2f}" for m in wanted)
        lines.append(f"| {r['id']} | {r['question'][:40]} | {r['gold_chapter']} | {vals} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    for m in wanted:
        v = scores.get(m)
        print(f"  {m:<20} {'—' if v is None else f'{v:.3f}'}")
    print("=" * 72)
    print(f"✓ 明细: {csv_path}\n✓ JSON: {json_path}\n✓ 报告: {md_path}")


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAGAS 系统评测（数据集生成 / 打分）")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="[主环境] 检索并生成评测数据集")
    g.add_argument("--name", required=True)
    g.add_argument("--qa", default="", help="QA 文件（默认 qa_sets/<name>_natural.json）")
    g.add_argument("--out", default="")
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--enable-rerank", action="store_true")
    g.add_argument("--max-ctx", type=int, default=8, help="最多保留几条 contexts")
    g.add_argument("--ctx-chars", type=int, default=1200, help="单条 context 截断字数")
    g.add_argument("--max-gt-chars", type=int, default=600)
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("gen-answers", help="[主环境] 用 ask 链路生成 answer（需 KEY）")
    a.add_argument("--dataset", required=True)
    a.add_argument("--api-key", default="")
    a.add_argument("--model", default="")
    a.set_defaults(func=cmd_gen_answers)

    d = sub.add_parser("dry-run", help="[RAGAS 环境] 数据集体检 + 自检（不调 LLM）")
    d.add_argument("--dataset", required=True)
    d.add_argument("--require-answer", action="store_true")
    d.set_defaults(func=cmd_dry_run)

    s = sub.add_parser("score", help="[RAGAS 环境] 计算指标（需 KEY）")
    s.add_argument("--dataset", required=True)
    s.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    s.add_argument("--api-key", default="")
    s.add_argument("--model", default="")
    s.add_argument("--embedding-model", default="")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--adapt-prompts", action="store_true")
    s.add_argument("--out-dir", default="ragas_out")
    s.set_defaults(func=cmd_score)
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "gen" and not args.qa:
        args.qa = f"qa_sets/{args.name}_natural.json"
    args.func(args)


if __name__ == "__main__":
    main()
