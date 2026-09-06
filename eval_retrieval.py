"""
eval_retrieval.py — RAG 检索质量评估工具

用「问题-预期命中章节」QA 集量化召回效果，输出 Recall@k：
    Recall@k = 命中「预期章节」的问题数 / 总问题数

用法:
    # 对某个已构建项目评估（需先 ingest，走与 ask 相同的检索通道）
    python eval_retrieval.py --name 盘龙 --qa qa_sets/panlong.json --top-k 1 3 5

    # 内置 demo 语料评估（无需 API Key、无需网络）
    python eval_retrieval.py --name demo --top-k 3

QA 集格式（JSON）:
    [
        {"question": "沈青在剑庐学到的第一课是什么？", "chapter": "第二章 剑庐第一课"},
        ...
    ]
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from config_manager import ConfigManager
from project_manager import ProjectManager
from rag_retriever import RAGRetriever
from utils import DEFAULT_EMBEDDING_MODEL, load_env_file

# demo 语料与问题卡（question/answer/chapter）定义于 novel_rag.py，单一事实源
from novel_rag import DEMO_QA

# 最先加载项目根目录 .env（幂等，仅补未设置键）
load_env_file()


def _enable_utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


# ===================== 内置 demo QA 集 =====================
# 由 novel_rag.py 导入（含 answer 参考答案字段，question/chapter 语义不变），
# 随 demo 语料同源维护，避免双份漂移。
# 未指定 --qa 时，默认用内置 demo QA 集评估（无需 API Key、无需联网）。


def _hit_in_expected(chapter: str, expected: str) -> bool:
    """命中判定：命中片段的章节与预期章节「同章」。

    章节名长度 < 4（如“第1章”）时要求完全相等；否则取
    前 4 个字符前缀比较（如“第二章 剑庐第一课”≈“第二章 剑庐…”）。
    """
    if not chapter or not expected:
        return False
    if chapter == expected:
        return True
    if len(expected) < 4:
        return False
    return chapter[:4] == expected[:4]


def build_retriever(project: dict, cfg: ConfigManager) -> Optional[RAGRetriever]:
    """按项目配置构建检索器（与 CLI ask 同一通道）"""
    vector_path = project.get("vector_db_path", "")
    if not vector_path or not os.path.isdir(vector_path):
        print(f"✗ 项目「{project.get('name')}」尚未构建向量库，请先运行 ingest。")
        return None
    embedding_spec = (
        cfg.get("embedding_model_path", "")
        or os.environ.get("EMBEDDING_MODEL", "").strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    return RAGRetriever.get_or_create(
        vector_path,
        reranker_model_path=cfg.get("reranker_model_path", ""),
        embedding_model_path=embedding_spec,
        vector_file=project.get("vector_file"),
        metadata_file=project.get("metadata_file"),
    )


def load_qa(qa_file: Optional[str]) -> List[Dict]:
    """加载 QA 集；未指定时返回内置 demo QA 集"""
    if qa_file:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa = json.load(f)
        if not isinstance(qa, list) or not qa:
            print(f"✗ QA 集格式错误或为空: {qa_file}")
            sys.exit(1)
        return qa
    print("ℹ 未指定 --qa，使用内置 demo QA 集（10+ 题，无版权风险）")
    return DEMO_QA


def run_eval(retriever: RAGRetriever, qa: List[Dict], top_k_values: List[int]) -> dict:
    """逐题检索，统计各 top_k 下的 Recall@k 与逐题命中明细"""
    recall = {k: 0 for k in top_k_values}
    details = []

    for i, item in enumerate(qa, 1):
        question = item.get("question", "")
        expected = item.get("chapter", "")
        if not question or not expected:
            print(f"⚠ 第 {i} 条 QA 缺少 question/chapter，跳过")
            continue

        max_k = max(top_k_values)
        hits = retriever.retrieve(question, top_k=max_k)
        hit_chapters = [h.get("chapter", "") for h in hits]

        per_k = {}
        for k in top_k_values:
            hit = any(_hit_in_expected(ch, expected)
                      for ch in hit_chapters[:k])
            if hit:
                recall[k] += 1
            per_k[f"recall@{k}"] = hit

        details.append({
            "question": question,
            "answer": item.get("answer", ""),
            "expected": expected,
            "top_chapters": hit_chapters[:max_k],
            **per_k,
        })
        status = "✓" if per_k[f"recall@{max(top_k_values)}"] else "✗"
        print(f"  {status} Q{i}: {question[:36]}")
        print(f"      预期: {expected} | 命中: {hit_chapters[:3]}")

    total = len(details)
    print("\n" + "=" * 52)
    print(f"QA 总数: {total}")
    for k in top_k_values:
        if total:
            print(f"  Recall@{k:<2} = {recall[k]}/{total} = "
                  f"{recall[k] / total:.1%}")
    print("=" * 52)
    return {"recall": {k: (recall[k] / total if total else 0)
                       for k in top_k_values}, "details": details}


def main():
    parser = argparse.ArgumentParser(
        description="RAG 检索质量评估（Recall@k）",
    )
    parser.add_argument("--name", help="项目名（默认最近使用的项目）")
    parser.add_argument("--qa", default=None, help="QA 集 JSON 路径（默认内置 demo QA）")
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5],
                        help="评估的 top_k 值（默认 1 3 5）")
    parser.add_argument("--json-out", default=None, help="逐题明细输出路径（可选）")
    args = parser.parse_args()

    top_k_values = sorted(set(args.top_k))
    pm = ProjectManager()
    project = pm.get_project_by_name(args.name) if args.name \
        else pm.get_current_project()
    if not project:
        tip = f"「{args.name}」" if args.name else ""
        print(f"✗ 找不到项目{tip}。请先运行: python novel_rag.py ingest <小说.txt> --name <名字>")
        sys.exit(1)

    retriever = build_retriever(project, ConfigManager())
    if retriever is None:
        sys.exit(1)

    qa = load_qa(args.qa)
    print(f"📊 开始评估项目「{project.get('name')}」"
          f"（top_k={top_k_values}，{len(qa)} 题）\n")
    result = run_eval(retriever, qa, top_k_values)

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 明细已保存: {args.json_out}")


if __name__ == "__main__":
    _enable_utf8_stdout()
    main()
