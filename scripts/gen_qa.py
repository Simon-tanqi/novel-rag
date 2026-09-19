"""
gen_qa.py — 从已构建向量库自动生成「锚定式」QA 集（无 LLM、纯规则、可复现）

设计动机：
  建立 60% 及格线评测口径时，QA 集必须满足两个条件：
    1) 答案可定位：gold answer 必须是向量库中真实存在的原文句子，
       否则任何检索系统都不可能命中，评测失去意义；
    2) 可复现：同一仓库 + 同一 seed 应生成完全相同的 QA 集。

实现思路：
  - 从 data/<name>/vector_db/metadata.json 均匀抽取 N 个片段；
  - 对每个片段选出「实体区分度最高」的句子（书中高频专名出现次数加权）；
  - 以该句为 gold answer，以句中最高频实体构造 question；
  - chapter 字段 = 片段所属章节标题，供「章节命中」判定使用。

用法：
    python scripts/gen_qa.py --name zhetian --count 20 --seed 42
    python scripts/gen_qa.py --name jszz --count 20 --out qa_sets/jszz.json

输出（JSON 数组，兼容 eval_retrieval.py 的 QA 格式）：
    [{"question": "...", "chapter": "第一章 ...", "answer": "原文句子", "entity": "实体", "chunk_id": 123}, ...]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 过滤高频功能词 / 常用语（避免把“说道”“什么”当实体）
_STOP = set("""一个 什么 怎么 自己 他们 我们 你们 这个 那个 因为 所以 但是 如果 已经
之后 然后 突然 直接 同时 开始 起来 下来 过去 回来 出来 上去 看到 看见 听到 说道
问道 笑道 一声 一眼 一步 手中 身上 脸上 眼中 心里 整个 如今 知道 时候 现在 就是
没有 不是 只是 还是 但是 不过 却 都 也 又 很 太 更 最 再 可 并 而 与 及 或 等
好 像 仿佛 似乎 应该 可能 大概 顿时 立刻 马上 终于 竟然 居然 难道 如何 为何 这时
那里 这里 前面 后面 旁边 上方 下方 虚空 天地 整个 无数 十分 非常 特别 一般 有些
同时 期间 其中 之中 之上 之下 之间 左右 上下 前后 内外 天地间 一瞬间 下一刻
这样 那样 怎样 如何 什么 如此 这么 那么 一样 一边 一阵 一切 所有 全部 咱们
说道 问道 笑道 大声 小声道 喃喃 冷哼 点头 摇头 抬头 低头 转身 回头 开口 闭口
一眼 一眼 一步 一瞬 一刻 一刻钟 一天 一夜 一年 一月 一时 一息 一刻 半晌 片刻
只见 但见 却是 便是 就是 就是 只是 不过 然而 反而 于是 接着 随后 随即 此刻
那时 这时 那里 这里 渐渐 缓缓 慢慢 快速 迅速 突然 忽然 陡然 骤然 猛然 狠狠
再也 从未 曾经 早已 已经 还未 尚未 正欲 正要 刚想 准备 打算 决定 开始 继续
传来 响起 发出 出现 消失 离开 回到 来到 走进 走出 进入 退后 后退 上前 靠近
众人 大家 两人 三人 几人 所有人 其他人 对方 此人 那人 何人 有人 无人 多少
什么 怎样 如何 为啥 为啥 怎么 为何 何以 难道 莫非 是否 能否 可否 是否""".split())

# 过滤“了/的/在/是”等高频虚词参与的 2-gram（防止“了一/这样”成为实体）
_DROP_2GRAM = re.compile(r"[的了在是与和就都也又很太更最再可并而及或等于从对把被让向为以之其那这哪]")

_SENT_SPLIT = re.compile(r"[。！？；…]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _ngrams(text: str, n: int):
    """按汉字滑窗取 n-gram（跨标点断开）"""
    for seg in _CJK.findall(text):
        if len(seg) < n:
            continue
        for i in range(len(seg) - n + 1):
            yield seg[i : i + n]


def build_freq(chunks: list[dict]) -> dict[str, int]:
    """统计全书 2/3-gram 词频（排除纯功能词）"""
    counter: Counter[str] = Counter()
    for c in chunks:
        text = c.get("text", "")
        for gram in _ngrams(text, 2):
            counter[gram] += 1
        for gram in _ngrams(text, 3):
            counter[gram] += 1
    # 过滤：仅保留“书中特有词”（全局出现 30 次以上视为书名级特征词）
    freq = {k: v for k, v in counter.items() if v >= 30 and k not in _STOP}
    # 额外剔除高频虚词参与的 2-gram（如“了一”“这样”）
    freq = {k: v for k, v in freq.items() if len(k) >= 3 or not _DROP_2GRAM.search(k)}
    return freq


def pick_sentence(chunk_text: str, freq: dict[str, int]) -> tuple[str | None, str | None]:
    """从片段中挑一句实体区分度最高的句子；返回 (句子, 实体)"""
    best_sent, best_entity, best_score = None, None, 0
    for sent in _SENT_SPLIT.split(chunk_text):
        sent = sent.strip()
        if not (15 <= len(sent) <= 80):
            continue
        # 句子内出现的书中特有词（去重后计数 + 长度加权；优先 3-gram 实体）
        terms = set()
        tri_terms = set()
        for gram in _ngrams(sent, 3):
            if gram in freq:
                terms.add(gram)
                tri_terms.add(gram)
        if not tri_terms:
            for gram in _ngrams(sent, 2):
                if gram in freq:
                    terms.add(gram)
        if not terms:
            continue
        # 长度加权（实体越长越具区分度），3-gram 额外加成
        score = sum(freq[t] * (len(t) * 2) for t in terms)
        entity = max(terms, key=lambda t: (freq[t] * (len(t) * 2), len(t)))
        if score > best_score:
            best_sent, best_entity, best_score = sent, entity, score
    return best_sent, best_entity


def main():
    parser = argparse.ArgumentParser(description="生成锚定式 QA 集")
    parser.add_argument("--name", required=True, help="项目名（data/<name>/vector_db）")
    parser.add_argument("--count", type=int, default=20, help="生成 QA 条数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="", help="输出路径（默认 qa_sets/<name>.json）")
    args = parser.parse_args()

    vec_dir = ROOT / "data" / args.name / "vector_db"
    meta_file = vec_dir / "metadata.json"
    if not meta_file.exists():
        print(f"✗ 未找到向量库元数据: {meta_file}（请先运行 ingest）")
        sys.exit(1)

    chunks = json.loads(meta_file.read_text(encoding="utf-8"))
    print(f"ℹ 加载 {len(chunks)} 个片段，统计全书词频 ...")
    freq = build_freq(chunks)
    print(f"ℹ 书中特有词表: {len(freq)} 个（全局出现 >=30 次）")

    rng = random.Random(args.seed)
    total = len(chunks)
    # 均匀抽样 + 轻微抖动，保证覆盖全书且可复现
    idxs = [int(round((i + 0.5) / args.count * total)) for i in range(args.count)]
    idxs = [min(max(i + rng.randint(-3, 3), 0), total - 1) for i in idxs]
    idxs = sorted(set(idxs))

    items = []
    used_entities: set[str] = set()
    for ci in idxs:
        chunk = chunks[ci]
        text = chunk.get("text", "")
        if len(text) < 60:
            continue
        sent, entity = pick_sentence(text, freq)
        if not sent or not entity:
            continue
        if entity in used_entities:
            continue  # 同一实体只保留一题，保证 QA 集覆盖面
        used_entities.add(entity)
        book = chunk.get("book_title") or args.name
        question = f"在《{book}》中，「{entity}」相关的原文描述是什么？"
        items.append({
            "question": question,
            "chapter": chunk.get("chapter_title") or chunk.get("chapter") or "",
            "answer": sent,
            "entity": entity,
            "chunk_id": chunk.get("chunk_id"),
        })
        if len(items) >= args.count:
            break

    out = Path(args.out) if args.out else ROOT / "qa_sets" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 已生成 {len(items)} 条 QA: {out}")
    for it in items[:5]:
        print(f"  - {it['question']}")


if __name__ == "__main__":
    main()
