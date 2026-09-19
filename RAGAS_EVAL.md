---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: ef5b15bc948ed1a80cab91ca175eedba_26f540b5b1e611f197a3525400248c00
    ReservedCode1: Rh++vm6hxEvuXVrFHAK603d9oXONpqSr1ZhxyXIOqwu830W7yQ6QKJ5n1sKtGcEsOdxFTCJTZFiaZBQB8KECuxrwQUr+3xQ8czGvL1w6aqThA/jc6D+TKbH+dspM6AZ6ORfK8DZaARuHOnKnsaUiU4/2l4eQ4jKUByuJwEoiilesEhOTdW2ZYWl6UnQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: ef5b15bc948ed1a80cab91ca175eedba_26f540b5b1e611f197a3525400248c00
    ReservedCode2: Rh++vm6hxEvuXVrFHAK603d9oXONpqSr1ZhxyXIOqwu830W7yQ6QKJ5n1sKtGcEsOdxFTCJTZFiaZBQB8KECuxrwQUr+3xQ8czGvL1w6aqThA/jc6D+TKbH+dspM6AZ6ORfK8DZaARuHOnKnsaUiU4/2l4eQ4jKUByuJwEoiilesEhOTdW2ZYWl6UnQ=
---

# RAGAS 系统评测指南

本文件说明如何在 `novel-rag` 上运行 [RAGAS](https://docs.ragas.io/) 系统评测，包括环境隔离、
数据管线、四项指标在**中文小说问答**场景下的口径与局限，以及没有 API KEY 时能跑到哪一步。

> 一句话结论：RAGAS **可以**接入本项目（Python 3.13 实测通过），但**必须装在独立虚拟环境**里，
> 且四项指标全部依赖 LLM，因此真正打分需要 API KEY；无 KEY 时可完成环境搭建、数据集生成与
> dry-run 自检。

---

## 1. 为什么必须独立环境

| 冲突点 | 说明 |
| --- | --- |
| `langchain` 版本 | ragas 0.4.3 仍 `import langchain_community.chat_models.vertexai`，该模块在 langchain 1.x 的 community 包中已被移除 → 主环境若装 1.x，ragas 直接 ImportError；反之把 langchain 降到 0.3.x 会顶掉主环境依赖 |
| `numpy` / `pandas` 版本 | ragas 依赖链会拉 `datasets` → `pyarrow`、`pandas 3.x`，与主流程固定的版本区间不一致 |
| 体积 | 评测链路需要 `torch`(CPU) + `transformers`，混装会让主环境从 ~300MB 膨胀到 1GB+ |

因此评测依赖单独放在 **`requirements-extras-ragas.txt`**，环境单独放在 **`.ragas_venv/`**
（已在 `.gitignore` 建议中忽略），主 `.venv` 完全不受影响。

实测版本组合（Python 3.13.1 / Windows 11）：

```
ragas 0.4.3
langchain 0.3.30 / langchain-core 0.3.86 / langchain-community 0.3.27 / langchain-openai 0.3.35
datasets 5.0.1  pandas 3.0.5  numpy 2.5.3
sentence-transformers 6.0.1  torch (CPU, +cpu)
```

> 若 `pip install torch` 报 `WinError 206 文件名或扩展名太长`：说明虚拟环境路径太深
> （例如放在临时目录里），把 `.ragas_venv` 建在项目根或浅盘符目录即可。

---

## 2. 一键搭建

```powershell
# 在项目根目录
powershell -ExecutionPolicy Bypass -File scripts/ragas_env_setup.ps1
# 国内镜像 / 跳过 torch
powershell -ExecutionPolicy Bypass -File scripts/ragas_env_setup.ps1 -SkipTorch -Mirror https://mirrors.cloud.tencent.com/pypi/simple
```

脚本做四件事：建 `.ragas_venv` → 升级 pip → 装 CPU 版 torch → 装 `requirements-extras-ragas.txt` → 自检 import。

---

## 3. 四步工作流

```powershell
# ── 阶段 A：项目主环境（.venv），无 KEY 也能跑 ──
python scripts/ragas_eval.py gen --name zhetian --qa qa_sets/zhetian_natural.json
python scripts/ragas_eval.py gen --name jszz    --qa qa_sets/jszz_natural.json
# → ragas_out/<name>_natural_dataset.json（question / contexts / ground_truth / 元数据）

# ── 阶段 B：RAGAS 独立环境（.ragas_venv），无 KEY 也能跑 ──
.\.ragas_venv\Scripts\python.exe scripts/ragas_eval.py dry-run --dataset ragas_out/zhetian_natural_dataset.json

# ── 阶段 C：两条需要 API KEY 的命令 ──
# C1 生成 answer（主环境，走与 novel_rag.py ask 相同的 prompt 链路）
python scripts/ragas_eval.py gen-answers --dataset ragas_out/zhetian_natural_dataset.json
# C2 打分（RAGAS 环境）
.\.ragas_venv\Scripts\python.exe scripts/ragas_eval.py score --dataset ragas_out/zhetian_natural_dataset.json
```

有用参数：

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `--top-k` | gen | 检索深度，默认取 `config.json` 的 `top_k` |
| `--enable-rerank` | gen | 生成 contexts 时启用精排（建议与线上配置一致） |
| `--max-ctx` / `--ctx-chars` | gen | 送进评测的上下文条数/截断字数（默认 8 条 × 1200 字） |
| `--metrics` | score | 逗号分隔，默认四项全开 |
| `--adapt-prompts` | score | 尝试把指标提示词适配为中文（额外 token 开销） |
| `--embedding-model` | score | 覆盖 answer_relevancy 用的向量模型（默认本地 `models/bge-small-zh-v1.5`） |

产出：`ragas_out/<dataset>_ragas_<时间戳>.{csv,json,md}`（逐题分数 + 平均值 + Markdown 报告）。

---

## 4. 四项指标的口径（本项目落地定义）

| 指标 | 依赖字段 | 计算方式 | 回答的问题 |
| --- | --- | --- | --- |
| `context_recall` | question + contexts + **ground_truth** | 把 ground_truth 拆成陈述句，逐句判断能否由 contexts 推出，取覆盖比例 | 该召回的情节是否都进了上下文 |
| `context_precision` | question + contexts + **ground_truth** | 逐个判断 context 与问题是否相关，按名次加权 | 捞回来的上下文是不是大部分有用 |
| `faithfulness` | **answer** + contexts | 把 answer 拆成陈述句，逐句判断能否由 contexts 支撑 | 答案有没有编造原文里没有的情节 |
| `answer_relevancy` | **answer** + question | 由 answer 反推问题，与原问题算语义相似度（本地 bge-small-zh 向量） | 是否答非所问 |

打分链条中 **judge LLM 与 judge embeddings 都是外挂的**：

- judge LLM：复用项目 `config.json` 里的 API（`_openai_base_url()` 会把
  `https://api.deepseek.com/v1/chat/completions` 归一化成 langchain 可用的 `base_url`）；
- judge embeddings：直接加载本地 `models/bge-small-zh-v1.5`，**不联网、不调用 API**，
  这样 `answer_relevancy` 不额外花钱（也避免了 `langchain-huggingface` 依赖）。

---

## 5. 中文小说场景的局限（判读必读）

1. **ground_truth 的形态决定前两项指标的绝对值。** 本项目默认取 QA 文件里人工核验过的
   `evidence` 原文依据句（20/20 条均为 evidence 来源）。这是"原文摘录"，与检索到的
   contexts 同源，会把 `context_recall` **抬高**；它反映的是"情节是否被覆盖"，不等价于
   "答案是否完整"。若要更严格，应把 ground_truth 改写成独立成句的参考答案。
2. **faithfulness 对"未找到相关信息"型答案是盲区。** 拒答型 answer 没有可验证陈述，
   分数可能落在 1.0 或 NaN（视 ragas 版本），不能直接与"正常作答"样本混算平均。
3. **answer_relevancy 对中文长答案偏保守。** 该指标的提示词模板是英文的，由 answer 反推
   问题时对中文长句可能欠采；`--adapt-prompts` 可缓解，但会额外消耗 token。
4. **context_precision 按"检索条目"打分，条目粒度会显著影响分数。** 本项目 contexts 是
   章节聚合后的父块/邻居扩展片段（较长），单条通常"部分相关"——分数会低于按短 chunk
   打分的系统，不宜跨系统直接比。
5. **专名 / 功法名 / 境界名**这类无实义专有词，judge 模型只能靠上下文猜，可能出现误判；
   建议对含大量专名的题单独看 `per_question` 明细而不是只看均值。
6. **成本**：每题约 4 个指标 × 若干次 LLM 调用（含陈述句拆分、逐条相关性判断），
   40 题规模的单次评测属于"几十到上百次调用"量级，建议先用 `--metrics context_recall`
   小规模试跑再全量。

---

## 6. 无 KEY 时能跑到哪一步（已实测）

| 环节 | 无 KEY | 说明 |
| --- | --- | --- |
| 环境搭建 + 版本自检 | ✅ 已完成 | `.ragas_venv` 内 `ragas 0.4.3 / langchain 0.3.30` import 通过 |
| 数据集生成（`gen`） | ✅ 已完成 | 遮天 20 题、绝世主宰 20 题，contexts 来自真实检索链路 |
| dry-run 自检 | ✅ 已完成 | 字段完整性、长度分布、ragas 指标对象与 `EvaluationDataset` 构建全部通过 |
| answer 生成（`gen-answers`） | ❌ 需 KEY | 走项目同款 prompt，保证评测对象与线上一致 |
| 打分（`score`） | ❌ 需 KEY | 四项指标全部依赖 judge LLM |

**KEY 到位后的完整命令**（假设用 DeepSeek）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
python scripts/ragas_eval.py gen-answers --dataset ragas_out/zhetian_natural_dataset.json
python scripts/ragas_eval.py gen-answers --dataset ragas_out/jszz_natural_dataset.json
.\.ragas_venv\Scripts\python.exe scripts/ragas_eval.py score --dataset ragas_out/zhetian_natural_dataset.json
.\.ragas_venv\Scripts\python.exe scripts/ragas_eval.py score --dataset ragas_out/jszz_natural_dataset.json
```

---

## 7. 与召回率评测的分工

| 层 | 脚本 | 需要 KEY | 回答什么 |
| --- | --- | --- | --- |
| 检索层 | `scripts/eval_config_compare.py`（配置对照）、`scripts/eval_recall_anchor.py` | 否 | 目标章节有没有被召回（Recall@k / MRR），即**60 分的前置条件** |
| 生成层 + 端到端 | `scripts/ragas_eval.py score` | 是 | 上下文够不够（context_*）、答案是否忠实（faithfulness）、是否答非所问（answer_relevancy） |

两者互补：检索层无成本、可回归；RAGAS 层需要 KEY、衡量"最终答案质量"。
*（内容由AI生成，仅供参考）*
