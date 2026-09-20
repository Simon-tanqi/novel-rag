# -*- coding: utf-8 -*-
"""query_rewriter.py — 检索前查询改写（元词 → 具体实体名）

问题：问「主角叫什么名字？」这类问题时，查询串只含「元词」——「主角」是
指代性词汇，而全书第三人称直呼主角名（如示例书的主角「陈曦」），书首登场
片段与该问句的语义相似度低、排不进 top_k；召回内容全是无关配角对白，
最终触发 Prompt 的「无相关信息」规则拒答。

方案：检索前做一次「查询改写」——当且仅当问题命中元词（主角 / 男主 /
女主 / 主人公 / 男一号 / 女一号 / 主角团 等）时，复用 APIClient 调一次
LLM（deepseek-chat，temperature=0，max_tokens≈150）把元词解析为具体人物
名，产出若干条改写查询；原查询 + 各改写查询一起多路召回（见
``rag_retriever.RAGRetriever.retrieve_multi``），再由原有的 RRF 融合、
章节聚合与 top_k 截断收尾。

设计约束（三条硬红线）：
1. **路由触发**：非元词问题不发起任何 LLM 调用，零额外开销；
2. **静默降级**：任何网络 / 解析异常一律回退为「不改写」，函数绝不抛错，
   不得影响原有问答链路；
3. **短调用**：单次请求、不重试、temperature=0、输出长度受限。

用法::

    from query_rewriter import detect_meta_terms, rewrite_query

    if detect_meta_terms(question):
        result = rewrite_query(question, api_client, novel_name="斗罗大陆")
        extra_queries = result.extra_queries   # 失败时为空列表
        print(result.summary)                  # 「查询改写：主角 → 陈曦」
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ===================== 元词表 =====================

# 指代性元词：本身不含具体实体信息，单独作为查询串时召回效果极差。
# 长词在前，命中长词后其包含的短词不再重复计（如「主角团」命中后不再报「主角」）。
META_TERMS = [
    "男主人公", "女主人公", "主角团", "男一号", "女一号",
    "主人公", "男主角", "女主角", "男主", "女主",
    "主角", "男配", "女配", "反派",
]

_SORTED_META_TERMS = sorted(set(META_TERMS), key=len, reverse=True)

# ===================== 改写请求参数 =====================

REWRITE_TEMPERATURE = 0.0     # 改写要确定性，不要发挥
REWRITE_MAX_TOKENS = 150      # 输出极短：一段 JSON 而已
REWRITE_MAX_RETRIES = 0       # 改写是「锦上添花」，失败即降级，不重试

MAX_ENTITIES = 3              # 最多接受几个实体名
MAX_QUERIES = 3               # 最多接受几条改写查询
MAX_ENTITY_CHARS = 20         # 实体名长度上限（防模型输出整句）
MAX_QUERY_CHARS = 60          # 改写查询长度上限

REWRITE_INSTRUCTION = (
    "你是小说检索系统的「查询改写器」。用户的问题里可能用「主角」「男主」"
    "「女主」等元词指代人物，而小说原文里一律直呼具体人名，直接用元词检索"
    "召回不到答案。你的任务：把这些元词解析为具体人物名，并给出改写后的"
    "检索查询。\n"
    "要求：\n"
    "1. 只输出一个 JSON 对象，不要输出解释、前后缀或 Markdown 代码块；\n"
    "2. 格式：{\"entities\": [\"人物名\"], \"queries\": [\"改写查询1\", \"改写查询2\"]}；\n"
    "3. entities / queries 中必须是具体人名，不得再出现元词；\n"
    "4. 改写查询要贴近原文的称呼方式（例如问「主角叫什么名字」可改写为"
    "「<人名>的姓名」「<人名> 是谁」）；\n"
    f"5. entities 至多 {MAX_ENTITIES} 个，queries 至多 {MAX_QUERIES} 条；\n"
    "6. 若你没有把握确定具体人名，输出 {\"entities\": [], \"queries\": []}，"
    "严禁编造。"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


# ===================== 结果结构 =====================


@dataclass
class RewriteResult:
    """查询改写结果（结构化）。

    Attributes:
        triggered:  是否命中元词、触发了改写流程（无论成功与否）
        entities:   解析出的具体实体名（如 ["陈曦"]）
        queries:    改写后的检索查询（如 ["陈曦的姓名"]）
        meta_terms: 命中的元词（如 ["主角"]）
        raw:        LLM 原始返回（仅用于排查）
        error:      失败原因；空串表示成功
    """

    triggered: bool = False
    entities: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    meta_terms: List[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""

    @property
    def succeeded(self) -> bool:
        """是否真正产出了可用的改写结果（失败时为 False → 回退原查询）"""
        return self.triggered and not self.error and bool(self.entities or self.queries)

    @property
    def extra_queries(self) -> List[str]:
        """供多路召回使用的改写查询；未成功时为空列表（回退原查询）"""
        return list(self.queries) if self.succeeded else []

    @property
    def summary(self) -> str:
        """UI 展示摘要，如「查询改写：主角 → 陈曦」；无改写时返回空串"""
        if not self.succeeded:
            return ""
        source = "、".join(self.meta_terms) if self.meta_terms else "元词"
        target = "、".join(self.entities) if self.entities else ""
        if not target:
            # 只有 queries、没解析出实体名时退化为展示查询
            target = "、".join(self.queries[:2])
        return f"查询改写：{source} → {target}"

    def to_dict(self) -> Dict[str, Any]:
        """结构化输出（便于日志 / 测试断言）"""
        return {
            "triggered": self.triggered,
            "succeeded": self.succeeded,
            "meta_terms": list(self.meta_terms),
            "entities": list(self.entities),
            "queries": list(self.queries),
            "summary": self.summary,
            "error": self.error,
        }


# ===================== 元词识别 =====================


def detect_meta_terms(question: str) -> List[str]:
    """识别问题中命中哪些元词（长词优先，不重复计其包含的短词）

    Args:
        question: 用户问题

    Returns:
        命中的元词列表（按长度降序）；未命中返回空列表
    """
    if not question:
        return []
    hits: List[str] = []
    for term in _SORTED_META_TERMS:
        if term in question and not any(term in h for h in hits):
            hits.append(term)
    return hits


def has_meta_terms(question: str) -> bool:
    """问题是否含元词（路由开关：仅此函数为 True 时才发起改写调用）"""
    return bool(detect_meta_terms(question))


# ===================== Prompt 构建 =====================


def build_rewrite_prompt(
    question: str,
    meta_terms: Optional[List[str]] = None,
    novel_name: str = "",
) -> str:
    """构建改写用的单轮 Prompt（APIClient 只发一条 user 消息，故指令与问题合一）"""
    terms = meta_terms if meta_terms is not None else detect_meta_terms(question)
    novel_part = f"正在检索的小说：《{novel_name}》。\n" if novel_name else ""
    return (
        f"{REWRITE_INSTRUCTION}\n\n"
        f"{novel_part}"
        f"用户问题：{question}\n"
        f"问题中的元词：{'、'.join(terms) if terms else '（未识别到）'}\n"
        f"请输出 JSON："
    )


# ===================== 响应解析 =====================


def _extract_json_obj(text: str) -> Optional[dict]:
    """从 LLM 返回文本中稳健地抽出 JSON 对象（容忍代码块 / 前后缀文本）"""
    if not text:
        return None
    candidates: List[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            # 容错：模型直接吐了列表
            return {"entities": obj}
    return None


def _clean_str_list(value: Any, max_items: int, max_chars: int) -> List[str]:
    """把 LLM 返回的字段规整为去空、去重、限长的字符串列表"""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []

    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip().strip('"\'“”').strip()
        if not text or len(text) > max_chars:
            continue
        if text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def parse_rewrite_response(
    raw: str,
    meta_terms: Optional[List[str]] = None,
    original_query: str = "",
) -> RewriteResult:
    """解析 LLM 返回，得到结构化改写结果（解析失败 → error 非空）"""
    meta_terms = list(meta_terms or [])
    result = RewriteResult(triggered=True, meta_terms=meta_terms, raw=raw or "")

    data = _extract_json_obj(raw or "")
    if data is None:
        result.error = "响应不是合法 JSON"
        return result

    entities = _clean_str_list(data.get("entities"), MAX_ENTITIES, MAX_ENTITY_CHARS)
    queries = _clean_str_list(data.get("queries"), MAX_QUERIES, MAX_QUERY_CHARS)

    # 过滤：改写结果里不允许再出现元词（否则等于没改写）
    entities = [e for e in entities if not detect_meta_terms(e)]

    cleaned_queries: List[str] = []
    for query in queries:
        # 改写查询必须引入新信息：不能仍含元词，也不能与原查询完全相同
        if detect_meta_terms(query):
            continue
        if query.strip() == (original_query or "").strip():
            continue
        if query not in cleaned_queries:
            cleaned_queries.append(query)

    result.entities = entities
    result.queries = cleaned_queries
    if not entities and not cleaned_queries:
        result.error = "模型未给出可用的具体实体名"
    return result


# ===================== 对外主入口 =====================


def rewrite_query(
    question: str,
    api_client: Any,
    novel_name: str = "",
    max_retries: int = REWRITE_MAX_RETRIES,
    temperature: float = REWRITE_TEMPERATURE,
    max_tokens: int = REWRITE_MAX_TOKENS,
) -> RewriteResult:
    """检索前查询改写（路由触发 + 静默降级）。

    Args:
        question: 用户原始问题
        api_client: 已构造好的 APIClient（复用现有 deepseek-chat 配置）
        novel_name: 当前小说名（用于提示模型判断人物）
        max_retries: 改写调用重试次数（默认 0：失败即降级）
        temperature: 采样温度（默认 0）
        max_tokens: 输出长度上限（默认 150）

    Returns:
        RewriteResult；任何异常都转成 ``error`` 字段，绝不抛出，
        调用方通过 ``result.extra_queries`` 取用（失败时为空列表 → 原查询）
    """
    meta_terms = detect_meta_terms(question)
    if not meta_terms:
        return RewriteResult(triggered=False)
    if api_client is None:
        result = RewriteResult(triggered=True, meta_terms=meta_terms,
                               error="未提供 APIClient")
        print(f"⚠ 查询改写不可用（{result.error}），回退原查询")
        return result

    prompt = build_rewrite_prompt(question, meta_terms, novel_name)
    try:
        raw = api_client.call_api(
            prompt,
            enable_thinking=False,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:  # 网络 / HTTP / 参数不兼容 / 任何异常 → 静默降级
        result = RewriteResult(
            triggered=True, meta_terms=meta_terms,
            error=f"{type(e).__name__}: {e}",
        )
        print(f"⚠ 查询改写失败（{result.error}），本次回退原查询")
        return result

    result = parse_rewrite_response(raw, meta_terms, original_query=question)
    if result.error:
        print(f"⚠ 查询改写解析失败（{result.error}），本次回退原查询")
    else:
        print(f"✓ {result.summary}（改写查询 {len(result.queries)} 条）")
    return result
