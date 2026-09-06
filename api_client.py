"""
api_client.py — API客户端
封装与大语言模型API的通信
支持 DeepSeek 及兼容 OpenAI API 格式的服务

设计要点（2026-09 重构）：
1. 失败即抛异常（APIClientError），不再把错误文本伪装成正常回复返回——
   调用方通过 try/except 区分「成功回复」与「调用失败」，避免 UI/CLI
   把错误信息当成 AI 回答展示。
2. 指数退避重试：对网络异常 / 429 / 5xx 自动重试（默认 3 次，
   1s → 2s → 4s），对 4xx 参数错误不重试（重试无意义）。
"""
import time

import requests
from typing import Optional


class APIClientError(Exception):
    """LLM API 调用失败（网络 / HTTP / 响应解析）。

    Attributes:
        status_code: HTTP 状态码（网络层失败时为 None）
        retryable:   是否属于「重试可能成功」的错误（网络抖动 / 429 / 5xx）
    """

    def __init__(self, message: str, status_code: Optional[int] = None,
                 retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class APIClient:
    """API客户端"""

    # 重试有意义的 HTTP 状态码：429 限流、5xx 服务端临时故障
    RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
    DEFAULT_TIMEOUT = 120

    def __init__(self, api_url: str, api_key: str, model_name: str,
                 max_retries: int = 3, timeout: Optional[int] = None,
                 retry_backoff: float = 1.0):
        """
        初始化API客户端

        Args:
            api_url: API端点URL
            api_key: API密钥
            model_name: 模型名称
            max_retries: 最大重试次数（默认 3；仅对网络异常/429/5xx 生效）
            timeout: 单次请求超时秒数（默认 120）
            retry_backoff: 重试基准退避秒数（按 2^n 指数增长）
        """
        self.api_url = self._normalize_url(api_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.retry_backoff = retry_backoff

    def _normalize_url(self, url: str) -> str:
        """标准化URL，确保以 /v1/chat/completions 结尾"""
        if not url.endswith("/v1/chat/completions"):
            if url.endswith("/"):
                url = url + "v1/chat/completions"
            else:
                url = url + "/v1/chat/completions"
        return url

    def _build_request(self, message: str, enable_thinking: bool) -> dict:
        """构造请求体"""
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": message}
            ],
            "temperature": 0.1,
        }
        if enable_thinking:
            data["thinking"] = {"type": "enabled"}
        return data

    def _request_once(self, data: dict) -> dict:
        """
        发送一次请求并解析响应。

        Returns:
            OpenAI 兼容格式的完整响应 JSON（dict）

        Raises:
            APIClientError: 网络异常 / 非 2xx / 响应解析失败
                （是否可重试由 error.retryable 标示）
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=data,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise APIClientError(
                f"请求超时（>{self.timeout}s）：{e}", retryable=True,
            ) from e
        except requests.exceptions.RequestException as e:
            raise APIClientError(
                f"网络请求失败：{e}\n请检查网络连接、API URL 是否正确。",
                retryable=True,
            ) from e

        if response.status_code >= 400:
            retryable = response.status_code in self.RETRYABLE_STATUS
            # 截断响应体，避免把超长错误页塞进异常消息
            detail = response.text[:300] if response.text else ""
            raise APIClientError(
                f"API 返回 HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
                retryable=retryable,
            )

        try:
            return response.json()
        except ValueError as e:
            raise APIClientError(
                f"API 响应不是合法 JSON（HTTP 200）：{response.text[:200]}"
            ) from e

    @staticmethod
    def _extract_reply(result: dict, enable_thinking: bool) -> str:
        """从 OpenAI 兼容响应中提取回复文本"""
        choices = result.get("choices") or []
        if not choices:
            raise APIClientError(f"API 响应缺少 choices 字段：{str(result)[:200]}")

        message_data = (choices[0] or {}).get("message", {}) or {}
        reasoning_content = message_data.get("reasoning_content", "") or ""
        content = message_data.get("content", "") or ""

        if enable_thinking and reasoning_content:
            return f"【思考过程】\n{reasoning_content}\n\n【最终回答】\n{content}"
        if content:
            return content
        raise APIClientError("模型返回了空内容（content 与 reasoning_content 均为空）。")

    def call_api(self, message: str, enable_thinking: bool = False,
                 max_retries: Optional[int] = None) -> str:
        """
        调用API发送消息（带指数退避重试）。

        Args:
            message: 用户消息（或完整Prompt）
            enable_thinking: 是否启用深度思考模式
            max_retries: 本次调用最大重试次数（None 使用构造参数）

        Returns:
            AI回复文本

        Raises:
            APIClientError: 重试耗尽后仍失败 / 4xx 参数错误 / 响应解析失败
        """
        data = self._build_request(message, enable_thinking)
        attempts = (max_retries if max_retries is not None
                    else self.max_retries) + 1
        last_error: Optional[APIClientError] = None

        for attempt in range(attempts):
            try:
                result = self._request_once(data)
                return self._extract_reply(result, enable_thinking)
            except APIClientError as e:
                last_error = e
                # 不可重试的错误（如 401/400）直接抛出
                if not e.retryable:
                    raise
                # 已用完重试次数
                if attempt >= attempts - 1:
                    break
                backoff = self.retry_backoff * (2 ** attempt)
                print(f"⚠ API 调用失败（{e.status_code or '网络'}），"
                      f"{backoff:.0f}s 后重试 "
                      f"({attempt + 1}/{attempts - 1}) ...")
                time.sleep(backoff)

        assert last_error is not None  # attempts >= 1，必然赋值
        raise APIClientError(
            f"API 调用多次重试后仍失败：{last_error.message}",
            status_code=last_error.status_code,
            retryable=True,
        )

    def test_connection(self) -> tuple:
        """
        测试API连接

        Returns:
            (是否成功, 消息)
        """
        try:
            reply = self.call_api("你好，请回复'连接成功'", enable_thinking=False)
            return True, reply or "连接成功"
        except APIClientError as e:
            return False, e.message
