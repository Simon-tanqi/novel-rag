# -*- coding: utf-8 -*-
"""
api_client 单元测试：指数退避重试、429/5xx 重试、4xx 不重试、响应解析。
（用 monkeypatch 替换 requests.post，不发起真实网络请求）
"""
import pytest

from api_client import APIClient, APIClientError


def _client(**kwargs):
    """构造一个指向任意 URL 的客户端（测试中不会真实请求）"""
    return APIClient("http://mock.example/v1/chat/completions",
                     "sk-test", "deepseek-chat", **kwargs)


def _ok_response(content="你好"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }


class TestCallApiSuccess:
    def test_normal_reply(self, monkeypatch):
        seen = []
        def fake_post(*a, **k):
            seen.append(1)
            return _FakeResp(200, _ok_response("连接成功"))
        monkeypatch.setattr("requests.post", fake_post)
        client = _client()
        reply = client.call_api("你好")
        assert reply == "连接成功"
        # 一次成功不应触发重试
        assert len(seen) == 1

    def test_retries_then_success(self, monkeypatch):
        """首次 500，重试后 200 → 应返回成功回复"""
        responses = iter([
            _FakeResp(500, {"error": "boom"}),
            _FakeResp(200, _ok_response("恢复成功")),
        ])
        monkeypatch.setattr("requests.post", lambda *a, **k: next(responses))
        client = _client(max_retries=2, retry_backoff=0.01)
        reply = client.call_api("你好")
        assert reply == "恢复成功"

    def test_retries_on_429(self, monkeypatch):
        """429 限流应重试"""
        responses = iter([
            _FakeResp(429, {"error": "rate limited"}),
            _FakeResp(200, _ok_response("OK")),
        ])
        monkeypatch.setattr("requests.post", lambda *a, **k: next(responses))
        client = _client(max_retries=2, retry_backoff=0.01)
        assert client.call_api("hi") == "OK"


class TestCallApiFailure:
    def test_4xx_no_retry(self, monkeypatch):
        """401/400 参数错误：不应重试，应直接抛错"""
        seen = []
        def fake_post(*a, **k):
            seen.append(1)
            return _FakeResp(401, {"error": "invalid key"})
        monkeypatch.setattr("requests.post", fake_post)
        client = _client(max_retries=3, retry_backoff=0.01)
        with pytest.raises(APIClientError) as exc:
            client.call_api("hi")
        assert exc.value.status_code == 401
        assert exc.value.retryable is False
        assert len(seen) == 1  # 只请求了一次

    def test_5xx_exhausted_raises(self, monkeypatch):
        """持续 500：重试耗尽后应抛 APIClientError"""
        seen = []
        def fake_post(*a, **k):
            seen.append(1)
            return _FakeResp(500, {"error": "boom"})
        monkeypatch.setattr("requests.post", fake_post)
        client = _client(max_retries=2, retry_backoff=0.01)
        with pytest.raises(APIClientError) as exc:
            client.call_api("hi")
        assert exc.value.retryable is True
        assert len(seen) == 3  # 初始 1 次 + 重试 2 次

    def test_empty_content_raises(self, monkeypatch):
        """模型返回空 content → 应抛异常而非静默返回空串"""
        monkeypatch.setattr(
            "requests.post",
            lambda *a, **k: _FakeResp(
                200, {"choices": [{"message": {"content": ""}}]}
            ),
        )
        client = _client()
        with pytest.raises(APIClientError):
            client.call_api("hi")

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setattr(
            "requests.post",
            lambda *a, **k: _FakeResp(200, None, raw_body="not-json"),
        )
        client = _client()
        with pytest.raises(APIClientError):
            client.call_api("hi")


class TestThinking:
    def test_thinking_prepends_reasoning(self, monkeypatch):
        payload = {"choices": [{"message": {
            "reasoning_content": "推理中…", "content": "最终答案",
        }}]}
        monkeypatch.setattr(
            "requests.post", lambda *a, **k: _FakeResp(200, payload)
        )
        client = _client()
        reply = client.call_api("hi", enable_thinking=True)
        assert "最终答案" in reply and "推理中…" in reply


class _FakeResp:
    """最小 requests.Response 替身"""

    def __init__(self, status_code, json_obj, raw_body=None):
        self.status_code = status_code
        self._json = json_obj
        # text 始终为字符串（请求层会对其切片截断）
        if raw_body is not None:
            self.text = raw_body
        elif json_obj is not None:
            import json as _json
            self.text = _json.dumps(json_obj, ensure_ascii=False)
        else:
            self.text = ""

    def json(self):
        if self._json is None:
            raise ValueError("No JSON object could be decoded")
        return self._json

    def raise_for_status(self):  # 兼容旧代码路径
        pass
