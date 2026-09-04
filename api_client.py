"""
api_client.py — API客户端
封装与大语言模型API的通信
支持 DeepSeek 及兼容 OpenAI API 格式的服务
"""
import requests
from typing import Optional


class APIClient:
    """API客户端"""

    def __init__(self, api_url: str, api_key: str, model_name: str):
        """
        初始化API客户端

        Args:
            api_url: API端点URL
            api_key: API密钥
            model_name: 模型名称
        """
        self.api_url = self._normalize_url(api_url)
        self.api_key = api_key
        self.model_name = model_name

    def _normalize_url(self, url: str) -> str:
        """标准化URL，确保以 /v1/chat/completions 结尾"""
        if not url.endswith("/v1/chat/completions"):
            if url.endswith("/"):
                url = url + "v1/chat/completions"
            else:
                url = url + "/v1/chat/completions"
        return url

    def call_api(self, message: str, enable_thinking: bool = False) -> str:
        """
        调用API发送消息

        Args:
            message: 用户消息（或完整Prompt）
            enable_thinking: 是否启用深度思考模式

        Returns:
            AI回复文本
        """
        response = None
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "temperature": 0.1
            }

            if enable_thinking:
                data["thinking"] = {"type": "enabled"}

            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()

            try:
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message_data = choice.get("message", {})

                    reasoning_content = message_data.get("reasoning_content", "")
                    content = message_data.get("content", "")

                    if enable_thinking and reasoning_content:
                        reply = f"【思考过程】\n{reasoning_content}\n\n【最终回答】\n{content}"
                    else:
                        reply = content

                    if not reply:
                        reply = "模型未返回有效回复，请稍后重试。"
                else:
                    reply = f"API响应格式异常：{result}"
            except Exception as e:
                reply = f"解析响应失败：{str(e)}\n原始响应：{result}"

            return reply

        except requests.exceptions.RequestException as e:
            error_details = "No response received"
            try:
                if response:
                    error_details = response.text
            except Exception:
                pass

            return (
                f"API调用失败: {str(e)}\n\n"
                f"错误详情: {error_details}\n\n"
                f"请检查：\n"
                f"1. API URL是否正确\n"
                f"2. API Key是否正确\n"
                f"3. 模型名称是否正确\n"
                f"4. 网络连接是否正常"
            )

        except KeyError as e:
            return (
                f"响应格式错误：缺少字段 {str(e)}\n\n"
                f"这可能是API响应格式不兼容导致的。请检查：\n"
                f"1. 是否选择了正确的模型\n"
                f"2. API是否支持该请求格式\n"
                f"3. 模型名称是否正确"
            )

        except Exception as e:
            return f"发生未知错误：{str(e)}\n\n类型: {type(e).__name__}"

    def test_connection(self) -> tuple:
        """
        测试API连接

        Returns:
            (是否成功, 消息)
        """
        try:
            reply = self.call_api("你好，请回复'连接成功'", enable_thinking=False)
            if "失败" not in reply and "错误" not in reply and "异常" not in reply:
                return True, "连接成功"
            return False, reply
        except Exception as e:
            return False, str(e)
