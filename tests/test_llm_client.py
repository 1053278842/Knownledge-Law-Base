import json
import unittest
from unittest.mock import MagicMock, patch

from llm_client import HttpLLMClient


class HttpLLMClientTest(unittest.TestCase):
    """验证通用 HTTP 大模型客户端的协议适配能力。"""

    @patch("llm_client.urlopen")
    def test_chat_uses_custom_protocol(self, mock_urlopen):
        """自定义请求头、请求体和响应解析器应全部生效。"""
        response = MagicMock()
        response.read.return_value = (
            '{"data":{"answer":"internal result"}}'.encode("utf-8")
        )
        response.__enter__.return_value = response
        mock_urlopen.return_value = response

        def header_builder(messages, model, temperature):
            return {
                "X-API-Key": "secret",
                "X-Tenant-Id": "legal",
            }

        def body_builder(messages, model, temperature):
            return {
                "input": messages,
                "parameters": {
                    "model": model,
                    "temperature": temperature,
                },
            }

        def response_parser(raw):
            return json.loads(raw)["data"]["answer"]

        client = HttpLLMClient(
            endpoint="https://intranet.example/api/llm/chat",
            header_builder=header_builder,
            body_builder=body_builder,
            response_parser=response_parser,
            timeout=12.5,
        )
        result = client.chat(
            [{"role": "user", "content": "hello"}],
            model="internal-model",
            temperature=0.2,
        )

        self.assertEqual("internal result", result)
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(
            "https://intranet.example/api/llm/chat",
            request.full_url,
        )
        self.assertEqual("POST", request.get_method())
        self.assertEqual("secret", request.get_header("X-api-key"))
        self.assertEqual("legal", request.get_header("X-tenant-id"))
        self.assertEqual(
            "application/json",
            request.get_header("Content-type"),
        )
        self.assertEqual(
            {
                "input": [{"role": "user", "content": "hello"}],
                "parameters": {
                    "model": "internal-model",
                    "temperature": 0.2,
                },
            },
            json.loads(request.data.decode("utf-8")),
        )
        mock_urlopen.assert_called_once_with(request, timeout=12.5)


if __name__ == "__main__":
    unittest.main()
