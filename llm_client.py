import json
from typing import Any, Callable, Mapping, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClient(Protocol):
    """大模型客户端协议，不依赖具体厂商 SDK。"""

    def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """发送对话消息并返回纯文本结果。

        :param messages: OpenAI 风格的 role/content 消息列表
        :param model: 可选的模型名称
        :param temperature: 采样温度
        :return: 模型生成的纯文本
        """
        ...


class CallableLLMClient:
    """将任意函数适配为 LLMClient。"""

    def __init__(self, chat_callable: Callable[..., str]):
        if not callable(chat_callable):
            raise TypeError("chat_callable 必须可调用")
        self._chat_callable = chat_callable

    def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """调用底层函数并校验文本返回值。"""
        result = self._chat_callable(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        if not isinstance(result, str):
            raise TypeError("LLM 客户端必须返回 str")
        return result


class LLMRequestException(Exception):
    """大模型 HTTP 请求异常。"""


HeaderBuilder = Callable[
    [list[dict], Optional[str], float],
    Mapping[str, str],
]
BodyBuilder = Callable[[list[dict], Optional[str], float], Any]
ResponseParser = Callable[[str], str]


class HttpLLMClient:
    """通过可配置的 HTTP 协议调用大模型接口。"""

    def __init__(
        self,
        endpoint: str,
        header_builder: HeaderBuilder,
        body_builder: BodyBuilder,
        response_parser: ResponseParser,
        timeout: float = 60.0,
    ):
        """初始化通用 HTTP 大模型客户端。

        :param endpoint: 完整的模型请求地址，不再自动拼接路径
        :param header_builder: 根据 messages、model、temperature 生成请求头
        :param body_builder: 根据 messages、model、temperature 生成 JSON 请求体
        :param response_parser: 将响应文本解析为模型输出字符串
        :param timeout: 单次请求超时时间，单位为秒
        """
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("LLM API endpoint 不能为空")
        if not callable(header_builder):
            raise TypeError("header_builder 必须可调用")
        if not callable(body_builder):
            raise TypeError("body_builder 必须可调用")
        if not callable(response_parser):
            raise TypeError("response_parser 必须可调用")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout 必须大于 0")

        self._endpoint = endpoint.strip()
        self._header_builder = header_builder
        self._body_builder = body_builder
        self._response_parser = response_parser
        self._timeout = float(timeout)

    def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """使用自定义协议发送对话请求并返回文本结果。

        :param messages: role/content 消息列表
        :param model: 可选的模型名称
        :param temperature: 采样温度
        :return: response_parser 解析后的模型文本
        :raises LLMRequestException: 网络请求或响应解析异常
        """
        headers = self._header_builder(messages, model, temperature)
        if not isinstance(headers, Mapping):
            raise TypeError("header_builder 必须返回 Mapping")

        normalized_headers = {
            str(key): str(value) for key, value in headers.items()
        }
        if not any(
            key.lower() == "content-type" for key in normalized_headers
        ):
            normalized_headers["Content-Type"] = "application/json"

        payload = self._body_builder(messages, model, temperature)
        try:
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "body_builder 返回值必须可序列化为 JSON"
            ) from exc

        request = Request(
            self._endpoint,
            data=request_body,
            headers=normalized_headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMRequestException(
                f"LLM API HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise LLMRequestException(
                f"LLM API 连接失败: {exc.reason}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise LLMRequestException(
                "LLM API 响应不是 UTF-8 编码"
            ) from exc

        try:
            content = self._response_parser(raw)
        except Exception as exc:
            raise LLMRequestException(
                f"LLM API 响应解析失败: {exc}"
            ) from exc

        if not isinstance(content, str):
            raise LLMRequestException(
                "response_parser 必须返回文本内容"
            )
        return content


class DeepSeekLLMClient:
    """通过 DeepSeek Chat Completions HTTP API 调用模型。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("DeepSeek API Key 不能为空")

        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        self._api_key = api_key.strip()
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """调用 DeepSeek Chat Completions 接口。

        :param messages: role/content 消息列表
        :param model: 覆盖构造时的模型名称
        :param temperature: 采样温度
        :return: 模型生成的文本
        :raises LLMRequestException: 网络请求或响应格式异常
        """
        payload = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        request = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMRequestException(
                f"DeepSeek API HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise LLMRequestException(
                f"DeepSeek API 连接失败: {exc.reason}"
            ) from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMRequestException("DeepSeek API 响应格式异常") from exc

        if not isinstance(content, str):
            raise LLMRequestException("DeepSeek API 未返回文本内容")
        return content


def invoke(
    client: LLMClient,
    messages: list[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """通过统一协议调用大模型。

    :param client: 任意实现 LLMClient 协议的对象
    :param messages: role/content 消息列表
    :param model: 可选的模型名称
    :param temperature: 采样温度
    :return: 模型生成的纯文本
    """
    if client is None:
        raise ValueError("LLM client 不能为空")
    if not hasattr(client, "chat"):
        raise TypeError("LLM client 必须实现 chat 方法")

    result = client.chat(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    if not isinstance(result, str):
        raise TypeError("LLM client 的 chat 方法必须返回 str")
    return result
