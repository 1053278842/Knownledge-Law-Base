import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from config import (
    get_chat_config,
    get_embedding_config,
    get_rerank_config,
)
from llm_client import (
    LLMClient,
    LLMRequestException,
    OpenAICompatibleChatClient,
)


class EmbeddingClient(Protocol):
    def embed_texts(
        self,
        texts: list[str],
        *,
        kind: str = "document",
    ) -> np.ndarray:
        ...


class RerankerClient(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        ...


_LOCAL_EMBEDDING_MODELS: dict[str, object] = {}
_LOCAL_RERANK_MODELS: dict[str, object] = {}


def _request_json(
    *,
    service: str,
    url: str,
    api_key: str,
    payload: dict,
    timeout: float,
) -> dict:
    if not url:
        raise ValueError(f"{service} URL 不能为空")
    if not api_key:
        raise ValueError(f"{service} API Key 不能为空")

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMRequestException(
            f"{service} HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise LLMRequestException(
            f"{service} 连接失败: {exc.reason}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise LLMRequestException(
            f"{service} 响应不是 UTF-8 编码"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMRequestException(f"{service} 响应不是合法 JSON") from exc
    if not isinstance(data, dict):
        raise LLMRequestException(f"{service} 响应必须是 JSON 对象")
    return data


def _validate_required(
    service: str,
    config: dict,
    field_names: tuple[str, ...],
) -> None:
    missing = [name for name in field_names if not config.get(name)]
    if missing:
        env_names = ", ".join(
            f"{service}_{name.upper()}" for name in missing
        )
        raise ValueError(f"{service} 配置不完整，请设置: {env_names}")


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("向量包含非法值，无法归一化")
    return vectors / norms


class LocalEmbeddingClient:
    """使用本地 SentenceTransformer 生成向量。"""

    def __init__(
        self,
        model_path: str,
        *,
        dim: int,
        batch_size: int,
        normalize: bool,
        query_prefix: str,
        document_prefix: str,
    ):
        self._model_path = model_path
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def _get_model(self):
        model = _LOCAL_EMBEDDING_MODELS.get(self._model_path)
        if model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[embedder] loading {self._model_path} ...")
            model = SentenceTransformer(self._model_path)
            _LOCAL_EMBEDDING_MODELS[self._model_path] = model
        return model

    def embed_texts(
        self,
        texts: list[str],
        *,
        kind: str = "document",
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype="float32")
        if kind not in {"document", "query"}:
            raise ValueError("kind 只支持 document 或 query")

        prefix = (
            self._query_prefix if kind == "query"
            else self._document_prefix
        )
        model = self._get_model()
        vectors = model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=self._normalize,
            show_progress_bar=kind == "document",
            batch_size=self._batch_size,
        )
        result = np.asarray(vectors, dtype="float32")
        if result.ndim != 2:
            result = np.atleast_2d(result)
        if self._dim and result.shape[1] != self._dim:
            raise ValueError(
                f"本地向量维度为 {result.shape[1]}，配置维度为 {self._dim}"
            )
        return result


class OpenAICompatibleEmbeddingClient:
    """调用 OpenAI 风格的 Embeddings HTTP 接口。"""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        channel: str,
        dim: int,
        batch_size: int,
        normalize: bool,
        query_prefix: str,
        document_prefix: str,
        timeout: float,
    ):
        if not channel:
            raise ValueError("向量化模型通道名不能为空")
        self._url = url
        self._api_key = api_key
        self._channel = channel
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._timeout = timeout

    def _parse_vectors(
        self,
        data: dict,
        *,
        expected_count: int,
    ) -> np.ndarray:
        items = data.get("data")
        if not isinstance(items, list):
            raise LLMRequestException("向量化响应缺少 data 数组")
        if len(items) != expected_count:
            raise LLMRequestException(
                "向量化响应数量与输入数量不一致: "
                f"{len(items)} != {expected_count}"
            )

        indexed_items = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise LLMRequestException("向量化 data 项必须是对象")
            index = item.get("index", position)
            try:
                index = int(index)
            except (TypeError, ValueError) as exc:
                raise LLMRequestException(
                    "向量化响应 index 必须是整数"
                ) from exc
            indexed_items.append((index, item))
        indexed_items.sort(key=lambda pair: pair[0])

        vectors = []
        for _, item in indexed_items:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise LLMRequestException(
                    "向量化响应缺少 embedding 数组"
                )
            vectors.append(vector)

        result = np.asarray(vectors, dtype="float32")
        if result.ndim != 2:
            raise LLMRequestException("向量化响应不是二维向量数组")
        if not np.all(np.isfinite(result)):
            raise LLMRequestException("向量化响应包含 NaN 或 Inf")
        if self._dim and result.shape[1] != self._dim:
            raise LLMRequestException(
                f"向量维度为 {result.shape[1]}，配置维度为 {self._dim}"
            )
        if not self._dim:
            self._dim = result.shape[1]
        return result

    def embed_texts(
        self,
        texts: list[str],
        *,
        kind: str = "document",
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype="float32")
        if kind not in {"document", "query"}:
            raise ValueError("kind 只支持 document 或 query")

        prefix = (
            self._query_prefix if kind == "query"
            else self._document_prefix
        )
        prepared = [prefix + text for text in texts]
        batches = []
        for start in range(0, len(prepared), self._batch_size):
            batch = prepared[start:start + self._batch_size]
            data = _request_json(
                service="向量化模型",
                url=self._url,
                api_key=self._api_key,
                payload={
                    "model": self._channel,
                    "input": batch,
                },
                timeout=self._timeout,
            )
            batches.append(
                self._parse_vectors(data, expected_count=len(batch))
            )

        result = np.concatenate(batches, axis=0)
        if self._normalize:
            result = _normalize_rows(result)
        return result.astype("float32", copy=False)


class LocalRerankerClient:
    """使用本地 CrossEncoder 对候选文本精排。"""

    def __init__(self, model_path: str, *, max_length: int):
        self._model_path = model_path
        self._max_length = max_length

    def _get_model(self):
        model = _LOCAL_RERANK_MODELS.get(self._model_path)
        if model is None:
            from sentence_transformers import CrossEncoder

            print(f"[reranker] loading {self._model_path} ...")
            model = CrossEncoder(self._model_path, max_length=self._max_length)
            _LOCAL_RERANK_MODELS[self._model_path] = model
        return model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        if not candidates:
            return []
        model = self._get_model()
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: -float(pair[1]),
        )
        return [
            {"rerank_score": float(score), **candidate}
            for candidate, score in ranked[:top_k]
        ]


class OpenAICompatibleRerankerClient:
    """调用独立 Rerank HTTP 接口并映射回原始候选。"""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        channel: str,
        timeout: float,
    ):
        if not channel:
            raise ValueError("精排模型通道名不能为空")
        self._url = url
        self._api_key = api_key
        self._channel = channel
        self._timeout = timeout

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        if not candidates or top_k <= 0:
            return []

        requested_top_n = min(top_k, len(candidates))
        data = _request_json(
            service="精排模型",
            url=self._url,
            api_key=self._api_key,
            payload={
                "model": self._channel,
                "query": query,
                "documents": [
                    candidate["text"] for candidate in candidates
                ],
                "top_n": requested_top_n,
            },
            timeout=self._timeout,
        )

        items = data.get("results")
        if items is None:
            items = data.get("data")
        if not isinstance(items, list):
            raise LLMRequestException(
                "精排响应缺少 results 或 data 数组"
            )

        ranked = []
        seen_indexes = set()
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise LLMRequestException("精排结果项必须是对象")
            raw_index = item.get("index", position)
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise LLMRequestException(
                    "精排结果 index 必须是整数"
                ) from exc
            if index < 0 or index >= len(candidates):
                raise LLMRequestException(
                    f"精排结果 index 超出候选范围: {index}"
                )
            if index in seen_indexes:
                continue

            raw_score = item.get(
                "relevance_score",
                item.get("rerank_score", item.get("score")),
            )
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise LLMRequestException(
                    "精排结果缺少可用分数"
                ) from exc

            ranked.append({
                "rerank_score": score,
                **candidates[index],
            })
            seen_indexes.add(index)

        ranked.sort(key=lambda item: -item["rerank_score"])
        return ranked[:top_k]


def get_chat_client() -> LLMClient:
    config = get_chat_config()
    _validate_required(
        "CHAT",
        config,
        ("url", "api_key", "channel"),
    )
    return OpenAICompatibleChatClient(
        url=config["url"],
        api_key=config["api_key"],
        channel=config["channel"],
        timeout=config["timeout"],
    )


def get_embedding_client() -> EmbeddingClient:
    config = get_embedding_config()
    if config["provider"] == "local":
        return LocalEmbeddingClient(
            config["model"],
            dim=config["dim"],
            batch_size=config["batch_size"],
            normalize=config["normalize"],
            query_prefix=config["query_prefix"],
            document_prefix=config["document_prefix"],
        )

    _validate_required(
        "EMBEDDING",
        config,
        ("url", "api_key", "channel"),
    )
    return OpenAICompatibleEmbeddingClient(
        url=config["url"],
        api_key=config["api_key"],
        channel=config["channel"],
        dim=config["dim"],
        batch_size=config["batch_size"],
        normalize=config["normalize"],
        query_prefix=config["query_prefix"],
        document_prefix=config["document_prefix"],
        timeout=config["timeout"],
    )


def get_reranker_client() -> RerankerClient:
    config = get_rerank_config()
    if config["provider"] == "local":
        return LocalRerankerClient(
            config["model"],
            max_length=config["max_length"],
        )

    _validate_required(
        "RERANK",
        config,
        ("url", "api_key", "channel"),
    )
    return OpenAICompatibleRerankerClient(
        url=config["url"],
        api_key=config["api_key"],
        channel=config["channel"],
        timeout=config["timeout"],
    )
