import numpy as np

from model_clients import get_embedding_client


def embed_documents(texts: list[str]) -> np.ndarray:
    """使用当前配置的向量模型批量处理文档。"""
    return get_embedding_client().embed_texts(texts, kind="document")


def embed_query(query: str) -> np.ndarray:
    """使用当前配置的向量模型处理查询。"""
    return get_embedding_client().embed_texts([query], kind="query")
