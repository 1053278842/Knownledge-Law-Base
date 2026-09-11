import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, QUERY_INSTRUCTION

_model = None

# 向量模型，用于将文档或查询转换为向量表示，便于后续检索
def _get_model() -> SentenceTransformer:
    """懒加载，全局只加载一次"""
    global _model
    if _model is None:
        print(f"[embedder] loading {EMBED_MODEL} ...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """文档批量向量化"""
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,   # 归一化 → 内积即余弦相似度
        show_progress_bar=True,
        batch_size=32,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """查询向量化（BGE 中文模型需要加 instruction 前缀）"""
    model = _get_model()
    vec = model.encode(
        [QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
    )
    return np.asarray(vec, dtype="float32")