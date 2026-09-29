from model_clients import get_reranker_client


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """使用当前配置的精排模型对候选文档排序。"""
    return get_reranker_client().rerank(
        query,
        candidates,
        top_k=top_k,
    )
