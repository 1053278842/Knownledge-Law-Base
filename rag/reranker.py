from sentence_transformers import CrossEncoder
from config import RERANK_MODEL    

_model = None

# 精排模型，计算问题与候选文档的相关性分数，返回 top-k
def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"[reranker] loading {RERANK_MODEL} ...")
        _model = CrossEncoder(RERANK_MODEL, max_length=512)  
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [{"rerank_score": float(s), **c} for c, s in ranked[:top_k]]