from functools import lru_cache
from rag.embedder import embed_query
from rag.store import VectorStore
from rag.bm25 import BM25Index
from rag.reranker import rerank

# RRF 融合
def _rrf(vec_results, bm25_results, k=60):
    score_map = {}
    info_map = {}
    for rank, r in enumerate(vec_results):
        i = r["id"]
        score_map[i] = score_map.get(i, 0) + 1 / (k + rank + 1)
        info_map[i] = r
    for rank, (i, _) in enumerate(bm25_results):
        score_map[i] = score_map.get(i, 0) + 1 / (k + rank + 1)
    return [info_map[i] for i, _ in sorted(score_map.items(), key=lambda x: -x[1])
            if i in info_map]


class Retriever:
    def __init__(self):
        self.store = VectorStore()
        self.bm25 = BM25Index()
        # 启动时构建 BM25（几千个 chunk 秒级完成）
        texts, ids = self.store.all_texts()
        if texts:
            self.bm25.build(texts, ids)
            print(f"[retriever] BM25 index built on {len(texts)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        recall_k: int = 20,      # 每路召回数量
        use_rerank: bool = True,
    ) -> list[dict]:
        # 1) 向量召回
        qv = embed_query(query)
        vec_results = self.store.search(qv, top_k=recall_k)

        # 2) BM25 召回
        bm25_results = self.bm25.search(query, top_k=recall_k)

        # 3) RRF 融合
        merged = _rrf(vec_results, bm25_results)[:recall_k]

        # 4) 精排
        if use_rerank and merged:
            return rerank(query, merged, top_k=top_k)

        return merged[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()