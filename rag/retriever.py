from functools import lru_cache
from rag.embedder import embed_query
from rag.store import VectorStore
from rag.bm25 import BM25Index
from rag.reranker import rerank


def _result_key(result: dict):
    """生成候选的 source + id 去重键。

    :param result: 候选文档，source 可为空，id 必须存在
    :return: (source, id) 元组；候选无效时返回 None
    """
    if not isinstance(result, dict) or result.get("id") is None:
        return None
    return result.get("source", ""), result["id"]


# RRF 融合
def _rrf(vec_results, bm25_results, k=60, id_to_doc=None):
    """融合单问题的向量和 BM25 排名。

    :param vec_results: 向量召回文档列表
    :param bm25_results: BM25 召回的 (id, score) 列表
    :param k: RRF 平滑参数
    :param id_to_doc: BM25 id 到文档的映射
    :return: 按 RRF 分数降序排列的文档列表
    """
    score_map = {}
    info_map = {}
    vec_info = {}
    for rank, result in enumerate(vec_results):
        key = _result_key(result)
        if key is None:
            continue
        score_map[key] = score_map.get(key, 0) + 1 / (k + rank + 1)
        info_map.setdefault(key, result)
        vec_info[result["id"]] = result

    for rank, (i, _) in enumerate(bm25_results):
        result = id_to_doc.get(i) if id_to_doc is not None else vec_info.get(i)
        key = _result_key(result) if result is not None else None
        if key is None:
            continue
        score_map[key] = score_map.get(key, 0) + 1 / (k + rank + 1)
        info_map.setdefault(key, result)

    ranked = sorted(score_map.items(), key=lambda item: -item[1])
    return [info_map[key] for key, _ in ranked]


def _rrf_many(result_lists, k=60):
    """融合多个问题的排名列表。

    :param result_lists: 每个问题的已融合排名列表
    :param k: RRF 平滑参数
    :return: 按累计 RRF 分数降序排列的文档列表
    """
    score_map = {}
    info_map = {}
    for results in result_lists:
        for rank, result in enumerate(results):
            key = _result_key(result)
            if key is None:
                continue
            score_map[key] = score_map.get(key, 0) + 1 / (k + rank + 1)
            info_map.setdefault(key, result)

    ranked = sorted(score_map.items(), key=lambda item: -item[1])
    return [info_map[key] for key, _ in ranked]


def _dedup_by_source_id(results):
    """按 source + id 对候选去重。

    :param results: 待去重文档列表
    :return: 保留首次出现顺序的文档列表
    """
    deduped = []
    seen = set()
    for result in results:
        key = _result_key(result)
        if key is None or key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


class Retriever:
    def __init__(self):
        self.store = VectorStore()
        self.bm25 = BM25Index()
        # 启动时构建 BM25（几千个 chunk 秒级完成）
        texts, ids = self.store.all_texts()
        if texts:
            self.bm25.build(texts, ids)
            print(f"[retriever] BM25 index built on {len(texts)} chunks")
        self.id_to_doc = {
            int(i): {"id": int(i), **metadata}
            for i, metadata in self.store.meta.items()
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        recall_k: int = 20,      # 每路召回数量
        use_rerank: bool = True,
    ) -> list[dict]:
        queries = [query] if isinstance(query, str) else list(query)
        print(len(queries), "queries:", queries)
        # 每路 query 分别做 向量 + BM25 召回
        per_query_results = []
        for q in queries:
            # 向量召回
            qv = embed_query(q)
            vec = self.store.search(qv, top_k=recall_k)
            # BM25 召回
            bm = self.bm25.search(q, top_k=recall_k)
            # RRF 融合
            fused = _rrf(vec, bm, id_to_doc=self.id_to_doc)
            per_query_results.append(_dedup_by_source_id(fused))

        # 多查询之间再做一次 RRF（外层 RRF）
        merged = _rrf_many(per_query_results)
        merged = _dedup_by_source_id(merged)

        if use_rerank and merged:
            return rerank(queries[0], merged, top_k=top_k)
        return merged[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
