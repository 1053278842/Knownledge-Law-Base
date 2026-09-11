import jieba
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.ids: list[int] = []
        self.bm25 = None

    def build(self, texts: list[str], ids: list[int]):
        """中文用 jieba 分词后再喂给 BM25"""
        self.ids = ids
        self.corpus_tokens = [list(jieba.cut(t)) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: -x[1])[:top_k]
        return ranked