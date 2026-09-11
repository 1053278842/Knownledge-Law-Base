import json
import hashlib
import faiss
import numpy as np
from config import INDEX_DIR, EMBED_DIM

INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "meta.json"


def file_hash(path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class VectorStore:
    def __init__(self):
        self.index = None
        self.meta: dict[str, dict] = {}       # str(id) -> {text, source, chunk_id}
        self.file_hashes: dict[str, str] = {}  # source -> md5
        self._next_id = 0
        self._load()

    def _load(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            data = json.loads(META_FILE.read_text(encoding="utf-8"))
            self.meta = data["meta"]
            self.file_hashes = data["file_hashes"]
            self._next_id = data["next_id"]
            print(f"[store] loaded {self.index.ntotal} vectors, "
                  f"{len(self.file_hashes)} files")

    def save(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(
            json.dumps({
                "meta": self.meta,
                "file_hashes": self.file_hashes,
                "next_id": self._next_id,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- 增量更新核心 ----------
    def needs_update(self, source: str, path) -> bool:
        """文件内容变了 或 从未索引过 才需要更新"""
        h = file_hash(path)
        return self.file_hashes.get(source) != h

    def remove_file(self, source: str):
        """删除某个文件的所有向量"""
        if self.index is None:
            return
        ids = [int(i) for i, m in self.meta.items() if m["source"] == source]
        if not ids:
            return
        self.index.remove_ids(np.array(ids, dtype="int64"))
        for i in ids:
            del self.meta[str(i)]
        self.file_hashes.pop(source, None)
        print(f"[store] removed {len(ids)} chunks from {source}")

    def add(self, vectors: np.ndarray, docs: list[dict], source: str, path):
        """添加新向量（注意：调用前应先 remove_file）"""
        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBED_DIM))

        ids = np.arange(self._next_id, self._next_id + len(docs), dtype="int64")
        self.index.add_with_ids(vectors, ids)

        for i, d in zip(ids, docs):
            self.meta[str(i)] = d
        self._next_id += len(docs)
        self.file_hashes[source] = file_hash(path)

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(query_vec, top_k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1 or str(i) not in self.meta:
                continue
            results.append({"score": float(score), "id": int(i), **self.meta[str(i)]})
        return results

    def all_texts(self) -> tuple[list[str], list[int]]:
        """给 BM25 用：返回所有文本和对应 id"""
        texts, ids = [], []
        for i, m in self.meta.items():
            texts.append(m["text"])
            ids.append(int(i))
        return texts, ids