import json
import hashlib
import faiss
import numpy as np
from config import INDEX_DIR, get_embedding_profile, get_index_dir


def file_hash(path) -> str:
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


class VectorStore:
    def __init__(self):
        self.index = None
        self.meta: dict[str, dict] = {}       # str(id) -> {text, source, chunk_id}
        self.file_hashes: dict[str, str] = {}  # source -> md5
        self._next_id = 0
        self.profile = get_embedding_profile()
        self._index_dir = get_index_dir()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._index_dir / "faiss.index"
        self._meta_file = self._index_dir / "meta.json"
        self._profile_file = self._index_dir / "profile.json"
        self._load()

    def _validate_profile(self):
        if self._profile_file.exists():
            saved_profile = json.loads(
                self._profile_file.read_text(encoding="utf-8")
            )
            if saved_profile.get("profile_id") != self.profile["profile_id"]:
                raise ValueError(
                    "当前 embedding 配置与索引 profile 不一致。"
                    f" 索引目录: {self._index_dir}"
                    " 请使用相同配置重新执行 index_docs.py。"
                )
            saved_dim = int(saved_profile.get("dim") or 0)
            current_dim = int(self.profile.get("dim") or 0)
            if saved_dim and current_dim and saved_dim != current_dim:
                raise ValueError(
                    f"索引向量维度为 {saved_dim}，"
                    f"当前配置维度为 {current_dim}"
                )
            return

        if self._index_file.exists() and self._index_dir != INDEX_DIR:
            raise ValueError(
                "索引目录缺少 profile.json，无法确认向量模型是否匹配。"
            )

    def _load(self):
        if self._index_file.exists() and self._meta_file.exists():
            self._validate_profile()
            self.index = faiss.read_index(str(self._index_file))
            data = json.loads(
                self._meta_file.read_text(encoding="utf-8")
            )
            self.meta = data["meta"]
            self.file_hashes = data["file_hashes"]
            self._next_id = data["next_id"]
            configured_dim = self.profile.get("dim") or 0
            index_dim = getattr(self.index, "d", None)
            if configured_dim and index_dim and configured_dim != index_dim:
                raise ValueError(
                    f"索引向量维度为 {index_dim}，"
                    f"当前配置维度为 {configured_dim}"
                )
            print(f"[store] loaded {self.index.ntotal} vectors, "
                  f"{len(self.file_hashes)} files")

    def save(self):
        faiss.write_index(self.index, str(self._index_file))
        self._profile_file.write_text(
            json.dumps(self.profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._meta_file.write_text(
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
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维数组")
        vector_dim = vectors.shape[1]
        configured_dim = self.profile.get("dim") or 0
        if configured_dim and vector_dim != configured_dim:
            raise ValueError(
                f"待写入向量维度为 {vector_dim}，"
                f"当前配置维度为 {configured_dim}"
            )
        if not configured_dim:
            self.profile["dim"] = vector_dim

        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vector_dim))
        elif getattr(self.index, "d", vector_dim) != vector_dim:
            raise ValueError(
                f"待写入向量维度为 {vector_dim}，"
                f"索引维度为 {self.index.d}"
            )

        ids = np.arange(self._next_id, self._next_id + len(docs), dtype="int64")
        self.index.add_with_ids(vectors, ids)

        for i, d in zip(ids, docs):
            self.meta[str(i)] = d
        self._next_id += len(docs)
        self.file_hashes[source] = file_hash(path)

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = np.asarray(query_vec, dtype="float32")
        if query_vec.ndim != 2:
            raise ValueError("query_vec 必须是二维数组")
        if getattr(self.index, "d", query_vec.shape[1]) != query_vec.shape[1]:
            raise ValueError(
                f"查询向量维度为 {query_vec.shape[1]}，"
                f"索引维度为 {self.index.d}"
            )
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
