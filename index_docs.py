import sys
from pathlib import Path

from rag.splitter import split_law_text
from rag.embedder import embed_documents
from rag.store import VectorStore


def index_dir(dir_path: str, glob: str = "*.txt"):
    files = list(Path(dir_path).rglob(glob))
    if not files:
        print(f"未找到文档：{dir_path}/{glob}")
        return

    store = VectorStore()
    updated = 0
    skipped = 0

    for f in files:
        source = f.name

        if not store.needs_update(source, f):
            skipped += 1
            print(f"  [跳过] {source}（未修改）")
            continue

        store.remove_file(source)

        raw = f.read_text(encoding="utf-8")
        
        # 改后
        chunk_dicts = split_law_text(raw)

        texts = []
        docs = []
        for i, c in enumerate(chunk_dicts):
            texts.append(c["text"])
            docs.append({
                "text": c["text"],
                "source": source,
                "chunk_id": i,
                "article_no": c["article_no"],     # ← 新增：法条编号
                "chapter": c["chapter"],            # ← 新增：所属章
                "sub_id": c["sub_id"],
            })

        if len(texts) == 0:
            continue

        vectors = embed_documents(texts)
        store.add(vectors, docs, source, f)
        updated += 1
        print(f"  [更新] {source}: {len(chunk_dicts)} chunks")

    store.save()
    total = store.index.ntotal if store.index is not None else 0
    print()
    print(f"完成：{updated} 更新，{skipped} 未变，共 {total} 向量")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "docs"
    index_dir(arg)