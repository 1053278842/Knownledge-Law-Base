from pathlib import Path

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

# 中文向量模型（512维，轻量）
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512

# 切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# BGE 中文模型的查询指令（官方推荐，提升检索效果）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："