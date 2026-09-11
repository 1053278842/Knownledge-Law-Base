from pathlib import Path

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

MODELS_DIR = BASE_DIR / "models"

# 本地路径，不再走 HuggingFace
EMBED_MODEL = str(MODELS_DIR / "bge-small-zh-v1.5")
RERANK_MODEL = str(MODELS_DIR / "bge-reranker-base")

EMBED_DIM = 512
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# ---------- 关键：禁用 HuggingFace 网络访问 ----------
import os
os.environ["HF_HUB_OFFLINE"] = "1"           # 完全离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # 屏蔽软链接警告
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"