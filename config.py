import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

MODELS_DIR = BASE_DIR / "models"

# 本地模型路径，不再走 HuggingFace。
EMBED_MODEL = str(MODELS_DIR / "bge-small-zh-v1.5")
RERANK_MODEL = str(MODELS_DIR / "bge-reranker-base")

EMBED_DIM = 512
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _env_first(*names: str, default: str = "") -> str:
    """Return the first environment variable that is explicitly set."""
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是布尔值")


def _resolve_provider(
    capability: str,
    *,
    default: str,
    allowed: set[str],
) -> str:
    provider = os.getenv(f"{capability}_PROVIDER", default).strip().lower()
    if provider not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{capability}_PROVIDER 只支持: {choices}")
    return provider


def get_embedding_provider() -> str:
    return _resolve_provider(
        "EMBEDDING",
        default="local",
        allowed={"local", "bocai"},
    )


def get_rerank_provider() -> str:
    return _resolve_provider(
        "RERANK",
        default="local",
        allowed={"local", "bocai"},
    )


def get_embedding_config() -> dict:
    provider = get_embedding_provider()
    if provider == "local":
        return {
            "provider": "local",
            "model": _env_first(
                "EMBEDDING_LOCAL_MODEL",
                default=EMBED_MODEL,
            ),
            "dim": _int_env("EMBEDDING_DIM", EMBED_DIM),
            "batch_size": _int_env("EMBEDDING_BATCH_SIZE", 32),
            "normalize": _bool_env("EMBEDDING_NORMALIZE", True),
            "query_prefix": _env_first(
                "EMBEDDING_QUERY_PREFIX",
                default=QUERY_INSTRUCTION,
            ),
            "document_prefix": _env_first(
                "EMBEDDING_DOCUMENT_PREFIX",
                default="",
            ),
            "timeout": _float_env("EMBEDDING_TIMEOUT", 60.0),
        }

    return {
        "provider": "bocai",
        "url": os.getenv("EMBEDDING_URL", "").strip(),
        "api_key": os.getenv("EMBEDDING_API_KEY", "").strip(),
        "channel": os.getenv("EMBEDDING_CHANNEL", "").strip(),
        "dim": _int_env("EMBEDDING_DIM", 0),
        "batch_size": _int_env("EMBEDDING_BATCH_SIZE", 32),
        "normalize": _bool_env("EMBEDDING_NORMALIZE", True),
        "query_prefix": _env_first(
            "EMBEDDING_QUERY_PREFIX",
            default="",
        ),
        "document_prefix": _env_first(
            "EMBEDDING_DOCUMENT_PREFIX",
            default="",
        ),
        "timeout": _float_env("EMBEDDING_TIMEOUT", 60.0),
    }


def get_rerank_config() -> dict:
    provider = get_rerank_provider()
    if provider == "local":
        return {
            "provider": "local",
            "model": _env_first(
                "RERANK_LOCAL_MODEL",
                default=RERANK_MODEL,
            ),
            "max_length": _int_env("RERANK_MAX_LENGTH", 512),
            "timeout": _float_env("RERANK_TIMEOUT", 60.0),
        }

    return {
        "provider": "bocai",
        "url": os.getenv("RERANK_URL", "").strip(),
        "api_key": os.getenv("RERANK_API_KEY", "").strip(),
        "channel": os.getenv("RERANK_CHANNEL", "").strip(),
        "timeout": _float_env("RERANK_TIMEOUT", 60.0),
    }


def get_chat_config() -> dict:
    return {
        "url": os.getenv("CHAT_URL", "").strip(),
        "api_key": os.getenv("CHAT_API_KEY", "").strip(),
        "channel": os.getenv("CHAT_CHANNEL", "").strip(),
        "timeout": _float_env("CHAT_TIMEOUT", 60.0),
    }


def get_embedding_profile() -> dict:
    """Build a stable identity for the configured embedding space."""
    embedding = get_embedding_config()
    profile_data = {
        "provider": embedding["provider"],
        "model": embedding.get("model") or embedding.get("channel", ""),
        "dim": embedding["dim"],
        "normalize": embedding["normalize"],
        "query_prefix": embedding["query_prefix"],
        "document_prefix": embedding["document_prefix"],
    }
    raw_profile = "|".join(
        f"{key}={profile_data[key]}"
        for key in (
            "provider",
            "model",
            "normalize",
            "query_prefix",
            "document_prefix",
        )
    )
    profile_data["profile_id"] = hashlib.sha1(
        raw_profile.encode("utf-8")
    ).hexdigest()[:12]
    return profile_data


def get_index_dir() -> Path:
    """Return the index directory for the current embedding profile."""
    profile = get_embedding_profile()
    default_embedding_model = str(Path(EMBED_MODEL).resolve())
    current_embedding_model = str(Path(profile["model"]).resolve())
    is_default_local = (
        profile["provider"] == "local"
        and current_embedding_model == default_embedding_model
        and profile["dim"] == EMBED_DIM
        and profile["normalize"] is True
        and profile["query_prefix"] == QUERY_INSTRUCTION
        and profile["document_prefix"] == ""
    )
    if is_default_local:
        return INDEX_DIR
    return INDEX_DIR / f"{profile['provider']}-{profile['profile_id']}"


# ---------- 关键：禁用 HuggingFace 网络访问 ----------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
