import os
from pathlib import Path
from dotenv import load_dotenv

# Project root is the directory this file lives in — absolute, CWD-independent
_ROOT = Path(__file__).parent.resolve()

load_dotenv(_ROOT / ".env")

# LLM
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Embedding
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

def _data_path(env_key: str, relative_default: str) -> Path:
    """Return an absolute Path for a data directory.
    Relative values (from .env or defaults) are anchored to the project root,
    not the shell working directory, so the app works from any cwd.
    """
    raw = Path(os.getenv(env_key, relative_default))
    return raw if raw.is_absolute() else (_ROOT / raw).resolve()

# Storage
CHROMA_PERSIST_DIR: Path = _data_path("CHROMA_PERSIST_DIR", "data/chroma_db")
BM25_INDEX_DIR: Path = _data_path("BM25_INDEX_DIR", "data/bm25_indexes")
UPLOADS_DIR: Path = _data_path("UPLOADS_DIR", "data/uploads")

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# Ensure all data directories exist on every startup
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(_ROOT / "data" / "embed_cache").mkdir(parents=True, exist_ok=True)
