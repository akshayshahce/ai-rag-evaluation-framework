# src/config.py
import os
from pydantic import BaseModel
from dotenv import load_dotenv


class AppConfig(BaseModel):
    # Single source of truth:
    llm_provider: str = "ollama"  # ollama | openai

    # Optional override (if you really want later)
    embed_provider: str | None = None  # None = derive from llm_provider

    openai_api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    hf_embed_model: str = "intfloat/e5-small-v2"

    chroma_persist_dir: str = "chroma_db"
    chroma_collection: str = "kbase"  # MUST be >= 3 chars now (kb will fail)

    data_dir: str = "data"
    chunk_size: int = 512
    chunk_overlap: int = 80
    top_k: int = 4

    max_context_chars: int = 2000


def load_config() -> AppConfig:
    load_dotenv()
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    # If EMBED_PROVIDER not set, derive from llm_provider
    embed_provider = os.getenv("EMBED_PROVIDER", "").strip().lower() or None
    if embed_provider is None:
        embed_provider = "openai" if llm_provider == "openai" else "huggingface"

    return AppConfig(
        llm_provider=llm_provider,
        embed_provider=embed_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        hf_embed_model=os.getenv("HF_EMBED_MODEL", "intfloat/e5-small-v2"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "kbase"),  # >=3 chars
        data_dir=os.getenv("DATA_DIR", "data"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "4")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "2000")),
    )