# src/providers.py
import os
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from .config import AppConfig


def configure_llama_index(cfg: AppConfig) -> None:
    # Always apply chunking rules
    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    provider = cfg.llm_provider.lower().strip()

    # --- LLM selection ---
    if provider == "ollama":
        # Safety: prevent accidental OpenAI calls from any dependency
        os.environ.pop("OPENAI_API_KEY", None)

        from llama_index.llms.ollama import Ollama
        Settings.llm = Ollama(model=cfg.ollama_model, base_url=cfg.ollama_base_url)

    elif provider == "openai":
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai.")

        os.environ["OPENAI_API_KEY"] = cfg.openai_api_key

        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(model=cfg.openai_model, api_key=cfg.openai_api_key)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {cfg.llm_provider}")

    # --- Embedding selection (derived from LLM_PROVIDER only) ---
    if provider == "ollama":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.hf_embed_model)

    elif provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(model=cfg.openai_embed_model, api_key=cfg.openai_api_key)