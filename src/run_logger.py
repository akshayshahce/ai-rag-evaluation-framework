from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import AppConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": str(source.get("file", "unknown")),
        "page": str(source.get("page", "?")),
        "score": _to_float(source.get("score")),
        "text": str(source.get("text", "")),
    }


def _get_app_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = out.stdout.strip()
        return value or None
    except Exception:
        return None


def _resolve_model(cfg: AppConfig) -> str:
    if cfg.llm_provider == "openai":
        return cfg.openai_model
    return cfg.ollama_model


def start_run(question: str, cfg: AppConfig) -> dict[str, Any]:
    provider_details: dict[str, Any]
    if cfg.llm_provider == "openai":
        provider_details = {"model": cfg.openai_model}
    else:
        provider_details = {"model": cfg.ollama_model, "base_url": cfg.ollama_base_url}

    return {
        "run_id": str(uuid4()),
        "timestamp_start": _utc_now_iso(),
        "provider": cfg.llm_provider,
        "model": _resolve_model(cfg),
        "provider_details": provider_details,
        "question": question,
        "top_k": cfg.top_k,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "max_context_chars": cfg.max_context_chars,
        "data_dir": cfg.data_dir,
        "chroma_persist_dir": cfg.chroma_persist_dir,
        "chroma_collection": cfg.chroma_collection,
        "app_version": _get_app_version(),
    }


def finish_run(
    run: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
    timings: dict[str, Any],
    error: str | None = None,
    context_used: str | None = None,
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(run)
    record["timestamp_end"] = _utc_now_iso()
    record["answer"] = answer
    record["context_used"] = context_used or ""
    record["retrieval"] = retrieval or {}
    sanitized_sources = [_sanitize_source(s) for s in sources]
    record["sources"] = sanitized_sources
    record["num_sources"] = len(sanitized_sources)
    record["timings_ms"] = {
        "retrieval_ms": _to_float(timings.get("retrieval_ms")),
        "generation_ms": _to_float(timings.get("generation_ms")),
        "total_ms": _to_float(timings.get("total_ms")),
        "index_build_ms": _to_float(timings.get("index_build_ms")),
    }
    record["error"] = error

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    day = record["timestamp_start"][:10]
    outfile = RUNS_DIR / f"{day}.jsonl"
    with outfile.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
