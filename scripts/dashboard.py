from __future__ import annotations

import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure `src` imports work when Streamlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.rag_pipeline import LocalRAG

RUNS_DIR = REPO_ROOT / "runs"
EVAL_CANDIDATE_PATHS = [REPO_ROOT / "eval" / "results.jsonl", REPO_ROOT / "outputs" / "results.jsonl"]


def safe_iso_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def estimate_tokens(record: dict[str, Any]) -> int:
    text = f"{record.get('question', '')}\n{record.get('answer', '')}"
    return max(1, len(text) // 4) if text.strip() else 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stringify_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda v: "" if v is None else str(v))
    return out


def list_run_files() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("*.jsonl"))


def load_run_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_file in list_run_files():
        with run_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    continue

    records.sort(key=lambda r: r.get("timestamp_start", ""), reverse=True)
    return records


def build_runs_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        timings = r.get("timings_ms", {}) or {}
        rows.append(
            {
                "run_id": r.get("run_id", ""),
                "timestamp": r.get("timestamp_start", ""),
                "question": r.get("question", ""),
                "provider": r.get("provider", ""),
                "model": r.get("model", ""),
                "top_k": r.get("top_k", None),
                "chunk_size": r.get("chunk_size", None),
                "chunk_overlap": r.get("chunk_overlap", None),
                "latency_ms": timings.get("total_ms", None),
                "num_sources": r.get("num_sources", 0),
                "token_estimate": estimate_tokens(r),
                "error": r.get("error", ""),
                "answer": r.get("answer", ""),
            }
        )
    return pd.DataFrame(rows)


def run_mode_label(cfg: Any) -> str:
    return "OpenAI enabled" if cfg.llm_provider == "openai" else "Local-only"


def render_copy_button(text: str) -> None:
    safe_text = json.dumps(text)
    components.html(
        f"""
        <button style=\"padding:6px 12px;border-radius:6px;border:1px solid #bbb;background:#f6f6f6;cursor:pointer;\"
                onclick='navigator.clipboard.writeText({safe_text})'>Copy JSON</button>
        """,
        height=42,
    )


def render_overview_page(cfg: Any, records: list[dict[str, Any]]) -> None:
    st.subheader("Home / Overview")

    total_runs = len(records)
    today_str = now_utc().date().isoformat()
    runs_today = sum(1 for r in records if str(r.get("timestamp_start", "")).startswith(today_str))

    latencies = [
        float((r.get("timings_ms", {}) or {}).get("total_ms"))
        for r in records
        if (r.get("timings_ms", {}) or {}).get("total_ms") is not None
    ]
    answers = [len(str(r.get("answer", ""))) for r in records if str(r.get("answer", "")).strip()]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_answer_len = sum(answers) / len(answers) if answers else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", total_runs)
    c2.metric("Runs today", runs_today)
    c3.metric("Avg latency (ms)", f"{avg_latency:.1f}")
    c4.metric("Avg answer length", f"{avg_answer_len:.1f} chars")

    st.markdown("### Environment")
    st.write(f"Mode: **{run_mode_label(cfg)}**")

    model_value = cfg.openai_model if cfg.llm_provider == "openai" else cfg.ollama_model
    config_rows = [
        ("LLM_PROVIDER", cfg.llm_provider),
        ("EMBED_PROVIDER", cfg.embed_provider),
        ("model", model_value),
        ("chunk_size", cfg.chunk_size),
        ("chunk_overlap", cfg.chunk_overlap),
        ("top_k", cfg.top_k),
        ("max_context_chars", cfg.max_context_chars),
        ("data_dir", cfg.data_dir),
        ("chroma_persist_dir", cfg.chroma_persist_dir),
        ("chroma_collection", cfg.chroma_collection),
    ]
    cfg_df = pd.DataFrame(config_rows, columns=["config", "value"])
    st.table(stringify_df(cfg_df, ["config", "value"]))


def render_runs_page(records: list[dict[str, Any]]) -> None:
    st.subheader("Runs Table")

    if not records:
        st.info("No run logs found under runs/*.jsonl yet.")
        return

    df = build_runs_dataframe(records)
    if df.empty:
        st.info("No valid run records found.")
        return

    min_dt = safe_iso_to_dt(df["timestamp"].min())
    max_dt = safe_iso_to_dt(df["timestamp"].max())
    if min_dt is None or max_dt is None:
        min_date = date.today()
        max_date = date.today()
    else:
        min_date = min_dt.date()
        max_date = max_dt.date()

    c1, c2, c3, c4 = st.columns([2, 2, 2, 4])
    date_range = c1.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    provider_filter = c2.selectbox("Provider", options=["all", "ollama", "openai"], index=0)
    min_sources = c3.number_input("Min sources", min_value=0, value=0, step=1)
    keyword = c4.text_input("Keyword search (question/answer)", value="").strip().lower()

    filtered = df.copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        ts = pd.to_datetime(filtered["timestamp"], errors="coerce", utc=True)
        mask = (ts.dt.date >= start_date) & (ts.dt.date <= end_date)
        filtered = filtered[mask.fillna(False)]

    if provider_filter != "all":
        filtered = filtered[filtered["provider"] == provider_filter]

    filtered = filtered[filtered["num_sources"] >= int(min_sources)]

    if keyword:
        qmask = filtered["question"].astype(str).str.lower().str.contains(keyword, na=False)
        amask = filtered["answer"].astype(str).str.lower().str.contains(keyword, na=False)
        filtered = filtered[qmask | amask]

    cols = [
        "run_id",
        "timestamp",
        "question",
        "provider",
        "model",
        "top_k",
        "chunk_size",
        "chunk_overlap",
        "latency_ms",
        "num_sources",
        "token_estimate",
        "error",
    ]
    shown = filtered[cols].copy()
    for col in ["run_id", "timestamp", "question", "provider", "model", "error"]:
        shown[col] = shown[col].map(lambda v: "" if v is None else str(v))
    for col in ["top_k", "chunk_size", "chunk_overlap", "latency_ms", "num_sources", "token_estimate"]:
        shown[col] = pd.to_numeric(shown[col], errors="coerce")
    shown["latency_ms"] = pd.to_numeric(shown["latency_ms"], errors="coerce").round(1)

    st.caption(f"Showing {len(shown)} run(s)")
    try:
        event = st.dataframe(
            shown,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
    except TypeError:
        event = None
        st.dataframe(shown, width="stretch", hide_index=True)

    selected_run_id = None
    selection = getattr(event, "selection", None)
    if isinstance(selection, dict):
        rows = selection.get("rows", [])
        if rows:
            row_idx = rows[0]
            if 0 <= row_idx < len(shown):
                selected_run_id = str(shown.iloc[row_idx]["run_id"])

    if not selected_run_id:
        selected_run_id = st.selectbox(
            "Run Detail",
            options=[""] + shown["run_id"].tolist(),
            format_func=lambda x: "Select a run" if x == "" else x,
        )

    if selected_run_id:
        st.session_state["selected_run_id"] = selected_run_id
        st.info("Run selected. Open the 'Run Detail' page from the sidebar for deep debug view.")


def render_run_detail_page(records: list[dict[str, Any]], cfg: Any) -> None:
    st.subheader("Run Detail")

    if not records:
        st.info("No run logs found under runs/*.jsonl yet.")
        return

    by_id = {str(r.get("run_id", "")): r for r in records}
    run_ids = [rid for rid in by_id.keys() if rid]
    if not run_ids:
        st.info("No run_id values found in logs.")
        return
    default_run = st.session_state.get("selected_run_id")

    chosen_id = st.selectbox(
        "Select run_id",
        options=run_ids,
        index=run_ids.index(default_run) if default_run in run_ids else 0,
    )
    st.session_state["selected_run_id"] = chosen_id

    run = by_id[chosen_id]
    timings = run.get("timings_ms", {}) or {}

    st.markdown("### Question")
    st.write(run.get("question", ""))

    st.markdown("### Final answer")
    st.write(run.get("answer", ""))

    st.markdown("### Latency breakdown")
    latency_rows = [
        ("index_build_ms", timings.get("index_build_ms")),
        ("retrieval_ms", timings.get("retrieval_ms")),
        ("generation_ms", timings.get("generation_ms")),
        ("total_ms", timings.get("total_ms")),
    ]
    latency_df = pd.DataFrame(latency_rows, columns=["metric", "ms"])
    st.table(stringify_df(latency_df, ["metric", "ms"]))

    st.markdown("### Retrieval parameters")
    retrieval = run.get("retrieval", {}) or {}
    retrieval_df = pd.DataFrame(
        [
            ("top_k", run.get("top_k")),
            ("similarity_metric", retrieval.get("similarity_metric") or "n/a"),
        ],
        columns=["param", "value"],
    )
    st.table(stringify_df(retrieval_df, ["param", "value"]))

    st.markdown("### Retrieved contexts")
    sources = run.get("sources", []) or []
    if not sources:
        st.info("No sources found for this run.")
    for i, src in enumerate(sources, start=1):
        label = (
            f"#{i} file={src.get('file', 'unknown')} "
            f"page={src.get('page', '?')} score={safe_float(src.get('score')):.4f}"
        )
        with st.expander(label):
            st.write(src.get("text", ""))

    st.markdown("### Context sent to LLM")
    context_used = run.get("context_used") or ""
    with st.expander("Show context"):
        st.text(context_used)

    st.markdown("### Provider details")
    provider_details = run.get("provider_details", {}) or {}
    if run.get("provider") == "openai":
        st.write({"provider": "openai", "model": provider_details.get("model") or run.get("model")})
    else:
        st.write(
            {
                "provider": "ollama",
                "base_url": provider_details.get("base_url", cfg.ollama_base_url),
                "model": provider_details.get("model") or run.get("model"),
            }
        )

    st.markdown("### Raw run JSON")
    raw = json.dumps(run, indent=2, ensure_ascii=False)
    render_copy_button(raw)
    st.code(raw, language="json")


def list_documents(data_dir: Path) -> pd.DataFrame:
    files = [p for p in data_dir.rglob("*") if p.is_file()]
    rows = []
    for p in sorted(files):
        rows.append(
            {
                "file": str(p),
                "last_modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                "size_bytes": p.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def load_chroma_stats(cfg: Any) -> dict[str, Any]:
    try:
        client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        collection = client.get_or_create_collection(cfg.chroma_collection)
        count = collection.count()
        return {
            "count": count,
            "persist_dir": cfg.chroma_persist_dir,
            "collection": cfg.chroma_collection,
        }
    except Exception as exc:
        return {
            "count": "n/a",
            "persist_dir": cfg.chroma_persist_dir,
            "collection": cfg.chroma_collection,
            "error": str(exc),
        }


def render_documents_page(cfg: Any) -> None:
    st.subheader("Documents / Index")

    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        st.warning(f"Data directory does not exist: {data_dir}")
    else:
        docs_df = list_documents(data_dir)
        st.markdown("### Files in data/")
        if docs_df.empty:
            st.info("No files found.")
        else:
            st.dataframe(docs_df, width="stretch", hide_index=True)

    st.markdown("### Chroma collection stats")
    stats = load_chroma_stats(cfg)
    st.json(stats)

    st.markdown("### Index actions")
    if st.button("Rebuild index", type="primary"):
        progress = st.progress(0)
        with st.spinner("Building index..."):
            try:
                progress.progress(15)
                rag = LocalRAG(cfg)
                progress.progress(45)
                rag.build_index()
                progress.progress(100)
                st.success("Index rebuild complete.")
            except Exception as exc:
                st.error(f"Index rebuild failed: {exc}")

    confirm_clear = st.checkbox("Confirm clear Chroma persist directory")
    if st.button("Clear Chroma"):
        if not confirm_clear:
            st.warning("Enable confirmation checkbox first.")
        else:
            try:
                persist = Path(cfg.chroma_persist_dir)
                if persist.exists():
                    shutil.rmtree(persist)
                st.success(f"Deleted: {persist}")
            except Exception as exc:
                st.error(f"Failed to clear Chroma persist dir: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                val = json.loads(line)
                if isinstance(val, dict):
                    rows.append(val)
            except json.JSONDecodeError:
                continue
    return rows


def normalize_eval_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized = []
    for r in rows:
        normalized.append(
            {
                "chunk_size": r.get("chunk_size") or (r.get("config", {}) or {}).get("chunk_size"),
                "answer_relevancy": r.get("answer_relevancy") or (r.get("metrics", {}) or {}).get("answer_relevancy"),
                "faithfulness": r.get("faithfulness") or (r.get("metrics", {}) or {}).get("faithfulness"),
            }
        )
    df = pd.DataFrame(normalized)
    df["chunk_size"] = pd.to_numeric(df["chunk_size"], errors="coerce")
    df["answer_relevancy"] = pd.to_numeric(df["answer_relevancy"], errors="coerce")
    df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce")
    return df.dropna(subset=["chunk_size"], how="any")


def render_evaluation_page() -> None:
    st.subheader("Evaluation")

    eval_path = next((p for p in EVAL_CANDIDATE_PATHS if p.exists()), None)
    if eval_path is None:
        st.info("No evaluation results found. Expected e.g. eval/results.jsonl or outputs/results.jsonl")
        return

    rows = read_jsonl(eval_path)
    if not rows:
        st.info(f"Evaluation file exists but has no valid rows: {eval_path}")
        return

    st.caption(f"Loaded: {eval_path}")
    df = normalize_eval_df(rows)
    if df.empty:
        st.info("Evaluation schema detected, but required metrics are missing.")
        return

    st.dataframe(df, width="stretch", hide_index=True)

    line_df = df.sort_values(by="chunk_size")
    st.markdown("#### Chunk Size vs Answer Relevancy")
    st.line_chart(line_df.set_index("chunk_size")["answer_relevancy"])

    st.markdown("#### Chunk Size vs Faithfulness")
    st.line_chart(line_df.set_index("chunk_size")["faithfulness"])


def main() -> None:
    st.set_page_config(page_title="RAG Dashboard", layout="wide")
    st.title("RAG Dashboard")

    auto_refresh = st.sidebar.toggle("Live refresh", value=True)
    refresh_seconds = st.sidebar.slider("Refresh every (sec)", min_value=2, max_value=30, value=5)
    if auto_refresh:
        st.sidebar.caption("Live updates enabled")

    page = st.sidebar.radio(
        "Navigation",
        ["Home / Overview", "Runs Table", "Run Detail", "Documents / Index", "Evaluation"],
    )

    run_every = f"{refresh_seconds}s" if auto_refresh else None

    @st.fragment(run_every=run_every)
    def render_content() -> None:
        cfg = load_config()
        records = load_run_records()

        if page == "Home / Overview":
            render_overview_page(cfg, records)
        elif page == "Runs Table":
            render_runs_page(records)
        elif page == "Run Detail":
            render_run_detail_page(records, cfg)
        elif page == "Documents / Index":
            render_documents_page(cfg)
        elif page == "Evaluation":
            render_evaluation_page()

    render_content()


if __name__ == "__main__":
    main()
