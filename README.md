# ai-rag-evaluation-framework

A simple, production-style **local-first RAG** project:
- Ingest PDFs/text from `data/`
- Chunk + embed them
- Store vectors in **ChromaDB**
- Retrieve relevant context for each question
- Generate an answer using **Ollama (local)** or **OpenAI (hosted)** — controlled by `.env`

---

## What this repo does

When you ask a question, the system:
1. **Searches your documents** to find the most relevant text chunks.
2. **Injects those chunks** into the prompt as context.
3. **Asks the LLM** to answer using ONLY that context.
4. Returns the answer + **sources** (file + page) so responses are explainable.

## What This Repo Covers

* Local document ingestion (PDF / text)
* Chunking strategies for context construction
* Embedding generation (HuggingFace or OpenAI)
* Vector storage using ChromaDB
* Retrieval-based grounding for LLM responses
* Source attribution with page-level provenance
* Provider switching via environment configuration
* Interactive chat interface for RAG pipelines

### Key Takeaway

**Retrieval quality matters more than model size.**

This repository allows experimentation with context engineering decisions and observing their impact on output quality when running fully locally or with hosted APIs.

---

## Architecture Overview

```
                ┌──────────────────────┐
                │ Local Documents      │
                │ (PDF / Text)         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Chunking / Parsing   │
                │ SentenceSplitter     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Embedding Model      │
                │ HF / OpenAI          │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Chroma Vector DB     │
                └──────────┬───────────┘
                           │
                Retrieval   ▼
                ┌──────────────────────┐
User Query ───► │ Context Injection    │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ LLM Response         │
                │ Ollama / OpenAI      │
                └──────────────────────┘
```

---

## Prerequisites

Ensure the following are installed:

* Python 3.10+
* pip
* Git
* (Recommended) virtualenv
* Ollama (for local LLM mode)

Download Ollama:

https://ollama.com/download

---

## Setup Instructions

### Pull Local Model (Ollama mode)

```
ollama serve
ollama pull llama3.1:8b
```

### Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Configure Environment

```
cp .env.example .env
```

Add documents into:

```
data/
```

---

## Running the Project

### Interactive RAG Chat

```
python -m scripts.chat
```

This will:

1. Index local documents
2. Store embeddings in Chroma
3. Retrieve relevant context
4. Generate grounded responses
5. Display source attribution

---

## Provider Switching

All runtime behavior is controlled via `.env`.

### Local Mode (default)

```
LLM_PROVIDER=ollama
```

Uses:

* Ollama LLM
* HuggingFace embeddings
* Fully offline execution

### OpenAI Mode

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

Uses:

* OpenAI LLM
* OpenAI embeddings

No code changes required — restart the chat script.

---

## Repository Structure

```
src/
 ├── config.py
 ├── providers.py
 ├── rag_pipeline.py

scripts/
 ├── chat.py
 ├── dashboard.py

 data/
 runs/
```

## Dashboard

Run the local dashboard:

```bash
streamlit run scripts/dashboard.py
```

The dashboard includes:

1. Home / Overview: totals, avg latency, avg answer length, config snapshot, mode (local-only vs OpenAI enabled)
2. Runs Table: search/filter over `runs/*.jsonl`, with provider/date/source/keyword filters
3. Run Detail: full retrieval + generation debug view per run
4. Documents / Index: data files, Chroma stats, rebuild and clear actions
5. Evaluation: optional visualization when eval results are present


## Run Logging

Every question asked in `python -m scripts.chat` is logged to append-only JSONL files:

- directory: `runs/`
- file pattern: `runs/YYYY-MM-DD.jsonl`
- one JSON object per run (with `run_id`)

## Data Format (JSONL Run Schema)

Each run record includes:

- `run_id`
- `timestamp_start`, `timestamp_end`
- `provider`, `model`
- `provider_details` (for ollama: `base_url` + model; for openai: model only)
- `question`, `answer`
- `top_k`, `chunk_size`, `chunk_overlap`, `max_context_chars`
- `data_dir`, `chroma_persist_dir`, `chroma_collection`
- `num_sources`, `sources` (`file`, `page`, `score`, `text`)
- `context_used` (actual context string sent to the model)
- `retrieval` (`top_k`, `similarity_metric` when available)
- `timings_ms` (`index_build_ms`, `retrieval_ms`, `generation_ms`, `total_ms`)
- `error` (nullable)
- `app_version` (git commit hash if available)

Secrets are not logged. `OPENAI_API_KEY` is never written to run logs.

---

## Future Enhancements

* Hybrid retrieval (BM25 + vectors)
* Re-ranking
* Guardrails / hallucination detection
* Dockerization
* Kubernetes deployment
* CI-based regression evaluation
* Metadata filtering
* Observability dashboards

---

## Author

Akshay Shah (Akshayshah.ce@gmail.com)
