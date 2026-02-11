# ai-rag-evaluation-framework

A production-oriented framework for building Retrieval-Augmented Generation (RAG) pipelines with switchable providers (local or hosted). The project focuses on clean configuration, reproducibility, and real-world architecture patterns for context-grounded LLM applications.

---

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

 data/
```

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
