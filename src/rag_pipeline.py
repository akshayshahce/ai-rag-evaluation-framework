# src/rag_pipeline.py
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import AppConfig
from .providers import configure_llama_index


@dataclass
class Source:
    file: str
    page: str
    score: float
    text: str


class LocalRAG:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        configure_llama_index(cfg)

        # Chroma collection name validation got stricter; keep >= 3 chars
        if len(cfg.chroma_collection) < 3:
            raise ValueError("CHROMA_COLLECTION must be at least 3 characters (example: kbase).")

        self._client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(cfg.chroma_collection)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)

        self._index: VectorStoreIndex | None = None

    def build_index(self) -> None:
        from llama_index.readers.file import PDFReader

        reader = SimpleDirectoryReader(
            self.cfg.data_dir,
            file_extractor={".pdf": PDFReader()},
        )
        docs = reader.load_data()

        storage = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_documents(docs, storage_context=storage)

    def _extract_sources(self, source_nodes: List[Any]) -> List[Source]:
        sources: List[Source] = []
        for sn in source_nodes or []:
            meta = sn.node.metadata or {}
            txt = (sn.node.get_text() or "")[: self.cfg.max_context_chars]

            page = (
                meta.get("page_label")
                or meta.get("page_number")
                or meta.get("page")
                or meta.get("page_index")
            )
            if isinstance(page, int):
                if "page_index" in meta and "page_label" not in meta and "page_number" not in meta and "page" not in meta:
                    page = str(page + 1)
                else:
                    page = str(page)
            elif page is None:
                page = "?"
            else:
                page = str(page)

            sources.append(
                Source(
                    file=meta.get("file_name") or meta.get("filename") or "unknown",
                    page=page,
                    score=float(getattr(sn, "score", 0.0) or 0.0),
                    text=txt,
                )
            )
        return sources

    def _build_context_used(self, sources: List[Source]) -> str:
        chunks: List[str] = []
        total_chars = 0
        for s in sources:
            header = f"[file={s.file} page={s.page} score={s.score:.4f}]"
            body = s.text or ""
            piece = f"{header}\n{body}\n"
            if total_chars + len(piece) > self.cfg.max_context_chars:
                remaining = self.cfg.max_context_chars - total_chars
                if remaining > 0:
                    chunks.append(piece[:remaining])
                break
            chunks.append(piece)
            total_chars += len(piece)
        return "\n".join(chunks).strip()

    def query(self, question: str) -> Dict[str, Any]:
        query_start = perf_counter()
        index_build_ms: float | None = None
        if self._index is None:
            build_start = perf_counter()
            self.build_index()
            index_build_ms = (perf_counter() - build_start) * 1000.0

        retrieval_start = perf_counter()
        source_nodes: List[Any] = []
        answer_text = ""

        try:
            retriever = self._index.as_retriever(similarity_top_k=self.cfg.top_k)
            source_nodes = retriever.retrieve(question)
            retrieval_ms = (perf_counter() - retrieval_start) * 1000.0

            generation_start = perf_counter()
            try:
                synthesizer = self._index.as_response_synthesizer()
            except Exception:
                from llama_index.core import get_response_synthesizer

                synthesizer = get_response_synthesizer()

            try:
                resp = synthesizer.synthesize(query=question, nodes=source_nodes)
            except TypeError:
                from llama_index.core import QueryBundle

                resp = synthesizer.synthesize(query=QueryBundle(query_str=question), nodes=source_nodes)
            generation_ms = (perf_counter() - generation_start) * 1000.0
            answer_text = str(getattr(resp, "response", resp))
        except Exception:
            # Fallback for compatibility with differing llama-index versions.
            qe = self._index.as_query_engine(similarity_top_k=self.cfg.top_k)
            retrieval_ms = None
            generation_start = perf_counter()
            resp = qe.query(question)
            answer_text = str(resp)
            generation_ms = (perf_counter() - generation_start) * 1000.0
            source_nodes = getattr(resp, "source_nodes", []) or []

        sources = self._extract_sources(source_nodes)
        context_used = self._build_context_used(sources)
        total_ms = (perf_counter() - query_start) * 1000.0

        return {
            "answer": answer_text,
            "sources": [s.__dict__ for s in sources],
            "context_used": context_used,
            "retrieval": {
                "top_k": self.cfg.top_k,
                "similarity_metric": None,
            },
            "timings_ms": {
                "index_build_ms": index_build_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        }
