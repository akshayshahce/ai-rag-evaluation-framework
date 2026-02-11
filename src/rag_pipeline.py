# src/rag_pipeline.py
from dataclasses import dataclass
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

    def query(self, question: str) -> Dict[str, Any]:
        if self._index is None:
            self.build_index()

        qe = self._index.as_query_engine(similarity_top_k=self.cfg.top_k)
        resp = qe.query(question)

        sources: List[Source] = []
        for sn in getattr(resp, "source_nodes", []) or []:
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

        return {
            "answer": str(resp),
            "sources": [s.__dict__ for s in sources],
        }