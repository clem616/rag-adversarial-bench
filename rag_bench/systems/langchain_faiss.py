"""LangChainFAISSSystem — reference RAGSystem implementation.

Uses LangChain + FAISS + sentence-transformers (all-MiniLM-L6-v2) for retrieval
and an OpenAI-compatible LLM endpoint for generation. No proprietary dependencies.

This is the public reference implementation for the RAG-Adversarial-Bench leaderboard.
Researchers can reproduce its results to establish a baseline before benchmarking
their own system.

Environment variables (read at call time so run_benchmark.py env fallbacks work):
    RAG_BENCH_LLM_BASE_URL   OpenAI-compatible endpoint; falls back to LLM_BASE_URL from .env
    RAG_BENCH_LLM_API_KEY    API key; falls back to LLM_API_KEY from .env
    RAG_BENCH_LLM_MODEL      Model name; falls back to LLM_MODEL from .env (default: gpt-4.1-mini)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag_bench.systems.base import RAGSystem

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _llm_base_url() -> str:
    return (
        os.environ.get("RAG_BENCH_LLM_BASE_URL")
        or os.environ.get("LLM_BASE_URL", "https://api.openai.com")
    )


def _llm_api_key() -> str:
    return (
        os.environ.get("RAG_BENCH_LLM_API_KEY")
        or os.environ.get("LLM_API_KEY", "")
    )


def _llm_model() -> str:
    return (
        os.environ.get("RAG_BENCH_LLM_MODEL")
        or os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    )


class LangChainFAISSSystem(RAGSystem):
    """Retrieval: FAISS + all-MiniLM-L6-v2.  Generation: OpenAI-compatible LLM."""

    def __init__(self, index_path: Path | None = None) -> None:
        self._index_path = index_path
        self._vectorstore: Any = None
        self._embeddings: Any = None
        self._llm_client: Any = None
        self._docs: dict[str, str] = {}
        self._init_embeddings()
        if index_path and Path(index_path).exists():
            self._load_index(index_path)

    def _init_embeddings(self) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import]
        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)

    def _load_index(self, path: Path) -> None:
        from langchain_community.vectorstores import FAISS  # type: ignore[import]
        self._vectorstore = FAISS.load_local(
            str(path), self._embeddings, allow_dangerous_deserialization=True
        )

    def ingest(self, documents: list[dict]) -> None:
        """Build or rebuild the FAISS index from the provided documents."""
        from langchain_community.vectorstores import FAISS  # type: ignore[import]
        from langchain_core.documents import Document  # type: ignore[import]

        lc_docs = []
        for doc in documents:
            self._docs[doc["doc_id"]] = doc.get("text", "")
            lc_docs.append(Document(
                page_content=doc.get("text", ""),
                metadata={"doc_id": doc["doc_id"], "title": doc.get("title", doc["doc_id"])},
            ))

        self._vectorstore = FAISS.from_documents(lc_docs, self._embeddings)
        if self._index_path:
            self._vectorstore.save_local(str(self._index_path))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self._vectorstore is None:
            raise RuntimeError("Index not built. Call ingest() first or provide index_path.")
        results = self._vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "text": doc.page_content,
                "doc_id": doc.metadata.get("doc_id", ""),
                "score": float(score),
            }
            for doc, score in results
        ]

    def generate(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        if not self._llm_client:
            base_url = _llm_base_url()
            api_key = _llm_api_key()
            if "openai.azure.com" in base_url:
                from openai import AzureOpenAI  # type: ignore[import]
                self._llm_client = AzureOpenAI(
                    azure_endpoint=base_url,
                    api_key=api_key or "placeholder",
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                )
            else:
                from openai import OpenAI  # type: ignore[import]
                self._llm_client = OpenAI(base_url=base_url, api_key=api_key or "placeholder")

        context = "\n\n---\n\n".join(c["text"] for c in chunks if c.get("text"))
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question based only on the provided context. "
                    "If the context does not contain enough information, say so."
                ),
            }
        ]
        if history:
            messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        })

        resp = self._llm_client.chat.completions.create(
            model=_llm_model(),
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""
