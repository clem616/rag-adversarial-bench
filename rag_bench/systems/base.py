"""RAGSystem: system-agnostic interface for plugging any RAG pipeline into the benchmark.

Implement this ABC to evaluate your system against the 8-pillar framework.
The benchmark pillars call only retrieve(), generate(), and classify_intent() —
all other system internals are hidden behind this interface.

Retrieval Tier pillars (P1, P2, P7, P8) call retrieve() directly.
Generation Tier pillars (P3, P4, P5, P6) call retrieve() then generate().
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RAGSystem(ABC):
    """Abstract interface for a retrieval-augmented generation system."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the top-k most relevant chunks for the query.

        Returns a list of dicts, each with at minimum:
            {
                "text":   str,    # chunk text
                "doc_id": str,    # document identifier (filename or ID)
                "score":  float,  # retrieval score (higher = more relevant)
            }
        Additional fields (e.g. metadata) are allowed and ignored by the benchmark.
        """

    @abstractmethod
    def generate(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """Generate an answer given the query and retrieved chunks.

        Args:
            query:   The user question.
            chunks:  Output of retrieve() — the context for generation.
            history: Optional list of prior turns, each {"role": str, "content": str}.

        Returns:
            The generated answer as a plain string.
        """

    def classify_intent(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> str:
        """Classify the query intent for OOD safe-refusal testing (Pillar 4).

        Override this method to enable the OOD pillar to test whether your
        system correctly routes off-topic queries away from document retrieval.

        Returns one of:
            "document_content"  — triggers retrieval + generation (default)
            "small_talk"        — routed away from retrieval (counts as safe refusal)
            "unknown"           — routed away from retrieval (counts as safe refusal)
            Any other string    — treated as safe refusal if not "document_content"

        Default implementation always returns "document_content".
        Systems that do not implement intent classification will be evaluated
        only on max_similarity < tau* for the OOD safe-refusal metric.
        """
        return "document_content"

    def ingest(self, documents: list[dict]) -> None:
        """Optional: ingest documents into the system's index.

        Called by the benchmark harness before evaluation when a document
        store needs to be populated. Override if your system requires explicit
        ingestion. Default is a no-op (assumes the index is pre-populated).

        Each document dict has at minimum:
            {
                "doc_id": str,   # unique document identifier
                "text":   str,   # full document text
                "title":  str,   # optional title
            }
        """
