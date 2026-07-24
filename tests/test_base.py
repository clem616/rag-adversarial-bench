"""Tests for rag_bench/systems/base.py — RAGSystem ABC."""
from __future__ import annotations

import pytest
from abc import ABC

from rag_bench.systems.base import RAGSystem


class ConcreteSystem(RAGSystem):
    """Minimal concrete implementation for testing the base class."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return [{"text": "chunk", "doc_id": "d1", "score": 1.0}]

    def generate(self, query: str, chunks: list[dict], history=None) -> str:
        return "answer"


class ConcreteWithClassify(RAGSystem):
    """Implementation that overrides classify_intent."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return []

    def generate(self, query: str, chunks: list[dict], history=None) -> str:
        return "x"

    def classify_intent(self, query: str, history=None) -> str:
        return "small_talk"


class TestRAGSystemABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            RAGSystem()  # type: ignore

    def test_concrete_instantiable(self):
        sys = ConcreteSystem()
        assert isinstance(sys, RAGSystem)


class TestDefaultClassifyIntent:
    def test_default_returns_document_content(self):
        sys = ConcreteSystem()
        assert sys.classify_intent("any query") == "document_content"

    def test_default_ignores_history(self):
        sys = ConcreteSystem()
        assert sys.classify_intent("q", history=[{"role": "user", "content": "hi"}]) == "document_content"

    def test_override_classify_intent(self):
        sys = ConcreteWithClassify()
        assert sys.classify_intent("hello") == "small_talk"


class TestDefaultIngest:
    def test_ingest_is_noop(self):
        sys = ConcreteSystem()
        # Should not raise and should return None
        result = sys.ingest([{"doc_id": "d1", "text": "text", "title": "Title"}])
        assert result is None

    def test_ingest_empty_list(self):
        sys = ConcreteSystem()
        result = sys.ingest([])
        assert result is None


class TestRetrieve:
    def test_retrieve_returns_list(self):
        sys = ConcreteSystem()
        result = sys.retrieve("what is X?")
        assert isinstance(result, list)

    def test_retrieve_top_k_parameter(self):
        sys = ConcreteSystem()
        result = sys.retrieve("q", top_k=3)
        assert isinstance(result, list)

    def test_retrieve_chunk_fields(self):
        sys = ConcreteSystem()
        chunks = sys.retrieve("q")
        for chunk in chunks:
            assert "text" in chunk
            assert "doc_id" in chunk
            assert "score" in chunk


class TestGenerate:
    def test_generate_returns_string(self):
        sys = ConcreteSystem()
        chunks = sys.retrieve("q")
        result = sys.generate("q", chunks)
        assert isinstance(result, str)

    def test_generate_with_history(self):
        sys = ConcreteSystem()
        chunks = sys.retrieve("q")
        history = [{"role": "user", "content": "hi"}]
        result = sys.generate("q", chunks, history=history)
        assert isinstance(result, str)

    def test_generate_with_none_history(self):
        sys = ConcreteSystem()
        chunks = sys.retrieve("q")
        result = sys.generate("q", chunks, history=None)
        assert isinstance(result, str)
