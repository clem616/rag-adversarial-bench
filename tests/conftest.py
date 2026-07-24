"""Shared fixtures for the rag_bench test suite."""
from __future__ import annotations

import sys
import os
from pathlib import Path

_RAG_BENCH_ROOT = Path(__file__).parent.parent
if str(_RAG_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_BENCH_ROOT))

import numpy as np
import pytest
from unittest.mock import MagicMock

from rag_bench.systems.base import RAGSystem


class DummySystem(RAGSystem):
    """Minimal concrete RAGSystem for testing."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return [
            {"text": f"chunk for {query}", "doc_id": "doc-001", "score": 0.9}
        ]

    def generate(self, query: str, chunks: list[dict], history=None) -> str:
        return f"answer for {query}"


@pytest.fixture
def dummy_system():
    return DummySystem()


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.encode.return_value = np.array([[0.1, 0.9, 0.0]])
    return model


@pytest.fixture
def tmp_report(tmp_path):
    return tmp_path / "report.md"
