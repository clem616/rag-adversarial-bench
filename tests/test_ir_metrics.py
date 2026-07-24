"""Tests for rag_bench/pillars/retrieval/ir_metrics.py."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from rag_bench.pillars.retrieval.ir_metrics import is_chunk_relevant


def _make_embedding_model(return_vec: np.ndarray | None = None):
    """Return a mock embedding model whose .encode() returns a fixed 2-D array."""
    model = MagicMock()
    if return_vec is None:
        return_vec = np.array([1.0, 0.0, 0.0])
    model.encode.return_value = np.array([return_vec])
    return model


class TestIsChunkRelevant:
    """Branch coverage for is_chunk_relevant."""

    # ── doc_id not in expected_doc_ids → False ───────────────────────────────

    def test_wrong_doc_id_returns_false(self):
        model = _make_embedding_model()
        gt_emb = np.array([1.0, 0.0, 0.0])
        result = is_chunk_relevant(
            chunk_text="Some text",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001", "doc-002"},
            chunk_doc_id="doc-999",
            threshold=0.5,
            embedding_model=model,
        )
        assert result is False
        model.encode.assert_not_called()

    # ── empty / whitespace chunk_text → False ────────────────────────────────

    def test_empty_text_returns_false(self):
        model = _make_embedding_model()
        gt_emb = np.array([1.0, 0.0, 0.0])
        result = is_chunk_relevant(
            chunk_text="",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001"},
            chunk_doc_id="doc-001",
            threshold=0.5,
            embedding_model=model,
        )
        assert result is False
        model.encode.assert_not_called()

    def test_whitespace_only_text_returns_false(self):
        model = _make_embedding_model()
        gt_emb = np.array([1.0, 0.0, 0.0])
        result = is_chunk_relevant(
            chunk_text="   \t\n",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001"},
            chunk_doc_id="doc-001",
            threshold=0.5,
            embedding_model=model,
        )
        assert result is False

    # ── cosine sim >= threshold → True ───────────────────────────────────────

    def test_high_similarity_returns_true(self):
        gt_emb = np.array([1.0, 0.0, 0.0])
        # chunk embedding identical to gt → cosine sim = 1.0 > threshold=0.5
        model = _make_embedding_model(return_vec=np.array([1.0, 0.0, 0.0]))
        result = is_chunk_relevant(
            chunk_text="relevant text",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001"},
            chunk_doc_id="doc-001",
            threshold=0.5,
            embedding_model=model,
        )
        assert result is True

    # ── cosine sim < threshold → False ───────────────────────────────────────

    def test_low_similarity_returns_false(self):
        gt_emb = np.array([1.0, 0.0, 0.0])
        # orthogonal → cosine sim = 0.0 < threshold=0.5
        model = _make_embedding_model(return_vec=np.array([0.0, 1.0, 0.0]))
        result = is_chunk_relevant(
            chunk_text="irrelevant text",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001"},
            chunk_doc_id="doc-001",
            threshold=0.5,
            embedding_model=model,
        )
        assert result is False

    # ── threshold boundary ────────────────────────────────────────────────────

    def test_exact_threshold_returns_true(self):
        """Similarity exactly equal to threshold is considered relevant (>=)."""
        gt_emb = np.array([1.0, 0.0])
        # We need a vector such that cosine_sim(gt_emb, chunk_emb) == threshold=0.5
        # cos(60°) = 0.5 → chunk_emb = [0.5, sqrt(3)/2]
        threshold = 0.5
        chunk_vec = np.array([0.5, (3**0.5) / 2])
        model = _make_embedding_model(return_vec=chunk_vec)
        result = is_chunk_relevant(
            chunk_text="borderline",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001"},
            chunk_doc_id="doc-001",
            threshold=threshold,
            embedding_model=model,
        )
        assert result is True

    # ── zero-vector gt_embedding (edge case) ─────────────────────────────────

    def test_zero_gt_embedding_returns_false(self):
        gt_emb = np.array([0.0, 0.0, 0.0])
        model = _make_embedding_model(return_vec=np.array([1.0, 0.0, 0.0]))
        result = is_chunk_relevant(
            chunk_text="some text",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-001"},
            chunk_doc_id="doc-001",
            threshold=0.5,
            embedding_model=model,
        )
        # cosine_similarity with zero vector returns 0.0 < 0.5 → False
        assert result is False

    # ── embedding model called with correct args ──────────────────────────────

    def test_embedding_model_called_correctly(self):
        gt_emb = np.array([1.0, 0.0])
        model = _make_embedding_model(return_vec=np.array([1.0, 0.0]))
        chunk_text = "my chunk text"
        is_chunk_relevant(
            chunk_text=chunk_text,
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-x"},
            chunk_doc_id="doc-x",
            threshold=0.0,
            embedding_model=model,
        )
        model.encode.assert_called_once_with([chunk_text], convert_to_numpy=True)

    # ── multiple expected doc ids ─────────────────────────────────────────────

    def test_one_of_several_expected_docs_matches(self):
        gt_emb = np.array([1.0, 0.0])
        model = _make_embedding_model(return_vec=np.array([1.0, 0.0]))
        result = is_chunk_relevant(
            chunk_text="text",
            gt_embedding=gt_emb,
            expected_doc_ids={"doc-A", "doc-B", "doc-C"},
            chunk_doc_id="doc-B",
            threshold=0.5,
            embedding_model=model,
        )
        assert result is True
