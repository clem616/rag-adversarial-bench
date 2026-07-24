"""Retrieval-tier helper: relevance judgement for IR metric computation.

Extracted from the private benchmark engine so it can be used by the
public RAGSystem-based evaluation path without any app.* dependencies.
"""

from __future__ import annotations

import numpy as np

from rag_bench.stats.stats_utils import cosine_similarity


def is_chunk_relevant(
    chunk_text: str,
    gt_embedding: np.ndarray,
    expected_doc_ids: set[str],
    chunk_doc_id: str,
    threshold: float,
    embedding_model: object,
) -> bool:
    """Return True iff the chunk belongs to an expected document AND its embedding
    cosine-similarity to the ground-truth context is >= threshold.

    Parameters
    ----------
    chunk_text      : text of the retrieved chunk
    gt_embedding    : embedding of the ground-truth context sentence
    expected_doc_ids: set of document IDs that contain the answer
    chunk_doc_id    : doc_id of this chunk (from RAGSystem.retrieve())
    threshold       : Youden-optimal tau* from null calibration
    embedding_model : sentence-transformers model with .encode()
    """
    if chunk_doc_id not in expected_doc_ids:
        return False
    if not chunk_text.strip():
        return False
    chunk_emb = embedding_model.encode([chunk_text], convert_to_numpy=True)[0]
    return cosine_similarity(gt_embedding, chunk_emb) >= threshold
