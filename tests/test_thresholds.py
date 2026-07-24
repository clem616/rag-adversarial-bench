"""Tests for rag_bench/thresholds.py — all constants present and correctly typed."""
from __future__ import annotations

import pytest
import rag_bench.thresholds as T


def test_monte_carlo_defaults():
    assert isinstance(T.MONTE_CARLO_PASSES, int)
    assert T.MONTE_CARLO_PASSES >= 1
    assert isinstance(T.SCORE_MEAN_MIN, (int, float))
    assert isinstance(T.SCORE_STD_MAX, (int, float))
    assert isinstance(T.JUDGE_AGREEMENT_MIN, float)


def test_embedding_calibration_constants():
    assert isinstance(T.NULL_SAMPLE_PAIRS, int)
    assert T.NULL_SAMPLE_PAIRS > 0
    assert 0.0 < T.NULL_ALPHA < 1.0
    assert isinstance(T.ROC_AUC_MIN, float)
    assert 0.0 < T.ROC_AUC_MIN <= 1.0


def test_ir_metrics_constants():
    assert isinstance(T.TOP_K, int)
    assert T.TOP_K >= 1
    assert 0.0 < T.MRR_MIN <= 1.0
    assert 0.0 < T.HIT_RATE_MIN <= 1.0
    assert 0.0 < T.PRECISION_AT_K_MIN <= 1.0


def test_confidence_interval_constant():
    assert 0.0 < T.CI_LEVEL < 1.0


def test_ood_constant():
    assert 0.0 < T.OOD_SAFE_REFUSAL_RATE_MIN <= 1.0


def test_multiturn_constants():
    assert 0.0 < T.REWRITE_SIMILARITY_MIN <= 1.0
    assert 0.0 < T.MULTITURN_HIT_RATE_MIN <= 1.0
    assert T.CONTEXT_DECAY_MAX_DROP > 0.0


def test_poisoning_constant():
    assert 0.0 < T.POISONING_GROUNDING_RATE_MIN <= 1.0


def test_position_bias_constant():
    assert T.POSITION_BIAS_MAX_COMP_STD > 0.0


def test_distractor_constants():
    assert T.DISTRACTOR_HALLUCINATION_MIN > 0.0
    assert T.DISTRACTOR_COMPLETENESS_MIN > 0.0


def test_external_domain_constants():
    assert 0.0 < T.EXTERNAL_HIT_RATE_MIN <= 1.0
    assert 0.0 < T.EXTERNAL_MRR_MIN <= 1.0
    assert T.EXTERNAL_SCORE_MEAN_MIN > 0.0


def test_retrieval_weights_sum_to_one():
    total = sum(T.RETRIEVAL_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"RETRIEVAL_WEIGHTS sum={total}"


def test_generation_weights_sum_to_one():
    total = sum(T.GENERATION_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"GENERATION_WEIGHTS sum={total}"


def test_retrieval_weights_keys():
    assert set(T.RETRIEVAL_WEIGHTS.keys()) == {"calibration", "ir", "position_bias", "distractor"}


def test_generation_weights_keys():
    assert set(T.GENERATION_WEIGHTS.keys()) == {"monte_carlo", "ood", "multiturn", "poisoned"}
