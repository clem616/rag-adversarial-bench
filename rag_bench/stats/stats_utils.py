from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import stats


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_null_threshold(
    embeddings: np.ndarray,
    n_pairs: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
    doc_numbers: list[str] | None = None,
) -> tuple[float, np.ndarray]:
    """Sample random chunk pairs and return (p_threshold, all_scores).

    When *doc_numbers* is provided, only pairs from **different** documents
    are sampled — this prevents intra-document similarity from inflating
    the null distribution (critical when the corpus has few documents).
    """
    rng = np.random.default_rng(rng_seed)
    n_chunks = len(embeddings)
    scores = np.empty(n_pairs, dtype=np.float64)
    collected = 0
    max_attempts = n_pairs * 20
    attempts = 0
    while collected < n_pairs and attempts < max_attempts:
        idx_a, idx_b = rng.choice(n_chunks, size=2, replace=False)
        attempts += 1
        if doc_numbers is not None and doc_numbers[idx_a] == doc_numbers[idx_b]:
            continue
        scores[collected] = cosine_similarity(embeddings[idx_a], embeddings[idx_b])
        collected += 1
    scores = scores[:collected]
    if len(scores) == 0:
        return 0.0, scores
    return float(np.percentile(scores, (1.0 - alpha) * 100)), scores


def compute_confidence_interval(
    scores: list[float] | np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Student-t confidence interval for the mean (appropriate for small n)."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return (float(arr[0]), float(arr[0]))
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    if se == 0.0:
        return (mean, mean)
    lo, hi = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return (float(lo), float(hi))


def compute_cohens_d(
    treatment: list[float] | np.ndarray,
    null: list[float] | np.ndarray,
) -> float:
    """Cohen's d effect size between treatment and null distributions."""
    t = np.asarray(treatment, dtype=np.float64)
    n = np.asarray(null, dtype=np.float64)
    nt, nn = len(t), len(n)
    if nt < 2 or nn < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((nt - 1) * np.var(t, ddof=1) + (nn - 1) * np.var(n, ddof=1))
        / (nt + nn - 2)
    )
    if pooled_std == 0.0:
        return 0.0
    return float((np.mean(t) - np.mean(n)) / pooled_std)


def compute_roc_auc(
    null_scores: np.ndarray,
    signal_scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute ROC AUC and return (auc, fpr_array, tpr_array).

    Positive class = signal (known ground-truth relevant pairs).
    Negative class = null (random cross-document pairs).
    AUC = P(signal_score > null_score) for a random pair.
    """
    from sklearn.metrics import auc as sklearn_auc
    from sklearn.metrics import roc_curve

    null = np.asarray(null_scores, dtype=np.float64)
    signal = np.asarray(signal_scores, dtype=np.float64)

    y_true = np.concatenate([np.zeros(len(null)), np.ones(len(signal))])
    y_score = np.concatenate([null, signal])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(sklearn_auc(fpr, tpr)), fpr, tpr


def compute_optimal_threshold_youden(
    null_scores: np.ndarray,
    signal_scores: np.ndarray,
    n_thresholds: int = 1000,
) -> tuple[float, float]:
    """Return (optimal_tau, max_J) by maximising Youden's J = TPR - FPR."""
    null = np.asarray(null_scores, dtype=np.float64)
    signal = np.asarray(signal_scores, dtype=np.float64)
    lo = float(min(null.min(), signal.min()))
    hi = float(max(null.max(), signal.max()))
    thresholds = np.linspace(lo, hi, num=n_thresholds)

    best_tau = lo
    max_j = -2.0
    for tau in thresholds:
        tpr = float(np.sum(signal >= tau) / len(signal))
        fpr = float(np.sum(null >= tau) / len(null))
        j = tpr - fpr
        if j > max_j:
            max_j = j
            best_tau = float(tau)

    return best_tau, max_j


def compute_mrr(reciprocal_ranks: list[float]) -> float:
    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))


def compute_hit_rate(hits: int, total: int) -> float:
    if total == 0:
        return 0.0
    return hits / total


def compute_precision_at_k(relevant_count: int, k: int) -> float:
    if k == 0:
        return 0.0
    return relevant_count / k


def judge_agreement_rate(verdicts: list[str]) -> float:
    """Fraction of runs whose verdict matches the majority verdict."""
    if not verdicts:
        return 0.0
    majority = Counter(verdicts).most_common(1)[0][0]
    return sum(1 for v in verdicts if v == majority) / len(verdicts)
