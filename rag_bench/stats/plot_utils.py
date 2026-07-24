"""plot_utils.py — Matplotlib figures for the RAG-Adversarial-Bench framework.

All public functions save files to the configured output directory and return the PNG Path.
Import is deferred so the module can be imported even when matplotlib is absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_OUTPUT_DIR = Path(os.environ.get("RAG_BENCH_OUTPUT_DIR", "./output"))
_ROC_PNG = _OUTPUT_DIR / "roc_curve.png"
_ROC_PDF = _OUTPUT_DIR / "roc_curve.pdf"


def plot_roc_calibration(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    tau_star: float,
    youden_j: float,
    p95: float,
    null_scores: np.ndarray,
    signal_scores: np.ndarray,
    output_png: Path = _ROC_PNG,
    output_pdf: Path = _ROC_PDF,
) -> Path:
    """Two-panel figure: ROC curve (left) + score distribution overlap (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from scipy.stats import gaussian_kde

    from rag_bench.stats.stats_utils import compute_cohens_d

    null = np.asarray(null_scores, dtype=np.float64)
    signal = np.asarray(signal_scores, dtype=np.float64)
    n_null = len(null)
    n_signal = len(signal)

    j_vals = tpr - fpr
    youden_idx = int(np.argmax(j_vals))
    fpr_youden = float(fpr[youden_idx])
    tpr_youden = float(tpr[youden_idx])

    fpr_p95 = float(np.sum(null >= p95) / n_null)
    tpr_p95 = float(np.sum(signal >= p95) / n_signal)

    cohens_d = compute_cohens_d(signal, null)

    fig, (ax_roc, ax_dist) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.38)

    ax_roc.plot(
        [0, 1], [0, 1],
        color="#999999", linestyle="--", linewidth=1.0,
        label="Random baseline  (AUC = 0.50)",
    )
    ax_roc.plot(
        fpr, tpr,
        color="#2166AC", linewidth=2.2,
        label=f"all-MiniLM-L6-v2  (AUC = {roc_auc:.4f})",
    )
    ax_roc.fill_between(fpr, tpr, alpha=0.08, color="#2166AC")
    ax_roc.scatter(
        [fpr_youden], [tpr_youden],
        color="#D6604D", s=90, zorder=6,
        label=f"Youden optimal  τ* = {tau_star:.3f}  (J = {youden_j:.3f})",
    )
    x_offset = 0.07 if fpr_youden < 0.6 else -0.22
    y_offset = -0.13 if tpr_youden > 0.3 else 0.08
    ax_roc.annotate(
        f"τ* = {tau_star:.3f}\n"
        f"TPR = {tpr_youden:.2f},  FPR = {fpr_youden:.2f}",
        xy=(fpr_youden, tpr_youden),
        xytext=(fpr_youden + x_offset, tpr_youden + y_offset),
        fontsize=8,
        color="#D6604D",
        arrowprops=dict(arrowstyle="->", color="#D6604D", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#D6604D", alpha=0.85),
    )
    ax_roc.scatter(
        [fpr_p95], [tpr_p95],
        color="#F4A582", s=65, zorder=6, marker="D",
        label=f"P95 threshold  τ_P95 = {p95:.3f}",
    )
    ax_roc.set_xlabel(
        "False Positive Rate\n(fraction of random cross-doc pairs scoring ≥ τ)",
        fontsize=9.5,
    )
    ax_roc.set_ylabel(
        "True Positive Rate\n(fraction of relevant GT pairs scoring ≥ τ)",
        fontsize=9.5,
    )
    ax_roc.set_title("Embedding Discriminability — ROC Curve", fontsize=11, fontweight="bold")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.25, linestyle=":")
    ax_roc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_roc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_roc.set_aspect("equal")

    x_lo = min(null.min(), signal.min()) - 0.05
    x_hi = max(null.max(), signal.max()) + 0.05
    x_grid = np.linspace(x_lo, x_hi, 500)
    kde_null = gaussian_kde(null, bw_method="scott")
    kde_signal = gaussian_kde(signal, bw_method="scott")
    y_null = kde_null(x_grid)
    y_signal = kde_signal(x_grid)

    ax_dist.fill_between(x_grid, y_null, alpha=0.30, color="#92C5DE",
                         label=f"Null  (N={n_null},  μ={null.mean():.3f},  σ={null.std():.3f})")
    ax_dist.fill_between(x_grid, y_signal, alpha=0.30, color="#F4A582",
                         label=f"Signal  (N={n_signal},  μ={signal.mean():.3f},  σ={signal.std():.3f})")
    ax_dist.plot(x_grid, y_null, color="#2166AC", linewidth=1.8)
    ax_dist.plot(x_grid, y_signal, color="#D6604D", linewidth=1.8)
    ax_dist.axvline(tau_star, color="#D6604D", linestyle="--", linewidth=1.6,
                    label=f"τ* = {tau_star:.3f}  (Youden)")
    ax_dist.axvline(p95, color="#F4A582", linestyle=":", linewidth=1.6,
                    label=f"τ_P95 = {p95:.3f}")
    y_min_curve = np.minimum(y_null, y_signal)
    ax_dist.fill_between(x_grid, y_min_curve, alpha=0.18, color="#888888",
                         label="Overlap region (irreducible error)")
    ax_dist.text(0.97, 0.96, f"Cohen's d = {cohens_d:.2f}",
                 transform=ax_dist.transAxes, ha="right", va="top", fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor="#666666", alpha=0.90))
    ax_dist.set_xlabel("Cosine Similarity Score", fontsize=9.5)
    ax_dist.set_ylabel("Kernel Density", fontsize=9.5)
    ax_dist.set_title("Score Distributions: Null vs Signal", fontsize=11, fontweight="bold")
    ax_dist.legend(fontsize=7.5, loc="upper left")
    ax_dist.grid(True, alpha=0.25, linestyle=":")
    ax_dist.set_xlim(x_lo, x_hi)
    ax_dist.set_ylim(bottom=0)

    fig.text(
        0.5, -0.04,
        f"Null: {n_null} random cross-document chunk pairs  ·  "
        f"Signal: {n_signal} ground-truth context vs. expected-document chunk pairs  ·  "
        f"Embedding model: all-MiniLM-L6-v2 (384-dim, cosine similarity)  ·  "
        f"ROC computed with sklearn.metrics.roc_curve (Pedregosa et al., 2011)",
        ha="center", fontsize=7.5, style="italic", color="#555555",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return output_png
