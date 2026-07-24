"""Public ReportWriter — generates the dual-scorecard benchmark report.

Standalone report writer for the RAG-Adversarial-Bench evaluation framework.
No proprietary dependencies — usable with any RAGSystem implementation.

Environment variables used:
    RAG_BENCH_GENERATOR_MODEL   Name of the system under test (default: "unknown")
    RAG_BENCH_JUDGE_MODEL       Name of the judge model used (default: "unknown")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rag_bench.thresholds import (
    GENERATION_WEIGHTS,
    OOD_SAFE_REFUSAL_RATE_MIN,
    POISONING_GROUNDING_RATE_MIN,
    RETRIEVAL_WEIGHTS,
)

_GENERATOR_MODEL = os.environ.get("RAG_BENCH_GENERATOR_MODEL", "unknown")
_JUDGE_MODEL = os.environ.get("RAG_BENCH_JUDGE_MODEL", "claude-haiku-4-5-20251001")


class ReportWriter:
    """Generates a markdown report and parallel JSON sidecar for a benchmark run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self.path.write_text(
            f"# RAG-Adversarial-Bench Evaluation Report\n\n"
            f"_Generated: {ts}_\n\n"
            f"- **System under test:** `{_GENERATOR_MODEL}`\n"
            f"- **Judge model:** `{_JUDGE_MODEL}`\n\n",
            encoding="utf-8",
        )
        self._data: dict = {
            "generated": ts,
            "generator_model": _GENERATOR_MODEL,
            "judge_model": _JUDGE_MODEL,
            "sections": {},
        }

    def _append(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # Section 1: Embedding Calibration
    # ------------------------------------------------------------------
    def write_calibration_section(
        self,
        p95: float,
        optimal_tau: float,
        youden_j: float,
        null_mean: float,
        null_std: float,
        signal_mean: float,
        signal_std: float,
        per_question: list[dict],
        roc_auc: float = 0.0,
        roc_plot_path: str | None = None,
    ) -> None:
        self._data["sections"]["calibration"] = {
            "p95": p95,
            "optimal_tau": optimal_tau,
            "youden_j": youden_j,
            "null_mean": null_mean,
            "null_std": null_std,
            "signal_mean": signal_mean,
            "signal_std": signal_std,
            "roc_auc": roc_auc,
            "roc_plot_path": roc_plot_path,
            "per_question": per_question,
        }
        lines = [
            "## 1. Embedding Calibration (Youden-Optimal Threshold + ROC AUC)\n\n",
            f"- **Null distribution:** mu={null_mean:.4f}, sigma={null_std:.4f}\n",
            f"- **Signal distribution:** mu={signal_mean:.4f}, sigma={signal_std:.4f}\n",
            f"- **P95 threshold (alpha=0.05):** {p95:.4f}\n",
            f"- **Youden-optimal threshold (tau*):** {optimal_tau:.4f}  |  J={youden_j:.4f}\n",
            f"- **ROC AUC:** {roc_auc:.4f}\n\n",
        ]
        if roc_plot_path:
            lines.append(f"![ROC Curve]({roc_plot_path})\n\n")
        lines += [
            "| Question ID | Max Sim | Beats tau*? | Cohen's d |\n",
            "|---|---|---|---|\n",
        ]
        for q in per_question:
            beats = "YES" if q["max_sim"] > optimal_tau else "NO"
            lines.append(
                f"| {q['id']} | {q['max_sim']:.4f} | {beats} | {q['cohens_d']:.2f} |\n"
            )
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 2: IR Metrics
    # ------------------------------------------------------------------
    def write_ir_metrics_section(
        self,
        per_question: list[dict],
        aggregate_mrr: float,
        aggregate_hit_rate: float,
        aggregate_precision: float,
    ) -> None:
        self._data["sections"]["ir"] = {
            "per_question": per_question,
            "aggregate_mrr": aggregate_mrr,
            "aggregate_hit_rate": aggregate_hit_rate,
            "aggregate_precision": aggregate_precision,
        }
        lines = [
            "## 2. Information Retrieval Metrics\n\n",
            "| Question ID | Hit@K | First Relevant Rank | Reciprocal Rank | Precision@K |\n",
            "|---|---|---|---|---|\n",
        ]
        for q in per_question:
            hit = "YES" if q["hit"] else "NO"
            rank_str = str(q["first_relevant_rank"]) if q["first_relevant_rank"] else "—"
            lines.append(
                f"| {q['id']} | {hit} | {rank_str} | {q['reciprocal_rank']:.3f} "
                f"| {q['precision_at_k']:.3f} |\n"
            )
        lines.append(
            f"| **Aggregate** | **HR={aggregate_hit_rate:.2f}** | | "
            f"**MRR={aggregate_mrr:.3f}** | **P@K={aggregate_precision:.3f}** |\n"
        )
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 3: Monte Carlo
    # ------------------------------------------------------------------
    def write_monte_carlo_section(self, per_question: list[dict], passes: int) -> None:
        self._data["sections"]["monte_carlo"] = {"passes": passes, "per_question": per_question}
        lines = [
            f"## 3. Monte Carlo Evaluation (M={passes} passes)\n\n",
            "| Question ID | mu_comp | sigma_comp | CI_95_comp | mu_hall | sigma_hall "
            "| CI_95_hall | Judge Agree (comp) | Judge Agree (hall) |\n",
            "|---|---|---|---|---|---|---|---|---|\n",
        ]
        for q in per_question:
            ci_c = f"[{q['ci_comp'][0]:.2f}, {q['ci_comp'][1]:.2f}]"
            ci_h = f"[{q['ci_hall'][0]:.2f}, {q['ci_hall'][1]:.2f}]"
            lines.append(
                f"| {q['id']} "
                f"| {q['mu_comp']:.2f} | {q['sigma_comp']:.2f} | {ci_c} "
                f"| {q['mu_hall']:.2f} | {q['sigma_hall']:.2f} | {ci_h} "
                f"| {q['agree_comp']:.0%} | {q['agree_hall']:.0%} |\n"
            )
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 4: OOD Safe Refusal
    # ------------------------------------------------------------------
    def write_ood_section(self, per_question: list[dict], safe_rate: float) -> None:
        self._data["sections"]["ood"] = {
            "safe_rate": safe_rate,
            "per_question": per_question,
        }
        lines = [
            "## 4. OOD Safe Refusal\n\n",
            f"- **Safe Refusal Rate:** {safe_rate:.0%}  "
            f"({'PASS' if safe_rate >= OOD_SAFE_REFUSAL_RATE_MIN else 'FAIL'})\n\n",
            "| Question ID | Domain | Safe? | Reason |\n",
            "|---|---|---|---|\n",
        ]
        for q in per_question:
            safe_str = "YES" if q["safe"] else "**NO**"
            lines.append(f"| {q['id']} | {q['domain']} | {safe_str} | {q['reason']} |\n")
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 5: Multi-Turn
    # ------------------------------------------------------------------
    def write_multiturn_section(
        self,
        per_trajectory: list[dict],
        mean_rewrite_sim: float,
        turn2_hit_rate: float,
        mean_context_decay: float,
    ) -> None:
        self._data["sections"]["multiturn"] = {
            "mean_rewrite_sim": mean_rewrite_sim,
            "turn2_hit_rate": turn2_hit_rate,
            "mean_context_decay": mean_context_decay,
            "per_trajectory": per_trajectory,
        }
        lines = [
            "## 5. Multi-Turn Conversational Rigor\n\n",
            f"- **Mean rewrite similarity:** {mean_rewrite_sim:.3f}\n",
            f"- **Turn-2+ hit rate:** {turn2_hit_rate:.0%}\n",
            f"- **Mean context decay:** {mean_context_decay:.2f}\n\n",
            "| Trajectory | Rewrite Sims | Hit Flags | Comp/Turn |\n",
            "|---|---|---|---|\n",
        ]
        for r in per_trajectory:
            sims_str = ", ".join(f"{s:.3f}" for s in r["rewrite_sims"]) or "—"
            hits_str = ", ".join("HIT" if h else "MISS" for h in r["hit_flags"])
            comp_str = ", ".join(f"{c:.1f}" for c in r["comp_per_turn"])
            lines.append(f"| {r['id']} | {sims_str} | {hits_str} | {comp_str} |\n")
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 6: Poisoned Context
    # ------------------------------------------------------------------
    def write_poisoned_context_section(
        self, per_question: list[dict], grounding_rate: float
    ) -> None:
        self._data["sections"]["poisoned"] = {
            "grounding_rate": grounding_rate,
            "per_question": per_question,
        }
        lines = [
            "## 6. Poisoned Context — Counterfactual Grounding\n\n",
            f"- **Grounding Rate:** {grounding_rate:.0%}  "
            f"({'PASS' if grounding_rate >= POISONING_GROUNDING_RATE_MIN else 'FAIL'})\n\n",
            "| ID | Grounded? | Reason |\n",
            "|---|---|---|\n",
        ]
        for q in per_question:
            g = "YES" if q["grounded"] else "**NO**"
            lines.append(f"| {q['id']} | {g} | {q['reason']} |\n")
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 7: Position Bias
    # ------------------------------------------------------------------
    def write_position_bias_section(self, per_question: list[dict]) -> None:
        self._data["sections"]["position_bias"] = {"per_question": per_question}
        lines = [
            "## 7. Position Bias\n\n",
            "| ID | Score@0 | Score@mid | Score@end | std_dev |\n",
            "|---|---|---|---|---|\n",
        ]
        for q in per_question:
            sbp = q["scores_by_position"]
            positions = sorted(sbp.keys())
            s = [str(sbp[p]) for p in positions]
            lines.append(
                f"| {q['id']} | {s[0] if len(s)>0 else '?'} "
                f"| {s[1] if len(s)>1 else '?'} "
                f"| {s[2] if len(s)>2 else '?'} "
                f"| {q['std_dev']:.2f} |\n"
            )
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Section 8: Distractor Injection
    # ------------------------------------------------------------------
    def write_distractor_section(self, per_question: list[dict]) -> None:
        self._data["sections"]["distractor"] = {"per_question": per_question}
        lines = [
            "## 8. Distractor Injection\n\n",
            "| ID | Distractor Doc | Completeness | Hallucination |\n",
            "|---|---|---|---|\n",
        ]
        for q in per_question:
            lines.append(
                f"| {q['id']} | {q['distractor_doc']} "
                f"| {q['comp_score']} | {q['hall_score']} |\n"
            )
        lines.append("\n---\n\n")
        self._append("".join(lines))

    # ------------------------------------------------------------------
    # Dual Scorecard: Retrieval Tier vs Generation Tier
    # ------------------------------------------------------------------
    def write_dual_scorecard(
        self,
        retrieval_results: dict,
        generation_results: dict,
    ) -> None:
        """Emit the Retrieval / Generation component scorecard.

        Parameters
        ----------
        retrieval_results : dict
            Keys: "calibration", "ir", "position_bias", "distractor"
            Each value: {"score": float 0-100, "pass": bool}

        generation_results : dict
            Keys: "monte_carlo", "ood", "multiturn", "poisoned"
            Each value: {"score": float 0-100, "pass": bool}
        """
        def weighted_score(results: dict, weights: dict[str, float]) -> float:
            available = {
                k: v for k, v in results.items()
                if v.get("score") is not None and not v.get("skipped")
            }
            if not available:
                return 0.0
            total_w = sum(weights[k] for k in available if k in weights)
            if total_w == 0:
                return 0.0
            return round(
                sum(available[k]["score"] * weights[k] / total_w
                    for k in available if k in weights), 1
            )

        retrieval_score = weighted_score(retrieval_results, RETRIEVAL_WEIGHTS)
        generation_score = weighted_score(generation_results, GENERATION_WEIGHTS)
        overall_score = round((retrieval_score + generation_score) / 2, 1)

        self._data["sections"]["dual_scorecard"] = {
            "retrieval_score": retrieval_score,
            "generation_score": generation_score,
            "overall_score": overall_score,
            "retrieval_pillars": retrieval_results,
            "generation_pillars": generation_results,
        }

        _PILLAR_LABELS = {
            "calibration": "P1 — Embedding Calibration",
            "ir": "P2 — IR Metrics",
            "monte_carlo": "P3 — Monte Carlo Stability",
            "ood": "P4 — OOD Safe Refusal",
            "multiturn": "P5 — Multi-Turn Coherence",
            "poisoned": "P6 — Poisoned Context",
            "position_bias": "P7 — Position Bias",
            "distractor": "P8 — Distractor Injection",
        }

        lines = [
            "## Component Scorecard — Retrieval Tier vs Generation Tier\n\n",
            "### Retrieval Tier\n\n",
            f"> Weighted score: **{retrieval_score}/100**\n\n",
            "| Pillar | Weight | Score | Status |\n",
            "|---|---|---|---|\n",
        ]
        for key, weight in RETRIEVAL_WEIGHTS.items():
            r = retrieval_results.get(key, {})
            score_val = r.get("score") or 0.0
            status = "—" if r.get("skipped") else ("PASS" if r.get("pass") else "**FAIL**")
            lines.append(
                f"| {_PILLAR_LABELS.get(key, key)} | {weight:.0%} "
                f"| {score_val:.1f} | {status} |\n"
            )

        lines += [
            "\n### Generation Tier\n\n",
            f"> Weighted score: **{generation_score}/100**\n\n",
            "| Pillar | Weight | Score | Status |\n",
            "|---|---|---|---|\n",
        ]
        for key, weight in GENERATION_WEIGHTS.items():
            r = generation_results.get(key, {})
            score_val = r.get("score") or 0.0
            status = "—" if r.get("skipped") else ("PASS" if r.get("pass") else "**FAIL**")
            lines.append(
                f"| {_PILLAR_LABELS.get(key, key)} | {weight:.0%} "
                f"| {score_val:.1f} | {status} |\n"
            )

        overall_status = "PASS" if (retrieval_score >= 60 and generation_score >= 60) else "FAIL"
        lines += [
            f"\n### Overall Score: **{overall_score}/100** — {overall_status}\n\n",
            "| Tier | Score | Pass Threshold |\n",
            "|---|---|---|\n",
            f"| Retrieval | {retrieval_score}/100 | ≥ 60 |\n",
            f"| Generation | {generation_score}/100 | ≥ 60 |\n",
            f"| **Overall** | **{overall_score}/100** | — |\n",
            "\n---\n\n",
        ]
        self._append("".join(lines))

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        self._append("\n---\n_End of report._\n")
        json_path = self.path.with_suffix(".json")
        json_path.write_text(
            json.dumps(self._data, indent=2, default=str), encoding="utf-8"
        )
