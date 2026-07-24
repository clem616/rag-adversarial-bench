"""Tests for rag_bench/report/report_writer.py — ReportWriter class."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_bench.report.report_writer import ReportWriter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def writer(tmp_path):
    """Return a fresh ReportWriter writing to a temp file."""
    return ReportWriter(tmp_path / "report.md")


@pytest.fixture
def report_path(tmp_path):
    return tmp_path / "report.md"


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_creates_markdown_file(self, tmp_path):
        path = tmp_path / "sub" / "report.md"
        rw = ReportWriter(path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "report.md"
        rw = ReportWriter(path)
        assert path.exists()

    def test_header_written(self, tmp_path):
        path = tmp_path / "report.md"
        rw = ReportWriter(path)
        content = path.read_text(encoding="utf-8")
        assert "RAG-Adversarial-Bench Evaluation Report" in content

    def test_data_dict_initialized(self, tmp_path):
        path = tmp_path / "report.md"
        rw = ReportWriter(path)
        assert "sections" in rw._data
        assert "generated" in rw._data

    def test_accepts_str_path(self, tmp_path):
        path = str(tmp_path / "report.md")
        rw = ReportWriter(path)
        assert Path(path).exists()


# ── write_calibration_section ─────────────────────────────────────────────────

class TestWriteCalibrationSection:
    def _call(self, rw: ReportWriter, **kwargs):
        defaults = dict(
            p95=0.42,
            optimal_tau=0.45,
            youden_j=0.85,
            null_mean=0.15,
            null_std=0.08,
            signal_mean=0.72,
            signal_std=0.10,
            per_question=[
                {"id": "Q1", "max_sim": 0.80, "cohens_d": 1.5},
                {"id": "Q2", "max_sim": 0.35, "cohens_d": 0.3},
            ],
            roc_auc=0.92,
            roc_plot_path=None,
        )
        defaults.update(kwargs)
        rw.write_calibration_section(**defaults)

    def test_section_written_to_file(self, writer, report_path):
        rw = writer
        self._call(rw)
        content = rw.path.read_text(encoding="utf-8")
        assert "Embedding Calibration" in content
        assert "P95" in content or "p95" in content.lower() or "0.4200" in content

    def test_data_dict_updated(self, writer):
        self._call(writer)
        assert "calibration" in writer._data["sections"]
        assert writer._data["sections"]["calibration"]["roc_auc"] == pytest.approx(0.92)

    def test_roc_plot_link_included_when_path_given(self, writer):
        self._call(writer, roc_plot_path="./roc.png")
        content = writer.path.read_text(encoding="utf-8")
        assert "roc.png" in content

    def test_roc_plot_link_absent_when_none(self, writer):
        self._call(writer, roc_plot_path=None)
        content = writer.path.read_text(encoding="utf-8")
        assert "![ROC" not in content

    def test_per_question_beats_tau_yes(self, writer):
        self._call(writer, optimal_tau=0.45, per_question=[
            {"id": "Q1", "max_sim": 0.90, "cohens_d": 2.0},
        ])
        content = writer.path.read_text(encoding="utf-8")
        assert "YES" in content

    def test_per_question_beats_tau_no(self, writer):
        self._call(writer, optimal_tau=0.80, per_question=[
            {"id": "Q1", "max_sim": 0.40, "cohens_d": 0.1},
        ])
        content = writer.path.read_text(encoding="utf-8")
        assert "NO" in content


# ── write_ir_metrics_section ──────────────────────────────────────────────────

class TestWriteIrMetricsSection:
    def _per_q(self, hit=True, rank=1):
        return {
            "id": "Q1",
            "hit": hit,
            "first_relevant_rank": rank if hit else None,
            "reciprocal_rank": 1.0 / rank if hit else 0.0,
            "precision_at_k": 0.6,
        }

    def test_section_written(self, writer):
        writer.write_ir_metrics_section(
            per_question=[self._per_q()],
            aggregate_mrr=0.75,
            aggregate_hit_rate=0.80,
            aggregate_precision=0.50,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "Information Retrieval Metrics" in content

    def test_hit_shown_as_yes(self, writer):
        writer.write_ir_metrics_section(
            per_question=[self._per_q(hit=True, rank=2)],
            aggregate_mrr=0.5,
            aggregate_hit_rate=1.0,
            aggregate_precision=0.4,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "YES" in content

    def test_miss_shown_as_no(self, writer):
        writer.write_ir_metrics_section(
            per_question=[self._per_q(hit=False)],
            aggregate_mrr=0.0,
            aggregate_hit_rate=0.0,
            aggregate_precision=0.0,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "NO" in content

    def test_first_relevant_rank_none_shows_dash(self, writer):
        writer.write_ir_metrics_section(
            per_question=[self._per_q(hit=False)],
            aggregate_mrr=0.0,
            aggregate_hit_rate=0.0,
            aggregate_precision=0.0,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "—" in content

    def test_data_dict_updated(self, writer):
        writer.write_ir_metrics_section(
            per_question=[self._per_q()],
            aggregate_mrr=0.75,
            aggregate_hit_rate=0.80,
            aggregate_precision=0.50,
        )
        assert "ir" in writer._data["sections"]


# ── write_monte_carlo_section ─────────────────────────────────────────────────

class TestWriteMonteCarloSection:
    def _per_q(self):
        return {
            "id": "Q1",
            "mu_comp": 4.2,
            "sigma_comp": 0.5,
            "ci_comp": (3.8, 4.6),
            "mu_hall": 4.0,
            "sigma_hall": 0.4,
            "ci_hall": (3.7, 4.3),
            "agree_comp": 0.8,
            "agree_hall": 1.0,
        }

    def test_section_written(self, writer):
        writer.write_monte_carlo_section(per_question=[self._per_q()], passes=5)
        content = writer.path.read_text(encoding="utf-8")
        assert "Monte Carlo" in content

    def test_passes_in_header(self, writer):
        writer.write_monte_carlo_section(per_question=[self._per_q()], passes=7)
        content = writer.path.read_text(encoding="utf-8")
        assert "M=7" in content

    def test_data_dict_updated(self, writer):
        writer.write_monte_carlo_section(per_question=[self._per_q()], passes=5)
        assert "monte_carlo" in writer._data["sections"]

    def test_empty_per_question(self, writer):
        writer.write_monte_carlo_section(per_question=[], passes=5)
        content = writer.path.read_text(encoding="utf-8")
        assert "Monte Carlo" in content


# ── write_ood_section ─────────────────────────────────────────────────────────

class TestWriteOodSection:
    def _per_q(self, safe=True):
        return {
            "id": "Q1",
            "domain": "general",
            "safe": safe,
            "reason": "max_sim < tau" if safe else "max_sim >= tau",
        }

    def test_section_written(self, writer):
        writer.write_ood_section(per_question=[self._per_q()], safe_rate=1.0)
        content = writer.path.read_text(encoding="utf-8")
        assert "OOD Safe Refusal" in content

    def test_safe_shown_as_yes(self, writer):
        writer.write_ood_section(per_question=[self._per_q(safe=True)], safe_rate=1.0)
        content = writer.path.read_text(encoding="utf-8")
        assert "YES" in content

    def test_unsafe_shown_as_bold_no(self, writer):
        writer.write_ood_section(per_question=[self._per_q(safe=False)], safe_rate=0.0)
        content = writer.path.read_text(encoding="utf-8")
        assert "**NO**" in content

    def test_pass_label_when_rate_high(self, writer):
        writer.write_ood_section(per_question=[self._per_q()], safe_rate=1.0)
        content = writer.path.read_text(encoding="utf-8")
        assert "PASS" in content

    def test_fail_label_when_rate_low(self, writer):
        writer.write_ood_section(per_question=[self._per_q(safe=False)], safe_rate=0.0)
        content = writer.path.read_text(encoding="utf-8")
        assert "FAIL" in content

    def test_data_dict_updated(self, writer):
        writer.write_ood_section(per_question=[], safe_rate=0.9)
        assert "ood" in writer._data["sections"]


# ── write_multiturn_section ───────────────────────────────────────────────────

class TestWriteMultiturnSection:
    def _per_traj(self):
        return {
            "id": "T1",
            "rewrite_sims": [0.85, 0.90],
            "hit_flags": [True, False],
            "comp_per_turn": [4.0, 3.5],
        }

    def test_section_written(self, writer):
        writer.write_multiturn_section(
            per_trajectory=[self._per_traj()],
            mean_rewrite_sim=0.87,
            turn2_hit_rate=0.75,
            mean_context_decay=0.3,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "Multi-Turn" in content

    def test_hit_shown_in_table(self, writer):
        writer.write_multiturn_section(
            per_trajectory=[self._per_traj()],
            mean_rewrite_sim=0.87,
            turn2_hit_rate=0.75,
            mean_context_decay=0.3,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "HIT" in content
        assert "MISS" in content

    def test_empty_rewrite_sims_shows_dash(self, writer):
        traj = {"id": "T1", "rewrite_sims": [], "hit_flags": [], "comp_per_turn": []}
        writer.write_multiturn_section(
            per_trajectory=[traj],
            mean_rewrite_sim=0.0,
            turn2_hit_rate=0.0,
            mean_context_decay=0.0,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "—" in content

    def test_data_dict_updated(self, writer):
        writer.write_multiturn_section(
            per_trajectory=[], mean_rewrite_sim=0.0, turn2_hit_rate=0.0, mean_context_decay=0.0
        )
        assert "multiturn" in writer._data["sections"]


# ── write_poisoned_context_section ────────────────────────────────────────────

class TestWritePoisonedContextSection:
    def test_section_written(self, writer):
        writer.write_poisoned_context_section(
            per_question=[{"id": "Q1", "grounded": True, "reason": "ok"}],
            grounding_rate=1.0,
        )
        content = writer.path.read_text(encoding="utf-8")
        assert "Poisoned Context" in content

    def test_grounded_yes(self, writer):
        writer.write_poisoned_context_section(
            per_question=[{"id": "Q1", "grounded": True, "reason": "good"}],
            grounding_rate=1.0,
        )
        assert "YES" in writer.path.read_text(encoding="utf-8")

    def test_ungrounded_bold_no(self, writer):
        writer.write_poisoned_context_section(
            per_question=[{"id": "Q1", "grounded": False, "reason": "bad"}],
            grounding_rate=0.0,
        )
        assert "**NO**" in writer.path.read_text(encoding="utf-8")

    def test_pass_when_rate_high(self, writer):
        writer.write_poisoned_context_section(per_question=[], grounding_rate=1.0)
        assert "PASS" in writer.path.read_text(encoding="utf-8")

    def test_fail_when_rate_low(self, writer):
        writer.write_poisoned_context_section(per_question=[], grounding_rate=0.0)
        assert "FAIL" in writer.path.read_text(encoding="utf-8")

    def test_data_dict_updated(self, writer):
        writer.write_poisoned_context_section(per_question=[], grounding_rate=0.5)
        assert "poisoned" in writer._data["sections"]


# ── write_position_bias_section ───────────────────────────────────────────────

class TestWritePositionBiasSection:
    def _per_q(self, n_positions=3):
        sbp = {i: float(4 - i * 0.1) for i in range(n_positions)}
        return {"id": "Q1", "scores_by_position": sbp, "std_dev": 0.2}

    def test_section_written(self, writer):
        writer.write_position_bias_section(per_question=[self._per_q()])
        content = writer.path.read_text(encoding="utf-8")
        assert "Position Bias" in content

    def test_data_dict_updated(self, writer):
        writer.write_position_bias_section(per_question=[self._per_q()])
        assert "position_bias" in writer._data["sections"]

    def test_fewer_than_three_positions(self, writer):
        """Should not raise when scores_by_position has fewer than 3 entries."""
        writer.write_position_bias_section(per_question=[self._per_q(n_positions=2)])
        content = writer.path.read_text(encoding="utf-8")
        assert "Position Bias" in content

    def test_empty_per_question(self, writer):
        writer.write_position_bias_section(per_question=[])
        content = writer.path.read_text(encoding="utf-8")
        assert "Position Bias" in content


# ── write_distractor_section ──────────────────────────────────────────────────

class TestWriteDistractorSection:
    def _per_q(self):
        return {
            "id": "Q1",
            "distractor_doc": "noise_doc.pdf",
            "comp_score": 4,
            "hall_score": 5,
        }

    def test_section_written(self, writer):
        writer.write_distractor_section(per_question=[self._per_q()])
        content = writer.path.read_text(encoding="utf-8")
        assert "Distractor" in content

    def test_distractor_doc_in_table(self, writer):
        writer.write_distractor_section(per_question=[self._per_q()])
        assert "noise_doc.pdf" in writer.path.read_text(encoding="utf-8")

    def test_data_dict_updated(self, writer):
        writer.write_distractor_section(per_question=[])
        assert "distractor" in writer._data["sections"]


# ── write_dual_scorecard ──────────────────────────────────────────────────────

class TestWriteDualScorecard:
    def _retrieval(self, score=80.0, pass_val=True):
        return {
            "calibration": {"score": score, "pass": pass_val, "skipped": False},
            "ir": {"score": score, "pass": pass_val, "skipped": False},
            "position_bias": {"score": score, "pass": pass_val, "skipped": False},
            "distractor": {"score": score, "pass": pass_val, "skipped": False},
        }

    def _generation(self, score=75.0, pass_val=True):
        return {
            "monte_carlo": {"score": score, "pass": pass_val, "skipped": False},
            "ood": {"score": score, "pass": pass_val, "skipped": False},
            "multiturn": {"score": score, "pass": pass_val, "skipped": False},
            "poisoned": {"score": score, "pass": pass_val, "skipped": False},
        }

    def test_scorecard_written(self, writer):
        writer.write_dual_scorecard(self._retrieval(), self._generation())
        content = writer.path.read_text(encoding="utf-8")
        assert "Component Scorecard" in content

    def test_overall_pass_when_both_tiers_pass(self, writer):
        writer.write_dual_scorecard(self._retrieval(80), self._generation(80))
        content = writer.path.read_text(encoding="utf-8")
        assert "PASS" in content

    def test_overall_fail_when_retrieval_low(self, writer):
        writer.write_dual_scorecard(self._retrieval(30, False), self._generation(80))
        content = writer.path.read_text(encoding="utf-8")
        assert "FAIL" in content

    def test_skipped_pillar_shown_as_dash(self, writer):
        r = self._retrieval()
        r["position_bias"] = {"score": None, "pass": False, "skipped": True}
        writer.write_dual_scorecard(r, self._generation())
        content = writer.path.read_text(encoding="utf-8")
        assert "—" in content

    def test_data_dict_updated(self, writer):
        writer.write_dual_scorecard(self._retrieval(), self._generation())
        assert "dual_scorecard" in writer._data["sections"]

    def test_scores_computed_correctly(self, writer):
        writer.write_dual_scorecard(self._retrieval(80), self._generation(80))
        sc = writer._data["sections"]["dual_scorecard"]
        assert sc["retrieval_score"] == pytest.approx(80.0)
        assert sc["generation_score"] == pytest.approx(80.0)
        assert sc["overall_score"] == pytest.approx(80.0)

    def test_empty_retrieval_results(self, writer):
        """When no results are available, score should be 0.0."""
        writer.write_dual_scorecard({}, self._generation(60))
        sc = writer._data["sections"]["dual_scorecard"]
        assert sc["retrieval_score"] == pytest.approx(0.0)

    def test_all_none_scores_returns_zero(self, writer):
        r = {k: {"score": None, "pass": False, "skipped": True} for k in ["calibration", "ir", "position_bias", "distractor"]}
        writer.write_dual_scorecard(r, self._generation(60))
        sc = writer._data["sections"]["dual_scorecard"]
        assert sc["retrieval_score"] == pytest.approx(0.0)

    def test_unknown_keys_with_scores_gives_zero_weighted(self, writer):
        """Keys present with non-None scores but absent from RETRIEVAL_WEIGHTS → total_w=0 → 0.0."""
        # This hits the `if total_w == 0: return 0.0` branch in the inner weighted_score.
        r = {"nonexistent_key": {"score": 80.0, "pass": True, "skipped": False}}
        writer.write_dual_scorecard(r, self._generation(60))
        sc = writer._data["sections"]["dual_scorecard"]
        assert sc["retrieval_score"] == pytest.approx(0.0)


# ── finalize ──────────────────────────────────────────────────────────────────

class TestFinalize:
    def test_finalize_appends_end_marker(self, writer):
        writer.finalize()
        content = writer.path.read_text(encoding="utf-8")
        assert "End of report" in content

    def test_finalize_writes_json_sidecar(self, writer):
        writer.finalize()
        json_path = writer.path.with_suffix(".json")
        assert json_path.exists()

    def test_json_sidecar_valid_json(self, writer):
        writer.finalize()
        json_path = writer.path.with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "generated" in data
        assert "sections" in data

    def test_json_includes_all_written_sections(self, writer):
        writer.write_ood_section(
            per_question=[{"id": "Q1", "domain": "test", "safe": True, "reason": "ok"}],
            safe_rate=1.0,
        )
        writer.finalize()
        json_path = writer.path.with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "ood" in data["sections"]
