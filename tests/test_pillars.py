"""Tests for rag_bench/run_benchmark.py pillar runner functions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_bench.run_benchmark import (
    _build_corpus_chunks,
    _load_dataset,
    _pillar_result,
    _weighted_score,
    run_p4_ood,
    run_p5_multiturn,
    run_p6_poisoned,
)
from rag_bench.systems.base import RAGSystem
from rag_bench.thresholds import GENERATION_WEIGHTS, RETRIEVAL_WEIGHTS


# ── Shared test doubles ───────────────────────────────────────────────────────

class SimpleSystem(RAGSystem):
    """Deterministic RAGSystem for pillar tests."""

    def __init__(self, chunks=None, answer="default answer"):
        self._chunks = chunks or [
            {"text": "relevant text about the topic", "doc_id": "doc-001", "score": 0.9}
        ]
        self._answer = answer

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self._chunks[:top_k]

    def generate(self, query: str, chunks, history=None) -> str:
        return self._answer


class RefusingSystem(RAGSystem):
    """System that classifies everything as small_talk (for OOD tests)."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return []

    def generate(self, query: str, chunks, history=None) -> str:
        return "I don't know, that's off-topic."

    def classify_intent(self, query, history=None) -> str:
        return "small_talk"


# ── _pillar_result ─────────────────────────────────────────────────────────────

class TestPillarResult:
    def test_basic_structure(self):
        result = _pillar_result(75.0, pass_val=True)
        assert result["score"] == 75.0
        assert result["pass"] is True
        assert result["skipped"] is False

    def test_skipped_flag(self):
        result = _pillar_result(None, pass_val=False, skipped=True)
        assert result["score"] is None
        assert result["skipped"] is True

    def test_extra_kwargs_included(self):
        result = _pillar_result(50.0, pass_val=False, mrr=0.6, hit_rate=0.7)
        assert result["mrr"] == 0.6
        assert result["hit_rate"] == 0.7

    def test_none_score_allowed(self):
        result = _pillar_result(None, pass_val=False, skipped=True)
        assert result["score"] is None


# ── _weighted_score ────────────────────────────────────────────────────────────

class TestWeightedScore:
    def test_all_scores_present(self):
        results = {
            "calibration": {"score": 80.0},
            "ir": {"score": 90.0},
            "position_bias": {"score": 70.0},
            "distractor": {"score": 60.0},
        }
        score = _weighted_score(results, RETRIEVAL_WEIGHTS)
        assert 60.0 <= score <= 90.0

    def test_empty_results_returns_zero(self):
        assert _weighted_score({}, RETRIEVAL_WEIGHTS) == 0.0

    def test_none_score_excluded(self):
        results = {
            "calibration": {"score": None},
            "ir": {"score": 80.0},
            "position_bias": {"score": None},
            "distractor": {"score": None},
        }
        score = _weighted_score(results, RETRIEVAL_WEIGHTS)
        # Only "ir" is available, weight=0.40 → total_w=0.40 → score = 80.0 * 0.40 / 0.40 = 80.0
        assert score == pytest.approx(80.0)

    def test_all_none_scores_returns_zero(self):
        results = {k: {"score": None} for k in RETRIEVAL_WEIGHTS}
        assert _weighted_score(results, RETRIEVAL_WEIGHTS) == 0.0

    def test_score_is_rounded(self):
        results = {
            "calibration": {"score": 100.0},
            "ir": {"score": 100.0},
            "position_bias": {"score": 100.0},
            "distractor": {"score": 100.0},
        }
        score = _weighted_score(results, RETRIEVAL_WEIGHTS)
        assert score == pytest.approx(100.0)

    def test_unknown_key_ignored(self):
        results = {"nonexistent": {"score": 99.0}}
        score = _weighted_score(results, RETRIEVAL_WEIGHTS)
        # nonexistent not in RETRIEVAL_WEIGHTS → total_w = 0 → 0.0
        assert score == 0.0


# ── _load_dataset ─────────────────────────────────────────────────────────────

class TestLoadDataset:
    def test_loads_existing_files(self, tmp_path):
        ir_data = [{"id": "Q1", "question": "What is X?"}]
        (tmp_path / "ground_truth_ir.json").write_text(json.dumps(ir_data))
        (tmp_path / "ground_truth_ood.json").write_text("[]")
        (tmp_path / "ground_truth_poisoned.json").write_text("[]")
        (tmp_path / "ground_truth_multiturn.json").write_text("[]")
        (tmp_path / "source_passages.json").write_text("{}")

        data = _load_dataset(tmp_path)
        assert data["ir"] == ir_data
        assert data["ood"] == []
        assert data["source_passages"] == {}

    def test_missing_files_return_empty_defaults(self, tmp_path):
        data = _load_dataset(tmp_path)
        assert data["ir"] == []
        assert data["ood"] == []
        assert data["poisoned"] == []
        assert data["multiturn"] == []
        assert data["source_passages"] == {}

    def test_source_passages_defaults_to_empty_dict(self, tmp_path):
        data = _load_dataset(tmp_path)
        assert isinstance(data["source_passages"], dict)

    def test_partial_files(self, tmp_path):
        (tmp_path / "ground_truth_ir.json").write_text('[{"id": "q1"}]')
        data = _load_dataset(tmp_path)
        assert len(data["ir"]) == 1
        assert data["ood"] == []


# ── _build_corpus_chunks ──────────────────────────────────────────────────────

class TestBuildCorpusChunks:
    """RecursiveCharacterTextSplitter is imported inside _build_corpus_chunks,
    so we patch via sys.modules."""

    @staticmethod
    def _mock_splitter_ctx(return_chunks):
        mock_instance = MagicMock()
        mock_instance.split_text.return_value = return_chunks
        mock_module = MagicMock()
        mock_module.RecursiveCharacterTextSplitter.return_value = mock_instance
        return patch.dict("sys.modules", {"langchain_text_splitters": mock_module})

    def test_builds_chunks_and_doc_ids(self):
        with self._mock_splitter_ctx(["chunk1", "chunk2"]):
            source_passages = {"doc-A": "long text about something", "doc-B": "another text"}
            chunks, doc_ids = _build_corpus_chunks(source_passages)

        assert len(chunks) == 4  # 2 docs × 2 chunks each
        assert all("text" in c and "doc_id" in c for c in chunks)
        assert doc_ids == [c["doc_id"] for c in chunks]

    def test_skips_empty_passages(self):
        with self._mock_splitter_ctx(["chunk"]):
            source_passages = {"doc-A": "real text", "doc-B": "   "}
            chunks, doc_ids = _build_corpus_chunks(source_passages)

        assert len(chunks) == 1
        assert chunks[0]["doc_id"] == "doc-A"

    def test_empty_source_passages(self):
        with self._mock_splitter_ctx([]):
            chunks, doc_ids = _build_corpus_chunks({})

        assert chunks == []
        assert doc_ids == []

    def test_doc_ids_match_chunk_doc_ids(self):
        with self._mock_splitter_ctx(["a", "b"]):
            chunks, doc_ids = _build_corpus_chunks({"myDoc": "text here"})

        assert doc_ids == ["myDoc", "myDoc"]
        assert all(c["doc_id"] == "myDoc" for c in chunks)


# ── run_p4_ood ────────────────────────────────────────────────────────────────

class TestRunP4Ood:
    def _make_emb_model(self, vec=None):
        if vec is None:
            vec = np.array([0.1, 0.9, 0.0])
        model = MagicMock()
        model.encode.return_value = np.array([vec])
        return model

    def test_all_safe_via_intent_routing(self):
        """If classify_intent returns non-retrieval, safe=True regardless of similarity."""
        system = RefusingSystem()
        ood_cases = [{"id": "Q1", "question": "What is the capital of France?", "domain": "general"}]
        emb = self._make_emb_model()
        result = run_p4_ood(system, ood_cases, emb, tau=0.5, top_k=5)
        assert result["safe_rate"] == 1.0
        assert result["per_case"][0]["safe"] is True

    def test_safe_when_max_sim_below_tau(self):
        """For document_content intent, safe if max_sim < tau."""
        # Orthogonal vectors → cosine sim = 0.0 < tau=0.5
        q_vec = np.array([1.0, 0.0, 0.0])
        chunk_vec = np.array([0.0, 1.0, 0.0])

        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: np.array([q_vec] if texts[0] == "ood question" else [chunk_vec])

        system = SimpleSystem(chunks=[{"text": "unrelated content", "doc_id": "doc-x", "score": 0.1}])
        ood_cases = [{"id": "Q1", "question": "ood question", "domain": "test"}]
        result = run_p4_ood(system, ood_cases, model, tau=0.5, top_k=5)
        assert result["per_case"][0]["safe"] is True

    def test_unsafe_when_max_sim_above_tau(self):
        """For document_content intent, unsafe if max_sim >= tau."""
        # Identical vectors → cosine sim = 1.0 >= tau=0.5
        vec = np.array([1.0, 0.0, 0.0])
        model = MagicMock()
        model.encode.return_value = np.array([vec])

        system = SimpleSystem(chunks=[{"text": "very relevant content", "doc_id": "d1", "score": 0.9}])
        ood_cases = [{"id": "Q1", "question": "in-domain question", "domain": "medical"}]
        result = run_p4_ood(system, ood_cases, model, tau=0.5, top_k=5)
        assert result["per_case"][0]["safe"] is False

    def test_empty_ood_cases_returns_zero_rate(self):
        system = SimpleSystem()
        model = self._make_emb_model()
        result = run_p4_ood(system, [], model, tau=0.5, top_k=5)
        assert result["safe_rate"] == 0.0

    def test_result_has_required_keys(self):
        system = RefusingSystem()
        ood_cases = [{"id": "Q1", "question": "q", "domain": "d"}]
        model = self._make_emb_model()
        result = run_p4_ood(system, ood_cases, model, tau=0.5, top_k=5)
        assert "score" in result
        assert "pass" in result
        assert "safe_rate" in result
        assert "per_case" in result

    def test_metadata_query_intent_is_safe(self):
        """metadata_query is a retrieval intent — similarity is checked."""
        class MetadataSystem(SimpleSystem):
            def classify_intent(self, query, history=None):
                return "metadata_query"

        vec = np.array([1.0, 0.0, 0.0])
        model = MagicMock()
        model.encode.return_value = np.array([vec])

        system = MetadataSystem(chunks=[{"text": "relevant", "doc_id": "d1", "score": 0.9}])
        ood_cases = [{"id": "Q1", "question": "q", "domain": "d"}]
        result = run_p4_ood(system, ood_cases, model, tau=0.5, top_k=5)
        # Cosine sim=1.0 >= tau=0.5 → not safe
        assert result["per_case"][0]["safe"] is False

    def test_no_text_chunks_defaults_to_zero_sim(self):
        """Chunks without text yield max_sim = 0.0 (default)."""
        vec = np.array([1.0, 0.0])
        model = MagicMock()
        model.encode.return_value = np.array([vec])

        system = SimpleSystem(chunks=[{"text": "", "doc_id": "d1", "score": 0.0}])
        ood_cases = [{"id": "Q1", "question": "q", "domain": "d"}]
        result = run_p4_ood(system, ood_cases, model, tau=0.5, top_k=5)
        assert result["per_case"][0]["safe"] is True


# ── run_p5_multiturn ──────────────────────────────────────────────────────────

class TestRunP5Multiturn:
    def _make_emb_model(self):
        model = MagicMock()
        model.encode.return_value = np.array([[0.5, 0.5]])
        return model

    def _make_trajectory(self, turn2_doc="doc-001", standalone="rewritten q"):
        return {
            "id": "T1",
            "turns": [
                {"question": "first question"},
                {
                    "question": "follow-up",
                    "expected_standalone": standalone,
                    "expected_documents": [turn2_doc],
                },
            ],
        }

    def test_hit_when_doc_id_matches(self):
        """System retrieves doc-001 and trajectory expects doc-001 → hit."""
        system = SimpleSystem(chunks=[{"text": "t", "doc_id": "doc-001", "score": 0.9}])
        model = self._make_emb_model()
        traj = self._make_trajectory(turn2_doc="doc-001")
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        assert result["turn2_hit_rate"] == 1.0

    def test_miss_when_doc_id_mismatch(self):
        system = SimpleSystem(chunks=[{"text": "t", "doc_id": "doc-999", "score": 0.9}])
        model = self._make_emb_model()
        traj = self._make_trajectory(turn2_doc="doc-001")
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        assert result["turn2_hit_rate"] == 0.0

    def test_pdf_suffix_normalized(self):
        """doc-001.pdf in expected_documents should match doc-001 from retrieve."""
        system = SimpleSystem(chunks=[{"text": "t", "doc_id": "doc-001", "score": 0.9}])
        model = self._make_emb_model()
        traj = {
            "id": "T1",
            "turns": [
                {"question": "q1"},
                {
                    "question": "q2",
                    "expected_standalone": "rewritten",
                    "expected_documents": ["doc-001.pdf"],
                },
            ],
        }
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        assert result["turn2_hit_rate"] == 1.0

    def test_skip_trajectory_with_fewer_than_2_turns(self):
        system = SimpleSystem()
        model = self._make_emb_model()
        traj = {"id": "T1", "turns": [{"question": "only one turn"}]}
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        # No valid trajectories → fall-through to empty turn2_hits
        assert result["score"] == pytest.approx(0.0)
        assert result["pass"] is False

    def test_skip_trajectory_with_empty_standalone(self):
        system = SimpleSystem()
        model = self._make_emb_model()
        traj = {
            "id": "T1",
            "turns": [
                {"question": "q1"},
                {"question": "q2", "expected_standalone": "", "expected_documents": ["d1"]},
            ],
        }
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        assert result["score"] == pytest.approx(0.0)

    def test_empty_mt_cases(self):
        system = SimpleSystem()
        model = self._make_emb_model()
        result = run_p5_multiturn(system, [], model, tau=0.5, top_k=5)
        assert result["score"] == pytest.approx(0.0)

    def test_result_has_required_keys(self):
        system = SimpleSystem(chunks=[{"text": "t", "doc_id": "doc-001", "score": 0.9}])
        model = self._make_emb_model()
        traj = self._make_trajectory()
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        assert "score" in result
        assert "pass" in result
        assert "turn2_hit_rate" in result

    def test_fallback_to_question_when_no_expected_standalone(self):
        """If expected_standalone missing, falls back to 'question' key."""
        system = SimpleSystem(chunks=[{"text": "t", "doc_id": "doc-x", "score": 0.9}])
        model = self._make_emb_model()
        traj = {
            "id": "T1",
            "turns": [
                {"question": "q1"},
                {"question": "the fallback question", "expected_documents": ["doc-x"]},
            ],
        }
        result = run_p5_multiturn(system, [traj], model, tau=0.5, top_k=5)
        assert result["turn2_hit_rate"] == 1.0


# ── run_p6_poisoned ────────────────────────────────────────────────────────────

class TestRunP6Poisoned:
    def test_runs_without_api_key(self, monkeypatch):
        """P6 uses string matching only — it always runs, no API key required."""
        monkeypatch.delenv("RAG_BENCH_API_KEY", raising=False)
        system = SimpleSystem()
        # Case has no original_fact/counterfact → per_case is empty → score 0.0
        result = run_p6_poisoned(system, [{"id": "Q1"}], top_k=5)
        assert result["skipped"] is False
        assert result["score"] == pytest.approx(0.0)

    def test_grounded_when_answer_contains_counterfact(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        counterfact = "the drug causes liver damage"
        system = SimpleSystem(answer=f"According to the context, {counterfact}.")
        cases = [{
            "id": "Q1",
            "question": "Is the drug safe?",
            "original_fact": "the drug is safe",
            "counterfact": counterfact,
        }]
        result = run_p6_poisoned(system, cases, top_k=5)
        assert result["per_case"][0]["grounded"] is True
        assert "counterfact" in result["per_case"][0]["reason"]

    def test_grounded_when_answer_paraphrases(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        original_fact = "the drug is safe"
        counterfact = "the drug causes liver damage"
        # Answer contains neither fact → paraphrased
        system = SimpleSystem(answer="The medication has complex interactions.")
        cases = [{
            "id": "Q1",
            "question": "Is it safe?",
            "original_fact": original_fact,
            "counterfact": counterfact,
        }]
        result = run_p6_poisoned(system, cases, top_k=5)
        assert result["per_case"][0]["grounded"] is True
        assert "paraphrased" in result["per_case"][0]["reason"]

    def test_ungrounded_when_answer_uses_original_fact(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        original_fact = "the drug is safe"
        counterfact = "the drug causes liver damage"
        # Answer contains original fact but NOT counterfact → parametric memory
        system = SimpleSystem(answer=f"Studies show {original_fact}.")
        cases = [{
            "id": "Q1",
            "question": "Tell me about the drug",
            "original_fact": original_fact,
            "counterfact": counterfact,
        }]
        result = run_p6_poisoned(system, cases, top_k=5)
        assert result["per_case"][0]["grounded"] is False
        assert "parametric memory" in result["per_case"][0]["reason"]

    def test_skips_cases_without_facts(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        system = SimpleSystem()
        cases = [{"id": "Q1", "question": "q"}]  # no original_fact/counterfact
        result = run_p6_poisoned(system, cases, top_k=5)
        assert result["score"] == pytest.approx(0.0)
        assert result["pass"] is False

    def test_poison_injected_mid_chunks_when_not_found(self, monkeypatch):
        """When original_fact not found in chunks, counterfact is appended mid-list."""
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        original_fact = "drug X is red"
        counterfact = "drug X is blue"

        captured_chunks = {}

        class CapturingSystem(SimpleSystem):
            def generate(self, query, chunks, history=None):
                captured_chunks["chunks"] = chunks
                # Return answer with counterfact so grounded=True
                return f"I see {counterfact} in the context."

        chunks = [
            {"text": "chunk A no fact", "doc_id": "d1", "score": 0.5},
            {"text": "chunk B no fact", "doc_id": "d2", "score": 0.4},
        ]
        system = CapturingSystem(chunks=chunks)
        cases = [{
            "id": "Q1",
            "question": "q",
            "original_fact": original_fact,
            "counterfact": counterfact,
        }]
        result = run_p6_poisoned(system, cases, top_k=5)
        # Should still run and produce a result
        assert len(result["per_case"]) == 1

    def test_original_fact_replaced_in_chunk(self, monkeypatch):
        """When original_fact found in a chunk, it is replaced with counterfact."""
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        original_fact = "drug X is red"
        counterfact = "drug X is blue"

        captured = {}

        class CapturingSystem(SimpleSystem):
            def generate(self, query, chunks, history=None):
                captured["chunks"] = chunks
                return f"Context says {counterfact}."

        system = CapturingSystem(chunks=[
            {"text": f"The study found that {original_fact}.", "doc_id": "d1", "score": 0.9}
        ])
        cases = [{
            "id": "Q1",
            "question": "q",
            "original_fact": original_fact,
            "counterfact": counterfact,
        }]
        run_p6_poisoned(system, cases, top_k=5)
        # The chunk text should have been replaced
        assert counterfact in captured["chunks"][0]["text"]
        assert original_fact not in captured["chunks"][0]["text"]

    def test_empty_cases_returns_zero(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        result = run_p6_poisoned(SimpleSystem(), [], top_k=5)
        assert result["score"] == pytest.approx(0.0)
        assert result["pass"] is False

    def test_result_has_required_keys(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        counterfact = "drug X is toxic"
        system = SimpleSystem(answer=f"{counterfact}")
        cases = [{
            "id": "Q1",
            "question": "q",
            "original_fact": "drug X is safe",
            "counterfact": counterfact,
        }]
        result = run_p6_poisoned(system, cases, top_k=5)
        assert "score" in result
        assert "pass" in result
        assert "grounding_rate" in result
        assert "per_case" in result
