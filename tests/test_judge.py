"""Tests for rag_bench/judge/claude_judge.py."""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

import rag_bench.judge.claude_judge as judge_mod
from rag_bench.judge.claude_judge import (
    JudgeUnavailable,
    _parse_judge_output,
    _get_api_key,
    _get_base_url,
    _get_model,
    _is_azure_foundry,
    is_available,
    _call,
    evaluate_completeness,
    evaluate_hallucination,
)


# ── Helper env-variable fixtures ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure API key is not set unless explicitly patched in a test."""
    monkeypatch.delenv("RAG_BENCH_API_KEY", raising=False)
    monkeypatch.delenv("RAG_BENCH_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("RAG_BENCH_JUDGE_MODEL", raising=False)


# ── _get_api_key ──────────────────────────────────────────────────────────────

class TestGetApiKey:
    def test_returns_none_when_not_set(self):
        assert _get_api_key() is None

    def test_returns_key_when_set(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test-key")
        assert _get_api_key() == "sk-test-key"

    def test_empty_string_treated_as_none(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "")
        assert _get_api_key() is None


# ── _get_base_url ─────────────────────────────────────────────────────────────

class TestGetBaseUrl:
    def test_default_url(self):
        assert _get_base_url() == "https://api.anthropic.com"

    def test_custom_url(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_JUDGE_BASE_URL", "https://custom.endpoint.com")
        assert _get_base_url() == "https://custom.endpoint.com"


# ── _get_model ────────────────────────────────────────────────────────────────

class TestGetModel:
    def test_default_model(self):
        assert _get_model() == "claude-haiku-4-5-20251001"

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_JUDGE_MODEL", "claude-opus-5")
        assert _get_model() == "claude-opus-5"


# ── _is_azure_foundry ─────────────────────────────────────────────────────────

class TestIsAzureFoundry:
    def test_azure_url_detected(self):
        assert _is_azure_foundry("https://myservice.services.ai.azure.com/api") is True

    def test_anthropic_url_not_azure(self):
        assert _is_azure_foundry("https://api.anthropic.com") is False

    def test_custom_non_azure_url(self):
        assert _is_azure_foundry("https://my-llm-proxy.example.com") is False


# ── is_available ──────────────────────────────────────────────────────────────

class TestIsAvailable:
    def test_false_when_no_api_key(self):
        assert is_available() is False

    def test_true_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-some-key")
        assert is_available() is True


# ── _parse_judge_output ───────────────────────────────────────────────────────

class TestParseJudgeOutput:
    def test_full_pass_output(self):
        raw = "VERDICT: PASS\nSCORE: 5\nREASON: All key points covered."
        result = _parse_judge_output(raw)
        assert result["verdict"] == "PASS"
        assert result["score"] == 5
        assert "All key points covered" in result["reason"]

    def test_full_fail_output(self):
        raw = "VERDICT: FAIL\nSCORE: 2\nREASON: Missing critical information."
        result = _parse_judge_output(raw)
        assert result["verdict"] == "FAIL"
        assert result["score"] == 2
        assert "Missing critical" in result["reason"]

    def test_case_insensitive_verdict(self):
        raw = "verdict: pass\nSCORE: 4\nREASON: ok"
        result = _parse_judge_output(raw)
        assert result["verdict"] == "PASS"

    def test_missing_verdict_defaults_to_fail(self):
        raw = "SCORE: 3\nREASON: some reason"
        result = _parse_judge_output(raw)
        assert result["verdict"] == "FAIL"

    def test_missing_score_defaults_to_zero(self):
        raw = "VERDICT: PASS\nREASON: good"
        result = _parse_judge_output(raw)
        assert result["score"] == 0

    def test_missing_reason_uses_raw(self):
        raw = "VERDICT: PASS\nSCORE: 4"
        result = _parse_judge_output(raw)
        # Reason defaults to raw.strip() when REASON not present
        assert result["reason"] == raw.strip()

    def test_returns_dict_with_required_keys(self):
        raw = "VERDICT: PASS\nSCORE: 5\nREASON: good"
        result = _parse_judge_output(raw)
        assert "verdict" in result
        assert "score" in result
        assert "reason" in result

    def test_empty_string(self):
        result = _parse_judge_output("")
        assert result["verdict"] == "FAIL"
        assert result["score"] == 0

    def test_score_boundary_values(self):
        for s in range(1, 6):
            raw = f"SCORE: {s}"
            result = _parse_judge_output(raw)
            assert result["score"] == s

    def test_score_out_of_range_not_matched(self):
        raw = "SCORE: 6"
        result = _parse_judge_output(raw)
        assert result["score"] == 0  # regex requires [1-5]


# ── _call ─────────────────────────────────────────────────────────────────────

class TestCall:
    def test_raises_judge_unavailable_when_no_key(self):
        with pytest.raises(JudgeUnavailable, match="RAG_BENCH_API_KEY"):
            _call("system", "user")

    def test_raises_judge_unavailable_on_network_error(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        with patch("rag_bench.judge.claude_judge._is_azure_foundry", return_value=False):
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("name or service not known")
            with patch("anthropic.Anthropic", return_value=mock_client):
                with pytest.raises(JudgeUnavailable, match="DNS/network"):
                    _call("system", "user")

    def test_raises_judge_unavailable_on_timeout(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        with patch("rag_bench.judge.claude_judge._is_azure_foundry", return_value=False):
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("request timed out")
            with patch("anthropic.Anthropic", return_value=mock_client):
                with pytest.raises(JudgeUnavailable, match="timed out"):
                    _call("system", "user")

    def test_re_raises_other_exceptions(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        with patch("rag_bench.judge.claude_judge._is_azure_foundry", return_value=False):
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = ValueError("unknown error")
            with patch("anthropic.Anthropic", return_value=mock_client):
                with pytest.raises(ValueError, match="unknown error"):
                    _call("system", "user")

    def test_successful_call_standard_anthropic(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-real")
        with patch("rag_bench.judge.claude_judge._is_azure_foundry", return_value=False):
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text="VERDICT: PASS\nSCORE: 5\nREASON: good")]
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_msg
            with patch("anthropic.Anthropic", return_value=mock_client):
                result = _call("sys prompt", "user prompt")
                assert "VERDICT" in result

    def test_successful_call_azure_foundry(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "azure-key")
        monkeypatch.setenv("RAG_BENCH_JUDGE_BASE_URL", "https://mymodel.services.ai.azure.com")
        with patch("rag_bench.judge.claude_judge._is_azure_foundry", return_value=True):
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text="VERDICT: FAIL\nSCORE: 2\nREASON: bad")]
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_msg
            with patch("anthropic.AnthropicFoundry", return_value=mock_client):
                result = _call("sys", "usr")
                assert "VERDICT" in result

    def test_call_with_non_anthropic_base_url_passes_base_url(self, monkeypatch):
        """When base_url != api.anthropic.com, base_url kwarg is passed to Anthropic()."""
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        monkeypatch.setenv("RAG_BENCH_JUDGE_BASE_URL", "https://proxy.example.com")
        with patch("rag_bench.judge.claude_judge._is_azure_foundry", return_value=False):
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text="VERDICT: PASS\nSCORE: 4\nREASON: ok")]
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_msg
            with patch("anthropic.Anthropic", return_value=mock_client) as MockAnthropic:
                _call("s", "u")
                call_kwargs = MockAnthropic.call_args[1]
                assert "base_url" in call_kwargs


# ── evaluate_completeness ─────────────────────────────────────────────────────

class TestEvaluateCompleteness:
    def test_returns_parsed_dict(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        fake_raw = "VERDICT: PASS\nSCORE: 4\nREASON: Good"
        with patch("rag_bench.judge.claude_judge._call", return_value=fake_raw):
            result = evaluate_completeness("What is X?", "X is Y.", "X should be Y.")
            assert result["verdict"] == "PASS"
            assert result["score"] == 4

    def test_propagates_judge_unavailable(self, monkeypatch):
        monkeypatch.delenv("RAG_BENCH_API_KEY", raising=False)
        with patch("rag_bench.judge.claude_judge._call", side_effect=JudgeUnavailable("no key")):
            with pytest.raises(JudgeUnavailable):
                evaluate_completeness("Q", "A", "ideal")


# ── evaluate_hallucination ────────────────────────────────────────────────────

class TestEvaluateHallucination:
    def test_returns_parsed_dict(self, monkeypatch):
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        fake_raw = "VERDICT: FAIL\nSCORE: 2\nREASON: Hallucination detected."
        with patch("rag_bench.judge.claude_judge._call", return_value=fake_raw):
            result = evaluate_hallucination("The answer.", ["Context chunk 1", "Context chunk 2"])
            assert result["verdict"] == "FAIL"
            assert result["score"] == 2

    def test_joins_retrieved_texts_with_separator(self, monkeypatch):
        """Verify that multiple retrieved texts are joined before being passed."""
        monkeypatch.setenv("RAG_BENCH_API_KEY", "sk-test")
        fake_raw = "VERDICT: PASS\nSCORE: 5\nREASON: ok"
        captured_user = {}

        def fake_call(system_prompt, user_prompt):
            captured_user["prompt"] = user_prompt
            return fake_raw

        with patch("rag_bench.judge.claude_judge._call", side_effect=fake_call):
            evaluate_hallucination("answer", ["text-A", "text-B"])

        assert "text-A" in captured_user["prompt"]
        assert "text-B" in captured_user["prompt"]

    def test_empty_retrieved_texts(self, monkeypatch):
        fake_raw = "VERDICT: PASS\nSCORE: 5\nREASON: ok"
        with patch("rag_bench.judge.claude_judge._call", return_value=fake_raw):
            result = evaluate_hallucination("an answer", [])
            assert "verdict" in result
