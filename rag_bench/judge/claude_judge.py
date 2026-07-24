"""Independent LLM judge backed by the Anthropic Claude API.

Completely separate from any RAG pipeline under test — eliminating circular bias.
Uses the Anthropic Messages API directly via HTTP.

Configuration (environment variables):
    RAG_BENCH_JUDGE_BASE_URL  — defaults to https://api.anthropic.com
    RAG_BENCH_JUDGE_MODEL     — defaults to claude-haiku-4-5-20251001
                                Set to claude-sonnet-4-6 for official/publication runs.
    RAG_BENCH_API_KEY         — required; raises JudgeUnavailable if absent
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _get_api_key() -> str | None:
    return os.environ.get("RAG_BENCH_API_KEY") or None


def _get_base_url() -> str:
    return os.environ.get("RAG_BENCH_JUDGE_BASE_URL", "https://api.anthropic.com")


def _get_model() -> str:
    return os.environ.get("RAG_BENCH_JUDGE_MODEL", "claude-haiku-4-5-20251001")


def _is_azure_foundry(url: str) -> bool:
    return "services.ai.azure.com" in url


class JudgeUnavailable(Exception):
    """Raised when RAG_BENCH_API_KEY is not set."""


def is_available() -> bool:
    return bool(_get_api_key())


def _call(system_prompt: str, user_prompt: str) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise JudgeUnavailable(
            "RAG_BENCH_API_KEY is not set — Claude judge is unavailable."
        )
    base_url = _get_base_url()
    model = _get_model()

    try:
        if _is_azure_foundry(base_url):
            from anthropic import AnthropicFoundry
            client = AnthropicFoundry(api_key=api_key, base_url=base_url)
        else:
            from anthropic import Anthropic
            kwargs: dict = {"api_key": api_key}
            if "api.anthropic.com" not in base_url:
                kwargs["base_url"] = base_url
            client = Anthropic(**kwargs)

        msg = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    except Exception as exc:
        err = str(exc)
        if "name or service not known" in err.lower() or "nameresolution" in err.lower():
            raise JudgeUnavailable(f"Claude API unreachable (DNS/network): {exc}") from exc
        if "timed out" in err.lower():
            raise JudgeUnavailable(f"Claude API timed out: {exc}") from exc
        raise


def _parse_judge_output(raw: str) -> dict:
    verdict = "FAIL"
    score = 0
    reason = raw.strip()

    m = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", raw, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()

    m = re.search(r"SCORE\s*:\s*([1-5])", raw)
    if m:
        score = int(m.group(1))

    m = re.search(r"REASON\s*:\s*(.+)", raw, re.IGNORECASE)
    if m:
        reason = m.group(1).strip()

    return {"verdict": verdict, "score": score, "reason": reason}


def _load_prompt(name: str) -> dict:
    import yaml
    path = PROMPTS_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _render(template: str, context: dict) -> str:
    for key, value in context.items():
        template = template.replace(f"{{{{ {key} }}}}", str(value))
    return template


def evaluate_completeness(
    question: str,
    generated_answer: str,
    ideal_answer_points: str,
) -> dict:
    """Completeness judge: does the answer cover the required key points?"""
    prompt = _load_prompt("judge_completeness")
    system = prompt["system_prompt"]
    user = _render(prompt["user_prompt"], {
        "question": question,
        "ideal_answer_points": ideal_answer_points,
        "generated_answer": generated_answer,
    })
    raw = _call(system, user)
    return _parse_judge_output(raw)


def evaluate_hallucination(
    generated_answer: str,
    retrieved_texts: list[str],
) -> dict:
    """Hallucination judge: is the answer grounded in the retrieved context?"""
    prompt = _load_prompt("judge_hallucination")
    system = prompt["system_prompt"]
    combined_context = "\n\n---\n\n".join(retrieved_texts)
    user = _render(prompt["user_prompt"], {
        "combined_context": combined_context,
        "generated_answer": generated_answer,
    })
    raw = _call(system, user)
    return _parse_judge_output(raw)
