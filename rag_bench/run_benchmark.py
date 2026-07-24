"""Public benchmark runner for RAG-Adversarial-Bench.

Usage:
    python -m rag_bench.run_benchmark \\
        --system langchain_faiss \\
        --dataset ./rag_bench/datasets/medical \\
        --output ./results/langchain_faiss.json

    # With extra options
    python -m rag_bench.run_benchmark \\
        --system my_package.MyRAGSystem \\
        --dataset ./rag_bench/datasets/medical \\
        --output ./results/my_system.json \\
        --top-k 5 \\
        --monte-carlo-passes 3 \\
        --skip-judge          # skip P3/P7/P8 (saves API cost; P6 always runs)

Supported --system values:
    langchain_faiss                built-in reference (LangChain + FAISS + all-MiniLM-L6-v2)
    dotted.module.path.ClassName   any RAGSystem subclass; must accept no constructor args

Environment variables:
    RAG_BENCH_API_KEY        Anthropic API key (required for P3/P7/P8 LLM judge calls).
                             P6 (Poisoned Context) uses string matching — no API key needed.
    RAG_BENCH_JUDGE_BASE_URL Override judge endpoint URL (default: https://api.anthropic.com).
    RAG_BENCH_JUDGE_MODEL    Override judge model (default: claude-haiku-4-5-20251001).
                             Can also be set via --judge-model CLI flag.
    MONTE_CARLO_PASSES       Override MC passes (default: 5)

Subset notes:
    P3 (Monte Carlo) evaluates the first 50 IR cases rather than the full set to bound API cost
    (50 cases × M passes × 2 judge calls per pass). P7/P8 evaluate the first 20 P2 hit cases.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rag_bench.judge.claude_judge import (
    JudgeUnavailable,
    evaluate_completeness,
    evaluate_hallucination,
    is_available as _judge_available,
)
from rag_bench.pillars.retrieval.ir_metrics import is_chunk_relevant
from rag_bench.report.report_writer import ReportWriter
from rag_bench.stats.stats_utils import (
    compute_confidence_interval,
    compute_hit_rate,
    compute_mrr,
    compute_null_threshold,
    compute_optimal_threshold_youden,
    compute_precision_at_k,
    compute_roc_auc,
    cosine_similarity,
    judge_agreement_rate,
)
from rag_bench.systems.base import RAGSystem
from rag_bench.thresholds import (
    CI_LEVEL,
    DISTRACTOR_COMPLETENESS_MIN,
    DISTRACTOR_HALLUCINATION_MIN,
    GENERATION_WEIGHTS,
    HIT_RATE_MIN,
    MONTE_CARLO_PASSES as _DEFAULT_MC_PASSES,
    MRR_MIN,
    MULTITURN_HIT_RATE_MIN,
    NULL_ALPHA,
    NULL_SAMPLE_PAIRS,
    OOD_SAFE_REFUSAL_RATE_MIN,
    POISONING_GROUNDING_RATE_MIN,
    POSITION_BIAS_MAX_COMP_STD,
    RETRIEVAL_WEIGHTS,
    ROC_AUC_MIN,
    SCORE_MEAN_MIN,
    TOP_K,
)

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

random.seed(42)
np.random.seed(42)

# P3 is capped at the first N IR cases to bound API cost (N × M passes × 2 judge calls).
# P7/P8 are capped at the first 20 P2 hit cases (adversarial splits are 20 cases each).
_P3_IR_CASES = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_embedding_model():
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


def _load_dataset(dataset_dir: Path) -> dict:
    files = {
        "ir": "ground_truth_ir.json",
        "ood": "ground_truth_ood.json",
        "poisoned": "ground_truth_poisoned.json",
        "multiturn": "ground_truth_multiturn.json",
        "source_passages": "source_passages.json",
    }
    data: dict[str, Any] = {}
    for key, fname in files.items():
        path = dataset_dir / fname
        if path.exists():
            data[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            data[key] = [] if key != "source_passages" else {}
    return data


def _build_corpus_chunks(source_passages: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Re-chunk all source passages and return (chunks, doc_ids)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    chunks: list[dict] = []
    for doc_id, text in source_passages.items():
        if not text.strip():
            continue
        for chunk_text in splitter.split_text(text):
            chunks.append({"text": chunk_text, "doc_id": doc_id})
    doc_ids = [c["doc_id"] for c in chunks]
    return chunks, doc_ids


def _pillar_result(score: float | None, *, pass_val: bool, skipped: bool = False, **extras) -> dict:
    return {"score": score, "pass": pass_val, "skipped": skipped, **extras}


def _weighted_score(results: dict, weights: dict[str, float]) -> float:
    available = {k: v for k, v in results.items() if v.get("score") is not None}
    if not available:
        return 0.0
    total_w = sum(weights[k] for k in available if k in weights)
    if total_w == 0:
        return 0.0
    return round(
        sum(available[k]["score"] * weights[k] / total_w for k in available if k in weights), 1
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── P1: Embedding Calibration ─────────────────────────────────────────────────

def run_p1_calibration(
    corpus_chunks: list[dict],
    corpus_doc_ids: list[str],
    ir_cases: list[dict],
    embedding_model: Any,
    precomputed_embs: "np.ndarray | None" = None,
) -> dict:
    _log("  [P1] Embedding calibration ...")
    if precomputed_embs is not None:
        all_embs = precomputed_embs
    else:
        texts = [c["text"] for c in corpus_chunks]
        all_embs = embedding_model.encode(texts, convert_to_numpy=True).astype(np.float32)

    p95, null_scores = compute_null_threshold(
        all_embs, n_pairs=NULL_SAMPLE_PAIRS, alpha=NULL_ALPHA, doc_numbers=corpus_doc_ids
    )

    signal_scores: list[float] = []
    for case in ir_cases:
        gt_text = case.get("relevance_context_for_ir") or case.get("ground_truth_context", "")
        if not gt_text:
            continue
        gt_emb = embedding_model.encode([gt_text], convert_to_numpy=True)[0]
        expected_ids = set(case.get("expected_documents", []))
        for idx, doc_id in enumerate(corpus_doc_ids):
            if doc_id in expected_ids:
                signal_scores.append(cosine_similarity(gt_emb, all_embs[idx]))

    if not signal_scores:
        return _pillar_result(0.0, pass_val=False, reason="No signal pairs found")

    signal_arr = np.array(signal_scores, dtype=np.float64)
    optimal_tau, youden_j = compute_optimal_threshold_youden(null_scores, signal_arr)
    roc_auc, _, _ = compute_roc_auc(null_scores, signal_arr)

    score = min(100.0, roc_auc / ROC_AUC_MIN * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=roc_auc >= ROC_AUC_MIN,
        roc_auc=roc_auc,
        optimal_tau=optimal_tau,
        youden_j=youden_j,
        p95=p95,
    )


# ── P2: IR Metrics ────────────────────────────────────────────────────────────

def run_p2_ir(
    system: RAGSystem,
    ir_cases: list[dict],
    embedding_model: Any,
    tau: float,
    top_k: int,
) -> dict:
    _log(f"  [P2] IR metrics ({len(ir_cases)} cases) ...")
    per_case: list[dict] = []

    for case in ir_cases:
        gt_text = case.get("relevance_context_for_ir") or case.get("ground_truth_context", "")
        gt_emb = embedding_model.encode([gt_text], convert_to_numpy=True)[0] if gt_text else None
        expected_ids = set(case.get("expected_documents", []))

        chunks = system.retrieve(case["question"], top_k=top_k)
        first_rel_rank = None
        rel_count = 0
        first_rel_chunk = None

        for rank, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            doc_id = chunk.get("doc_id", "")
            if gt_emb is not None and text and is_chunk_relevant(
                text, gt_emb, expected_ids, doc_id, tau, embedding_model
            ):
                rel_count += 1
                if first_rel_rank is None:
                    first_rel_rank = rank
                    first_rel_chunk = chunk

        hit = first_rel_rank is not None
        per_case.append({
            "id": case["id"],
            "hit": hit,
            "first_relevant_rank": first_rel_rank,
            "reciprocal_rank": (1.0 / first_rel_rank) if hit else 0.0,
            "precision_at_k": compute_precision_at_k(rel_count, len(chunks)),
            "first_relevant_chunk": first_rel_chunk,
            "retrieved_chunks": chunks,
        })

    hits = sum(1 for r in per_case if r["hit"])
    total = len(per_case)
    mrr = compute_mrr([r["reciprocal_rank"] for r in per_case])
    hit_rate = compute_hit_rate(hits, total)
    avg_p = float(np.mean([r["precision_at_k"] for r in per_case])) if per_case else 0.0

    score = min(100.0, (hit_rate / HIT_RATE_MIN + mrr / MRR_MIN) / 2 * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(mrr >= MRR_MIN and hit_rate >= HIT_RATE_MIN),
        mrr=mrr,
        hit_rate=hit_rate,
        avg_precision=avg_p,
        per_case=per_case,
    )


# ── P3: Monte Carlo Stability ─────────────────────────────────────────────────

def run_p3_monte_carlo(
    system: RAGSystem,
    ir_cases: list[dict],
    top_k: int,
    passes: int,
) -> dict:
    if not _judge_available():
        _log("  [P3] Monte Carlo SKIPPED (RAG_BENCH_API_KEY not set)")
        return _pillar_result(None, pass_val=False, skipped=True)

    _log(f"  [P3] Monte Carlo ({len(ir_cases)} cases × {passes} passes) ...")
    per_case: list[dict] = []
    all_comp: list[float] = []

    for case in ir_cases:
        comp_scores: list[int] = []
        hall_scores: list[int] = []
        comp_verdicts: list[str] = []
        hall_verdicts: list[str] = []

        for _ in range(passes):
            chunks = system.retrieve(case["question"], top_k=top_k)
            answer = system.generate(case["question"], chunks)
            texts = [c.get("text", "") for c in chunks]
            try:
                c_res = evaluate_completeness(
                    case["question"], answer, case.get("ideal_answer_points", "")
                )
                h_res = evaluate_hallucination(answer, texts)
                comp_scores.append(c_res["score"])
                hall_scores.append(h_res["score"])
                comp_verdicts.append(c_res["verdict"])
                hall_verdicts.append(h_res["verdict"])
            except JudgeUnavailable:
                return _pillar_result(None, pass_val=False, skipped=True)

        mu_comp = float(np.mean(comp_scores))
        sigma_comp = float(np.std(comp_scores))
        all_comp.extend(comp_scores)
        ci_comp = compute_confidence_interval(comp_scores, CI_LEVEL)
        per_case.append({
            "id": case["id"],
            "mu_comp": mu_comp,
            "sigma_comp": sigma_comp,
            "ci_comp": ci_comp,
            "mu_hall": float(np.mean(hall_scores)),
            "sigma_hall": float(np.std(hall_scores)),
            "ci_hall": compute_confidence_interval(hall_scores, CI_LEVEL),
            "agree_comp": judge_agreement_rate(comp_verdicts),
            "agree_hall": judge_agreement_rate(hall_verdicts),
            "comp_scores": comp_scores,
            "hall_scores": hall_scores,
        })

    overall_mu = float(np.mean(all_comp)) if all_comp else 0.0
    score = min(100.0, overall_mu / SCORE_MEAN_MIN * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(overall_mu >= SCORE_MEAN_MIN),
        overall_mu_comp=overall_mu,
        passes=passes,
        per_case=per_case,
    )


# ── P4: OOD Safe Refusal ──────────────────────────────────────────────────────

def run_p4_ood(
    system: RAGSystem,
    ood_cases: list[dict],
    embedding_model: Any,
    tau: float,
    top_k: int,
) -> dict:
    _log(f"  [P4] OOD safe refusal ({len(ood_cases)} cases) ...")
    per_case: list[dict] = []
    _RETRIEVAL_INTENTS = {"document_content", "metadata_query"}

    for case in ood_cases:
        intent = system.classify_intent(case["question"])
        if intent not in _RETRIEVAL_INTENTS:
            per_case.append({
                "id": case["id"], "domain": case.get("domain", "?"),
                "safe": True, "reason": f"intent={intent} (non-retrieval route)"
            })
            continue

        chunks = system.retrieve(case["question"], top_k=top_k)
        q_emb = embedding_model.encode([case["question"]], convert_to_numpy=True)[0]
        max_sim = max(
            (cosine_similarity(q_emb, embedding_model.encode([c.get("text", "")], convert_to_numpy=True)[0])
             for c in chunks if c.get("text")),
            default=0.0,
        )
        safe = max_sim < tau
        per_case.append({
            "id": case["id"], "domain": case.get("domain", "?"),
            "safe": safe,
            "reason": f"max_sim={max_sim:.3f} {'<' if safe else '>='} tau={tau:.3f}",
        })

    safe_count = sum(1 for r in per_case if r["safe"])
    total = len(per_case)
    safe_rate = compute_hit_rate(safe_count, total)
    score = min(100.0, safe_rate / OOD_SAFE_REFUSAL_RATE_MIN * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(safe_rate >= OOD_SAFE_REFUSAL_RATE_MIN),
        safe_rate=safe_rate,
        per_case=per_case,
    )


# ── P5: Multi-Turn Coherence ──────────────────────────────────────────────────

def _normalize_doc_id(doc_id: str) -> str:
    """Strip .pdf suffix so raw filenames fall back to a stem comparison."""
    if doc_id.endswith(".pdf"):
        return doc_id[:-4]
    return doc_id


def run_p5_multiturn(
    system: RAGSystem,
    mt_cases: list[dict],
    embedding_model: Any,
    tau: float,
    top_k: int,
) -> dict:
    _log(f"  [P5] Multi-turn coherence ({len(mt_cases)} trajectories) ...")
    turn2_hits: list[bool] = []

    for traj in mt_cases:
        turns = traj.get("turns", [])
        if len(turns) < 2:
            continue
        turn2 = turns[1]
        standalone = turn2.get("expected_standalone", turn2.get("question", ""))
        if not standalone:
            continue
        # Normalize: strip .pdf so raw filenames don't silently miss normalized doc_ids
        expected_ids = {_normalize_doc_id(d) for d in turn2.get("expected_documents", [])}

        chunks = system.retrieve(standalone, top_k=top_k)
        hit = any(_normalize_doc_id(c.get("doc_id", "")) in expected_ids for c in chunks)
        turn2_hits.append(hit)

    if not turn2_hits:
        return _pillar_result(0.0, pass_val=False, reason="No valid multi-turn cases")

    hit_rate = compute_hit_rate(sum(turn2_hits), len(turn2_hits))
    score = min(100.0, hit_rate / MULTITURN_HIT_RATE_MIN * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(hit_rate >= MULTITURN_HIT_RATE_MIN),
        turn2_hit_rate=hit_rate,
        n_trajectories=len(turn2_hits),
    )


# ── P6: Poisoned Context ──────────────────────────────────────────────────────

def run_p6_poisoned(
    system: RAGSystem,
    poisoned_cases: list[dict],
    top_k: int,
) -> dict:
    # P6 uses string matching only — no LLM judge call, no API key required.
    _log(f"  [P6] Poisoned context ({len(poisoned_cases)} cases) ...")
    per_case: list[dict] = []

    for case in poisoned_cases:
        original_fact = case.get("original_fact", "")
        counterfact = case.get("counterfact", "")
        if not original_fact or not counterfact:
            continue

        chunks = system.retrieve(case["question"], top_k=top_k)
        poisoned_chunks = []
        injected = False
        for chunk in chunks:
            text = chunk.get("text", "")
            if not injected and original_fact.lower() in text.lower():
                text = text.replace(original_fact, counterfact)
                injected = True
            poisoned_chunks.append({**chunk, "text": text})

        if not injected and poisoned_chunks:
            mid = len(poisoned_chunks) // 2
            poisoned_chunks[mid] = {
                **poisoned_chunks[mid],
                "text": poisoned_chunks[mid].get("text", "") + f" Note: {counterfact}",
            }

        answer = system.generate(case["question"], poisoned_chunks)
        cf_lower = counterfact.lower()
        of_lower = original_fact.lower()
        ans_lower = answer.lower()

        if cf_lower in ans_lower:
            grounded, reason = True, "answer reflects counterfact"
        elif of_lower not in ans_lower:
            grounded, reason = True, "answer paraphrased (neither fact present)"
        else:
            grounded, reason = False, "answer uses original fact (parametric memory)"

        per_case.append({
            "id": case["id"],
            "description": case.get("description", case["id"]),
            "grounded": grounded,
            "reason": reason,
        })

    if not per_case:
        return _pillar_result(0.0, pass_val=False, reason="No valid poisoned cases")

    grounded_count = sum(1 for r in per_case if r["grounded"])
    grounding_rate = compute_hit_rate(grounded_count, len(per_case))
    score = min(100.0, grounding_rate / POISONING_GROUNDING_RATE_MIN * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(grounding_rate >= POISONING_GROUNDING_RATE_MIN),
        grounding_rate=grounding_rate,
        per_case=per_case,
    )


# ── P7: Position Bias ─────────────────────────────────────────────────────────

def run_p7_position_bias(
    system: RAGSystem,
    p2_per_case: list[dict],
    top_k: int,
    ir_cases: list[dict] | None = None,
) -> dict:
    if not _judge_available():
        _log("  [P7] Position bias SKIPPED (RAG_BENCH_API_KEY not set)")
        return _pillar_result(None, pass_val=False, skipped=True)

    hit_cases = [r for r in p2_per_case if r.get("hit") and r.get("first_relevant_chunk")]
    sample = hit_cases[:20]
    if not sample:
        return _pillar_result(0.0, pass_val=False, reason="No hit cases from P2")

    _log(f"  [P7] Position bias ({len(sample)} cases × 3 positions) ...")
    case_by_id = {c["id"]: c for c in (ir_cases or [])}
    per_case: list[dict] = []

    for r in sample:
        relevant_chunk = r["first_relevant_chunk"]
        other_chunks = [c for c in r["retrieved_chunks"] if c is not relevant_chunk][:top_k - 1]
        base_case = case_by_id.get(r["id"])
        question = base_case["question"] if base_case else r["id"]

        scores_by_pos: dict[int, float] = {}
        positions = [0, max(1, len(other_chunks) // 2), len(other_chunks)]

        for pos in positions:
            chunk_list = other_chunks[:pos] + [relevant_chunk] + other_chunks[pos:]
            try:
                answer = system.generate(question, chunk_list)
                j_res = evaluate_completeness(question, answer, relevant_chunk.get("text", "")[:200])
                scores_by_pos[pos] = float(j_res["score"])
            except JudgeUnavailable:
                return _pillar_result(None, pass_val=False, skipped=True)

        if len(scores_by_pos) < 2:
            continue
        std_dev = float(np.std(list(scores_by_pos.values())))
        per_case.append({"id": r["id"], "scores_by_position": scores_by_pos, "std_dev": std_dev})

    if not per_case:
        return _pillar_result(0.0, pass_val=False, reason="No valid position bias results")

    avg_std = float(np.mean([r["std_dev"] for r in per_case]))
    score = max(0.0, (1 - avg_std / POSITION_BIAS_MAX_COMP_STD) * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(avg_std <= POSITION_BIAS_MAX_COMP_STD),
        avg_std_dev=avg_std,
        per_case=per_case,
    )


# ── P8: Distractor Injection ──────────────────────────────────────────────────

def run_p8_distractor(
    system: RAGSystem,
    p2_per_case: list[dict],
    source_passages: dict[str, str],
    top_k: int,
    ir_cases: list[dict],
) -> dict:
    if not _judge_available():
        _log("  [P8] Distractor injection SKIPPED (RAG_BENCH_API_KEY not set)")
        return _pillar_result(None, pass_val=False, skipped=True)

    hit_cases = [r for r in p2_per_case if r.get("hit")][:20]
    if not hit_cases:
        return _pillar_result(0.0, pass_val=False, reason="No hit cases from P2")

    all_doc_ids = list(source_passages.keys())
    _log(f"  [P8] Distractor injection ({len(hit_cases)} cases) ...")
    per_case: list[dict] = []

    case_by_id = {c["id"]: c for c in ir_cases}

    for r in hit_cases:
        base_case = case_by_id.get(r["id"])
        if not base_case:
            continue
        expected_ids = set(base_case.get("expected_documents", []))
        distractor_ids = [d for d in all_doc_ids if d not in expected_ids]
        if not distractor_ids:
            continue

        distractor_doc = random.choice(distractor_ids)
        distractor_text = source_passages[distractor_doc][:400]
        distractor_chunk = {"text": distractor_text, "doc_id": distractor_doc, "score": 0.0}

        chunks = list(r["retrieved_chunks"])
        mid = len(chunks) // 2
        chunks_with_distractor = chunks[:mid] + [distractor_chunk] + chunks[mid:]

        question = base_case["question"]
        try:
            answer = system.generate(question, chunks_with_distractor)
            c_res = evaluate_completeness(
                question, answer, base_case.get("ideal_answer_points", "")
            )
            h_res = evaluate_hallucination(answer, [c.get("text", "") for c in chunks_with_distractor])
        except JudgeUnavailable:
            return _pillar_result(None, pass_val=False, skipped=True)

        per_case.append({
            "id": r["id"],
            "distractor_doc": distractor_doc,
            "comp_score": c_res["score"],
            "hall_score": h_res["score"],
        })

    if not per_case:
        return _pillar_result(0.0, pass_val=False, reason="No valid distractor results")

    avg_comp = float(np.mean([r["comp_score"] for r in per_case]))
    avg_hall = float(np.mean([r["hall_score"] for r in per_case]))
    score = min(100.0, (avg_comp / DISTRACTOR_COMPLETENESS_MIN + avg_hall / DISTRACTOR_HALLUCINATION_MIN) / 2 * 100)
    return _pillar_result(
        round(score, 1),
        pass_val=(avg_comp >= DISTRACTOR_COMPLETENESS_MIN and avg_hall >= DISTRACTOR_HALLUCINATION_MIN),
        avg_completeness=avg_comp,
        avg_hallucination=avg_hall,
        per_case=per_case,
    )


# ── System loader ─────────────────────────────────────────────────────────────

def _load_system(system_name: str) -> RAGSystem:
    if system_name == "langchain_faiss":
        from rag_bench.systems.langchain_faiss import LangChainFAISSSystem
        return LangChainFAISSSystem()
    parts = system_name.rsplit(".", 1)
    if len(parts) != 2:
        print(f"ERROR: --system must be 'langchain_faiss' or 'dotted.module.ClassName'")
        sys.exit(1)
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_env_fallbacks() -> None:
    """Read .env and set RAG_BENCH_* vars from LLM_* vars if not already set."""
    env_file = Path(__file__).parent.parent / ".env"
    env_vars: dict[str, str] = {}
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore[import]
            load_dotenv(env_file, override=False)
        except ImportError:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        env_vars[k.strip()] = v.strip()
            for k, v in env_vars.items():
                os.environ.setdefault(k, v)

    # Set RAG_BENCH_* from LLM_* fallbacks if not already set
    if not os.environ.get("RAG_BENCH_API_KEY"):
        fallback_key = os.environ.get("LLM_API_KEY_CLAUDE") or os.environ.get("LLM_API_KEY")
        if fallback_key:
            os.environ["RAG_BENCH_API_KEY"] = fallback_key
    if not os.environ.get("RAG_BENCH_JUDGE_BASE_URL"):
        fallback_url = os.environ.get("LLM_BASE_URL_CLAUDE")
        if fallback_url:
            os.environ["RAG_BENCH_JUDGE_BASE_URL"] = fallback_url
    if not os.environ.get("RAG_BENCH_JUDGE_MODEL"):
        fallback_model = (
            os.environ.get("LLM_MODEL_CLAUDE_HAIKU")
            or os.environ.get("LLM_MODEL_CLAUDE")
        )
        if fallback_model:
            os.environ["RAG_BENCH_JUDGE_MODEL"] = fallback_model


def main() -> None:
    _load_env_fallbacks()
    parser = argparse.ArgumentParser(
        description="RAG-Adversarial-Bench: run all 8 pillars against a RAGSystem implementation."
    )
    parser.add_argument("--system", required=True, help="'langchain_faiss' or 'module.ClassName'")
    parser.add_argument("--dataset", required=True, type=Path, help="Directory with ground truth JSON files")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path for results")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--monte-carlo-passes", type=int, default=_DEFAULT_MC_PASSES)
    parser.add_argument("--judge-model", type=str, default=None,
                        help="LLM judge model (default: claude-haiku-4-5-20251001). "
                             "Set to claude-sonnet-4-6 for official runs.")
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip P3/P7/P8 LLM judge calls (saves API cost). "
             "P6 always runs — it uses string matching and needs no API key.",
    )
    args = parser.parse_args()

    if args.judge_model:
        os.environ["RAG_BENCH_JUDGE_MODEL"] = args.judge_model

    if args.skip_judge:
        os.environ.pop("RAG_BENCH_API_KEY", None)

    _log(f"\n{'='*60}")
    _log(f"RAG-Adversarial-Bench")
    _log(f"  system  : {args.system}")
    _log(f"  dataset : {args.dataset}")
    _log(f"  top_k   : {args.top_k}")
    _log(f"  mc_passes: {args.monte_carlo_passes}")
    _log(f"  judge   : {'ENABLED' if _judge_available() else 'DISABLED for P3/P7/P8 (--skip-judge or no API key)'}")
    _log(f"{'='*60}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    _log("[1/3] Loading dataset ...")
    data = _load_dataset(args.dataset)
    ir_cases = data["ir"]
    if not ir_cases:
        _log("ERROR: ground_truth_ir.json not found or empty. Run the ingestion script first.")
        sys.exit(1)
    _log(f"  IR cases: {len(ir_cases)}, OOD: {len(data['ood'])}, "
         f"poisoned: {len(data['poisoned'])}, multiturn: {len(data['multiturn'])}")
    _log(f"  P3 will evaluate first {min(_P3_IR_CASES, len(ir_cases))} of {len(ir_cases)} IR cases (API cost cap).")

    # ── Corpus & embedding ────────────────────────────────────────────────────
    _log("[2/3] Building corpus embeddings ...")
    embedding_model = _load_embedding_model()
    corpus_chunks, corpus_doc_ids = _build_corpus_chunks(data["source_passages"])
    _log(f"  {len(corpus_chunks)} chunks from {len(data['source_passages'])} documents")

    # ── Ingest into system ────────────────────────────────────────────────────
    _log("[3/3] Ingesting corpus into RAGSystem ...")
    system = _load_system(args.system)
    ingest_docs = [
        {"doc_id": c["doc_id"], "text": c["text"], "title": c["doc_id"]}
        for c in corpus_chunks
    ]
    system.ingest(ingest_docs)
    _log(f"  Ingested {len(ingest_docs)} chunks into '{args.system}'")

    # ── Pre-compute corpus embeddings once (reused by P1, P4, P5) ────────────
    _log("  Pre-computing corpus embeddings ...")
    corpus_texts = [c["text"] for c in corpus_chunks]
    corpus_embs = embedding_model.encode(corpus_texts, convert_to_numpy=True).astype(np.float32)
    _log(f"  {len(corpus_embs)} embeddings ready.")

    # ── Run pillars ───────────────────────────────────────────────────────────
    _log("\n[Pillars]")

    p1 = run_p1_calibration(
        corpus_chunks, corpus_doc_ids, ir_cases, embedding_model,
        precomputed_embs=corpus_embs,
    )
    tau = p1.get("optimal_tau", p1.get("p95", 0.5))

    p2 = run_p2_ir(system, ir_cases, embedding_model, tau, args.top_k)
    p3 = run_p3_monte_carlo(system, ir_cases[:_P3_IR_CASES], args.top_k, args.monte_carlo_passes)
    p4 = run_p4_ood(system, data["ood"], embedding_model, tau, args.top_k)
    p5 = run_p5_multiturn(system, data["multiturn"], embedding_model, tau, args.top_k)
    p6 = run_p6_poisoned(system, data["poisoned"], args.top_k)
    p7 = run_p7_position_bias(system, p2.get("per_case", []), args.top_k, ir_cases)
    p8 = run_p8_distractor(system, p2.get("per_case", []), data["source_passages"], args.top_k, ir_cases)

    retrieval = {"calibration": p1, "ir": p2, "position_bias": p7, "distractor": p8}
    generation = {"monte_carlo": p3, "ood": p4, "multiturn": p5, "poisoned": p6}

    retrieval_score = _weighted_score(retrieval, RETRIEVAL_WEIGHTS)
    generation_score = _weighted_score(generation, GENERATION_WEIGHTS)
    overall_score = round((retrieval_score + generation_score) / 2, 1)

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = args.output.with_suffix(".md")
    writer = ReportWriter(report_path)
    writer.write_dual_scorecard(retrieval, generation)
    writer.finalize()

    # ── Results JSON ──────────────────────────────────────────────────────────
    results = {
        "system": args.system,
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "monte_carlo_passes": args.monte_carlo_passes,
        "retrieval_score": retrieval_score,
        "generation_score": generation_score,
        "overall_score": overall_score,
        "pillars": {
            "p1_calibration": {k: v for k, v in p1.items() if k != "per_case"},
            "p2_ir": {k: v for k, v in p2.items() if k != "per_case"},
            "p3_monte_carlo": {k: v for k, v in p3.items() if k != "per_case"},
            "p4_ood": {k: v for k, v in p4.items() if k != "per_case"},
            "p5_multiturn": {k: v for k, v in p5.items() if k != "per_case"},
            "p6_poisoned": {k: v for k, v in p6.items() if k != "per_case"},
            "p7_position_bias": {k: v for k, v in p7.items() if k != "per_case"},
            "p8_distractor": {k: v for k, v in p8.items() if k != "per_case"},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"RESULTS — {args.system}")
    _log(f"  Retrieval Score : {retrieval_score}/100  "
         f"(P1={p1.get('score', '?')} P2={p2.get('score', '?')} "
         f"P7={p7.get('score', 'skip')} P8={p8.get('score', 'skip')})")
    _log(f"  Generation Score: {generation_score}/100  "
         f"(P3={p3.get('score', 'skip')} P4={p4.get('score', '?')} "
         f"P5={p5.get('score', '?')} P6={p6.get('score', 'skip')})")
    _log(f"  Overall Score   : {overall_score}/100")
    _log(f"  Report          : {report_path}")
    _log(f"  Results JSON    : {args.output}")
    _log(f"{'='*60}\n")


if __name__ == "__main__":
    main()
