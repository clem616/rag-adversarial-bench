"""Generate LaTeX tables from RAG-Adversarial-Bench JSON result files.

Reads one or more result JSON files (from run_benchmark.py --output) and emits
ready-to-paste LaTeX tabular environments for the arXiv paper.

Usage:
    uv run python3 scripts/generate_latex_tables.py \\
        huggingface_dataset/results/reference_langchain_faiss.json \\
        [huggingface_dataset/results/my_system.json ...]

    # With a custom output file
    uv run python3 scripts/generate_latex_tables.py \\
        huggingface_dataset/results/reference_langchain_faiss.json \\
        --output arxiv_preprint/tables.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PILLAR_LABELS: dict[str, str] = {
    "p1_calibration":  "P1 — Embedding Calibration",
    "p2_ir":           "P2 — IR Metrics",
    "p3_monte_carlo":  "P3 — Monte Carlo Stability",
    "p4_ood":          "P4 — OOD Safe Refusal",
    "p5_multiturn":    "P5 — Multi-Turn Coherence",
    "p6_poisoned":     "P6 — Poisoned Context",
    "p7_position_bias":"P7 — Position Bias",
    "p8_distractor":   "P8 — Distractor Injection",
}

_TIER_ORDER = {
    "retrieval": ["p1_calibration", "p2_ir", "p7_position_bias", "p8_distractor"],
    "generation": ["p3_monte_carlo", "p4_ood", "p5_multiturn", "p6_poisoned"],
}


def _score_str(score: float | None, skipped: bool = False) -> str:
    if skipped or score is None:
        return "—"
    return f"{score:.1f}"


def _status_str(pass_: bool | None, skipped: bool = False) -> str:
    if skipped or pass_ is None:
        return "—"
    return r"\textbf{PASS}" if pass_ else r"\textbf{FAIL}"


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    pillars = data.get("pillars", {})
    return {
        "system": data.get("system", path.stem),
        "dataset": data.get("dataset", "—"),
        "retrieval_score": data.get("retrieval_score"),
        "generation_score": data.get("generation_score"),
        "overall_score": data.get("overall_score"),
        "pillars": pillars,
    }


def generate_per_pillar_table(results: list[dict], caption: str = "", label: str = "") -> str:
    systems = [r["system"] for r in results]
    col_spec = "ll" + "c" * len(systems)
    header_cells = " & ".join(r"\textbf{" + s + "}" for s in systems)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\textbf{{Tier}} & \\textbf{{Pillar}} & {header_cells} \\\\",
        r"\midrule",
    ]

    for tier_name, pillar_keys in _TIER_ORDER.items():
        lines.append(f"\\multicolumn{{{2+len(systems)}}}{{l}}{{\\textit{{{tier_name.title()} Tier}}}} \\\\")
        for pk in pillar_keys:
            label_str = _PILLAR_LABELS.get(pk, pk)
            cells = []
            for r in results:
                pdata = r["pillars"].get(pk, {})
                skipped = pdata.get("skipped", False)
                score = pdata.get("score")
                cells.append(_score_str(score, skipped))
            tier_cell = ""
            lines.append(f"& {label_str} & {' & '.join(cells)} \\\\")
        lines.append(r"\midrule")

    # Scorecard row
    ret_cells = " & ".join(_score_str(r["retrieval_score"]) for r in results)
    gen_cells = " & ".join(_score_str(r["generation_score"]) for r in results)
    ov_cells = " & ".join(_score_str(r["overall_score"]) for r in results)
    lines += [
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Retrieval Score}}}} & {ret_cells} \\\\",
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Generation Score}}}} & {gen_cells} \\\\",
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Overall Score}}}} & {ov_cells} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_scorecard_table(results: list[dict], caption: str = "", label: str = "") -> str:
    systems = [r["system"] for r in results]
    col_spec = "l" + "ccc" * len(systems)
    sub_header = " & ".join(r"\textbf{Ret} & \textbf{Gen} & \textbf{Overall}" for _ in systems)
    sys_header = " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{s}}}}}" for s in systems)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\textbf{{System}} & {sys_header} \\\\",
        r"\cmidrule(lr){2-4}" * len(systems),
        f"& {sub_header} \\\\",
        r"\midrule",
    ]

    for r in results:
        row_cells = " & ".join([
            _score_str(r["retrieval_score"]),
            _score_str(r["generation_score"]),
            _score_str(r["overall_score"]),
        ] * 1)
        lines.append(f"{r['system']} & {row_cells} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from benchmark results.")
    parser.add_argument("result_files", nargs="+", type=Path, help="Result JSON files.")
    parser.add_argument("--output", type=Path, default=None, help="Output .tex file (default: stdout).")
    parser.add_argument("--caption", type=str, default="RAG-Adversarial-Bench results (Medical dataset).",
                        help="Table caption.")
    parser.add_argument("--label", type=str, default="tab:main_results", help="LaTeX label.")
    args = parser.parse_args()

    results = []
    for p in args.result_files:
        if not p.exists():
            print(f"WARNING: {p} not found — skipping.", file=sys.stderr)
            continue
        results.append(load_result(p))

    if not results:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    out_parts = [
        "% -------------------------------------------------------",
        "% Auto-generated by scripts/generate_latex_tables.py",
        "% -------------------------------------------------------",
        "",
        generate_per_pillar_table(results, caption=args.caption, label=args.label),
        "",
        generate_scorecard_table(results, caption="Summary scorecard.", label="tab:scorecard"),
    ]

    output = "\n".join(out_parts)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Tables written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
