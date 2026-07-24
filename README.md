# rag-adversarial-bench

An eight-pillar adversarial evaluation framework for Retrieval-Augmented Generation systems.

Plug in your RAG system, run the benchmark, get a dual Retrieval / Generation scorecard. The Medical reference dataset (150 IR cases from 42 FDA guidance documents + 150 PubMedQA cases) is included and ready to use.

---

## Requirements

- Python 3.10 or higher
- **Anthropic API key** (`sk-ant-...`) — required for the three LLM-judge pillars (P3/P7/P8). P6 runs without a key. Sign up at [console.anthropic.com](https://console.anthropic.com).
- **OpenAI-compatible LLM endpoint** — required by the LangChain-FAISS reference system for its `generate()` step. Set `RAG_BENCH_LLM_BASE_URL` and `RAG_BENCH_LLM_API_KEY` to point at any OpenAI-compatible API (OpenAI, Azure OpenAI, a local llama.cpp server, etc.). If you benchmark **your own system**, you control the generation step and this is not needed.

---

## Install

**Step 1 — clone the repository:**

```bash
git clone https://github.com/rag-adversarial-bench/rag-adversarial-bench
cd rag-adversarial-bench
```

**Step 2 — install the package:**

```bash
pip install -e ".[judge,faiss]"
```

This installs everything needed to run all 8 pillars including the LLM judge and the LangChain-FAISS reference system. If you only want to run a subset:

```bash
pip install -e .                         # P1/P2/P4/P5/P6 only — no API key required
pip install -e ".[judge]"               # + LLM judge for P3/P7/P8
pip install -e ".[judge,faiss]"         # + LangChain-FAISS reference system (recommended)
```

---

## Quickstart: run the reference system immediately

This requires no custom code — the LangChain-FAISS reference system and the Medical dataset are included.

**Step 1 — set your API keys:**

```bash
# For the LLM judge (P3/P7/P8) — skip if using --skip-judge
export RAG_BENCH_API_KEY=sk-ant-...

# For the LangChain-FAISS reference system's generation step
export RAG_BENCH_LLM_BASE_URL=https://api.openai.com   # or any OpenAI-compatible endpoint
export RAG_BENCH_LLM_API_KEY=sk-...
export RAG_BENCH_LLM_MODEL=gpt-4.1-mini                # or your model name
```

**Step 2 — run all 8 pillars:**

```bash
python -m rag_bench.run_benchmark \
  --system rag_bench.systems.langchain_faiss.LangChainFAISSSystem \
  --dataset rag_bench/datasets/medical \
  --output results/langchain_faiss.json
```

The benchmark builds a FAISS index from the included dataset, runs all 8 pillars, and writes results to `results/langchain_faiss.json`. A Markdown scorecard is printed to stdout.

Expected output: Overall 91.3/100 (Retrieval 93.6, Generation 89.0) with `claude-sonnet-4-6` judge, 5 Monte Carlo passes.

**No API key?** Run P1/P2/P4/P5/P6 at zero cost (P6 uses string matching — no judge needed):

```bash
python -m rag_bench.run_benchmark \
  --system rag_bench.systems.langchain_faiss.LangChainFAISSSystem \
  --dataset rag_bench/datasets/medical \
  --output results/langchain_faiss_free.json \
  --skip-judge
```

---

## Quickstart: benchmark your own RAG system

**Step 1 — implement the `RAGSystem` interface:**

Create a Python file (e.g. `my_system.py`) with the following structure:

```python
from rag_bench.systems.base import RAGSystem

class MySystem(RAGSystem):
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Query your retrieval system here.
        # Must return a list of dicts, each with:
        #   "doc_id": str   — identifier matching the dataset's expected_documents field
        #   "text":   str   — the retrieved passage text
        #   "score":  float — relevance score (higher = more relevant)
        ...

    def generate(self, query: str, chunks: list[dict], history=None) -> str:
        # Generate an answer from the retrieved chunks.
        # chunks is the same list of dicts returned by retrieve().
        # Must return the answer as a plain string.
        ...
```

**Step 2 — set your API key** (required for P3/P7/P8; skip with `--skip-judge` to run P1/P2/P4/P5/P6 only):

```bash
export RAG_BENCH_API_KEY=sk-ant-...
```

**Step 3 — run the benchmark:**

```bash
python -m rag_bench.run_benchmark \
  --system my_system.MySystem \
  --dataset rag_bench/datasets/medical \
  --output results/my_system.json
```

**Step 4 — read your scorecard:**

The file `results/my_system.json` contains the full pillar breakdown. A Markdown scorecard is also printed to stdout. Submit your results to the leaderboard — reach out via the dataset discussion tab.

---

## Eight pillars

| ID | Pillar | Tier | Requires API key |
|---|---|---|---|
| P1 | Embedding Calibration | Retrieval | No |
| P2 | IR Metrics (Hit@5, MRR) | Retrieval | No |
| P7 | Position Bias | Retrieval | Yes |
| P8 | Distractor Injection | Retrieval | Yes |
| P3 | Monte Carlo Stability (first 50 IR cases) | Generation | Yes |
| P4 | OOD Safe Refusal | Generation | No |
| P5 | Multi-Turn Coherence | Generation | No |
| P6 | Poisoned Context Grounding | Generation | No¹ |

**Retrieval Score** = weighted average of P1 / P2 / P7 / P8  
**Generation Score** = weighted average of P3 / P4 / P5 / P6  
**Overall Score** = (Retrieval + Generation) / 2

¹ P6 (Poisoned Context) uses string matching to check whether the generated answer reflects the injected counterfact or the original fact. No LLM judge is called; no API key is required. `--skip-judge` does not skip P6.

---

## Medical reference dataset

The dataset ships inside the repository at `rag_bench/datasets/medical/`. It contains:

```
300 IR cases   — 150 IR cases from 42 FDA guidance documents (public domain) + 150 PubMedQA (MIT)
50  OOD cases  — out-of-domain refusal queries
20  Poisoned   — counterfactual FDA passages (P6: string-matching evaluation, no judge)
20  Multi-turn — conversation chains with standalone rewrites
```

> **P3 note**: Monte Carlo Stability is evaluated on the first 50 of 300 IR cases to cap API cost (50 × M passes × 2 judge calls). Position bias (P7) and distractor injection (P8) use the first 20 P2 hit cases.

To benchmark against your own domain dataset (generated by `domain-ingestion-engine`):

```bash
python -m rag_bench.run_benchmark \
  --system my_system.MySystem \
  --dataset ./ground_truth/legal \
  --output results/my_system_legal.json
```

---

## Judge configuration

Three Generation Tier pillars (P3/P7/P8) use an independent LLM judge. The judge is deliberately separate from the system under test to avoid circular evaluation bias. P6 (Poisoned Context) uses string matching only and requires no API key.

```bash
# Anthropic API (default)
export RAG_BENCH_API_KEY=sk-ant-...

# Azure AI Foundry
export RAG_BENCH_JUDGE_BASE_URL=https://your-resource.services.ai.azure.com/anthropic
export RAG_BENCH_API_KEY=your-azure-key
```

Control cost via model and pass count:

```bash
# Fast and cheap — use for development
RAG_BENCH_JUDGE_MODEL=claude-haiku-4-5 MONTE_CARLO_PASSES=3 python -m rag_bench.run_benchmark ...

# Official — use for leaderboard submission
RAG_BENCH_JUDGE_MODEL=claude-sonnet-4-6 MONTE_CARLO_PASSES=5 python -m rag_bench.run_benchmark ...
```

---

## Repository structure

```
rag_bench/
├── run_benchmark.py            — 8-pillar CLI entry point
├── thresholds.py               — pass/fail thresholds (calibrated on Medical dataset)
├── systems/
│   ├── base.py                 — RAGSystem ABC (implement this for your system)
│   └── langchain_faiss.py      — reference implementation (FAISS + all-MiniLM-L6-v2)
├── pillars/
│   └── retrieval/
│       └── ir_metrics.py       — Hit@K, MRR, Precision@K (pure functions)
├── judge/
│   └── claude_judge.py         — independent LLM judge (completeness + hallucination)
├── stats/
│   ├── stats_utils.py          — bootstrapped CI, cosine similarity, ROC AUC
│   └── plot_utils.py           — ROC curve and score distribution plots
├── report/
│   └── report_writer.py        — dual scorecard Markdown report
└── datasets/
    └── medical/                — Medical reference dataset (FDA + PubMedQA)
        ├── ground_truth_ir.json
        ├── ground_truth_ood.json
        ├── ground_truth_poisoned.json
        ├── ground_truth_multiturn.json
        └── source_passages.json

scripts/
└── generate_latex_tables.py    — parse benchmark JSON → LaTeX tables for the paper
```

---

## CLI reference

```
python -m rag_bench.run_benchmark [OPTIONS]

  --system TEXT              Dotted path to RAGSystem subclass (e.g. my_system.MySystem)
  --dataset PATH             Directory containing ground_truth_*.json files
  --output PATH              Where to write the results JSON
  --skip-judge               Skip P3/P7/P8 judge calls. P1/P2/P4/P5/P6 still run — no API key needed for those pillars.
  --monte-carlo-passes INT   Number of independent generation passes for P3 [default: 5]
  --judge-model TEXT         LLM judge model [default: claude-haiku-4-5]
  --top-k INT                Retrieval depth [default: 5]
```

---

## Development setup

To run the test suite locally:

```bash
git clone https://github.com/rag-adversarial-bench/rag-adversarial-bench
cd rag-adversarial-bench
pip install uv          # if not already installed
uv sync                 # installs dev dependencies (pytest, pytest-cov, anthropic)
uv run pytest tests/ -v # 232 tests, no real API calls made
```

---

## v2 roadmap — call for contributions

This is a v1 release. Open problems we are inviting the community to address:

- **Second domain dataset** — a Legal or Finance instantiation of `domain-ingestion-engine` that validates domain-agnostic generalisability
- **Human GT audit** — 50-case spot-check with inter-annotator agreement on the Medical IR cases
- **Alternative judge models** — replication with GPT-4o, Gemini, or a local open-weight model to quantify judge-model sensitivity
- **Larger adversarial splits** — expanding poisoned/multiturn sets from 20 to 100+ cases to reduce confidence intervals
- **Additional reference systems** — BM25, hybrid dense+sparse, any commercial RAG API

Issues, leaderboard submissions, and domain PRs welcome.

---

## License

Apache 2.0. See `LICENSE`.

The Medical reference dataset is governed by the source licenses of its components:
- FDA guidance documents: United States public domain (17 U.S.C. § 105)
- PubMedQA cases: MIT license (`qiaojin/PubMedQA` on HuggingFace)

---

## Citation

The accompanying paper is currently under review. Citation will be added here upon publication.
