# rag-adversarial-bench

An evaluation framework for retrieval-augmented generation (RAG) systems: eight checks, from standard retrieval metrics to adversarial cases such as a false statement or an unrelated passage placed in the context.

Plug in your RAG system by implementing two methods, run the benchmark, and get a scorecard split into a retrieval tier and a generation tier. A medical reference dataset (300 questions from 42 public FDA guidance documents and PubMedQA) is included.

This is a v1 built as a learning project and an evaluation study, alongside my internship work on RAG in a regulated environment. Running it, then examining what each check really measures, taught me as much as building it. The section [What each check measures, and its limits](#what-each-check-measures-and-its-limits) is the honest account of that, and it is worth reading before trusting any score this tool produces.

---

## Requirements

- Python 3.10 or higher
- **Anthropic API key** (`sk-ant-...`) for the language-model judge used by P3, P7 and P8. Without it, run with `--skip-judge`.
- **OpenAI-compatible LLM endpoint** for the generation step of the LangChain-FAISS reference system. Set `RAG_BENCH_LLM_BASE_URL` and `RAG_BENCH_LLM_API_KEY` to any OpenAI-compatible API (OpenAI, Azure OpenAI, a local llama.cpp server, etc.). If you benchmark your own system, you control generation and this is not needed.

---

## Install

```bash
git clone https://github.com/clem616/rag-adversarial-bench
cd rag-adversarial-bench
pip install -e ".[judge,faiss]"
```

This installs the judge and the reference system. Subsets:

```bash
pip install -e .                         # core only
pip install -e ".[judge]"                # + language-model judge for P3/P7/P8
pip install -e ".[judge,faiss]"          # + LangChain-FAISS reference system
```

---

## Quickstart: run the reference system

**Step 1: set the keys.**

```bash
# Judge (P3/P7/P8); not needed with --skip-judge
export RAG_BENCH_API_KEY=sk-ant-...

# Generation step of the LangChain-FAISS reference system
export RAG_BENCH_LLM_BASE_URL=https://api.openai.com   # or any OpenAI-compatible endpoint
export RAG_BENCH_LLM_API_KEY=sk-...
export RAG_BENCH_LLM_MODEL=gpt-4.1                     # model used for the reference run; any model name works
```

**Step 2: run all eight checks.**

```bash
python -m rag_bench.run_benchmark \
  --system rag_bench.systems.langchain_faiss.LangChainFAISSSystem \
  --dataset rag_bench/datasets/medical \
  --output results/langchain_faiss.json
```

The benchmark cuts the dataset's documents into chunks, gives them to the system to index, runs the eight checks, and writes the results to `results/langchain_faiss.json`. A Markdown scorecard is printed to stdout.

### Reference result

Run of 17 July 2026 on the included medical dataset: generation with `gpt-4.1`, judge `claude-sonnet-4-6`, 5 Monte Carlo passes.

| Tier | Check | Measured | Score | Verdict |
|---|---|---|---|---|
| Retrieval | P1 Embedding Calibration | ROC AUC 0.698 | 87.2 | FAIL |
| Retrieval | P2 IR Metrics | Hit@5 0.913, MRR 0.846 | 100.0 | PASS |
| Retrieval | P7 Position Bias | score std dev 0.204 | 83.0 | PASS |
| Retrieval | P8 Distractor Injection | completeness 3.45, hallucination 4.35 (out of 5) | 100.0 | FAIL |
| Generation | P3 Monte Carlo Stability | mean completeness 2.9 (out of 5) | 72.5 | FAIL |
| Generation | P4 OOD Safe Refusal | safe rate 0.84 | 100.0 | PASS |
| Generation | P5 Multi-Turn Coherence | turn-2 hit rate 0.85 | 100.0 | PASS |
| Generation | P6 Poisoned Context | grounding rate 0.75 | 93.8 | FAIL |

**Overall 90.7/100** (Retrieval 93.6, Generation 87.8).

How to read it: each score is the measured value divided by its pass threshold, capped at 100, so 100 means "threshold met", not "perfect". The overall verdict only requires each tier to reach 60, which is why this run passes overall while four checks fail. Read the per-check verdicts, not the total.

To reproduce this setting, add `--judge-model claude-sonnet-4-6` to the command above; the default judge is `claude-haiku-4-5`. Results vary slightly between runs because generation is not deterministic. The distractor choice in P8 is seeded and does not vary.

**Without a judge key**, P1, P2, P4, P5 and P6 still run. P6 checks the answer by text matching, so it calls the system's generation step but not the judge.

```bash
python -m rag_bench.run_benchmark \
  --system rag_bench.systems.langchain_faiss.LangChainFAISSSystem \
  --dataset rag_bench/datasets/medical \
  --output results/langchain_faiss_free.json \
  --skip-judge
```

---

## Quickstart: benchmark your own RAG system

**Step 1: implement the `RAGSystem` interface** in a file such as `my_system.py`.

```python
from rag_bench.systems.base import RAGSystem

class MySystem(RAGSystem):
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Return a list of dicts, each with:
        #   "doc_id": str   — identifier matching the dataset's expected_documents field
        #   "text":   str   — the retrieved passage
        #   "score":  float — relevance score (higher = more relevant)
        ...

    def generate(self, query: str, chunks: list[dict], history=None) -> str:
        # Generate an answer from the chunks and return it as plain text.
        ...
```

Two methods are optional:

- `classify_intent(query)` returns a label. P4 treats a question as safely handled when the label is anything other than `"document_content"` or `"metadata_query"`. The default always returns `"document_content"`.
- `ingest(documents)` receives the chunks cut by the benchmark, to build your index.

**Step 2: set the judge key** (or use `--skip-judge`).

```bash
export RAG_BENCH_API_KEY=sk-ant-...
```

**Step 3: run.**

```bash
python -m rag_bench.run_benchmark \
  --system my_system.MySystem \
  --dataset rag_bench/datasets/medical \
  --output results/my_system.json
```

`results/my_system.json` contains the per-check breakdown, and a Markdown scorecard is printed to stdout.

---

## The eight checks

| ID | Check | Tier | Judge needed |
|---|---|---|---|
| P1 | Embedding Calibration | Retrieval | No |
| P2 | IR Metrics (Hit@5, MRR) | Retrieval | No |
| P7 | Position Bias | Retrieval | Yes |
| P8 | Distractor Injection | Retrieval | Yes |
| P3 | Monte Carlo Stability (first 50 questions) | Generation | Yes |
| P4 | OOD Safe Refusal | Generation | No |
| P5 | Multi-Turn Coherence | Generation | No |
| P6 | Poisoned Context Grounding | Generation | No |

Weights are in `rag_bench/thresholds.py`:

```
Retrieval  = 0.30 × P1 + 0.40 × P2 + 0.15 × P7 + 0.15 × P8
Generation = 0.40 × P3 + 0.20 × P4 + 0.20 × P5 + 0.20 × P6
Overall    = (Retrieval + Generation) / 2
```

A chunk counts as relevant when it comes from an expected document and its cosine similarity to the reference passage is at least `tau`. P1 computes `tau` at each run, as the threshold that best separates related from unrelated passages (Youden's J). P2 and P4 use it.

---

## What each check measures, and its limits

The names describe what each check aims at. This section describes what the code does today, and where the two differ.

| Check | What the code does | Known limits |
|---|---|---|
| P1 Embedding Calibration | Compares similarities of related passages with 1,000 random pairs from different documents; reports ROC AUC and sets `tau`. | Never calls the system under test, so every system gets the same P1, yet it weighs 30% of the retrieval score. It describes the dataset and the benchmark's embedding model. |
| P2 IR Metrics | Hit@5, MRR and Precision@5 on 300 questions. | Relevance depends on `tau` and on `all-MiniLM-L6-v2`, the model the reference system also uses, which favours it. A system with a different embedding model can find correct passages that this model scores below `tau`. The cap makes 0.75 and 0.95 both score 100. |
| P3 Monte Carlo Stability | Generates 5 answers for each of the first 50 questions; the judge scores completeness and hallucination. | The verdict is the mean completeness, not the stability: the spread, confidence intervals and agreement between passes are reported but do not enter the verdict. The hallucination score is not used either. |
| P4 OOD Safe Refusal | For 50 off-topic questions: safe if `classify_intent` routes the question away from retrieval, otherwise safe if the best retrieved chunk is below `tau`. | Never reads the generated answer, so it does not check that the system refused. A system that always refuses and one that always invents an answer get the same score if their retrieval is the same. |
| P5 Multi-Turn Coherence | For 20 conversations, retrieves with the dataset's rewritten version of the second turn, and counts a hit if an expected document comes back. | The system never sees the conversation or the ambiguous follow-up, and nothing is generated. It is a retrieval test on a clean question. |
| P6 Poisoned Context | Replaces a true fact with a false one in the retrieved context, generates, and matches text: grounded if the answer contains the false fact or neither fact; not grounded if it contains the true one. | An answer with neither fact, including "I don't know", counts as grounded. Exact matching misses paraphrases, and an answer that quotes the false fact to reject it counts as grounded. The check measures obedience to the context, which is wanted for a document assistant but also means one corrupted document misleads the system. |
| P7 Position Bias | For 20 questions, places the relevant chunk at the start, middle and end of the context; the judge scores completeness; reports the average standard deviation. | Measures consistency, not quality: uniformly poor answers pass. Only 20 questions and 3 positions; the judge's reference is the first 200 characters of the chunk. It tests the generator but counts in the retrieval tier. |
| P8 Distractor Injection | For 20 questions, inserts 400 characters of an unrelated document in the middle of the context; the judge scores completeness and hallucination. | The score averages the two ratios before capping, so a high hallucination score hides a completeness shortfall: the reference run scores 100 and fails. An unrelated passage is an easy distractor; one on a neighbouring topic would be harder. It tests the generator but counts in the retrieval tier. |

Limits that apply to the whole framework:

- **The judge is not validated.** No sample has been scored by people to measure agreement with the judge. When the judge's reply cannot be parsed, the case counts as FAIL with a score of 0, which mixes judge errors into the system's score.
- **The dataset is not reviewed.** The FDA questions, reference answers, off-topic questions, poisoned facts and conversations were generated by a language model. PubMedQA and FDA guidance are public, so the generator may already know the answers, which can inflate P3.
- **The thresholds were set by hand** on this medical dataset, so passing them says little about another domain.
- **The total hides failures**, as the reference run shows. Read the verdicts.
- **The benchmark does the chunking itself** (800 characters, 100 of overlap), so a system cannot be tested with its own chunking.
- **The unit tests check the code, not the measurements.** The 232 tests verify that each function computes what it should on known inputs. They do not show that each check measures what its name says.

### Known issue

When `--judge-model` is given on the command line, the Markdown report header still shows the default judge model, because the report reads the model name at import, before the option is applied. The judge actually called is the one passed on the command line.

---

## Medical reference dataset

Included at `rag_bench/datasets/medical/`:

```
300 IR cases   — 150 generated from 42 FDA guidance documents (public domain) + 150 from PubMedQA (MIT)
50  OOD cases  — off-topic questions
20  Poisoned   — true facts and their false counterparts
20  Multi-turn — two-turn conversations with a rewritten standalone question
```

P3 uses the first 50 IR cases to cap cost (50 × passes × 2 judge calls). P7 and P8 use the first 20 questions that P2 retrieved correctly.

A dataset for another domain can be generated with [domain-ingestion-engine](https://github.com/clem616/domain-ingestion-engine):

```bash
python -m rag_bench.run_benchmark \
  --system my_system.MySystem \
  --dataset ./ground_truth/legal \
  --output results/my_system_legal.json
```

---

## Judge configuration

P3, P7 and P8 use a language-model judge, separate from the system under test so that a model does not grade its own output. It runs at temperature 0 and scores completeness and hallucination from 1 to 5.

```bash
# Anthropic API (default)
export RAG_BENCH_API_KEY=sk-ant-...

# Azure AI Foundry
export RAG_BENCH_JUDGE_BASE_URL=https://your-resource.services.ai.azure.com/anthropic
export RAG_BENCH_API_KEY=your-azure-key
```

Cost depends on the judge model and the number of passes:

```bash
# Faster and cheaper, for development
RAG_BENCH_JUDGE_MODEL=claude-haiku-4-5 MONTE_CARLO_PASSES=3 python -m rag_bench.run_benchmark ...

# Reference setting
RAG_BENCH_JUDGE_MODEL=claude-sonnet-4-6 MONTE_CARLO_PASSES=5 python -m rag_bench.run_benchmark ...
```

A full run with 5 passes makes about 350 generation calls and 600 judge calls.

---

## Repository structure

```
rag_bench/
├── run_benchmark.py            — CLI entry point and the eight checks
├── thresholds.py               — pass thresholds and tier weights (set on the medical dataset)
├── systems/
│   ├── base.py                 — RAGSystem base class (implement this for your system)
│   └── langchain_faiss.py      — reference system (FAISS + all-MiniLM-L6-v2)
├── pillars/
│   └── retrieval/
│       └── ir_metrics.py       — Hit@K, MRR, Precision@K
├── judge/
│   └── claude_judge.py         — language-model judge (completeness and hallucination)
├── stats/
│   ├── stats_utils.py          — cosine similarity, ROC AUC, Youden threshold, Student-t intervals
│   └── plot_utils.py           — ROC curve and score distribution plots
├── report/
│   └── report_writer.py        — Markdown scorecard
└── datasets/
    └── medical/                — medical reference dataset (FDA + PubMedQA)

scripts/
└── generate_latex_tables.py    — benchmark JSON to LaTeX tables
```

---

## CLI reference

```
python -m rag_bench.run_benchmark [OPTIONS]

  --system TEXT              Dotted path to a RAGSystem subclass (e.g. my_system.MySystem)
  --dataset PATH             Directory containing the ground_truth_*.json files
  --output PATH              Where to write the results JSON
  --skip-judge               Skip P3/P7/P8; the other checks still run
  --monte-carlo-passes INT   Generation passes for P3 [default: 5]
  --judge-model TEXT         Judge model [default: claude-haiku-4-5]
  --top-k INT                Retrieval depth [default: 5]
```

---

## Development

```bash
git clone https://github.com/clem616/rag-adversarial-bench
cd rag-adversarial-bench
pip install uv          # if not already installed
uv sync                 # installs dev dependencies (pytest, pytest-cov, anthropic)
uv run pytest tests/ -v # 232 tests, no real API calls
```

---

## Roadmap

What I would change first, in order:

1. **Validate the judge** against human scores on a sample of 50 to 100 answers (Cohen's kappa on verdicts, Spearman on scores), and repeat with a second judge model.
2. **Make the score and the verdict agree**: score each check from the weakest of its ratios, and require every check to pass for an overall pass.
3. **Make P4 read the answer** and check for an actual refusal.
4. **Give P5 the conversation**: the raw follow-up and the history, then judge the answer.
5. **Separate "neither fact" in P6** into its own category, and replace text matching with an entailment model or a judge question.
6. **Test the checks with control systems** whose behaviour is known (one that always refuses, one that always invents, one that ignores the context), to show that each verdict moves as it should.
7. **Define relevance without an embedding model**, from labelled passage positions, so that P2 does not favour systems that use `all-MiniLM-L6-v2`.
8. **Audit the dataset**: have people review a sample of the generated questions and reference answers, and add a second domain.
9. **Report uncertainty on the totals**, with a bootstrap over questions.

---

## How it was built

I designed what to test and how, and built it with an AI coding assistant, reviewing and running everything. The limits above come from going back over the code and the reference run afterwards.

---

## License

Apache 2.0. See `LICENSE`.

The medical reference dataset follows the licences of its sources:
- FDA guidance documents: United States public domain (17 U.S.C. § 105)
- PubMedQA cases: MIT licence (`qiaojin/PubMedQA` on Hugging Face)
