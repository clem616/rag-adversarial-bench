import os

# --- Monte Carlo ---
MONTE_CARLO_PASSES = int(os.environ.get("MONTE_CARLO_PASSES", "5"))
SCORE_MEAN_MIN = 4.0
SCORE_STD_MAX = 1.0
JUDGE_AGREEMENT_MIN = 0.60
CI_LOWER_BOUND_MIN = 2.5

# --- Embedding Calibration ---
NULL_SAMPLE_PAIRS = 1000
NULL_ALPHA = 0.05
EFFECT_SIZE_MIN = 0.5
CALIBRATION_PASS_RATE = 0.70
ROC_AUC_MIN = 0.80

# --- IR Metrics ---
TOP_K = 5
MRR_MIN = 0.70
HIT_RATE_MIN = 0.70
PRECISION_AT_K_MIN = 0.20

# --- Confidence Interval ---
CI_LEVEL = 0.95

# --- OOD / Adversarial ---
OOD_SAFE_REFUSAL_RATE_MIN = 0.80

# --- Multi-Turn ---
REWRITE_SIMILARITY_MIN = 0.70
MULTITURN_HIT_RATE_MIN = 0.60
CONTEXT_DECAY_MAX_DROP = 1.5

# --- Poisoned Context ---
POISONING_GROUNDING_RATE_MIN = 0.60

# --- Position Bias ---
POSITION_BIAS_MAX_COMP_STD = 1.2

# --- Distractor Injection ---
DISTRACTOR_HALLUCINATION_MIN = 3.5
DISTRACTOR_COMPLETENESS_MIN = 3.5

# --- External Domain IR (relaxed — harder than the golden corpus) ---
# Used for PubMedQA and any future externally-sourced domain datasets.
EXTERNAL_HIT_RATE_MIN = 0.65
EXTERNAL_MRR_MIN = 0.60
EXTERNAL_SCORE_MEAN_MIN = 3.5
EXTERNAL_CI_LOWER_BOUND_MIN = 2.5

# --- Scorecard weights (Retrieval Tier) ---
# P1 Embedding Calibration, P2 IR Metrics, P7 Position Bias, P8 Distractor
RETRIEVAL_WEIGHTS = {"calibration": 0.30, "ir": 0.40, "position_bias": 0.15, "distractor": 0.15}

# --- Scorecard weights (Generation Tier) ---
# P3 Monte Carlo, P4 OOD, P5 Multi-Turn, P6 Poisoned Context
GENERATION_WEIGHTS = {"monte_carlo": 0.40, "ood": 0.20, "multiturn": 0.20, "poisoned": 0.20}
