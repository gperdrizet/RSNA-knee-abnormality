'''
Configuration constants for label extraction.
'''

import os
from pathlib import Path

# Prompt version selector: 'v1' is the original prompt. 'v2' adds one narrow
# rule on top of v1 - read the whole report body (not just the conclusion)
# specifically for Effusion, based on phi4's own round-1 calibration errors
# (run 20260909_phi4). Broader/other candidate rules were tried and
# rejected as annotation noise or unsupported by the report text - see
# prompt.py's docstring and notebooks/02-label_extraction.ipynb section 4.
PROMPT_VERSION = os.environ.get('PROMPT_VERSION', 'v1')

# Repository layout
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = REPO_ROOT / 'data' / 'raw'
TRAIN_CSV    = RAW_DATA_DIR / 'train.csv'
OUTPUT_ROOT  = REPO_ROOT / 'data' / 'pipeline' / 'label_extraction'

# The 12 target conditions (order matters for output column order) ---
CONDITIONS = [
    'ACL',
    'MCL',
    'Medial Meniscus',
    'Lateral Meniscus',
    'Medial OA',
    'Lateral OA',
    'PF OA',
    'Effusion',
    'Synovitis',
    "Baker's",
    'Contusion',
    'Fracture',
]

# Short human-facing descriptions used in the prompt
CONDITION_DESCRIPTIONS = {
    'ACL': 'Anterior cruciate ligament abnormality (injury, tear, signal increase, disruption)',
    'MCL': 'Medial collateral ligament abnormality (injury, tear, signal increase, edema at the MCL)',
    'Medial Meniscus': 'Medial meniscus tear or other parenchymal abnormality (NOT degenerative only)',
    'Lateral Meniscus': 'Lateral meniscus tear or other parenchymal abnormality (NOT degenerative only)',
    'Medial OA': 'Osteoarthritic changes of the medial femorotibial compartment (bone marrow edema pattern of chronic OA, cartilage loss, subchondral sclerosis)',
    'Lateral OA': 'Osteoarthritic changes of the lateral femorotibial compartment',
    'PF OA': 'Patellofemoral osteoarthritis / patellar cartilage abnormality (chondropathy, cartilage loss)',
    'Effusion': 'Joint effusion (any degree) or joint fluid collection',
    'Synovitis': 'Synovitis / synovial thickening / reactive synovium',
    "Baker's": "Baker's (popliteal) cyst or popliteal cyst abnormality",
    'Contusion': 'Bone contusion / bone marrow edema acute pattern (injury-related, not chronic OA pattern)',
    'Fracture': 'Fracture (any site in the knee: patella, femoral condyle, tibial plateau, etc.)',
}

# Environment variable names (read from .env)
ENV_BASE_URL = 'BASE_URL'
ENV_MODEL    = 'MODEL'
ENV_API_KEY  = 'API_KEY'

# Local llama.cpp servers: one API key shared across N base URLs, each with
# its own model. Work is split across these round-robin.
ENV_LOCAL_API_KEY      = 'LOCAL_API_KEY'
ENV_LOCAL_MODEL        = 'LOCAL_MODEL'
ENV_LOCAL_URLS         = ['LOCAL_URL_A', 'LOCAL_URL_B']

# --- Two-model ensemble (phi4 + qwen7B) ---------------------------------
# When ENSEMBLE_MODE=true, extraction calls two independently-deployed
# models (one dedicated llama.cpp server/GPU each) instead of one, and
# combines their per-condition votes via ENSEMBLE_RULESET. Same shared
# LOCAL_API_KEY, distinct URL/model per side.
ENSEMBLE_MODE = os.environ.get('ENSEMBLE_MODE', 'false').lower() == 'true'

ENV_ENSEMBLE_MODEL_A_URL  = 'ENSEMBLE_MODEL_A_URL'   # phi4 server
ENV_ENSEMBLE_MODEL_A_NAME = 'ENSEMBLE_MODEL_A_NAME'
ENV_ENSEMBLE_MODEL_B_URL  = 'ENSEMBLE_MODEL_B_URL'   # qwen7B server
ENV_ENSEMBLE_MODEL_B_NAME = 'ENSEMBLE_MODEL_B_NAME'

# Per-condition combination policy - 'AND' requires both models to vote
# positive; 'model_a'/'model_b' use only that model's vote. Derived from
# the 58 gold-row policy sweep and confirmed stable under leave-one-out
# resampling - see notebooks/02-label_extraction.ipynb section 4.
ENSEMBLE_RULESET = {c: 'AND' for c in CONDITIONS}
ENSEMBLE_RULESET.update({c: 'model_a' for c in ['PF OA', 'Fracture']})
ENSEMBLE_RULESET.update({c: 'model_b' for c in ['Effusion', 'Synovitis', 'Contusion']})

# API settings
# Sampling temperature for the majority-vote samples (all N samples use this;
# the vote over 3 samples is the determinism mechanism, not temperature=0).
SAMPLE_TEMPERATURE         = 0.4
DEFAULT_TEMPERATURE        = 0.0 # kept for reference / single-sample calls
DEFAULT_SAMPLES_PER_REPORT = 3   # majority-vote samples per report

# 2 local servers x 3 slots each (phi4 deployment, --parallel 3 - backed off
# further from 6 after nvidia-smi dmon showed sm~100%/mem~20%, i.e.
# compute-bound on prefill contention, not VRAM-bound) = 6 concurrent
# requests. Keep this in sync with whichever --parallel value the
# currently-deployed llama-server processes use (see
# notebooks/02-label_extraction.ipynb section 1.3). Only applies to the
# single-model round-robin pool (ENV_LOCAL_URLS); in ENSEMBLE_MODE, each
# side is one dedicated server (phi4 --parallel 4, qwen7B --parallel 4) so
# effective concurrency is capped by the smaller of the two - pass
# --workers 4 when running with ENSEMBLE_MODE=true.
DEFAULT_WORKERS        = 8
SLOTS_PER_LOCAL_SERVER = 3
TIMEOUT_SECONDS        = 600

# Hard cap on completion length: 12 conditions + short evidence quotes fit
# comfortably well under this. Without a cap, a model stuck in a repetition
# loop (missing EOS) runs until TIMEOUT_SECONDS instead of failing fast.
MAX_TOKENS = 1024

# Flat retry policy: on any API error, sleep this long and retry once.
# (No token bucket, no exponential backoff, the gateway is ours to scale.)
RETRY_SLEEP_SECONDS = 15

# Output
LABEL_SOURCE_GOLD   = 'gold'
LABEL_SOURCE_PSEUDO = 'pseudo'
MERGED_CSV_NAME     = 'train_labeled_v2.csv'

# Calibration (58 gold rows) output subdirectory, one JSON per uid:
# {cond: [value, evidence, agreement]}
CALIBRATION_DIR_NAME = 'calibration_rows'
