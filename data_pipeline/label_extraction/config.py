'''
Configuration constants for label extraction.
'''

from pathlib import Path

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
# its own model (4 slots each). Work is split across these round-robin.
ENV_LOCAL_API_KEY      = 'LOCAL_API_KEY'
ENV_LOCAL_MODEL        = 'LOCAL_MODEL'
ENV_LOCAL_URLS         = ['LOCAL_URL_A', 'LOCAL_URL_B']
SLOTS_PER_LOCAL_SERVER = 4

# API settings
# Sampling temperature for the majority-vote samples (all N samples use this;
# the vote over 3 samples is the determinism mechanism, not temperature=0).
SAMPLE_TEMPERATURE         = 0.4
DEFAULT_TEMPERATURE        = 0.0       # kept for reference / single-sample calls
DEFAULT_SAMPLES_PER_REPORT = 3  # majority-vote samples per report

# 2 local servers x 4 slots each = 8 concurrent requests fit comfortably.
DEFAULT_WORKERS = 8
TIMEOUT_SECONDS = 600

# Flat retry policy: on any API error, sleep this long and retry once.
# (No token bucket, no exponential backoff — the gateway is ours to scale.)
RETRY_SLEEP_SECONDS = 15

# Output
LABEL_SOURCE_GOLD   = 'gold'
LABEL_SOURCE_PSEUDO = 'pseudo'
MERGED_CSV_NAME     = 'train_labeled_v2.csv'

# Calibration (58 gold rows) output subdirectory, one JSON per uid:
# {cond: [value, evidence, agreement]}
CALIBRATION_DIR_NAME = 'calibration_rows'
