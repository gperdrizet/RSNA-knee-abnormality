'''
Configuration file for notebook paths and settings.
'''
import sys
from pathlib import Path

# Make the repo root importable
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Raw data paths
DATA_DIR         = '../data/raw'
TRAIN_CSV        = f'{DATA_DIR}/train.csv'
TRAIN_SERIES_CSV = f'{DATA_DIR}/train_series.csv'
TEST_CSV         = f'{DATA_DIR}/test.csv'
TEST_SERIES_CSV  = f'{DATA_DIR}/test_series.csv'

# Pseudolabeled data paths
PIPELINE_DATA_DIR    = '../data/pipeline'
LABEL_EXTRACTION_DIR = f'{PIPELINE_DATA_DIR}/label_extraction'
EXTRACTION_RUN_DIR   = f'{LABEL_EXTRACTION_DIR}/20260910_ensemble_full'
MERGED_CSV_PATH      = f'{EXTRACTION_RUN_DIR}/train_labeled_v2.csv'
CLEANED_CSV_PATH     = '../data/pseudolabeled-train.csv'

# Imaging data paths
IMAGING_DATA_DIR = '../data/raw/train_series'