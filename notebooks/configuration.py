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