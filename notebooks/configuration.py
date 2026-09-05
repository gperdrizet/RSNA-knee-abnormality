'''
Configuration file for notebook paths and settings.
'''

import os


# Raw data paths
DATA_DIR         = os.path.join('data', 'raw')
TRAIN_CSV        = os.path.join(DATA_DIR, 'train.csv')
TRAIN_SERIES_CSV = os.path.join(DATA_DIR, 'train_series.csv')
TEST_CSV         = os.path.join(DATA_DIR, 'test.csv')
TEST_SERIES_CSV  = os.path.join(DATA_DIR, 'test_series.csv')