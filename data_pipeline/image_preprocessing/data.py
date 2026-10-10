'''
Series metadata loading for the image preprocessing stage.

``load_series_meta(split)`` returns the per-series frame
(StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression,
Anatomical_Plane) for one split; ``load_pseudolabels()`` returns the
study-level pseudo labels from the label extraction stage. Both are
cached in-process keyed on file mtime (the Luigi graph calls them once
per task).
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.image_preprocessing import config

_series_cache: dict[str, tuple[pd.DataFrame, float]] = {}
_pseudo_cache: tuple[pd.DataFrame, float] | None = None


def load_series_meta(split: str) -> pd.DataFrame:
    '''Return the series metadata frame for one split ('train'/'test').'''

    if split not in config.SPLITS:
        raise ValueError(f'unknown split {split!r}; expected one of {config.SPLITS}')

    csv = config.TRAIN_SERIES_CSV if split == 'train' else config.TEST_SERIES_CSV
    mtime = csv.stat().st_mtime

    cached = _series_cache.get(split)
    if cached is not None and cached[1] == mtime:
        return cached[0]

    df = pd.read_csv(csv)
    cols = ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive',
            'Fat_Suppression', 'Anatomical_Plane']
    _series_cache[split] = (df[cols].reset_index(drop=True), mtime)
    return _series_cache[split][0]


def load_all_series() -> pd.DataFrame:
    '''Series metadata for all splits with a ``split`` column.'''

    frames = [df.assign(split=s) for s, df in
              ((s, load_series_meta(s)) for s in config.SPLITS)]
    return pd.concat(frames, ignore_index=True)


def series_dir(split: str, study_uid: str, series_uid: str) -> str:
    '''Filesystem path of one DICOM series directory.'''

    root = config.TRAIN_SERIES_DIR if split == 'train' else config.TEST_SERIES_DIR
    return str(root / study_uid / series_uid)


def load_pseudolabels() -> pd.DataFrame:
    '''Study-level labels (StudyInstanceUID + 12 conditions + label_source)
    from the label extraction stage.'''

    global _pseudo_cache
    mtime = config.PSEUDOLABELED_CSV.stat().st_mtime

    if _pseudo_cache is not None and _pseudo_cache[1] == mtime:
        return _pseudo_cache[0]

    df = pd.read_csv(config.PSEUDOLABELED_CSV)

    # The cleaned CSV (notebooks/03-label-cleanup.ipynb) drops the
    # label_source column but keeps pseudo_status, which is null exactly
    # on the 58 gold rows (gold rows keep their original labels and never
    # receive a pseudo_status).  Reconstruct label_source from that.
    if 'label_source' not in df.columns:
        df['label_source'] = np.where(
            df['pseudo_status'].isna(), 'gold', 'pseudo'
        )

    keep = (['StudyInstanceUID', 'label_source']
            + [c for c in df.columns
               if c not in ('StudyInstanceUID', 'label_source', 'Report')
               and not c.endswith('_agreement') and not c.endswith('_evidence')])
    _pseudo_cache = (df[keep].reset_index(drop=True), mtime)
    return _pseudo_cache[0]
