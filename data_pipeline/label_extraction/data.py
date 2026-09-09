'''
Loading the raw training data.

``load_train_reports`` returns a small (uid, report, is_labeled) frame.
The parse result is cached in-process, keyed on the CSV mtime, because the
Luigi graph calls it once per task (4,349 times) and re-parsing each time
would waste minutes.
'''

from __future__ import annotations

import pandas as pd

from data_pipeline.label_extraction import config

_reports_cache: pd.DataFrame | None = None
_reports_cache_mtime: float | None = None


def load_train_reports() -> pd.DataFrame:
    """Return (StudyInstanceUID, Report, is_labeled) for the whole dataset."""
    global _reports_cache, _reports_cache_mtime
    mtime = config.TRAIN_CSV.stat().st_mtime
    if _reports_cache is not None and _reports_cache_mtime == mtime:
        return _reports_cache
    df = pd.read_csv(config.TRAIN_CSV)
    label_cols = [c for c in df.columns
                  if c not in ('StudyInstanceUID', 'Report')]
    _reports_cache = pd.DataFrame({
        'StudyInstanceUID': df['StudyInstanceUID'],
        'Report': df['Report'],
        'is_labeled': df[label_cols].notna().any(axis=1),
    })
    _reports_cache_mtime = mtime
    return _reports_cache


def unlabeled_uids() -> list[str]:
    """UIDs that do not yet have gold labels (the work to be done)."""
    return list(load_train_reports().loc[
        ~load_train_reports()['is_labeled'], 'StudyInstanceUID'])
