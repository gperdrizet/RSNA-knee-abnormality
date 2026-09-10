'''Helper functions for 02-label_extraction.ipynb section 2 (calibration review).'''

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

import data_pipeline.label_extraction.config as pipeline_config


def load_calibration_cmp(run_id: str, label_cols: list, gold: pd.DataFrame) -> pd.DataFrame:
    '''Load one calibration run's predictions joined against gold.'''
    cdir = pipeline_config.OUTPUT_ROOT / run_id / 'calibration_rows'
    rows = []

    for f in sorted(cdir.glob('*.json')):
        pred = json.loads(f.read_text())
        rec = {'StudyInstanceUID': f.stem}

        for c in label_cols:
            v, *_ = pred.get(c, [0, '', 1.0])
            rec[c] = v

        rows.append(rec)

    preds_r = pd.DataFrame(rows).set_index('StudyInstanceUID').loc[gold.index]

    return gold.join(preds_r, lsuffix='_label', rsuffix='_pred')


def run_model_name(run_id: str) -> str:
    meta_f = pipeline_config.OUTPUT_ROOT / run_id / 'run_meta.json'

    if meta_f.exists():
        return json.loads(meta_f.read_text()).get('model', run_id)

    return run_id


def failed_uid_count(run_id: str) -> int:
    '''Rows a run gave up on (e.g. context overflow) - these load as empty/
    all-zero predictions, so a non-zero count means degraded, not missing,
    data. See failed_uids.txt written by CalibrateGoldRow/record_failure.'''
    f = pipeline_config.OUTPUT_ROOT / run_id / 'failed_uids.txt'

    return len(f.read_text().splitlines()) if f.exists() else 0


def kappa(g, p) -> float:
    '''Cohen's kappa.'''

    g, p = np.asarray(g), np.asarray(p)
    po = (g == p).mean()
    pg = g.mean(); pp = p.mean()
    pe = pg * pp + (1 - pg) * (1 - pp)

    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def per_condition_metrics(df: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    '''Per-condition accuracy/kappa/AUC/AP/error-counts for one trial.'''
    recs = []

    for c in label_cols:
        g = (df[f'{c}_label'].fillna(0) == 1).astype(int).values
        p = (df[f'{c}_pred'].fillna(0) == 1).astype(int).values

        recs.append({
            'condition': c,
            'gold_pos': int(g.sum()),
            'pred_pos': int(p.sum()),
            'accuracy': float((g == p).mean()),
            'kappa': kappa(g, p),
            'roc_auc': roc_auc_score(g, p),
            'avg_precision': average_precision_score(g, p),
            'false_neg': int(((g == 1) & (p == 0)).sum()),
            'false_pos': int(((g == 0) & (p == 1)).sum()),
        })

    return pd.DataFrame(recs).set_index('condition')


def mcnemar_bias(df: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    '''Per-condition McNemar exact test on the discordant (gold vs. LLM) pairs.'''
    recs = []

    for c in label_cols:
        g = (df[f'{c}_label'].fillna(0) == 1).astype(int).values
        p = (df[f'{c}_pred'].fillna(0) == 1).astype(int).values

        false_pos = int(((g == 0) & (p == 1)).sum())  # labeler over-calls
        false_neg = int(((g == 1) & (p == 0)).sum())  # labeler under-calls
        n_discordant = false_pos + false_neg

        # McNemar table: rows/cols don't matter, only the off-diagonal (discordant) counts.
        table = [[0, false_neg], [false_pos, 0]]
        pvalue = mcnemar(table, exact=True).pvalue if n_discordant > 0 else float('nan')

        recs.append({
            'condition': c,
            'false_pos': false_pos,
            'false_neg': false_neg,
            'n_discordant': n_discordant,
            'bias_direction': 'over-calls' if false_pos > false_neg else ('under-calls' if false_neg > false_pos else 'balanced'),
            'mcnemar_p': pvalue,
        })

    bias_tbl = pd.DataFrame(recs).set_index('condition').sort_values('mcnemar_p')
    bias_tbl['flag'] = np.where(bias_tbl['mcnemar_p'] < 0.05, '*** significant bias ***', '')

    return bias_tbl


def bias_severity(bias_tbl: pd.DataFrame) -> float:
    '''Sum of -log10(p) over significantly-biased conditions - rewards both
    fewer and less-confident biases, unlike a plain significant-count.'''
    sig_p = bias_tbl.loc[bias_tbl['mcnemar_p'] < 0.05, 'mcnemar_p']

    return float((-np.log10(sig_p)).sum())
