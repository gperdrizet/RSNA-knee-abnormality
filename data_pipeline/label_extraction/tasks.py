'''
Luigi tasks for label extraction.

Graph (one job):

    TrainDataAvailable
        |
        +----> ExtractLabels(run_id, uid)   # one per unlabeled report
                   |
                   +----> BuildLabeledDataset(run_id)  # train_labeled_v2.csv

``ExtractLabels`` is idempotent at the file level: each task writes
``labels/{uid}.csv`` atomically (json + tmp + rename), so an interrupted
run can be resumed - Luigi simply skips tasks whose output already exists.
Rows where every LLM sample failed are written with null labels +
``label_status='failed'`` and appended to ``failed_uids.txt`` so the
dataset build still completes; 402s stop the whole run.

``CalibrateGoldRow(run_id, uid)`` mirrors ``ExtractLabels`` but runs against
the 58 gold reports instead, writing ``calibration_rows/{uid}.json`` with
``{cond: [value, evidence, agreement]}`` for the calibration notebook to
load directly - no separate backfill script needed to re-run calibration.
'''

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import luigi
import pandas as pd

from data_pipeline.label_extraction import config, data, llm

logger = logging.getLogger(__name__)

_FAILED_LOCK = threading.Lock()


def run_dir(run_id: str) -> Path:
    d = config.OUTPUT_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)

    return d


def labels_dir(run_id: str) -> Path:
    d = run_dir(run_id) / 'labels'
    d.mkdir(parents=True, exist_ok=True)

    return d


def calibration_dir(run_id: str) -> Path:
    d = run_dir(run_id) / config.CALIBRATION_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)

    return d


def record_failure(run_id: str, uid: str, reason: str) -> None:
    '''Append a failed uid + reason to the run's failed_uids.txt (thread-safe).'''

    f = run_dir(run_id) / 'failed_uids.txt'

    with _FAILED_LOCK:
        with open(f, 'a', encoding='utf-8') as fh:
            fh.write(f'{uid}\t{reason}\n')


class TrainDataAvailable(luigi.Task):
    '''Static dependency: the raw train.csv exists.'''

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(str(config.TRAIN_CSV))

    def run(self) -> None:
        if not self.output().exists():
            raise FileNotFoundError(f'Missing input: {config.TRAIN_CSV}')


class ExtractLabels(luigi.Task):
    '''Extract pseudo labels for one unlabeled report.'''

    run_id = luigi.Parameter()
    uid = luigi.Parameter()

    def requires(self):
        return TrainDataAvailable()

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(str(labels_dir(self.run_id) /
                                     f'{self.uid}.csv'))

    def run(self) -> None:
        reports = data.load_train_reports()
        row = reports.loc[reports['StudyInstanceUID'] == self.uid]

        if row.empty:
            raise ValueError(f'UID {self.uid} not found in train.csv')
    
        row = row.iloc[0]

        if bool(row['is_labeled']):
            raise ValueError(f'UID {self.uid} is already gold-labeled')

        report = row['Report']

        if not isinstance(report, str) or not report.strip():
            raise ValueError(f'UID {self.uid}: missing/empty report')

        logger.info(
            'ExtractLabels start uid=%s run=%s',
            self.uid, self.run_id
        )

        status = 'pseudo'

        try:
            result = llm.extract_report(report, uid=self.uid)

            if result.get('unparseable'):
                status = 'unparseable'
    
        except llm.BalanceExhaustedError:

            # Fail fast: the run must stop until the 402 cause is fixed.
            raise

        except llm.ExtractionError as e:
            logger.critical(
                'ExtractLabels giving up on uid=%s: %s',
                self.uid, e
            )

            record_failure(self.run_id, self.uid, f'extraction_error: {e}')

            result = {'unparseable': True, 'labels': {}, 'agreement': {}}
            status = 'failed'

        if status != 'pseudo':
            record_failure(self.run_id, self.uid, status)

        records = result_to_records(self.uid, result)
        records['label_status'] = status
        out = Path(self.output().path)
        tmp = out.with_suffix(out.suffix + '.tmp')

        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(records, fh)

        tmp.replace(out)


def result_to_records(uid: str, result: dict) -> dict:
    '''Flatten an extraction result into an output record (dict).

    Evidence strings live ONLY in these per-UID files, not in the merged
    CSV - they are for spot-checking, not modelling.
    '''

    row = {'StudyInstanceUID': uid}

    for cond in config.CONDITIONS:
        if result.get('unparseable'):
            row[cond] = None
            row[f'{cond}_agreement'] = None
            row[f'{cond}_evidence'] = None

        else:
            v, ev = result['labels'][cond]
            row[cond] = v
            row[f'{cond}_agreement'] = result.get('agreement', {}).get(cond)
            row[f'{cond}_evidence'] = ev

    return row


class CalibrateGoldRow(luigi.Task):
    '''Extract labels + agreement for one gold-labeled report (calibration).'''

    run_id = luigi.Parameter()
    uid = luigi.Parameter()

    def requires(self):
        return TrainDataAvailable()

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(str(calibration_dir(self.run_id) /
                                     f'{self.uid}.json'))

    def run(self) -> None:
        reports = data.load_train_reports()
        row = reports.loc[reports['StudyInstanceUID'] == self.uid]

        if row.empty:
            raise ValueError(f'UID {self.uid} not found in train.csv')

        row = row.iloc[0]

        if not bool(row['is_labeled']):
            raise ValueError(f'UID {self.uid} is not gold-labeled')

        report = row['Report']

        if not isinstance(report, str) or not report.strip():
            raise ValueError(f'UID {self.uid}: missing/empty report')

        logger.info(
            'CalibrateGoldRow start uid=%s run=%s',
            self.uid, self.run_id
        )

        try:
            result = llm.extract_report(report, uid=self.uid)

        except llm.BalanceExhaustedError:
            raise

        except llm.ExtractionError as e:
            logger.critical(
                'CalibrateGoldRow giving up on uid=%s: %s',
                self.uid, e
            )

            record_failure(self.run_id, self.uid, f'extraction_error: {e}')
            result = {'unparseable': True, 'labels': {}, 'agreement': {}}

        if result.get('unparseable'):
            record_failure(self.run_id, self.uid, 'unparseable')
            record = {}
        else:
            record = {
                cond: [value, evidence, result['agreement'].get(cond, 1.0)]
                for cond, (value, evidence) in result['labels'].items()
            }

        out = Path(self.output().path)
        tmp = out.with_suffix(out.suffix + '.tmp')

        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(record, fh)

        tmp.replace(out)


class RunCalibration(luigi.Task):
    '''Umbrella task: calibrate every gold row for one run id.'''

    run_id = luigi.Parameter()

    def requires(self):
        reports = data.load_train_reports()
        gold_uids = reports.loc[reports['is_labeled'], 'StudyInstanceUID']

        return [TrainDataAvailable()] + [
            CalibrateGoldRow(run_id=self.run_id, uid=u) for u in gold_uids
        ]

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(
            str(calibration_dir(self.run_id) / '_complete'))

    def run(self) -> None:
        meta = {
            'run_id': self.run_id,
            'model': os.environ.get(config.ENV_LOCAL_MODEL, 'default'),
            'urls': [os.environ[name] for name in config.ENV_LOCAL_URLS
                     if os.environ.get(name)],
            'samples_per_report': config.DEFAULT_SAMPLES_PER_REPORT,
            'completed_at': datetime.now(timezone.utc).isoformat(),
        }

        meta_path = run_dir(self.run_id) / 'run_meta.json'
        meta_path.write_text(json.dumps(meta, indent=2))

        with open(self.output().path, 'w', encoding='utf-8') as fh:
            fh.write('done\n')


class BuildLabeledDataset(luigi.Task):
    '''Merge gold rows + pseudo rows into ``train_labeled_v2.csv``.'''

    run_id = luigi.Parameter()

    def requires(self):
        reports = data.load_train_reports()
        unlabeled = reports.loc[~reports['is_labeled'], 'StudyInstanceUID']
        deps = [TrainDataAvailable()]

        deps.extend(
            ExtractLabels(run_id=self.run_id, uid=u)
            for u in unlabeled
        )

        return deps

    def output(self) -> luigi.Target:

        return luigi.LocalTarget(
            str(run_dir(self.run_id) / config.MERGED_CSV_NAME))

    def run(self) -> None:
        reports = data.load_train_reports()
        raw = pd.read_csv(config.TRAIN_CSV)
        label_cols = [c for c in raw.columns
                      if c not in ('StudyInstanceUID', 'Report')]
    
        gold_mask = reports['is_labeled'].values

        # Gold block: original labels, plus a provenance column.
        gold = raw.loc[gold_mask].copy()
        gold['label_source'] = config.LABEL_SOURCE_GOLD

        # Pseudo block: read the per-UID output files.
        pseudo_rows = []

        for f in sorted(labels_dir(self.run_id).glob('*.csv')):
            if f.name.endswith('.tmp'):
                continue

            with open(f, encoding='utf-8') as fh:
                pseudo_rows.append(json.load(fh))

        if not pseudo_rows:
            raise RuntimeError('No pseudo-label outputs found')

        pseudo = pd.DataFrame(pseudo_rows)
        unlabeled_uids = list(raw.loc[~gold_mask, 'StudyInstanceUID'])
        missing = set(unlabeled_uids) - set(pseudo['StudyInstanceUID'])

        if missing:
            raise ValueError(
                f'{len(missing)} unlabeled UIDs have no output; '
                f'e.g. {list(missing)[:3]}'
            )

        pseudo = pseudo.set_index('StudyInstanceUID').loc[unlabeled_uids] \
            .reset_index()

        pseudo['label_source'] = config.LABEL_SOURCE_PSEUDO
        pseudo = pseudo.rename(columns={'label_status': 'pseudo_status'})

        merged = pd.concat([gold, pseudo], ignore_index=True)

        n_fail = int(pseudo['pseudo_status'].isin(
            ['failed', 'unparseable']).sum()
        )

        logger.info(
            'Merge: %d rows total (%d gold, %d pseudo, %d '
            'failed/unparseable)', len(merged), len(gold),
            len(pseudo), n_fail
        )

        assert len(merged) == len(raw), 'merged row count != train row count'

        # Gold integrity: labels must match the originals exactly.
        merged_g = (
            merged.loc[merged['label_source'] == config.LABEL_SOURCE_GOLD]
            .sort_values('StudyInstanceUID').reset_index(drop=True)
        )

        gold_orig = gold.sort_values('StudyInstanceUID').reset_index(drop=True)

        for col in label_cols:
            assert (merged_g[col].reset_index(drop=True) == gold_orig[col].reset_index(drop=True)).all(), \
                f'Gold integrity check failed for column {col}'
        
        assert (merged_g['StudyInstanceUID'].reset_index(drop=True).tolist() == gold_orig['StudyInstanceUID'].tolist()), \
            'Gold integrity check failed for StudyInstanceUID'

        merged.to_csv(self.output().path, index=False)
        logger.info('Wrote %s (%d rows)', self.output().path, len(merged))
