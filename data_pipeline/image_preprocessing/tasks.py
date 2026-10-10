'''
Luigi tasks for image preprocessing.

Graph (one job per run id):

    SeriesDataAvailable
        |
        +----> PreprocessSeries(run_id, split, study_uid, series_uid)
        |           # one per series, train + test
        |
        +----> BuildManifest(run_id)   # manifest.csv + run_meta.json

``PreprocessSeries`` is idempotent at the file level: each task writes
an atomic per-series sidecar JSON (status + geometry + normalization
provenance) next to the ``.nii.gz`` volume, so an interrupted run
resumes by skipping completed series.  A series that fails to
preprocess (corrupt DICOM, degenerate geometry, zero variance) writes a
sidecar with ``status='failed'`` + reason and is appended to
``failed_series.txt`` - the manifest build still completes so failures
are visible in one place.  Delete a failed sidecar (+ its .nii.gz if
any) to re-process it.

``BuildManifest`` joins the per-series sidecars with the study-level
pseudo labels (data/pseudolabeled-train.csv from the label extraction
stage) on StudyInstanceUID, producing a single CSV that maps every
preprocessed volume to its labels - the entry point for downstream
pseudo-labeling / model training.

All output lives under ``config.OUTPUT_ROOT/<run_id>/`` so runs are
side-by-side and easy to diff.
'''

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import luigi
import numpy as np
import pandas as pd

from data_pipeline.image_preprocessing import config, data, preprocess, volume

logger = logging.getLogger(__name__)

_FAILED_LOCK = threading.Lock()


def run_dir(run_id: str) -> Path:
    d = config.OUTPUT_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)

    return d


def series_out_dir(run_id: str, split: str, study_uid: str) -> Path:
    d = run_dir(run_id) / split / study_uid
    d.mkdir(parents=True, exist_ok=True)

    return d


def sidecar_path(run_id: str, split: str, study_uid: str,
                 series_uid: str) -> Path:
    return series_out_dir(run_id, split, study_uid) / \
        (f'{series_uid}{config.SIDECAR_SUFFIX}')


def nifti_path(run_id: str, split: str, study_uid: str,
               series_uid: str) -> Path:
    return series_out_dir(run_id, split, study_uid) / \
        (f'{series_uid}{config.NIFTI_SUFFIX}')


def record_failure(run_id: str, split: str, series_uid: str,
                   reason: str) -> None:
    '''Append a failed series + reason to the run's failed_series.txt
    (thread-safe).'''

    f = run_dir(run_id) / config.FAILED_LOG_NAME

    with _FAILED_LOCK:
        with open(f, 'a', encoding='utf-8') as fh:
            fh.write(f'{split}\t{series_uid}\t{reason}\n')


def _atomic_json(path: Path, record: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')

    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1)

    tmp.replace(path)


class SeriesDataAvailable(luigi.Task):
    '''Static dependency: the raw series CSVs + DICOM trees exist.'''

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(str(config.TRAIN_SERIES_CSV))

    def run(self) -> None:
        missing = [p for p in
                   (config.TRAIN_SERIES_CSV, config.TEST_SERIES_CSV,
                    config.TRAIN_SERIES_DIR, config.TEST_SERIES_DIR)
                   if not p.exists()]

        if missing:
            raise FileNotFoundError(
                f'missing raw inputs: {[str(p) for p in missing]}'
            )


def _series_for_split(split: str, limit: int = 0) -> list[dict]:
    '''(study_uid, series_uid) rows for one split, optionally limited
    (deterministic: first N rows of the series CSV).'''

    df = data.load_series_meta(split)
    if limit > 0:
        df = df.head(limit)

    return [
        {'study_uid': r.StudyInstanceUID, 'series_uid': r.SeriesInstanceUID}
        for r in df.itertuples()
    ]


class PreprocessSeries(luigi.Task):
    '''Preprocess one DICOM series to a standardized NIfTI volume.'''

    run_id = luigi.Parameter()
    split = luigi.Parameter()
    study_uid = luigi.Parameter()
    series_uid = luigi.Parameter()

    def requires(self):
        return SeriesDataAvailable()

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(str(
            sidecar_path(self.run_id, self.split, self.study_uid,
                         self.series_uid)
        ))

    def run(self) -> None:
        src = data.series_dir(self.split, self.study_uid, self.series_uid)

        logger.info(
            'PreprocessSeries start run=%s split=%s series=%s',
            self.run_id, self.split, self.series_uid
        )

        record: dict = {
            'StudyInstanceUID': self.study_uid,
            'SeriesInstanceUID': self.series_uid,
            'split': self.split,
        }

        # Series flags from the competition metadata (for the manifest +
        # downstream sequence selection).
        meta_df = data.load_series_meta(self.split)
        row = meta_df.loc[meta_df['SeriesInstanceUID'] == self.series_uid]

        if row.empty:
            raise ValueError(
                f'series {self.series_uid} not in {self.split}_series.csv'
            )

        row = row.iloc[0]
        record.update({
            'Fluid_Sensitive': int(row['Fluid_Sensitive']),
            'Fat_Suppression': int(row['Fat_Suppression']),
            'Anatomical_Plane': row['Anatomical_Plane'],
        })

        nii = nifti_path(self.run_id, self.split, self.study_uid,
                         self.series_uid)

        try:
            img, rec = preprocess.preprocess_series(
                src, plane=str(row['Anatomical_Plane']))

        except (volume.SeriesReconstructionError, ValueError) as e:
            record['status'] = 'failed'
            record['reason'] = str(e)
            logger.critical(
                'PreprocessSeries giving up on %s: %s',
                self.series_uid, e
            )
            record_failure(self.run_id, self.split, self.series_uid, str(e))

        else:
            # Atomic: tmp then rename, so an interrupted write never
            # leaves a truncated volume.  The tmp name must end in
            # '.nii.gz' so write_nifti's suffix guard is a no-op
            # (with_suffix('.gz.tmp') produced 'x.nii.gz.tmp', which
            # write_nifti rewrote as 'x.nii.gz.tmp.nii.gz' and the
            # rename below then failed with FileNotFoundError).
            tmp_nii = nii.with_name(nii.name + '.tmp.nii.gz')
            preprocess.write_nifti(img, str(tmp_nii))
            tmp_nii.replace(nii)
            record.update(rec)
            record['nii'] = f'{self.split}/{self.study_uid}/{nii.name}'

        _atomic_json(Path(self.output().path), record)


class BuildManifest(luigi.Task):
    '''Join per-series sidecars with the study-level pseudo labels into
    ``manifest.csv`` (+ ``run_meta.json``).'''

    run_id = luigi.Parameter()
    splits = luigi.Parameter(default='train,test')
    limit = luigi.IntParameter(default=0)

    def requires(self):
        splits = self._splits()
        deps = [SeriesDataAvailable()]

        for split in splits:
            for s in _series_for_split(split, self.limit):
                deps.append(PreprocessSeries(
                    run_id=self.run_id, split=split,
                    study_uid=s['study_uid'], series_uid=s['series_uid']
                ))

        return deps

    def _splits(self) -> list[str]:
        return [s for s in self.splits.split(',') if s]

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(
            str(run_dir(self.run_id) / config.MANIFEST_NAME))

    def run(self) -> None:
        splits = self._splits()
        rows = []

        for split in splits:
            for s in _series_for_split(split, self.limit):
                p = sidecar_path(self.run_id, split, s['study_uid'],
                                 s['series_uid'])

                with open(p, encoding='utf-8') as fh:
                    rows.append(json.load(fh))

        manifest = pd.DataFrame(rows)

        # Flatten the nested provenance for a tidy wide CSV.
        def flat(r: dict) -> dict:
            out = {k: v for k, v in r.items()
                   if k not in ('volume', 'normalization')}
            vol = r.get('volume', {})
            norm = r.get('normalization', {})
            out.update({f'src_{k}': v for k, v in vol.items()})
            out.update({f'norm_{k}': v for k, v in norm.items()
                        if k != 'method'})
            out['norm_method'] = norm.get('method', '')

            return out

        manifest = manifest.apply(flat, result_type='expand', axis=1)

        # Join the study-level pseudo labels (label extraction stage).
        label_cols: list[str] = []

        if 'train' in splits:
            pseudo = data.load_pseudolabels()
            label_cols = [c for c in pseudo.columns
                          if c != 'StudyInstanceUID']
            manifest = manifest.merge(
                pseudo, on='StudyInstanceUID', how='left'
            )

        # Column order: identity, flags, status, geometry, norm, labels.
        identity = ['split', 'StudyInstanceUID', 'SeriesInstanceUID',
                    'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane',
                    'status', 'reason', 'nii']

        # 'reason' only exists on failed rows and 'nii' only on ok rows;
        # backfill so the fixed column order is always valid.
        for col in identity:
            if col not in manifest.columns:
                manifest[col] = np.nan
        rest = [c for c in manifest.columns
                if c not in identity and c not in label_cols]
        manifest = manifest[identity + rest + label_cols]

        manifest = manifest.sort_values(
            ['split', 'StudyInstanceUID', 'SeriesInstanceUID']
        ).reset_index(drop=True)

        manifest.to_csv(self.output().path, index=False)

        n_ok = int((manifest['status'] == 'ok').sum())
        n_failed = int((manifest['status'] == 'failed').sum())
        logger.info(
            'Manifest: %d series (%d ok, %d failed)',
            len(manifest), n_ok, n_failed
        )

        meta = {
            'run_id': self.run_id,
            'splits': splits,
            'limit': self.limit,
            'n_series': len(manifest),
            'n_ok': n_ok,
            'n_failed': n_failed,
            'orientation': config.TARGET_ORIENTATION,
            'target_box_voxels': config.TARGET_BOX_VOXELS,
            'voxel_spacing_mmm': list(config.VOXEL_SPACING_MMM),
            'grid_pad_voxels': config.GRID_PAD_VOXELS,
            'normalization': {
                'clip_percentiles': [config.CLIP_PERCENTILE_LOW,
                                     config.CLIP_PERCENTILE_HIGH],
                'zclip_sigma': config.ZCLIP_SIGMA,
                'output_dtype': config.OUTPUT_DTYPE,
            },
            'completed_at': datetime.now(timezone.utc).isoformat(),
        }

        (run_dir(self.run_id) / config.RUN_META_NAME).write_text(
            json.dumps(meta, indent=2)
        )
