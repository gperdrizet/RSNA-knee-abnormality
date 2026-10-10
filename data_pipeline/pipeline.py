'''
CLI for the data pipeline stages.

Usage:
    python3 -m data_pipeline.pipeline build --run 20260705_001 --workers 8
    python3 -m data_pipeline.pipeline preprocess --run 20261009_full --workers 8

Subcommands:
    build         Run the label extraction job for one run id
                  (extraction + dataset build)
    calibrate     Run extraction against just the 58 gold rows, capturing
                  value/evidence/agreement per condition (data/pipeline/
                  label_extraction/<run>/calibration_rows/*.json)
    preprocess    Run the image preprocessing job for one run id
                  (align/normalize/resample every series to standardized
                  NIfTI volumes + manifest with pseudo labels)

Every run id writes to data/pipeline/<stage>/<run>/ so runs are
side-by-side and easy to diff.  Each item's output is cached at the
file level, so re-running the same run id skips completed work.

Luigi scheduler: runs use a local (in-memory) scheduler by default so it
never depends on a remote/redis setup.
'''

import argparse
from typing import Optional, Sequence

import luigi

from data_pipeline.image_preprocessing import tasks as preproc_tasks
from data_pipeline.label_extraction import tasks
from data_pipeline.logging_config import setup_logging


def build_parser() -> argparse.ArgumentParser:
    '''Creates and returns the argument parser for the CLI.'''

    p = argparse.ArgumentParser(description='Label extraction pipeline')

    p.add_argument(
        '--logging-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity'
    )

    sub = p.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)

    common.add_argument(
        '--run', required=True,
        help='Run id (free-form, e.g. 20260705_001); output goes to '
        'data/pipeline/label_extraction/<run>/'
    )

    common.add_argument(
        '--workers', type=int, default=8,
        help='Number of worker threads used by Luigi (default 8)'
    )
    
    # Local scheduler: in-memory, no external services.
    common.add_argument(
        '--local-scheduler', action='store_true', default=True,
        help='Use an in-memory scheduler (default)'
    )

    sp = sub.add_parser(
        'build', parents=[common],
        help='Run the build job (extraction + merged dataset) for one run'
    )

    sub.add_parser(
        'calibrate', parents=[common],
        help='Re-run extraction against the 58 gold rows for one run'
    )

    pp = sub.add_parser(
        'preprocess', parents=[common],
        help='Run the image preprocessing job for one run '
             '(align/normalize/resample series to NIfTI + manifest)'
    )

    pp.add_argument(
        '--splits', default='train,test',
        help='Comma-separated splits to process (default: train,test)'
    )

    pp.add_argument(
        '--limit', type=int, default=0,
        help='Only process the first N series of each split (debugging; '
             'default 0 = all)'
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging(args.logging_level)

    if args.command == 'build':
        task_objs = [tasks.BuildLabeledDataset(run_id=args.run)]

    elif args.command == 'calibrate':
        task_objs = [tasks.RunCalibration(run_id=args.run)]

    elif args.command == 'preprocess':
        task_objs = [preproc_tasks.BuildManifest(
            run_id=args.run, splits=args.splits, limit=args.limit
        )]

    luigi.build(
        task_objs,
        workers=args.workers,
        local_scheduler=True
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
