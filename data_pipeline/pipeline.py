'''
CLI for the key-label extraction pipeline.

Usage:
    python3 -m data_pipeline.pipeline build --run 20260705_001 --workers 8

Subcommands:
    build      Run the full job for one run id (extraction + dataset build)

Every run id writes to data/pipeline/label_extraction/<run>/ so runs are
side-by-side and easy to diff.  Each UID's output is cached at the
file level, so re-running the same run id skips completed labels.

Luigi scheduler: runs use a local (in-memory) scheduler by default so it
never depends on a remote/redis setup.
'''

import argparse
from typing import Optional, Sequence

import luigi

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

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging(args.logging_level)

    if args.command == 'build':
        task_objs = [tasks.BuildLabeledDataset(run_id=args.run)]

    luigi.build(
        task_objs,
        workers=args.workers,
        local_scheduler=True
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
