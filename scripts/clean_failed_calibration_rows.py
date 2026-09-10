#!/usr/bin/env python3
'''
Remove failed calibration rows for a label-extraction run so calibration can
be safely re-run.

Deletes, under data/pipeline/label_extraction/<run_id>/:
  - calibration_rows/{uid}.json for each uid listed in failed_uids.txt
  - calibration_rows/_complete (the run's completion marker)
  - failed_uids.txt itself

Writes a simplified one-line-per-uid summary of the failures to
cleaned_errors.txt for future reference before deleting failed_uids.txt.

Usage:
    python scripts/clean_failed_calibration_rows.py <run_id>
'''

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / 'data' / 'pipeline' / 'label_extraction'
CALIBRATION_DIR_NAME = 'calibration_rows'

# Collapses the noisy per-sample JSON error blob down to a short label.
_ERROR_PATTERNS = [
    (re.compile(r'Context size has been exceeded'), 'context_size_exceeded'),
    (re.compile(r'length limit was reached'), 'length_limit_reached'),
    (re.compile(r'^unparseable$'), 'unparseable'),
]


def simplify_reason(reason: str) -> str:
    for pattern, label in _ERROR_PATTERNS:
        if pattern.search(reason):
            return label

    return reason.strip().splitlines()[0][:80]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', help='Run id, e.g. 20260910_phi4_prompt_v2')
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.run_id
    calibration_dir = run_dir / CALIBRATION_DIR_NAME
    failed_uids_path = run_dir / 'failed_uids.txt'
    complete_marker = calibration_dir / '_complete'
    cleaned_errors_path = run_dir / 'cleaned_errors.txt'

    if not run_dir.is_dir():
        sys.exit(f'No such run directory: {run_dir}')

    if not failed_uids_path.exists():
        sys.exit(f'No failed_uids.txt found at {failed_uids_path}; nothing to clean')

    reasons_by_uid: dict[str, str] = {}

    with open(failed_uids_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\n')

            if not line:
                continue

            uid, _, reason = line.partition('\t')
            reasons_by_uid.setdefault(uid, simplify_reason(reason))

    with open(cleaned_errors_path, 'w', encoding='utf-8') as fh:
        for uid, reason in sorted(reasons_by_uid.items()):
            fh.write(f'{uid}\t{reason}\n')

    print(f'Wrote {len(reasons_by_uid)} simplified errors to {cleaned_errors_path}')

    removed, missing = 0, 0

    for uid in reasons_by_uid:
        row_path = calibration_dir / f'{uid}.json'

        if row_path.exists():
            row_path.unlink()
            removed += 1
        else:
            missing += 1

    print(f'Removed {removed} calibration row(s); {missing} already absent')

    if complete_marker.exists():
        complete_marker.unlink()
        print(f'Removed {complete_marker}')

    failed_uids_path.unlink()
    print(f'Removed {failed_uids_path}')


if __name__ == '__main__':
    main()
