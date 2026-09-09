'''
Logging configuration for the data pipeline.

Sets up the ``data_pipeline`` root logger with:
    - an INFO stream handler to stderr (console)
    - a DEBUG rotating file handler under ``logs/``

Usage:
    from data_pipeline.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
'''

import logging
import logging.handlers
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOG_FILE = LOGS_DIR / 'data_pipeline.log'

_FMT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
_DATEFMT = '%Y-%m-%d %H:%M:%S'

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    '''Configure the ``data_pipeline`` root logger (idempotent).'''

    global _configured

    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger('data_pipeline')
    root.setLevel(logging.DEBUG)

    # Console: INFO and above
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(console)

    # File: DEBUG, rotated
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=50_000_000, backupCount=5, encoding='utf-8'
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(file_handler)

    _configured = True
    root.info('Logging configured. File: %s', LOG_FILE)
