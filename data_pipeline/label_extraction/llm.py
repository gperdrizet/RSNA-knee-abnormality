'''
LLM extraction for the RSNA knee-abnormality label-extraction pipeline.

Uses LangChain's ``with_structured_output`` to get typed Pydantic objects
back from the model.  Each report is sampled 3×; per-condition majority
vote produces the final label.

Architecture kept intentionally flat for a bootcamp audience: one file,
clear functions, no abstractions beyond what the framework gives us.
'''

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from data_pipeline.label_extraction import config
from data_pipeline.label_extraction.prompt import build_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema (the "shape" we ask the model to produce)
# ---------------------------------------------------------------------------


class ConditionLabel(BaseModel):
    '''Label for a single condition.'''

    value: int = Field(ge=0, le=1, description="1 if present, 0 if absent")
    evidence: str = Field(default="", description="Short verbatim quote from the report supporting the value; empty string if none")

    class Config:
        title = "ConditionLabel"


class ReportLabels(BaseModel):
    '''Structured output: one entry per condition plus an unparseable flag.'''

    unparseable: bool = Field(default=False, description="True only if the report is truncated, gibberish, or clearly not a knee MRI report")

    labels: Dict[str, ConditionLabel] = Field(
        description=f"One entry per each of these 12 conditions: {', '.join(config.CONDITIONS)}"
    )

    class Config:
        title = "ReportLabels"


# ---------------------------------------------------------------------------
# Client (module-level lazy singleton)
# ---------------------------------------------------------------------------

_client: Optional[ChatOpenAI] = None


def get_client() -> ChatOpenAI:
    '''Return a shared ChatOpenAI client or create one on first call.'''

    global _client

    if _client is None:
        load_dotenv(os.path.join(config.REPO_ROOT, '.env'))

        _client = ChatOpenAI(
            base_url=os.environ['BASE_URL'],
            api_key=os.environ['API_KEY'],
            model=os.environ.get('MODEL', 'default'),
            temperature=config.DEFAULT_TEMPERATURE,
            timeout=600,
        )

    return _client


# ---------------------------------------------------------------------------
# Extraction (single report → N samples → majority vote)
# ---------------------------------------------------------------------------


def extract_report(report: str, uid: str = "?") -> Dict:
    '''Call the LLM n_samples× and return a majority-vote result.

    Returns a dict of the same shape the old ``extract.py`` produced:
        {"unparseable": bool,
         "labels": {cond: (value:int, evidence:str)},
         "agreement": {cond: fraction:float}}

    Raises ``ExtractionError`` if ALL samples fail (transport or
    schema-validation), so the caller can log and skip.
    '''

    n = config.DEFAULT_SAMPLES_PER_REPORT

    # Sample at temp 0.4 for all N calls (matches calibration behavior);
    # the majority vote over 3 samples is the determinism mechanism.
    client = get_client().model_copy(
        update={'temperature': config.SAMPLE_TEMPERATURE}
    )

    sc = client.with_structured_output(ReportLabels)
    messages = build_messages(report)
    tag = f'uid={uid}'

    samples: List[ReportLabels] = []
    errors: List[str] = []

    for i in range(n):
        try:
            result: ReportLabels = sc.invoke(messages)
            samples.append(result)

        except Exception as e:
            if _is_balance_error(e):
                logger.critical(
                    'Balance exhausted (402) on %s — stopping run. '
                    'Top up and re-run; completed rows are cached.', tag
                )

                raise BalanceExhaustedError('HTTP 402 from the gateway') from e

            # Simple fixed retry: sleep 15s, retry once more
            logger.warning(
                'Sample %d/%d %s failed: %s (retrying once)',
                i + 1, n, tag, e
            )

            time.sleep(config.RETRY_SLEEP_SECONDS)

            try:
                result = sc.invoke(messages)
                samples.append(result)

            except BalanceExhaustedError:
                raise

            except Exception as e2:
                errors.append(f'sample {i+1}/{n}: {e2}')

                logger.error(
                    'Sample %d/%d %s failed after retry: %s',
                    i + 1, n, tag, e2
                )

    if not samples:
        raise ExtractionError(
            f'All {n} samples failed for {tag}: {errors}')

    # Majority vote
    if len(samples) == 1:
        s = samples[0]

        if s.unparseable:
            return {
                'unparseable': True, 'labels': {},
                'agreement': {c: 1.0 for c in config.CONDITIONS}
            }

        labels = {
            c: (s.labels[c].value, s.labels[c].evidence)
            for c in config.CONDITIONS if c in s.labels
        }

        agreement = {c: 1.0 for c in config.CONDITIONS}

        return {'unparseable': False, 'labels': labels, 'agreement': agreement}

    return _majority_vote(samples, tag)


class ExtractionError(Exception):
    '''Raised when all LLM samples for a report fail.'''
    pass


class BalanceExhaustedError(Exception):
    '''402 from the gateway, the run must stop, not retry.'''
    pass


def _is_balance_error(e: Exception) -> bool:
    '''True if the exception is an API 402 (payment / balance exhausted).'''

    return getattr(e, 'status_code', None) == 402 or '402' in type(e).__name__


def _majority_vote(samples: List[ReportLabels], tag: str = "?") -> Dict:
    '''Per-condition majority vote across N structured-output samples.'''

    parseable = [s for s in samples if not s.unparseable]

    if not parseable:
        return {
            'unparseable': True, 'labels': {},
            'agreement': {c: 1.0 for c in config.CONDITIONS}
        }

    labels: Dict[str, tuple] = {}
    agreement: Dict[str, float] = {}

    for cond in config.CONDITIONS:
        votes: List[int] = []
        evidences: List[str] = []

        for s in parseable:
            if cond not in s.labels:
                continue

            votes.append(s.labels[cond].value)
            evidences.append(s.labels[cond].evidence)

        if not votes:
            votes = [0]
            evidences = ['']

        ones = sum(votes)
        value = 1 if ones * 2 > len(votes) else 0
        agmt = max(ones, len(votes) - ones) / len(votes)

        # Pick the longest evidence quote that agrees with the final value
        chosen_ev = ''

        for v, ev in zip(votes, evidences):
            if v == value and len(ev) > len(chosen_ev):
                chosen_ev = ev

        labels[cond] = (value, chosen_ev)
        agreement[cond] = agmt

    result = {'unparseable': False, 'labels': labels, 'agreement': agreement}
    positives = [c for c, (v, _) in labels.items() if v == 1]
    min_ag = min(agreement.values(), default=1.0)

    logger.info(
        'Extraction done %s: unparseable=False positives=%s '
        'min_agreement=%.2f failures=%d',
        tag, positives, min_ag,
        len(samples) - len(parseable)
    )

    return result
