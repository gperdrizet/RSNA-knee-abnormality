'''
LLM extraction for the RSNA knee-abnormality label-extraction pipeline.

Uses LangChain's ``with_structured_output`` to get typed Pydantic objects
back from the model.  Each report is sampled 3×; per-condition majority
vote produces the final label.

Two extraction paths share the same N-sample majority-vote core
(``_sample_and_vote``): ``extract_report`` (one model) and
``extract_report_ensemble`` (phi4 + qwen7B, combined per condition via
``config.ENSEMBLE_RULESET`` when ``config.ENSEMBLE_MODE`` is set - see
notebooks/02-label_extraction.ipynb section 4).

Architecture kept intentionally flat for a bootcamp audience: one file,
clear functions, no abstractions beyond what the framework gives us.
'''

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List

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
    evidence: str = Field(
        default="",
        description="Short verbatim quote from the report supporting the value; empty string if none"
    )

    class Config:
        title = "ConditionLabel"


class ReportLabels(BaseModel):
    '''Structured output: one entry per condition plus an unparseable flag.'''

    unparseable: bool = Field(
        default=False,
        description="True only if the report is truncated, gibberish, or clearly not a knee MRI report"
    )

    labels: Dict[str, ConditionLabel] = Field(
        description=f"One entry per each of these 12 conditions: {', '.join(config.CONDITIONS)}"
    )

    class Config:
        title = "ReportLabels"


# ---------------------------------------------------------------------------
# Client pool: one ChatOpenAI per local llama.cpp server, round-robin by uid
# ---------------------------------------------------------------------------

_clients: List[ChatOpenAI] = []
_clients_lock = threading.Lock()


def _build_clients() -> List[ChatOpenAI]:
    '''Build one ChatOpenAI client per configured local server endpoint.'''

    load_dotenv(os.path.join(config.REPO_ROOT, '.env'))

    api_key = os.environ[config.ENV_LOCAL_API_KEY]
    model = os.environ.get(config.ENV_LOCAL_MODEL, 'default')
    urls = [
        os.environ[name] for name in config.ENV_LOCAL_URLS
        if os.environ.get(name)
    ]

    if not urls:
        raise RuntimeError(
            f'No local server URLs found; set {config.ENV_LOCAL_URLS} in .env')

    clients = [
        ChatOpenAI(
            base_url=url,
            api_key=api_key,
            model=model,
            temperature=config.DEFAULT_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            timeout=600,
        )
        for url in urls
    ]

    logger.info('LLM client pool: %d server(s): %s', len(clients), urls)

    return clients


def get_clients() -> List[ChatOpenAI]:
    '''Return the shared client pool, building it on first call.'''

    global _clients

    if not _clients:
        with _clients_lock:
            if not _clients:
                _clients = _build_clients()

    return _clients


def _client_for(uid: str) -> ChatOpenAI:
    '''Pick a client for this uid: a stable hash spreads uids evenly and
    deterministically across the available servers.'''

    clients = get_clients()
    idx = hash(uid) % len(clients)

    return clients[idx], idx


# ---------------------------------------------------------------------------
# Ensemble client pool: one ChatOpenAI per model (phi4 + qwen7B), each on
# its own dedicated llama.cpp server/GPU - no round-robin, one fixed
# client per model.
# ---------------------------------------------------------------------------

_ensemble_clients: Dict[str, ChatOpenAI] = {}
_ensemble_clients_lock = threading.Lock()


def _build_ensemble_clients() -> Dict[str, ChatOpenAI]:
    '''Build one ChatOpenAI client per ensemble model side ('model_a',
    'model_b'), reading its URL/model name from its own env vars.'''

    load_dotenv(os.path.join(config.REPO_ROOT, '.env'))

    api_key = os.environ[config.ENV_LOCAL_API_KEY]
    clients = {}

    for key, url_env, model_env in [
        ('model_a', config.ENV_ENSEMBLE_MODEL_A_URL, config.ENV_ENSEMBLE_MODEL_A_NAME),
        ('model_b', config.ENV_ENSEMBLE_MODEL_B_URL, config.ENV_ENSEMBLE_MODEL_B_NAME),
    ]:
        url = os.environ.get(url_env)

        if not url:
            raise RuntimeError(
                f'ENSEMBLE_MODE requires {url_env} to be set in .env')

        clients[key] = ChatOpenAI(
            base_url=url,
            api_key=api_key,
            model=os.environ.get(model_env, 'default'),
            temperature=config.DEFAULT_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            timeout=600,
        )

    logger.info(
        'Ensemble client pool: model_a=%s (%s) model_b=%s (%s)',
        os.environ.get(config.ENV_ENSEMBLE_MODEL_A_NAME, 'default'),
        os.environ[config.ENV_ENSEMBLE_MODEL_A_URL],
        os.environ.get(config.ENV_ENSEMBLE_MODEL_B_NAME, 'default'),
        os.environ[config.ENV_ENSEMBLE_MODEL_B_URL],
    )

    return clients


def get_ensemble_clients() -> Dict[str, ChatOpenAI]:
    '''Return the shared ensemble client pool, building it on first call.'''

    global _ensemble_clients

    if not _ensemble_clients:
        with _ensemble_clients_lock:
            if not _ensemble_clients:
                _ensemble_clients = _build_ensemble_clients()

    return _ensemble_clients


# ---------------------------------------------------------------------------
# Extraction (single report → N samples → majority vote)
# ---------------------------------------------------------------------------


def extract_report(report: str, uid: str = "?") -> Dict:
    '''Call one model N times and return a majority-vote result.

    Returns a dict of the same shape the old ``extract.py`` produced:
        {"unparseable": bool,
         "labels": {cond: (value:int, evidence:str)},
         "agreement": {cond: fraction:float}}

    Raises ``ExtractionError`` if ALL samples fail (transport or
    schema-validation), so the caller can log and skip.
    '''

    base_client, endpoint_idx = _client_for(uid)
    tag = f'uid={uid} endpoint={endpoint_idx}'

    return _sample_and_vote(base_client, report, tag)


def extract_report_ensemble(report: str, uid: str = "?") -> Dict:
    '''Two-model ensemble extraction: phi4 (model_a) and qwen7B (model_b)
    each independently do their own N-sample majority vote, then
    per-condition results are combined via ``config.ENSEMBLE_RULESET``.

    Same return shape as ``extract_report``. If one model's samples all
    fail or come back unparseable, its votes default to 0/absent for
    combination purposes (conservative: an 'AND' condition can't turn
    positive off a model that produced nothing).
    '''

    clients = get_ensemble_clients()
    per_model: Dict[str, Dict] = {}
    errors: Dict[str, BaseException] = {}

    def _run(key: str, client: ChatOpenAI) -> None:
        try:
            per_model[key] = _sample_and_vote(client, report, f'uid={uid} model={key}')
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread below
            errors[key] = e

    # Run both models concurrently (each on its own GPU/server) rather than
    # sequentially, so the ensemble costs one round-trip, not two.
    threads = [threading.Thread(target=_run, args=(key, client)) for key, client in clients.items()]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    if 'model_a' in errors and 'model_b' in errors:
        raise errors['model_a']

    for key in ('model_a', 'model_b'):
        if key in errors:
            if isinstance(errors[key], BalanceExhaustedError):
                raise errors[key]

            logger.warning(
                'Ensemble uid=%s: %s failed entirely (%s); treating its votes as absent',
                uid, key, errors[key]
            )
            per_model[key] = {
                'unparseable': True, 'labels': {},
                'agreement': {c: 1.0 for c in config.CONDITIONS}
            }

    if per_model['model_a']['unparseable'] and per_model['model_b']['unparseable']:
        return {'unparseable': True, 'labels': {}, 'agreement': {}}

    labels: Dict[str, tuple] = {}
    agreement: Dict[str, float] = {}

    for cond in config.CONDITIONS:
        a_val, a_ev = per_model['model_a']['labels'].get(cond, (0, ''))
        b_val, b_ev = per_model['model_b']['labels'].get(cond, (0, ''))
        a_ag = per_model['model_a']['agreement'].get(cond, 1.0)
        b_ag = per_model['model_b']['agreement'].get(cond, 1.0)

        policy = config.ENSEMBLE_RULESET.get(cond, 'AND')

        if policy == 'model_a':
            value, evidence, agmt = a_val, a_ev, a_ag
        elif policy == 'model_b':
            value, evidence, agmt = b_val, b_ev, b_ag
        else:  # 'AND'
            value = 1 if (a_val == 1 and b_val == 1) else 0
            evidence = a_ev if len(a_ev) >= len(b_ev) else b_ev
            agmt = min(a_ag, b_ag)

        labels[cond] = (value, evidence)
        agreement[cond] = agmt

    positives = [c for c, (v, _) in labels.items() if v == 1]
    logger.info('Ensemble extraction done uid=%s positives=%s', uid, positives)

    return {'unparseable': False, 'labels': labels, 'agreement': agreement}


def _sample_and_vote(base_client: ChatOpenAI, report: str, tag: str) -> Dict:
    '''Call ``base_client`` N times at ``config.SAMPLE_TEMPERATURE`` and
    return the majority-vote result (shape documented on
    ``extract_report``). Shared by the single-model and ensemble paths.
    '''

    n = config.DEFAULT_SAMPLES_PER_REPORT
    client = base_client.model_copy(
        update={'temperature': config.SAMPLE_TEMPERATURE}
    )

    sc = client.with_structured_output(ReportLabels)
    messages = build_messages(report)

    samples: List[ReportLabels] = []
    errors: List[str] = []

    for i in range(n):
        try:
            result: ReportLabels = sc.invoke(messages)
            samples.append(result)

        except Exception as e:
            if _is_balance_error(e):
                logger.critical(
                    'Balance exhausted (402) on %s: stopping run. '
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
