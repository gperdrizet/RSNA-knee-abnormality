# LLM label extraction pipeline: plan

**Status:** Planned (2026-09-05)
**Stack:** Luigi + `openai` SDK + Promptly gateway (Qwen3.8-27B)

## Overview

Extract 12 binary knee abnormality labels from the ~4,350 unlabeled `Report` texts in
`data/raw/train.csv` using the Qwen3.8-27B model served via our Promptly API gateway.
Built as a Luigi task graph inside a new `data_pipeline/label_extraction/` Python
package, designed so future preprocessing stages can be added as sibling Luigi tasks.

## Locked decisions

- **Framework:** Luigi (SQLite dependency store).
- **Endpoint:** Promptly gateway (OpenAI-compatible `/v1/chat/completions`), per the [Promptly agents doc](https://github.com/gperdrizet/promptly/blob/main/AGENTS.md). Uses the `openai` python SDK with the env vars in `.env`: `BASE_URL`, `MODEL=default` (field accepted but ignored, server uses loaded model), and `API_KEY` (bearer).
- **Endpoint facts that drive design** (from Promptly README/AGENTS.md):
  - Single slot deployment, 262,144 token context. Each added worker divides the context equally, so 8 workers gives 32k per slot. That is ample for ~2k token prompts, so high parallelism is safe on the VRAM side.
  - Rate limits are 120 req/min per IP and **60 req/min per API key**. The key limit is the throughput ceiling, not the GPU. The client paces at ~48 req/min and treats 429 as retryable with backoff. Full run estimate: 13k calls / 48 per min is about **4.5 hours** plus model latency.
  - The loaded model is a hybrid thinking/instruct model and the server defaults `reasoning_effort` to `medium`. We send `reasoning_effort: low` for full runs (structured extraction does not need deep reasoning, a big latency saving) and `medium` in calibration where speed is irrelevant. Config constant, switchable per task.
  - Responses include the non-standard `reasoning_content` field. The client always reads `choices[0].message.content` only. `reasoning_content` is logged at DEBUG level only.
  - **A 402 means the token balance is exhausted: fail fast, log at CRITICAL, and do not retry.** Token estimate: ~13k calls x ~2.5k tokens is about 30-35M tokens for the full run. Check or top up the balance before the full run.
- **Concurrent workers:** default `--workers 8` on the CLI. The context per slot math leaves massive headroom and throughput is rate limit bound regardless. Bump only if 429s dominate backoff time.
- **Extraction policy:** 3-sample majority voting per report via `temperature` sampling. Per condition agreement (2/3 or 3/3) is stored as free confidence. About 13k+ API calls in total.
- **Outputs:** one CSV of pseudo labels per report, plus a merged `train_labeled_v2.csv` with a `label_source` column (`gold` or `pseudo`). Gold rows are never overwritten.
- **Scope:** train reports only. The test set is imaging only (no reports), so label extraction is pure preprocessing. There is no test report task.

## Module layout

    data_pipeline/
        __init__.py
        logging_config.py          # logging setup -> logs/ (shared by all stages)
        pipeline.py                # Luigi task graph + entrypoint
        label_extraction/
            __init__.py
            config.py              # constants: condition names, env var names, paths, prompt template
            client.py              # OpenAI-SDK client for Promptly gateway, with retries, backoff, and 402 handling
            prompt.py              # builds system/user prompts; multilingual glossary per condition
            extract.py             # single extraction logic: prompt -> JSON -> validated schema
            tasks.py               # Luigi tasks (see task graph below)
            schema.py              # validation of extracted JSON
    requirements.txt               # luigi, openai, python-dotenv, pandas, pydantic

(`.env` already contains `BASE_URL`, `MODEL`, and `API_KEY`, so no changes are needed.)

`pipeline.py` sits at the `data_pipeline/` level (not inside `label_extraction/`) because
it will host tasks for all future stages. `label_extraction/` holds the domain logic.

## Luigi task graph

Granularity: per report tasks (4,407 of them). Parallelism is controlled by `luigi --local-scheduler --workers N`.

1. `ExtractLabels(report_uid, run)`: one Luigi task per row in `train.csv`.
   - Input: raw `train.csv` (single static dependency `TrainCsvAvailable`).
   - Logic: read that row's `Report`. Skip immediately if it is already gold labeled (output is the gold passthrough) or if `Report` is missing or truncated.
   - Call `extract.py` **3 times** with temperature sampling, parse the JSON, and majority vote each of the 12 conditions.
   - Output: `data/pipeline/label_extraction/{run}/labels/{uid}.csv` (one row: uid + 12 binary labels + 12 agreement columns, e.g. `acl_agree=3`).
   - Retry policy: Luigi `RETRY_COUNT` for transport errors. Per call retries in `client.py` with exponential backoff.
2. `MergeLabels(run)`: input is all `ExtractLabels` outputs.
   - Writes `data/pipeline/label_extraction/{run}/train_labeled_v2.csv`:
     - Gold rows (58): gold labels, `label_source='gold'`.
     - Pseudo rows: voted labels, `label_source='pseudo'`, with `*_agreement` columns retained.
     - Rows that failed extraction or had empty reports are excluded. They are logged and listed in a `failed_uids.txt` side file.
3. `CalibrateAgainstGold(run)`: input is the `MergeLabels` output.
   - **Runs before the full run.** On the 58 gold rows it re-predicts via the same 3-sample path and writes `data/pipeline/label_extraction/{run}/calibration_report.csv/json` with per condition accuracy + Cohen's kappa. Human-in-the-loop gate: inspect, iterate on the glossary/prompt, bump the `run` id, and start over.

The run id (e.g. `20260905_001`) is passed via CLI (`--run`) so prompt iterations do not clobber prior outputs.

## Key implementation details

### Prompt design (`prompt.py`)

- System + user prompt returning **strict JSON only**: `{"<condition_snake_case>": {"label": 0|1, "evidence": "<short quote>"}}` for all 12 conditions.
- Multilingual glossary embedded per condition (seed terms observed in data: ES/Dutch/French/EN). E.g. effusion = "derrame / effusion / liquide articulaire / vloeistof", meniscus tear = "rotura de menisco / Meniscusscheur / déchirure du ménisque / tear".
- The negation and hedging policy is made explicit in the prompt. Default: positive only for definite findings. The final policy is decided by checking how the 58 gold rows treat "suspicious"/"possible" during calibration.
- Truncation handling: the model outputs `{"unparseable": true}` when the report is cut off mid-section. Those rows are dropped with a log line.
- Few-shot: 2-3 exemplar report-to-label pairs pulled from the 58 gold rows, injected into the prompt.

### Client (`client.py`)

- `openai` python SDK: `OpenAI(base_url=<BASE_URL from .env>, api_key=<API_KEY from .env>)`. Calls use `model=<MODEL from .env>` ("default"), `temperature` (e.g. 0.4), `max_tokens` sized for the JSON output, and per-task `reasoning_effort` (`low` for full runs, `medium` for calibration). `reasoning_effort` is a Promptly extension sent via `extra_body`, behind a config flag, and is dropped gracefully if a 400 occurs.
- Retry policy (in call, 3 attempts, exponential backoff + jitter): retryable on 429, 5xx (`502` = upstream llama server error), and timeouts. **Not retryable:** 402 (balance exhausted, raises a distinct `BalanceExhaustedError`, logs at CRITICAL, fails fast) and 400/401 (config errors).
- Simple shared pacing (token bucket ~48 req/min) inside the client so 8 workers stay under the per key rate limit without hammering it.
- Distinguished exception types make Luigi-level `RETRY_COUNT` vs in-call retries easy to audit.

### Logging (`logging_config.py`)

- `logging` module. Root logger `data_pipeline` at INFO to console and DEBUG to file `logs/data_pipeline.log` (RotatingFileHandler).
- Per task INFO lines: start/finish, n_samples, agreement counts. DEBUG: full prompts and raw model outputs (DEBUG level only, since verbose).
- All module code logs via `logger = logging.getLogger(__name__)`.

## Phases and execution order

- **Phase A, scaffolding** (parallelizable): package layout, `requirements.txt`, `logging_config.py`, `config.py`.
- **Phase B, extraction core:** `client.py`, `schema.py`, `prompt.py`, `extract.py`.
- **Phase C, plumbing:** `tasks.py` and the `pipeline.py` entry point (CLI: `python -m data_pipeline.pipeline extract --run RUN --workers N`, default `N=8`).
- **Phase D, calibration:** run `CalibrateAgainstGold` on the 58 gold rows only (174 calls, `reasoning_effort: medium`). Iterate on the glossary/prompt until per condition accuracy is acceptable. **The user reviews the calibration report before the full run.**
- **Phase E, full run:** **pre-flight check the token balance** (need ~30-35M tokens for 13k calls), then run all `ExtractLabels` (default `--workers 8`), then `MergeLabels`, then spot checks.

Parallelism note: throughput is bound by the 60 req/min per key rate limit, not GPU slots.
Default `--workers 8` and the client's ~48 req/min token bucket + 429 backoff absorb any overshoot.
Watch the logs for 429 density and adjust if needed.

## Verification

1. **Unit smoke test:** `client.py` + `extract.py` on 5 hard gold rows (negations, multilingual, truncated). Check the parsed JSON and the votes.
2. **Calibration gate:** per condition accuracy vs the 58 gold rows is recorded in `calibration_report.json`. Conditions below threshold get glossary/prompt fixes and a re-run (new `run` id).
3. **Full run checks:**
   - Row counts: merged CSV = 58 gold + (unlabeled, failed). `failed_uids.txt` explains the delta.
   - `label_source` column integrity: gold rows match the original `train.csv` labels exactly (asserted in `MergeLabels`).
   - Spot check 20 random pseudo rows: read the report and sanity check the labels against the `evidence` quotes.
   - Label distributions per condition look plausible compared to the 58 gold rows.
4. **Logs:** `logs/data_pipeline.log` contains per task lines, with no unhandled exceptions in a full run.

## Scope boundaries

- **In:** train report extraction, the calibration harness, and the merged CSV artifact.
- **Out (future stages, but the task graph is ready to extend):** DICOM preprocessing, per study aggregation (`train_series.csv` maps multiple studies per patient and is the natural next Luigi stage), and any model training. **Test report extraction is not on the roadmap** (test set is imaging only).
- **Out:** fine tuning the LLM, any changes to the Promptly deployment, and anything touching the 58 gold labels.

## Assumptions and further considerations

- The gateway fronts llama.cpp, which supports `temperature` sampling for the 3-sample voting. Plan B (only if sampling proves degenerate or is ignored): 3 passes with distinct prompt variants (normal, negation focused, condition focused).
- `reasoning_effort` is a Promptly extension forwarded verbatim to the model. If a future deployed model rejects it, the client tolerates the 400 by dropping the field (config flag).
- Report lengths are short (radiology notes, well under 2k tokens), so even the 32k per worker context split is ample. No chunking is needed.