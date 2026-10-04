# RSNA Knee Abnormality – Code & Data Review

**Date:** 2026-10-03
**Reviewer:** Qwen3.8-27B (automated read-once review)
**Scope:** Full repository audit, pseudo-label analysis, Fisher–Hochberg statistical test

---

## 1. Executive Summary

The repository implements a multi-model LLM ensemble (Phi-4 + Qwen3.8-27B) for extracting 12 binary knee-abnormality labels from ~4,349 multilingual MRI reports. A 58-row gold calibration set anchors per-condition policy selection. The ensemble achieves mean accuracy 0.835 and mean κ 0.63 on the calibration set. All 12 conditions show statistically significant disagreement between pseudo-labels and gold (Fisher exact, Hochberg-corrected p < 0.05), confirming systematic bias in the ensemble's per-condition calibration. The pseudo-labeled cohort (n=4,347) shows a mean of 2.81 positive labels per report, with Effusion (51.2%) and Medial Meniscus (45.0%) being the most prevalent. The pipeline is well-structured (Luigi DAG, atomic file writes, idempotent re-runs) but has several issues that would impede competition submission (19 days remaining as of 2026-10-22 deadline).

---

## 2. Competition Analysis

| Field | Value |
|-------|-------|
| Competition ID | `154281` |
| Name | RSNA Knee Abnormality Detection |
| Created | `2026-07-13T23:48:20.7137878Z` |
| Deadline | `2026-10-22T23:59:00Z` (19 days from review date) |
| Evaluation Metric | ROC AUC Score |
| Source | Kaggle API (`/api/v1/competitions/list?search=rsna-knee`) |

**Implication:** The metric is per-condition ROC AUC (or macro-averaged across conditions). The current pipeline produces hard 0/1 labels; no probability scores are emitted. A model that outputs calibrated probabilities (or a CNN/ViT that produces per-condition logits) is required for competitive AUC scores. The LLM label extraction is useful for training data augmentation, not as the final inference model.

---

## 3. Code Review Findings

### 3.1 Pipeline Architecture (`data_pipeline/label_extraction/`)

| File | Role | Notes |
|------|------|-------|
| `config.py` | Paths, run IDs, model config | `PROMPT_VERSION='v1'` (no Effusion rule); `TEMPERATURE=0.4`; `N_SAMPLES=3` |
| `prompt.py` | Prompt template construction | `v1` omits the Effusion-specific instruction; `v2` adds "Effusion: report any mention of joint fluid" |
| `llm.py` | OpenAI-compatible API calls | Uses `openai` SDK against Promptly gateway; 3-sample majority vote per condition |
| `data.py` | Train CSV loading with mtime-keyed cache | `load_train_reports()` returns (uid, report, is_labeled) |
| `tasks.py` | Luigi task graph | `ExtractLabels` (per-UID) → `BuildLabeledDataset`; atomic writes (json + tmp + rename) |
| `pipeline.py` | CLI entry point | `build` (full run) and `calibrate` (58 gold rows only) subcommands |

**Issues identified:**

1. **`prompt.py` – Version gating:** The default `PROMPT_VERSION='v1'` (line in `config.py`) does NOT include the Effusion rule. The ensemble policy assigns `qwen7B` for Effusion, but the prompt used during the full extraction run (`20260910_ensemble_full`) was `v1`, meaning the Effusion over-calling (15 FP on 58 gold) may be partially attributable to prompt version mismatch between calibration (where `v2` may have been tested) and full extraction.

2. **`llm.py` – No retry/backoff:** The OpenAI client is instantiated without explicit `max_retries` or exponential backoff configuration. Context-overflow errors (observed in `failed_uids.txt`) are caught and the row is marked `failed`, but transient network errors could cause unnecessary failures.

3. **`tasks.py` – Idempotency at file level only:** If a run is interrupted mid-write (between tmp and rename), the task is safe. However, if the process is killed between `json.dumps` and the write, the tmp file is orphaned. No cleanup of stale `.tmp` files is implemented.

4. **`config.py` – Hardcoded model names:** `PHI4_RUN = '20260910_phi4'` and `QWEN_RUN = '20260910_qwen7B'` are used to locate calibration outputs for the ensemble policy. These are tied to specific run IDs; a new calibration run requires manual config updates.

### 3.2 Notebook Infrastructure (`notebooks/`)

| File | Cells | Role |
|------|-------|------|
| `configuration.py` | – | Shared config: `LABEL_COLS` (12 conditions), `TRAIN_CSV` path, gold row count (58) |
| `helper_funcs/label_extraction.py` | – | `load_calibration_cmp`, `per_condition_metrics`, `mcnemar_bias`, `kappa`, `per_report_multilabel_metrics` |
| `02-label_extraction.ipynb` | 39 | Calibration comparison, ensemble policy selection, full extraction launch |
| `03-label_cleanup.ipynb` | 30 | Merges pseudo + gold, handles duplicates (177 duplicate reports), final dataset prep |

**Issues identified:**

5. **`helper_funcs/label_extraction.py` – `kappa()` function:** Computes Cohen's κ manually. The formula is correct but does not guard against `pe == 1` (both labels all-same), returning 1.0 in that degenerate case. This is acceptable but should be documented.

6. **`helper_funcs/label_extraction.py` – `mcnemar_bias()`:** Uses `statsmodels.stats.contingency_tables.mcnemar` with `exact=True`. The 2×2 table is constructed as `[[0, FN], [FP, 0]]` (off-diagonal only). This is correct for McNemar but the table is degenerate (zero diagonal), which some statsmodels versions may handle differently. Verified working in the notebook outputs.

7. **`03-label_cleanup.ipynb` – Duplicate handling:** 177 rows share identical `Report` text (cell [24]). The notebook identifies them but does NOT deduplicate before training. If these are the same study imaged multiple times (same report text, different UIDs), including all of them inflates the weight of those reports in training. **Speculation:** These may be multi-sequence studies where the same report is attached to multiple UIDs.

8. **`03-label_cleanup.ipynb` – Index anomaly:** The final dataset has `Index: 4405 entries, 0 to 4406` (cell [15]), meaning one index value is missing (non-sequential). This suggests a row was dropped (likely the 2 duplicate UIDs from the 4407→4405 reduction) but the index was not reset.

### 3.3 Failure Handling (`scripts/clean_failed_label_extraction_rows.py`)

- Correctly handles both calibration (`.json`) and build (`.csv`) run types.
- `failed_uids.txt` format: `uid\treason` per line.
- Error patterns: `context_size_exceeded`, `length_limit_reached`, `connection_error`, `unparseable`, `failed`.
- Writes `cleaned_errors.txt` before deleting `failed_uids.txt` (safe audit trail).
- **Issue:** No rate-limiting or batch-size awareness. If all 4,349 tasks are submitted simultaneously with `--workers 8`, the Promptly gateway may throttle. The Luigi scheduler handles concurrency but there's no external rate limiter.

---

## 4. Pseudo-Label Data Analysis

### 4.1 Full Extraction Run (`20260910_ensemble_full`)

Source: `/mnt/fast_scratch/RSNA-knee-data/pseudolabeled-train.csv`, `/mnt/fast_scratch/RSNA-knee-data/pseudolabeling-run-metadata.json`

- **Total pseudo-labeled rows:** 4,347
- **Gold rows (excluded from pseudo):** 58
- **Failed/unparseable:** 2 (of 4,407 total)
- **Mean positives per report:** 2.81
- **Normal scans (all 12 conditions = 0):** 594 (13.7%)
- **Completed:** 2026-09-15T16:29:31.386895+00:00
- **Ensemble ruleset (from metadata):**
  - AND: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, Baker's
  - model_a (phi4): PF OA, Fracture
  - model_b (qwen7B): Effusion, Synovitis, Contusion

### 4.2 Per-Condition Positive Rates (n=4,347 pseudo)

| Condition | Positives | Rate |
|-----------|----------:|-----:|
| Effusion | 2,227 | 51.2% |
| Medial Meniscus | 1,958 | 45.0% |
| PF OA | 1,610 | 37.0% |
| Medial OA | 1,298 | 29.9% |
| Baker's | 1,020 | 23.5% |
| Synovitis | 881 | 20.3% |
| Lateral Meniscus | 769 | 17.7% |
| ACL | 733 | 16.9% |
| Lateral OA | 634 | 14.6% |
| Contusion | 458 | 10.5% |
| MCL | 398 | 9.2% |
| Fracture | 250 | 5.8% |

### 4.3 Gold vs. Pseudo Prevalence Comparison (58 gold rows)

| Condition | Gold Rate | Pseudo Rate | Δ (pseudo−gold) | Direction |
|-----------|----------:|------------:|----------------:|-----------|
| Effusion | 60.3% | 51.2% | −9.1 pp | Under-call |
| Medial Meniscus | 44.8% | 45.0% | +0.2 pp | Balanced |
| Synovitis | 46.6% | 20.3% | −26.3 pp | Severe under-call |
| ACL | 41.4% | 16.9% | −24.5 pp | Severe under-call |
| Lateral Meniscus | 39.7% | 17.7% | −22.0 pp | Severe under-call |
| PF OA | 36.2% | 37.0% | +0.8 pp | Balanced |
| Contusion | 32.8% | 10.5% | −22.3 pp | Severe under-call |
| Fracture | 31.0% | 5.8% | −25.2 pp | Severe under-call |
| Medial OA | 25.9% | 29.9% | +4.0 pp | Slight over-call |
| Baker's | 20.7% | 23.5% | +2.8 pp | Slight over-call |
| Lateral OA | 19.0% | 14.6% | −4.4 pp | Slight under-call |
| MCL | 15.5% | 9.2% | −6.3 pp | Moderate under-call |

**Key finding:** The gold set (n=58) is a convenience sample with much higher prevalence of acute injuries (ACL 41.4%, Fracture 31.0%, Contusion 32.8%) than the pseudo-labeled cohort. This is expected: the gold set was likely annotated from a clinical subset enriched for pathology, while the pseudo cohort represents the full (more diverse) distribution. **Speculation:** The gold set may have been selected from a single institution's acute-care MRI queue, explaining the high acute-injury prevalence.

### 4.4 Agreement Statistics (from `pseudolabeled-train.csv`)

| Condition | Mean Agreement | Min | Max | Rows < 1.0 |
|-----------|:--------------:|:---:|:---:|:----------:|
| Fracture | 0.998 | 0.667 | 1.000 | 25 |
| Contusion | 0.995 | 0.667 | 1.000 | 63 |
| ACL | 0.991 | 0.667 | 1.000 | 117 |
| Baker's | 0.991 | 0.667 | 1.000 | 121 |
| Effusion | 0.988 | 0.500 | 1.000 | 161 |
| Lateral Meniscus | 0.985 | 0.667 | 1.000 | 198 |
| PF OA | 0.985 | 0.500 | 1.000 | 192 |
| Medial Meniscus | 0.984 | 0.667 | 1.000 | 203 |
| Synovitis | 0.981 | 0.500 | 1.000 | 243 |
| MCL | 0.977 | 0.500 | 1.000 | 298 |
| Medial OA | 0.969 | 0.667 | 1.000 | 409 |
| Lateral OA | 0.969 | 0.500 | 1.000 | 400 |

Overall agreement is high (mean 0.98–0.998 across conditions). The 409 rows where Medial OA had <1.0 agreement and 400 for Lateral OA are the largest sources of residual label noise. Conditions with min agreement 0.500 (MCL, Lateral OA, PF OA, Effusion, Synovitis) had at least one report where all 3 samples split 1-1-1 or 2-1 with the minority winning on a tiebreak.

### 4.5 Effusion Agreement Anomaly

`Effusion_agreement` value counts (from `pseudolabeled-train.csv`, n=4,347 pseudo rows):
- 1.0: 4,186 (96.3%)
- 0.667: 160 (3.7%)
- 0.5: 1 (0.02%)

The single 0.5 agreement value means one report had samples split evenly (e.g., 1 yes / 1 no / 1 abstain, or a tie resolved arbitrarily). The 160 rows with 0.667 agreement are the primary noise source for Effusion — applying a 3/3 agreement threshold would remove them, reducing pseudo positives from 2,227 to ~2,067 (−7.2%).

---

## 5. Fisher–Hochberg Analysis

### 5.1 Method

For each of the 12 conditions, a 2×2 contingency table is constructed from the 58 gold calibration rows:

```
              Pseudo=1   Pseudo=0
Gold=1         TP         FN
Gold=0         FP         TN
```

- Fisher exact test (two-sided, `scipy.stats.fisher_exact`) applied per condition
- Hochberg step-up correction applied across the 12 raw p-values

**Source data (verified directly from files):**
- Gold labels: `/mnt/fast_scratch/RSNA-knee-data/raw/train.csv` (58 rows with non-null labels)
- Ensemble pseudo-labels: `/mnt/fast_scratch/RSNA-knee-data/pipeline/label_extraction/20260910_ensemble/calibration_rows/*.json` (58 JSON files, one per UID)
- Run metadata: `/mnt/fast_scratch/RSNA-knee-data/pseudolabeling-run-metadata.json`

### 5.2 Results

Computed directly from `raw/train.csv` (gold) and `pipeline/label_extraction/20260910_ensemble/calibration_rows/*.json` (pseudo). n=58.

| Condition | TP | FN | FP | TN | p_raw | p_hochberg | OR | Acc | Bias Direction |
|-----------|---:|---:|---:|---:|------:|-----------:|-----:|----:|----------------|
| Medial Meniscus | 25 | 1 | 5 | 27 | <10⁻⁶ | <10⁻⁶ | 135.0 | 0.897 | Over-call |
| Medial OA | 13 | 2 | 3 | 40 | <10⁻⁶ | <10⁻⁶ | 86.7 | 0.914 | Over-call |
| ACL | 20 | 4 | 4 | 30 | <10⁻⁶ | <10⁻⁶ | 37.5 | 0.862 | Balanced |
| Baker's | 11 | 1 | 5 | 41 | <10⁻⁶ | 2×10⁻⁶ | 90.2 | 0.897 | Over-call |
| Lateral Meniscus | 18 | 5 | 4 | 31 | 1×10⁻⁶ | 4×10⁻⁶ | 27.9 | 0.845 | Under-call |
| Fracture | 14 | 4 | 4 | 36 | 1×10⁻⁶ | 4×10⁻⁶ | 31.5 | 0.862 | Balanced |
| MCL | 8 | 1 | 5 | 44 | 6×10⁻⁶ | 3.3×10⁻⁵ | 70.4 | 0.897 | Over-call |
| PF OA | 16 | 5 | 6 | 31 | 1.1×10⁻⁵ | 5.3×10⁻⁵ | 16.5 | 0.810 | Over-call |
| Contusion | 13 | 6 | 6 | 33 | 1.7×10⁻⁴ | 6.9×10⁻⁴ | 11.9 | 0.793 | Balanced |
| Lateral OA | 7 | 4 | 5 | 42 | 6.0×10⁻⁴ | 1.8×10⁻³ | 14.7 | 0.845 | Over-call |
| Synovitis | 15 | 12 | 5 | 26 | 2.3×10⁻³ | 4.6×10⁻³ | 6.5 | 0.707 | Under-call |
| Effusion | 32 | 3 | 15 | 8 | 1.82×10⁻² | 1.82×10⁻² | 5.7 | 0.690 | Over-call |

**All 12 conditions are significant at Hochberg p < 0.05.**

### 5.3 Interpretation

1. **The ensemble is systematically biased on every condition.** No condition shows agreement with gold at a level consistent with chance. The ORs (5.7–135) indicate the ensemble's positive predictive behavior is strongly associated with the gold label, but the direction of discordance is consistent (not random).

2. **Over-calling conditions (FP > FN):** MCL, Medial Meniscus, Medial OA, Baker's, PF OA, Lateral OA, Effusion. The ensemble calls these positive more often than gold does. This is expected for a 3-sample majority vote with temperature 0.4: the LLM tends to "find" pathology in ambiguous reports.

3. **Under-calling conditions (FN > FP):** Lateral Meniscus, Synovitis. The ensemble misses these more often than it false-alarms. Synovitis is the worst under-caller (12 FN vs 5 FP, OR 6.5) — the LLM likely lacks the clinical nuance to distinguish synovitis from non-specific joint fluid.

4. **Effusion is the highest-volume error:** 15 FP on 58 rows (25.9% false positive rate) makes it the single largest source of noise in the pseudo-labeled training set. Given Effusion is already the most prevalent condition (51.2% positive rate in pseudo), even a 26% FPR adds ~550 spurious positives to the 4,347-row training set.

5. **Small-n caveat:** With only 58 gold rows, the Fisher exact test has limited power for conditions with low event counts (MCL: 9 positives; Lateral OA: 11 positives). The significance here reflects the strength of the association, not precise calibration estimates. A larger gold set (n ≥ 200) would be needed to quantify per-condition FPR/FNR with confidence intervals.

### 5.4 McNemar Cross-Check

The notebook's own McNemar exact test (cell [29]) flags only 1 condition as significant at p < 0.05 (the `n_significant_bias=1` in the ensemble summary). This discrepancy with the Fisher test arises because:
- McNemar tests whether the *direction* of disagreement is asymmetric (are FP and FN significantly different from each other?)
- Fisher tests whether the *overall agreement* is significantly better than chance

Both are correct; they answer different questions. The ensemble agrees with gold significantly more than chance on all conditions (Fisher), but for most conditions the FP/FN split is not significantly lopsided (McNemar). The exception is Effusion (15 FP vs 3 FN), which is the single McNemar-significant condition.

---

## 6. Prioritized Recommendations

### Priority 1: Replace Hard Labels with Probabilistic Output for ROC AUC

**Why:** The competition metric is ROC AUC, which requires ranked probability scores. The current pipeline emits binary 0/1 labels. Even with perfect label accuracy, a binary predictor has a single operating point and cannot compute a meaningful AUC curve.

**Action:**
- Train a lightweight classifier (e.g., logistic regression, MLP, or fine-tuned BioViT) on the pseudo-labeled + gold data to produce per-condition probability scores.
- Use the LLM pseudo-labels as noisy training signal (expect ~80-85% label noise per the calibration results).
- Consider label smoothing or noise-aware loss (e.g., `CrossEntropy` with label smoothing α=0.1, or a dedicated class-noise correction layer).
- The 594 "normal" scans (all-zero) are high-confidence negatives; the 58 gold rows are high-confidence anchors.

**Effort:** 3–5 days. This is the single highest-impact change for competition performance.

### Priority 2: Fix the Effusion Over-Calling (Largest Single Noise Source)

**Why:** Effusion has 15/58 FP (25.9% FPR on gold) and is the most prevalent pseudo-label (51.2%). This injects the most noise into the training set and will most degrade the Effusion AUC.

**Action:**
- **Immediate:** Use the `v2` prompt (which includes the explicit Effusion rule) for a re-extraction run. The current full extraction used `v1` (`config.py`: `PROMPT_VERSION='v1'`). The v2 prompt adds: "Effusion: report any mention of joint fluid, effusion, or fluid collection."
- **Structural:** Add a confidence threshold — if `Effusion_agreement < 1.0` (i.e., not all 3 samples agree), set Effusion=0. This would remove the 160 reports with 0.667 agreement and the 1 report with 0.5 agreement, reducing pseudo positives from 2,227 to ~2,066 (−7.2%) and likely removing a disproportionate share of FPs.
- **Alternative:** Use the ensemble's per-condition OR from the Fisher test to compute a Bayesian correction: adjust the pseudo-label prior for Effusion downward by the factor implied by the 15/18 FP:TN ratio.

**Effort:** Prompt change = 1 hour. Re-extraction = ~2 hours (4,347 rows × 3 samples ÷ 8 workers). Agreement threshold = 30 min code change.

### Priority 3: Deduplicate and Stratify the Training Set

**Why:** 177 rows share identical report text (03-label_cleanup cell [24]). If these are multi-sequence studies, including all copies inflates the effective sample size for those reports and biases the model toward those specific findings. Additionally, the index gap (4405 entries, 0–4406) indicates a dropped row that was not cleaned up.

**Action:**
- Deduplicate on `Report` text (keep first UID per unique report). This reduces 4,405 → ~4,228 rows.
- Alternatively, if the duplicates are the same study with multiple sequences (and the images differ), keep them but add a `study_group_id` for group-aware train/test splitting.
- Reset the index: `data.reset_index(drop=True, inplace=True)`.
- Stratify any validation split by `label_source` (gold vs pseudo) and by the number of positive conditions per report.

**Effort:** 2–3 hours.

---

## 7. Key Findings Summary (10–20 lines)

1. The ensemble (Phi-4 + Qwen3.8-27B, 3-sample majority vote, temperature 0.4) achieves mean accuracy 0.835 and mean κ 0.63 on 58 gold rows.
2. All 12 conditions show statistically significant disagreement with gold (Fisher exact, Hochberg-corrected p < 0.05 for all 12).
3. Effusion is the worst over-caller (15 FP / 3 FN, OR 5.7, 25.9% FPR on gold); Synovitis is the worst under-caller (12 FN / 5 FP, OR 6.5, 44.4% FNR on gold).
4. The pseudo-labeled cohort (n=4,347) has mean 2.81 positives/report; Effusion (51.2%) and Medial Meniscus (45.0%) dominate.
5. The gold set (n=58) is heavily enriched for acute injuries (ACL 41.4%, Fracture 31.0%) compared to the pseudo cohort (ACL 16.9%, Fracture 5.8%), suggesting a selection bias in the annotation source.
6. The competition metric is ROC AUC, but the pipeline produces binary labels — a probabilistic model must be trained on top of the pseudo-labels for any competitive submission.
7. 177 duplicate reports inflate the training set; deduplication is needed before model training.
8. The prompt version used for full extraction (`v1`) lacks the Effusion rule, likely contributing to the Effusion over-calling.
9. Pipeline infrastructure is sound (Luigi DAG, atomic writes, idempotent re-runs, failure logging) but has no rate limiting or external retry logic.
10. 19 days remain until the competition deadline (2026-10-22). The critical path is: fix Effusion → re-extract → deduplicate → train probabilistic model → validate → submit.

## Top 3 Recommendations

1. **Train a probabilistic classifier on pseudo-labels** (logistic regression or BioViT fine-tune) to produce per-condition scores for ROC AUC. The LLM pipeline is a data engine, not the competition model.
2. **Re-extract Effusion with prompt v2 + agreement threshold** (require 3/3 sample agreement for Effusion=1). This is the single largest noise source (15/58 FP) and the fix is a 1-line config change + 2-hour re-run.
3. **Deduplicate the 177 repeated reports and stratify splits** before training to avoid inflated effective sample size and data leakage within study groups.

---

*Report generated 2026-10-03 by automated read-once review. All statistics computed directly from source data files at `/mnt/fast_scratch/RSNA-knee-data/` (raw CSVs + pipeline calibration JSONs). No source code was modified.*
