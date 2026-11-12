# Phase A — M1 Classwise Descriptive Checkpoint (12 November 2026)

**Status:** completed descriptive calculation for the **17 officially settled forecasts through 11 November 2026**, not a final Phase A result or an independent provenance attestation. This checkpoint **does not** modify Notebook 12, the frozen M1 deployment, submitted probabilities, historical receipts, or the original accepted-first-30 cohort.

**Source:** executed `notebooks/12_es_arena_prospective_evaluation_v0_3_2_public.ipynb` supplied for review, specifically its displayed main descriptive sample and historical-baseline outputs. The analyzed notebook copy has SHA-256 `fd38e9ecb7ca2d94da7a3719b859788eaddae38cd68b50caf8af99551948080f`. The local research repository should rerun Notebook 12 for later checkpoints; these numbers are a *snapshot*, not a live feed.

## 1. Cohort and provenance

- Officially settled, descriptively eligible outcomes: **17**; correct top-class classifications: **9**.
- Accepted submissions documented through November 12: **18**; November 12 remains outside the *settled* cohort at this checkpoint.
- The accepted first-30 Notebook 12 analysis is unchanged. This is not the separate proposed eligible-opportunity cohort; the complete historical opportunity census remains unverified.
- Strict independent server-UTC-certification subset: **0**. Native scored receipts support inclusion in the descriptive sample but are not externally signed UTC acceptance-time attestations. The October 20 legacy record is included with explicit lower-provenance status.

## 2. Main performance and baselines

| Comparator | N | Accuracy | Unscaled 3-class Brier | Log loss |
|---|---:|---:|---:|---:|
| Frozen M1 Logistic Regression | 17 | **52.94%** (9/17) | **0.610223** | **1.009760** |
| Uniform class probabilities (1/3 each) | 17 | Undefined (argmax tie) | 0.666667 | 1.098612 |
| Historical frozen training-only class prior | 17 | 47.06% (8/17) | 0.670001 | 1.105669 |
| Always bearish, hard class only | 17 | 41.18% (7/17) | Not defined | Not defined |
| Always neutral, hard class only | 17 | 11.76% (2/17) | Not defined | Not defined |
| Always bullish, hard class only | 17 | 47.06% (8/17) | Not defined | Not defined |

The hard-class rules have no supplied full probability vectors; therefore **do not** assign them artificial Brier or log-loss values. The frozen training prior is a descriptive benchmark: Notebook 12 notes a historical 3,855-versus-3,835 training-population/smoothing discrepancy. Do not relabel this as a prospectively frozen primary benchmark.

M1's Brier is approximately **8.47% lower than the uniform Brier** for these same 17 outcomes (relative Brier reduction, not an estimated return or demonstrated market edge). Accuracy exceeds the always-bullish rule by **one correct prediction, 5.88 percentage points**.

## 3. Classwise confusion matrix

Rows are official Arena outcomes, columns are the predicted top-probability class.

| Official outcome \ Predicted class | Bearish | Neutral | Bullish | Support |
|---|---:|---:|---:|---:|
| Bearish | 1 | 1 | 5 | 7 |
| Neutral | 0 | 0 | 2 | 2 |
| Bullish | 0 | 0 | 8 | 8 |
| **Predicted count** | **1** | **1** | **15** | **17** |

### Per-class diagnostics

| Class | True instances | Predicted instances | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Bearish | 7 | 1 | 100.00% | 14.29% | 25.00% |
| Neutral | 2 | 1 | 0.00% | 0.00% | 0.00% |
| Bullish | 8 | 15 | 53.33% | 100.00% | 69.57% |

- **Balanced accuracy:** (1/7 + 0/2 + 8/8) / 3 = **38.10%**.
- **Macro F1:** mean of classwise F1 = **31.52%**.
- **Bullish predictions:** 15/17 = **88.24%**.
- Precision for bearish is 100% only because there was **one** bearish prediction and it was correct; it is not a robust precision estimate.
- Neutral recall of 0% arises from just two actual neutral outcomes. All metrics are highly sample-sensitive.

## 4. Interpretation and reporting constraints

1. The model shows marked **top-class bullish concentration** and uneven performance on bearish and neutral outcomes. High bullish recall does not imply reliable three-class discrimination.
2. The current 17-session descriptive result cannot establish persistent probabilistic skill, statistical superiority, profitability, or classwise calibration.
3. Use official Arena labels for this benchmark; do not replace them with self-reconstructed close-to-close market classifications.
4. Retain the original full-precision probabilities and challenge IDs. Do not backfill missed opportunities from weekdays or retroactively replace the scored probability vector with a revision.
5. Repeat this report at the actual final Phase A boundary, and disclose any differences between the accepted-first-30 cohort and the proposed eligible-first-30 cohort, plus missing/failed/cancelled and unresolved records.

**Research conclusion:** Kopei's requested basic classwise and constant-direction descriptive diagnostics can be reported at the 17-settlement checkpoint. The *final* Phase A evaluation remains pending and the complete eligible-opportunity census remains unverified.
