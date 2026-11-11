# Frozen ES M1 — bullish-preference diagnostic

**Research checkpoint:** 2026-11-11  
**Status:** exploratory, read-only, descriptive model-behavior audit  
**Scope:** 16 native saved ES Arena forecast files; **not** a settled-outcome evaluation or a new prospectively registered experiment  
**Research branch:** `experiment/headline-arena-es-daily`

## Executive finding

The frozen ES-only three-class Logistic Regression has a learned intercept advantage for **bullish**. Using the preserved native feature vectors, that intercept accounts for most of the *mean* bullish-versus-bearish class-score advantage. In contrast, the observed ES features **add substantially** to the bullish-versus-neutral advantage. This explains an important part of the high rate of bullish class selections, but does **not** demonstrate predictive skill or identify a causal market mechanism.

The original 16 native probability vectors were reproduced exactly (`maximum absolute difference = 0`) using the frozen local pipeline and the 12 feature values retained in each original forecast JSON. This is **local mathematical consistency**, not independently timestamp-certified model execution.

## Frozen model and inputs

- Declared joblib SHA-256: `5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d`
- `StandardScaler` followed by scikit-learn three-class `LogisticRegression`; L2, `C=0.01`, `solver=lbfgs`.
- Feature-complete historical training observations: 3,835; training cutoff: 2026-09-30.
- Twelve lagged features drawn from prior eligible completed ES sessions.
- Fitted scikit-learn class/probability column order: **`bearish`, `bullish`, `neutral`**. The class-order check is essential: it is **not** `bearish`, `neutral`, `bullish`.
- The original 16 native forecast JSONs contain `feature_order`, `features`, `probabilities`, `model_sha256`, and challenge/source-date fields. These saved inputs were used as-is; no Databento raw data was reconstructed for this model-behavior audit.

## Validated local model-behavior checkpoint

| Diagnostic | Result |
|---|---:|
| Native saved forecasts checked | 16 / 16 |
| Recomputed vs saved probability max absolute error | 0 |
| Bullish `argmax` forecasts | 14 |
| Neutral `argmax` forecasts | 1 |
| Bearish `argmax` forecasts | 1 |
| Mean predicted P(bullish) | 0.412781 |
| Mean predicted P(neutral) | 0.269135 |
| Mean predicted P(bearish) | 0.318084 |
| Mean first-versus-second probability margin | 0.079952 |
| Median first-versus-second probability margin | 0.075832 |

A bullish `argmax` class label does **not** imply probability greater than 50%; it is simply the largest of the three probabilities.

### Intercept-only diagnostic

At a **zero standardized feature vector** (not an actual submitted forecast), the fitted intercepts yield approximately:

| Class | Intercept | Probability at zero features |
|---|---:|---:|
| Bearish | −0.12379685 | 0.29264736 |
| Bullish | +0.14819026 | 0.38412017 |
| Neutral | −0.02439342 | 0.32323247 |

The intercept-only probabilities are **not** simply the historical training-class frequencies; the intercepts were learned jointly with feature coefficients and regularization.

### Pairwise score decomposition

In a multiclass linear classifier, for classes *a* and *b*, `score(a) - score(b)` equals their intercept difference plus the sum of `(coefficient(a,j) - coefficient(b,j)) × standardized_feature(j)`.

| Mean score component across 16 native forecasts | Bullish − bearish | Bullish − neutral |
|---|---:|---:|
| Fitted intercept difference | +0.271987 | +0.172584 |
| Mean sum of feature contributions | −0.004351 | +0.329019 |
| Mean total class-score difference | +0.267636 | +0.501603 |

These are **logit/class-score differences**, not percentage-point differences in predicted probabilities.

The largest mean feature effects include five-session average returns, five-session volatility, and lagged log-volume. Effects are conditional on the frozen standardized feature values and fitted coefficients; they do **not** establish that any feature causes a future market direction.

For interpretation only, mathematically removing the fitted class intercepts while retaining saved feature contributions changes the `argmax` distribution from **14 bullish / 1 neutral / 1 bearish** to **7 bullish / 5 neutral / 4 bearish**. This counterfactual is *not* a trained replacement, an accuracy evaluation, or a proposed change to the official M1.

## Outstanding input validation — important

The saved feature-contribution reports exposed an **unusually large negative lagged ES return** affecting several early-November forecasts. Its carry-through into successive lag positions is internally consistent, but **the original market observation, session boundaries, continuity and contract mapping have not yet been independently rechecked against the immutable Notebook 11/Databento source evidence** for this new diagnostic.

This outstanding source-level check prevents interpreting the apparent later-November change in class probabilities as a verified market regime shift. If the input is found to be anomalous, retain all original records and document the discrepancy; do not silently correct or overwrite frozen forecasts.

## Cohort and audit limitations

1. These **16 native** saved forecasts are **not identical** to the previously reported **16 officially settled** evaluation cohort: the native set includes a subsequently prepared record, while the earlier settled set includes the separately flagged October 20 legacy submission. The two denominators must never be interchanged.
2. This audit does not join official Arena settlements, assess accuracy or Brier skill, or establish economic profitability.
3. Reproduction of saved probabilities from a matching local artifact demonstrates conditional computational consistency; it does not independently prove execution at submission time, all original input availability timestamps, or completeness of the eligible-opportunity calendar.
4. Arena server `created_at` timestamps lacking an explicit UTC offset remain a separate provenance limitation. No retrospective UTC certification is claimed.
5. These are 16 temporally adjacent observations. No significance, calibration, stable directional bias or market-regime causal inference is claimed from this small selected sample.
6. Original licensed market data, model bytes, full private receipt and source files, private forecast-level CSV reports and challenge identifiers are **not** included in this release. External reviewers can inspect the verification method and synthetic tests but cannot reproduce the local real-record result without authorized private evidence.

## Public implementation and safe reproduction

Public, reviewed-for-obvious-paths-and-secrets code:

- `tools/diagnostics/inspect_frozen_m1_readonly.py`
- `tools/diagnostics/analyze_frozen_m1_forecasts_readonly.py`
- `tools/diagnostics/test_m1_diagnostics.py`

Run synthetic regression tests without private artifacts:

```bat
python -m unittest discover -s tools/diagnostics -p "test_m1_diagnostics.py" -v
```

When authorized local model and saved native forecast files are available, run **from the repository root**, with the original paths preserved:

```bat
set "DIAG=data\processed\es_arena_research_v0_1\diagnostics\m1_bullish_audit_20261111"
set "MODEL=data\processed\es_arena_nb10_v0_1\deployment_ab4d5c14423352c7\es_daily_lr_deployment_v0_1.joblib"
python tools\diagnostics\inspect_frozen_m1_readonly.py "%MODEL%"
python tools\diagnostics\analyze_frozen_m1_forecasts_readonly.py --model "%MODEL%" --forecasts "data\processed\es_arena_live\forecasts" --output "%DIAG%"
```

Both scripts load joblib only after hash verification; **joblib deserialization must be restricted to trusted local artifacts**. The analyzer writes output reports in the requested private output directory and does not rewrite the original model, source snapshots, forecasts or receipts. Do not pass a public or input-source folder as `--output`.

The scripts' unit tests use deliberately **synthetic** fitted models and JSON forecasts, including altered-data rejection checks. Passing tests proves the tested behavior of the utilities, not the correctness of private economic source records.

## Next actions

1. Validate the unusual early-November lagged source observation against the original immutable ES input and Databento audit evidence.
2. Complete the official settled-cohort confusion matrix, balanced accuracy, per-class recall, constant-direction benchmark and training-prior benchmark (without altering the frozen Phase A evaluation metrics).
3. Complete the eligible-challenge opportunity calendar and a sanitized, challenge-indexed provenance package.
4. Preserve M1 without retraining throughout the original 30-opportunity Phase A; develop future model changes in separate versioned research configurations and freeze all selection and missingness rules before challenge 31.

**No M1 re-training, forecast replacement, Arena submission, Hugging Face deployment, or historical record modification was performed as part of this diagnostic.**
