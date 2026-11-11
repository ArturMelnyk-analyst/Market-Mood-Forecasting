# ES Daily Forecasting — Research Status and Reproducibility Checkpoint

**Project:** Market Mood Forecasting  
**Research stream:** Headline Arena, E-mini S&P 500 (ES) daily probabilistic classification  
**Checkpoint date:** 2026-11-11  
**Status:** Preliminary prospective evaluation; M1 baseline still active  
**Git branch:** `experiment/headline-arena-es-daily`

> **Research-only disclaimer.** These are provisional forecasting results from a small, sequential sample. They do **not** establish a statistically reliable edge, calibrated probabilities, investable performance, or an economic return. No trades or profitability claims follow from these metrics.

## 1. Objective and experiment boundaries

Assess the prospective, out-of-sample quality of daily probability forecasts for three official Arena ES outcomes: **bearish, neutral, bullish**. The local historical label proxy uses the predeclared ±0.3% ES session-return boundaries (neutral includes the boundaries), but **Arena's official settlement is authoritative** for the Arena evaluation. Any disagreement between historical proxy and official settlement must be disclosed rather than silently reconciled.

The original **M1** is frozen throughout its initial study. This checkpoint documents the *existing* model and the evaluation infrastructure, not a model redesign or evidence that a newer model has been deployed.

## 2. Frozen M1 specification

| Property | Recorded implementation |
| --- | --- |
| Estimator | `scikit-learn` pipeline: `StandardScaler` + 3-class `LogisticRegression` |
| Regularization | L2; `C=0.01` |
| Solver and iterations | `lbfgs`; `max_iter=2000` |
| Class weighting and separate calibration | `class_weight=None`; no independently fitted probability calibrator |
| Market input | ES futures, Databento `GLBX.MDP3` / `ohlcv-1m`, source definition `ES.v.0` |
| Predictors | 12 pre-defined lagged ES return, range, volume, mean-return, absolute-return and volatility features, derived from eligible completed prior sessions |
| Historical fitting cutoff | 2026-09-30 |
| Feature-complete training rows | 3,835 |
| Deployment ID | `ab4d5c14423352c7` |
| Recorded model `.joblib` SHA-256 | `5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d` |

The feature-complete training-label counts are **1,092 bearish, 1,305 neutral and 1,438 bullish**. A separate upstream count contains 3,855 target-eligible labels before the rolling-feature warm-up exclusions. Notebooks using the smoothed 3,855-label reference and those using the unsmoothed 3,835-row fitting distribution must identify these as **different baselines**.

The original model bytes, licensed market-data files and private deployment manifests are retained locally. A published hash is an identity claim that can be checked against the original bytes; by itself it does not prove which model executed for each forecast.

## 3. Prospectively recorded cohort as of this checkpoint

| Measure | Count |
| --- | ---: |
| Original accepted Arena forecasts observed | **17** |
| Native forecast JSON + scored-receipt records | **16** |
| Recovered legacy record (2026-10-20) | **1** |
| Locally imported official Arena settlements | **16** |
| Native accepted records with imported outcomes | **15** |
| Accepted record not yet in local settled sample (2026-11-11) | **1** |
| Research peer-forecast vectors imported | **52** |
| Separate calendar/opportunity-ledger rows currently populated | **0** |

The 16 scored observations cover **2026-10-20 through 2026-11-10**. The 2026-11-11 accepted prediction was **not included** in these metrics because its official settlement was not yet recorded in the local evaluated dataset. Later settlement imports will change the sample; this document is a dated, non-updating checkpoint.

All 16 imported official directions matched the corresponding local ES-direction proxy **for those 16 checked dates**. This does not imply that future Arena settlements must match the proxy.

## 4. Preliminary model performance (n = 16)

These descriptive statistics were computed from original M1 probabilities linked to accepted forecasts and locally imported official Arena outcomes in Notebook 12 v0.3.2. The **multiclass Brier** here is the *unscaled sum of three class-wise squared probability errors per observation, averaged across the sample*; lower values are better.

| Metric / comparator | Result |
| --- | ---: |
| M1 directional accuracy | **8/16 = 50.0%** |
| M1 mean maximum predicted probability | **0.4137** |
| M1 unscaled three-class Brier score | **0.621310** |
| M1 multiclass log loss | **1.026332** |
| Uniform three-class Brier (1/3 each) | **0.666667** |
| Frozen *training-frequency* prior Brier | **0.675155** |
| Descriptive Brier skill versus uniform | **+0.0680** (approximately +6.8%) |

The Brier skill score is `1 - (M1 Brier / uniform Brier)`. This is **not the Arena platform's market Brier skill score**, nor a return-on-investment statistic. Differences on 16 time-dependent observations are not established evidence of stable advantage.

### Directional concentration and calibration warning

The settled outcomes comprised **7 bearish, 2 neutral and 7 bullish**. M1's largest probability favored **bullish on 14/16 dates**, bearish on 1 date and neutral on 1 date. It therefore exhibits a substantial bullish-prediction concentration. Accuracy alone can conceal this imbalance; class-wise recall, calibration, Brier and log loss should be evaluated together.

A peer with complete overlap across the 16 settled observations recorded approximately **0.616938 Brier**, compared with M1's **0.621310**, with both at **50% directional accuracy**. This is a descriptive paired comparison only; the peer dataset contains uneven coverage and does not establish which model will be better prospectively.

No model-selection claim is made from retrospective inspection of these preliminary comparisons.

## 5. Provenance audit: scope, passes and unresolved items

The read-only `verify_es_forecast_provenance.py` audit is separate from the model's daily inference and official submission workflow.

| Audit result (checkpoint) | Count / finding |
| --- | --- |
| Original frozen local model artifact checks | Passed locally |
| Native forecast files parsed | 16/16 |
| Native scored receipts matched to forecast evidence | 16/16 |
| Contradictory/corrupted native records detected | 0 |
| Strictly verified complete native local chain | **0** |
| Native records with remaining provenance limitations | **16 (`INCOMPLETE`)** |
| Historical recovered record | **1 (`LEGACY_LIMITATION`)** |

**Why are native records still incomplete?** Original server `created_at` values in the preserved Arena responses lack an explicit timezone, so their UTC meaning and precise deadline relationship are not independently certified. The separate, timezone-aware local `recorded_at_utc` is a record of the client saving the response; it is **not** independent server-time certification. Also, a local retrieval/processing timestamp does not establish Databento's original first-publication timestamp.

A special near-deadline case occurred on **2026-11-06**: the unzoned server creation string was close to the cutoff, while the locally recorded receipt was after it. These facts remain preserved without inventing a timezone or declaring a definitive late/early finding from incomplete evidence.

**October 20** was reconstructed from earlier console evidence. It is retained transparently in the descriptive cohort and specifically flagged as legacy; its original model execution and full native artifact/receipt chain cannot be upgraded retrospectively.

The strict provenance subset currently has **n = 0**. The descriptive accepted-and-officially-settled cohort currently has **n = 16**. The two populations serve different questions and must never be conflated.

## 6. November 4–10 evidence backfill

Official outcomes and peer comparison records were updated for **November 4, 5, 6, 9 and 10** using dated, retained official Arena source pages and SHA-256-pinned, chronological importer checks.

The revision-aware importer distinguishes a peer's **current scored probability vector** from historical revisions visible on the same page. Prior revisions are **not counted as additional scored peer forecasts**. The five imported outcomes expanded the local settlement table from **11 to 16** rows, and the eligible peer-vector table from **39 to 52** rows at this checkpoint.

The source HTML archives, content fingerprints, original peer revision evidence and canonical private CSVs remain outside this proposed public release pending a rights/privacy review. A public methods repository can explain the validation logic but cannot independently replay these historical private inputs without access to them.

## 7. Public reproducibility versus private evidence

**Suitable for review and potential publication after source/output screening:**

- A cleared-output, manually sanitized copy of Notebook 12 v0.3.2, including the analysis and per-challenge verification logic.
- `verify_es_forecast_provenance.py` and its synthetic tests.
- The revision-aware source importer (`import_arena_peer_day_v1_1.py`) and suitable portable tests, if fixtures are authorized for redistribution.
- This dated research report, methodological specifications and frozen artifact fingerprint declarations.

**Retained privately (not included in the Git commit):** Databento licensed market-data bytes, original authenticated Arena HTML, full private forecast/attempt/receipt evidence, the M1 joblib/manifest, backup CSVs, local executed notebooks and unfinished ROI/paper-trading Notebook 13.

For a fully independent historical audit, a reviewer would need access to authorized original artifacts and a challenge-by-challenge ledger linking model/source bytes, forecast bytes, receipt bytes and official outcomes. Hash matches demonstrate consistency of the retained records, **not proof of original model execution, no leakage, or historical pre-deadline public publication**. Git author/commit dates created now cannot retroactively prove earlier forecasts existed before their deadlines.

## 8. Forward experiment and commitments

| Phase | Planned official/experimental routing | Evaluation purpose |
| --- | --- | --- |
| **A: initial 30 eligible challenge opportunities** | Original frozen M1 remains the official Arena model; log *every* challenge opportunity, failed attempt and missed deadline | Baseline and operational quality; currently 17 accepted |
| **B: next 60 scheduled challenge opportunities** | M1 remains the single official submission; research variants produce local shadow forecasts; primary selection restricted to at most four **predeclared** finalists including M1 | Prospective paired comparison and model selection |
| **C: next 60 scheduled challenge opportunities** | Freeze the Phase B-selected winner once. If switching is permitted and the predeclared selection rule is met, use a separately versioned official adapter; otherwise retain M1. Keep unchanged M1 as shadow comparator when a challenger is official | Untouched prospective confirmation |

The **120 subsequent opportunities** are 60 shared market outcomes for selection plus 60 later shared market outcomes for confirmation—not 120 independent trials per model. Serial dependence, missingness, paired multiclass Brier differences, benchmark choice, minimum improvement, tie/fallback rules and uncertainty method must be frozen *before Phase B*. Failed/missing shadow predictions must remain in coverage accounting; the observation window must not be silently extended to collect a favorable number of successful pairs.

Research candidates may test GDELT, VIX, Google Trends and published unemployment data, with point-in-time availability controls and, where feasible, a matched-training ES-only comparison. New sources are hypotheses, not proven feature improvements. The original frozen M1, its historical probabilities, the current local daily submission program and the existing Hugging Face deployment remain unchanged.

A prospective cryptographic commitment/reveal process for each shadow model is **planned, not yet implemented in the initial historical cohort**. It should bind versioned canonical probability records, challenge/model/data identifiers and unpredictable nonces; publish commitments before each challenge deadline; then reveal all records (including failures) under a fixed policy after settlement. Publishing this report today is **not** a substitute for pre-deadline commitments.

## 9. Remaining work and release checklist

1. Retain complete local originals and backups; do not rewrite forecasts, receipts, source snapshots or historical settlements to improve audit status.
2. Establish an opportunity calendar for the first 30 study, using actual listings, attempted submissions, failures and official receipts; avoid inferring missingness solely from accepted predictions.
3. Seek authoritative confirmation of Arena's timezone convention or independent acceptance-time evidence, retaining the unzoned original server values meanwhile.
4. Import the November 11 official settlement only after it is actually resolved and authenticated source evidence passes checksum-pinned validation.
5. Publish only manually reviewed code, portable tests, cleared/sanitized notebook copies and this report. Exclude licensed evidence, credentials, original private results and unfinished ROI work.
6. Freeze the 60+60 experimental protocol, finalist-selection rule, source information cutoffs and public commitment procedure before the first Phase B opportunity.
7. Report all future evaluation snapshots with an immutable date, Git commit and honest sample/coverage denominators; keep confirmatory results separate from the Phase B model-selection sample.

## 10. Interpretive conclusion

This checkpoint documents a **working, frozen M1 research baseline and a substantially strengthened local audit workflow**. The 16-settlement descriptive comparison is preliminary. The existing provenance limitations and incomplete opportunity calendar remain visible, and no claim of statistically demonstrated predictive superiority or financial profitability is justified.

**This document reports the November 11 local research checkpoint only.** It does not assert that these files have already been pushed to GitHub, that the historical forecast archive is fully independently reproducible, or that the planned research upgrades have already been implemented.
