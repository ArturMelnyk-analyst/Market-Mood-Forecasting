# Headline Arena ES Daily Forward-Validation Protocol

**Project:** Market Mood Forecasting  
**Experiment:** ES Daily Forward Experiment  
**Experiment version:** v0.1  
**Parent system:** Market Mood Forecasting v1.2.0 (frozen)  
**Research branch:** `experiment/headline-arena-es-daily`  
**Protocol revision date:** 2026-10-17  
**Status:** Notebook 08 completed; Notebook 09 evaluation protocol specified below, pending approval and a separate documentation commit; no predictive advantage claimed.

> **Research integrity rule:** Historical data construction, historical model evaluation, and genuinely prospective Headline Arena forecasts are three different forms of evidence. They must never be presented as interchangeable.

---

## 1. Purpose and scientific question

This is a separate daily CME E-mini S&P 500 futures (ES) probabilistic forecasting experiment. It does **not** repurpose the frozen weekly S&P 500 downside-risk model as a daily forecast.

The initial question is:

> Can a model using only 12 pre-specified, previous-completed-session ES features deliver a lower chronological out-of-sample **summed multiclass Brier score** than a historical class-frequency baseline?

Subsequent, separately versioned questions are whether additional event/sentiment data improve forecasts beyond the ES-only baseline and whether any historical improvement persists in independently settled prospective forecasts.

A null or negative result is an acceptable and reportable outcome. The purpose is reliable evidence, not retrospectively maximizing a leaderboard score.

## 2. Separation from stable Market Mood Forecasting v1.2.0

The existing v1.2.0 system remains a frozen, weekly, binary next-week S&P 500 downside-risk forecast using its own features, validation, explainability, and deployment. The ES experiment has a different instrument, target, horizon, class structure, deadline, and evaluation contract.

- No modification of v1.2.0 model weights, published historical metrics, deployment, or inference behavior is authorized by this protocol.
- The experimental branch and its pull request remain separate from `main`; a passing Notebook 08 does **not** authorize a merge.
- Potential integration or a v1.3.0 release requires a later, explicit review.

## 3. Three evidence layers and their status

| Evidence layer | Definition | Current state |
|---|---|---|
| Data engineering | Reconstruct and audit historical ES sessions, labels, and predictors | **Completed in Notebook 08** |
| Historical prediction | Issue chronological, out-of-sample probability vectors on frozen historical rows | **Not yet performed; Notebook 09** |
| Prospective external validation | Submit timestamped predictions before actual Arena deadlines; obtain independent settlements | **Not yet performed** |

Notebook 08's 3,855 historical labels are **synthetic Arena-style historical outcomes**, not 3,855 historical live Arena submissions. Its eight aligned challenges validate provider/settlement reconstruction on those selected cases; they do not establish predictive skill.

## 4. Arena-style market and timing contract

- **Instrument:** CME E-mini S&P 500 futures (ES), consistent with the individual Arena challenge's frozen instrument and contract-mapping rules. SPX cash or SPY is not an interchangeable target.
- **Session reference:** 18:00 US Eastern (`America/New_York`) Globex opening observation for the trading session ending the following product day.
- **Forecast cutoff:** Nominally 10:00 ET for US-exchange daily futures, but **the challenge API's `deadline` field is authoritative for each live challenge**. Never infer a fixed UTC deadline or assume special schedules are impossible.
- **Session end:** 17:00 ET close under the challenge's stored settlement convention. For historical one-minute OHLCV reconstruction, the normal final interval is the bar timestamped 16:59 ET; that is a session-close proxy, not a claim about the exchange's separately published official daily settlement price.
- **Target return:** `session_return = session_close / session_open - 1`.
- **Timezone handling:** All local session/calendar logic uses `America/New_York`, with timezone-aware conversion and DST handling. Live records retain source UTC timestamps as well.

An individual Arena challenge's frozen settlement contract (price sources, fallback, timing, threshold, boundary ownership, cancellation) always overrides generic wording for **that** challenge. API fields must be inspected before any submission.

## 5. Frozen historical v0.1 outcome definition

**Notebook 08 uses a fixed ±0.30 percentage-point neutral threshold**, which replaces the obsolete ±0.20% statement in the earlier version of this document:

| Class | Frozen historical v0.1 definition |
|---|---|
| Bearish | `session_return < -0.003` |
| Neutral | `-0.003 <= session_return <= +0.003` |
| Bullish | `session_return > +0.003` |

Both exact boundaries are neutral. The number **0.30** is expressed in percentage points; the equivalent decimal return magnitude is **0.003**.

**Live-contract incompatibility rule:** The platform's challenge-specific `dead_zone_pct` remains authoritative. If a future live challenge's threshold or settlement contract differs from the frozen 0.30% historical specification, do **not** silently treat the historical model's labels/probabilities as directly comparable or submit an incompatible model. First document the difference, assess target compatibility, and retrain/revalidate or version the experiment as needed. No retroactive relabeling of the v0.1 artifact is permitted.

## 6. Completed Notebook 08 dataset and provenance

**Producer:** `notebooks/08_es_daily_data_and_events.ipynb`  
**Producer Git commit:** `bd928eeb370a8f8a9178f59c37b64a1163392dbc`  
**Market-data provider:** Databento  
**Dataset:** `GLBX.MDP3`  
**Continuous selector:** `ES.v.0` (`stype_in=continuous`)  
**Historical schema:** `ohlcv-1m`  
**History start (UTC):** `2010-06-06T00:00:00Z`  
**History end (UTC, exclusive):** `2026-09-30T21:00:00Z` (17:00 ET)  
**Local output:** `data/processed/es_daily_modeling_v0_1.parquet`  
**Metadata:** `data/processed/es_daily_modeling_v0_1_metadata.json`  
**SHA-256 of modeling parquet:** `6787e64fe1c0c3508d68cd2d98cc554c6e5da19f320c7080ad7e34831b9b5db5`

| Dataset property | Verified Notebook 08 result |
|---|---:|
| Historical minute records | 5,704,833 |
| Historical sessions constructed | 4,203 |
| Eligible historical sessions | 3,855 |
| Structurally excluded sessions | 348 |
| Final model-table columns | 16 |
| ES-only predictor columns | 12 |
| Raw files verified against frozen manifest | 20 |
| Selected Arena challenges reconstructed | 8 / 8 |

Notebook 08 completed the raw SHA-256 integrity gate, target and feature audits, full parquet value roundtrip, metadata serialization, and `NOTEBOOK 08 FINAL CHECKPOINT: PASS`. These are data-processing assertions—not historical forecast results.

Notebook 09 **must check the parquet SHA-256 against metadata and this protocol and stop on a mismatch**. It must not silently rebuild, update, or overwrite Notebook 08 data. A materially changed dataset requires an explicitly versioned replacement, with new provenance and a reviewed protocol amendment.

The raw purchased data and derived market-data artifacts remain local and excluded from public Git commits pending any required redistribution-rights review. Public notebooks and their saved outputs must be reviewed for credentials and licensed material.

## 7. Frozen v0.1 feature and information-availability contract

Notebook 08's 12 predictors, and only these, form the initial ES-only model input set:

```text
es_return_lag1
es_return_lag2
es_return_lag3
es_return_lag5
es_range_pct_lag1
es_log_volume_lag1
es_return_mean_5
es_return_mean_20
es_vol_5
es_vol_20
es_abs_return_mean_5
es_abs_return_mean_20
```

Feature-source semantics:

1. Use **previous target-eligible, completed sessions only**.
2. An excluded historical session is not a source observation for lag or rolling features.
3. No feature may use the current session's realized return, high, low, close, volume, target, or post-deadline information.
4. Rolling windows use the previous 5 or 20 **valid eligible source sessions**, not 5 or 20 consecutive calendar rows.
5. Initial rolling-feature warm-up NaNs are legitimate and preserved by Notebook 08; its first 20 modeling rows contain incomplete predictors.
6. Notebook 08 performs no model fitting, global scaling, imputation, resampling, or class balancing.

The full **10:00 ET information set** (including same-day data available before 10:00) is **not** part of this v0.1 ES-only baseline. Its absence is a conservative design choice, not evidence that such information is unusable.

Before live use, audit that continuous futures contract mapping and its roll decisions are reproducible from information available as of each prediction time; a date-effective historical mapping alone is not sufficient proof of a fully point-in-time trading implementation.

## 8. Historical data-quality qualifications

Notebook 08 has material coverage and comparability limitations that must be visible alongside future model results:

- Historical exchange-session rules changed over the 2010–2026 sample. Early Fridays, in particular, may not support a modern 17:00 ET session-end observation and are disproportionately excluded.
- **2,080 of the 3,855 eligible sessions** carry deviations from the expected minute-bar count. This is a diagnostic flag, **not** proof that every missing minute is corrupted: trading hours, gaps and no-trade intervals may contribute.
- Some otherwise eligible high-volatility sessions have unusually low observed bar coverage.
- A selected set of eight 2026 Arena settlements cannot prove settlement equivalence in every earlier market regime.
- Eligibility-based filtering can create systematic time/weekday/regime selection effects.

Do not retrospectively change the frozen v0.1 label or eligibility rules merely to improve model performance. Analyze these issues through predefined sensitivity tests and report exceptions, exclusions and denominator changes explicitly.

## 9. Hypothesis and preregistered evaluation hierarchy

**Primary model question (v0.1, Notebook 09):** Does a regularized, multinomial Logistic Regression model using the frozen 12 features reduce the **summed multiclass Brier score** relative to the expanding historical class-prior baseline on unseen, chronological sessions?

**Primary comparison:** Logistic Regression vs expanding class-prior probabilities, on exactly the same scored dates.

**Secondary comparisons:** Uniform (`1/3` per class) and trailing 252-eligible-session class-prior baselines; multiclass log loss, calibration, classwise metrics, and directional accuracy. Any more-complex model is exploratory unless registered in an amendment before test inspection.

**No claim of superiority** may be based on training metrics, random train/test splits, development-fold performance alone, or a selected favorable subperiod.

## 10. Chronological splits and walk-forward schedule — proposed for approval

These are the intended **calendar boundaries** to approve and commit **before any Notebook 09 training or scoring**:

| Role | Calendar sessions | Permitted use |
|---|---|---|
| Initial training history | First eligible session through 2018-12-31 | Establish training-only estimates and preprocessing |
| Development / selection | 2019-01-01 through 2023-12-31 | Expanding-window, chronological quarterly validation; choose allowed hyperparameters and model rules |
| Locked final evaluation | 2024-01-01 through 2026-09-30 | One designated model-evaluation run with no post-hoc redesign |

Notebook 09 setup may **count rows** and verify date boundaries, eligibility, ordering, feature availability and fold feasibility before any model is fitted; it must print the exact number of scored sessions in each period. If the boundaries are infeasible, amend this protocol **before** looking at scores.

Use evaluation blocks keyed to calendar quarters, and fit the feature-dependent model at each quarter's start using eligible, settled prior-session labels only. Training expands over time. Final-period test predictions may incorporate earlier **already settled** final-period rows at later scheduled quarterly refits; this is legitimate walk-forward forecasting, but any such updating rule must be applied mechanically and cannot be redesigned in response to test performance.

The final period is *locked from model selection*, not literally unseen in every sense: Notebook 08 already produced descriptive historical and validation diagnostics, and selected Arena challenges were examined.

## 11. Baselines: exact information rules

**Baseline U — Uniform:** `[1/3, 1/3, 1/3]` for every scored session. This is a fixed reference, not an estimate of market structure.

**Baseline E — Expanding class prior (primary benchmark):** For each prediction, calculate the class distribution using only eligible sessions whose labels were settled strictly **before** that prediction's deadline. Apply symmetric Laplace smoothing with `alpha=1` per class to avoid zero probabilities. This baseline may update after a previously predicted day settles; it must never see the present or future target.

**Baseline R — Rolling class prior:** Use the most recent **252 target-eligible, already settled** sessions before each prediction, also with `alpha=1` per class. If fewer than 252 precede a scored date, report the limitation and use the exact fallback rule approved before scoring (default proposal: all available settled sessions with the same smoothing).

Class order is fixed as `bearish`, `neutral`, `bullish`. All baselines and models must be scored over **identical dates** in each head-to-head comparison. The 20-row feature warm-up period is excluded from the primary feature-dependent comparison sample, but the source history may still be used to build valid later features and train baselines where permitted.

## 12. Primary model, preprocessing and hyperparameters

**Primary feature-dependent model:** scikit-learn multinomial Logistic Regression (three classes), with L2 regularization, `class_weight=None`, a training-only median imputer if genuinely needed beyond frozen warm-up exclusions, and training-only `StandardScaler` fitted within an explicit pipeline. Any sample exclusion must be determined from feature availability, never from target outcomes or apparent model performance.

**Predeclared candidate regularization strengths:** `C ∈ {0.01, 0.1, 1.0, 10.0}`. Fix `max_iter=2000` and a compatible multinomial solver supported by the installed scikit-learn version. Library/API differences must be documented before interpreting results. Convergence warnings are reported and investigated, not silently ignored.

Select `C` using only the 2019–2023 chronological development forecasts and their mean summed multiclass Brier score. If tied within floating-point tolerance, choose the **more strongly regularized** (smaller `C`) candidate. Freeze the selected rule prior to evaluating 2024–2026.

During the locked evaluation, refit the **same** pipeline and selected `C` at each calendar-quarter boundary using only prior eligible settled data; do not retune `C` from locked-test performance.

No SMOTE, shuffled validation, automatic class weighting, global feature scaling or global median imputation is allowed in this first v0.1 comparison.

## 13. Primary and secondary scoring

For observed one-hot outcome `y_(t,k)` and predicted probabilities `p_(t,k)` over the fixed class order, the **summed** multiclass Brier score is

\[
BS = \frac{1}{N}\sum_{t=1}^{N}\sum_{k\in\{bearish,neutral,bullish\}}(p_{t,k}-y_{t,k})^2.
\]

- **Lower is better.** This definition is not the variant that divides the per-row sum by three.
- **Primary improvement:** `ΔBS = BS(LogisticRegression) - BS(expanding prior)`. Negative values favor the model.
- Report both raw scores, `ΔBS`, the number of scored sessions, and a paired uncertainty interval—not only a percentage improvement.
- Check that predicted probabilities are finite, individually in `[0,1]` and sum to one to tolerance `1e-10` before scoring.
- Secondary: multiclass log loss with explicitly defined zero clipping (if needed), per-class calibration/reliability, class prevalence, accuracy, macro F1, and confusion matrix.
- Report performance separately for all registered development and locked-test periods; do not combine them into one flattering headline number.

**Arena terminology:** Its directional leaderboard score is confidence-weighted and **not** Brier. The platform uses a submitted full three-class vector in its Brier *diagnostic*. Before claiming numerical equivalence, verify the platform's exact diagnostic implementation and normalization on actual settled, submitted vectors. A local historical Brier score is not an Arena leaderboard score.

## 14. Model selection, calibration and uncertainty

- Development-period model/hyperparameter selection ends **before** evaluating the locked final period.
- The **uncalibrated** Logistic Regression probabilities are the primary v0.1 test model. An explicit calibration layer is **not** automatically enabled.
- If calibration appears justified, develop and register a separate calibrated specification using only earlier chronological, training-side calibration data; do not calibrate against final-test outcomes or silently replace the primary result.
- For the primary comparison, report a paired confidence interval for `ΔBS` that respects temporal dependence. Proposed predeclared method: a moving-block bootstrap of per-session paired Brier differences, block length 20 consecutive eligible scored sessions, 2,000 replicates, random seed 42. Report that the block length is a sensitivity assumption; optionally compare additional prespecified lengths as secondary robustness checks.
- A favorable point estimate alone is not evidence of statistically established superiority. If uncertainty includes no improvement, report the result as inconclusive.
- Report **all** predeclared primary comparisons, including null or negative results; exploratory follow-ups are labeled exploratory and versioned when material.

## 15. Predeclared historical sensitivity analyses

Keep the full eligible sample and the planned evaluation periods as the **primary** setup. Predeclare separate secondary analyses for:

1. **Market-hours regime:** pre-2012 vs post-2012 vs post-2021 observations, with explicit dates, support and denominators. Where a test period contains only later regimes, report that limitation rather than invent an earlier-regime test.
2. **Coverage:** outcomes and performance among sessions above or below a preselected observed bar-coverage threshold (proposal: 95% of that era's expected minute count); missing bars are an imperfect quality proxy, not a proven corruption label.
3. **Weekdays, contract rolls, and time periods:** compare conditional performance where sample sizes support interpretation; do not infer predictability from tiny cells.
4. **Volatility/stress:** a pre-specified, past-information-only classification of regimes; never define regimes using the future return being predicted.
5. **Continuity and mapping:** independently review selected contract-roll and historical calendar dates for point-in-time availability and data-provider mapping assumptions.

These analyses are diagnostic. If a subset performs better, that does not replace the predeclared primary result or justify retrospective sample selection.

## 16. Prediction artifacts and reproducibility

Notebook 09 must retain all out-of-sample predictions, including failures and poor results, in a machine-readable table with at least:

```text
session_date
forecast_cutoff_et
fold_id
evaluation_period
model_name
model_version
train_end_session_date
p_bearish
p_neutral
p_bullish
actual_target
```

It must additionally preserve: dataset and metadata SHA-256, producer and evaluator Git commits, protocol revision, exact feature order, preprocessing and estimator settings, selected hyperparameters, library versions, class order, random seeds, fold definitions, calibration method (if any), the number of scored rows, and exclusions.

Metrics must be recomputable from the saved per-session probability vectors and outcomes; a summary-only score is insufficient. Derived predictions remain private until reviewed for licensing and dissemination rights.

## 17. Future event-risk and sentiment experiment (out of scope for Notebook 09 v0.1)

A later experimental version may add previously known macroeconomic releases, VIX information, timestamped news, and event-risk features. Candidates from the original protocol include counts, severity, time since last event, and topic groups (geopolitical, trade, energy, banking, macro) over 1-, 3-, 5-, and 20-day lookbacks.

Every such record must have an auditable *first-known/publication timestamp*, revision history where relevant, source, and a feature-availability cutoff. An event may not be selected or rated for severity using the subsequent market move. The procedure “observe a large move → search for a matching headline → label it predictive” is prohibited.

For continuity with the original exploratory plan, candidate *future-version* features include:

```text
event_count_last_1d
event_count_last_3d
event_count_last_5d
event_count_last_20d
event_severity_last_3d
event_severity_last_5d
event_severity_last_20d
major_event_last_1d
major_event_last_5d
days_since_last_event
geopolitical_event_last_5d
trade_event_last_5d
energy_event_last_5d
banking_event_last_5d
macro_event_last_5d
```

These are research candidates, **not** columns in the frozen Notebook 08 table and **not** authorized additions to the Notebook 09 primary v0.1 comparison. Feature definitions, first-known timestamps and available-data licensing would need their own versioned protocol before implementation.

The eventual event-aware model must be compared against the ES-only model on **the same walk-forward folds and scored dates**. No claim that sentiment improves forecasting is established by Notebook 08 or by this protocol.

## 18. Live Headline Arena probability submission policy (future stage)

A future live agent should submit the full probability vector:

```json
{"bearish": 0.60, "neutral": 0.35, "bullish": 0.05}
```

The values must be nonnegative and sum to one, and the exact live API schema, permissions and endpoint should be checked against current documentation before first submission. A full vector preserves the model's asymmetric uncertainty. Legacy direction-plus-confidence encoding is not the primary planned format.

The live workflow is **manual and supervised first**:

1. Retrieve a currently open ES challenge and its `challenge_id`.
2. Verify instrument, expiration/contract mapping, open/close reference, `deadline`, `resolve_at`, `dead_zone_pct`, and frozen settlement rules.
3. Reject or version any mismatch with the model's frozen target or information-availability contract.
4. Load the frozen model and features; validate that every feature was knowable by the challenge cutoff.
5. Generate probabilities; verify class order, valid values and sum-to-one.
6. Submit before the authoritative API deadline; retain the timestamped response and its `counts_for_score` status.
7. After settlement, record the platform outcome, platform directional score, and diagnostic Brier values when available; reproduce the relevant local diagnostic under a verified normalization.
8. Keep both unsuccessful and successful attempts in the audit trail, without deleting inconvenient outcomes.

Automation is a **later, separately authorized change**, not implied by the existence of a completed notebook.

## 19. Submission revisions and forward audit trail

The initial live strategy is **one model-generated forecast per eligible challenge**, without discretionary revisions in response to later market movement. A correction for a documented technical submission mistake is allowed only if the API deadline and challenge rules permit it; retain the original and correction records.

Store an append-only record, where possible, including:

```text
challenge_id
instrument
challenge_date
deadline_utc
resolve_at_utc
dead_zone_pct
experiment_version
model_version
git_commit
dataset_sha256
feature_cutoff_utc
p_bearish
p_neutral
p_bullish
argmax_direction
max_probability
submitted_at_utc
submission_id
counts_for_score
settled_outcome
local_brier
arena_brier
arena_directional_score
revision_reference
```

No result is counted as an externally settled, scored forecast until the platform confirms that status. Cancelled, open, unscored and late paper signals are reported separately.

## 20. Secrets, licenses and publication

- Store keys/tokens in environment variables or a local secret store, never committed in `.env`, notebooks, logs, PR descriptions or Git history.
- Keep `.venv/`, purchased `data/raw/databento/`, and locally derived `data/processed/` outputs outside public Git staging unless applicable redistribution rights have been reviewed and explicitly approved.
- Review saved `.ipynb` outputs and text-based manifests for licensed data, local paths, private metadata, and possible secrets before public Git publication.
- Document enough processing and provenance for authorized reproduction without distributing raw licensed market data.
- Remain on the experimental Git branch; do not merge into stable `main` merely because research milestones pass.

## 21. Experiment freeze, change control and milestone gates

**Notebook 08 gate — complete:** Data validation, historical label/feature construction, checksum/roundtrip verification and successful final execution. Producer commit and dataset hash are recorded above.

**Notebook 09 pre-fit gate — still required:** Review and approve this revised protocol, confirm calendar split counts and feasible folds, commit and push the reviewed protocol **before producing any model scores**, and retain the immutable dataset SHA-256. If a substantive design choice changes at this point, document the amendment before training.

**Notebook 09 evidence gate — future:** Save complete chronological out-of-sample probability vectors and all predeclared metrics, report calibration and uncertainty, carry out registered sensitivity tests, and preserve negative results.

**Live readiness gate — future:** Freeze model artifact, feature timing, preprocessing, decision rules, API submission behavior and revision policy; verify point-in-time target compatibility; run manual dry-runs where supported.

**Prospective milestones — future:** One technically scored submission; approximately five settlement checks; ten technically clean forecasts; approximately 30 independently settled forecasts as an initial public-track-record milestone. Thirty is **not**, by itself, strong statistical evidence of a modest forecasting advantage.

Any material change after results are seen must be disclosed; changes made after live forecasts start require a clearly identified new experiment version. Historic evaluations and prospective results must remain separately reported.

## 22. Promotion criteria

A future integration into Market Mood Forecasting or a proposed v1.3.0 requires a separate decision supported by:

- complete historical out-of-sample evaluation against prespecified baselines;
- demonstrated data and feature availability without leakage;
- calibration and uncertainty analysis;
- adequate historical-regime and coverage robustness;
- reliable API/forecast-timing integration;
- a meaningful, independently settled prospective record;
- an explicit review of operational risk and continued separation from the weekly v1.2.0 forecast.

No automatic merge or promotion is authorized.

## 23. Open decisions to resolve before the Notebook 09 protocol commit

**These are implementation sign-offs, not permission to search for better-performing evaluation rules after seeing scores.**

1. Verify exact eligible-session counts and quarterly fold feasibility under the proposed 2010–2018 / 2019–2023 / 2024–2026 calendar partition.
2. Approve the exact `LogisticRegression` solver/version implementation and any genuinely necessary imputation rule; the preprocessing rule must remain training-only.
3. Approve quarterly refit cadence, `C` grid, expanding/rolling baseline rules, comparison dates and the `alpha=1` smoothing convention.
4. Approve summed Brier as primary, paired-block-bootstrap uncertainty specification, and final-test access restrictions.
5. Decide whether additional point-in-time instrument-roll audit evidence is required before interpreting the final historical test or submitting live predictions.
6. Verify the existing experimental GitHub PR and leave it unmerged; commit this protocol on the experimental branch.

Once approved, mark these decisions as frozen with a new protocol Git commit. Notebook 09 may then be created and run according to that exact specification.

## 24. Core scientific principle

> The integrity of a forecast made before an outcome exists is more important than an impressive historical score. A good experiment makes it possible to reject our hypothesis honestly.

---

### References and reproducibility anchors

- Canonical Headline Arena methodology (reviewed 2026-10-17): https://headlinearena.com/methodology
- Experimental Notebook 08 producer: `notebooks/08_es_daily_data_and_events.ipynb`
- Notebook 08 Git commit: `bd928eeb370a8f8a9178f59c37b64a1163392dbc`
- Frozen parquet SHA-256: `6787e64fe1c0c3508d68cd2d98cc554c6e5da19f320c7080ad7e34831b9b5db5`
- Frozen metadata: `data/processed/es_daily_modeling_v0_1_metadata.json` (local, not published)
