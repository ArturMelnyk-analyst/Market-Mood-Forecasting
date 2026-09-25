# Headline Arena ES Daily Forward-Validation Protocol

**Project:** Market Mood Forecasting  
**Experiment:** ES Daily Forward Experiment  
**Experiment version:** v0.1  
**Status:** Experimental  
**Parent project:** Market Mood Forecasting v1.2.0  

---

## 1. Purpose

This experiment extends the Market Mood Forecasting project with a separate prospective forward-validation workflow.

The existing Market Mood Forecasting v1.2.0 model remains frozen and continues to represent the weekly S&P 500 downside-risk forecasting system.

The Headline Arena experiment does **not** reuse the v1.2.0 weekly prediction as a daily forecast.

Instead, it defines a new forecasting problem specifically aligned with the Headline Arena CME E-mini S&P 500 futures challenge.

The purpose of this experiment is to evaluate whether a leakage-safe probabilistic market model can produce useful daily probability estimates on observations that did not exist when the model was developed.

Historical walk-forward results and live Headline Arena results will therefore be reported separately.

---

## 2. Relationship to Market Mood Forecasting v1.2.0

### Frozen v1.2.0 model

The existing deployed project predicts:

> Probability of a next-week S&P 500 downside move.

Characteristics:

- weekly forecasting horizon
- binary classification
- Logistic Regression
- leakage-safe engineered features
- event-risk features
- walk-forward validation
- SHAP explainability
- Gradio deployment
- Hugging Face deployment

The v1.2.0 model and its reported results will not be retrospectively modified as part of the ES daily experiment.

### New ES daily experiment

The new experiment predicts:

> The probability that the CME E-mini S&P 500 futures session will settle bearish, neutral, or bullish according to the Headline Arena challenge definition.

This is a different:

- instrument,
- horizon,
- target,
- class structure,
- and evaluation contract.

The experiment is therefore developed independently from the frozen weekly v1.2.0 forecasting model.

---

## 3. Forecasting Contract

### Instrument

CME E-mini S&P 500 futures (ES), using the challenge instrument and contract definition supplied by Headline Arena.

Historical data used for model development must correspond as closely as possible to the same ES futures market and session definition.

SPX cash, SPY, or ordinary U.S. equity-market session prices must not be substituted for the ES target without explicit justification.

### Forecast horizon

One exchange session.

### Session reference open

18:00 ET Globex session open.

### Settlement

17:00 ET settlement-day close relative to the corresponding session open.

### Submission deadline

The Headline Arena API challenge `deadline` field is authoritative.

The implementation must use the API-provided timestamp rather than manually assuming a fixed UTC time.

This is necessary because daylight-saving-time changes and special market schedules can affect clock conversions.

### Target return

The target return is:

`session_return = (session_close / session_open) - 1`

### Target classes

- **Bullish:** session return > +0.20%
- **Neutral:** -0.20% <= session return <= +0.20%
- **Bearish:** session return < -0.20%

Exactly +0.20% and exactly -0.20% resolve as neutral.

The challenge-specific `dead_zone_pct` supplied by Headline Arena remains authoritative if the platform contract changes in the future.

---

## 4. Probabilistic Output Contract

The model will produce a complete three-class probability distribution:

- P(bearish)
- P(neutral)
- P(bullish)

The probabilities must:

- be non-negative,
- each lie between 0 and 1,
- sum to 1 within the Headline Arena API tolerance.

Example:

```json
{
  "bearish": 0.60,
  "neutral": 0.35,
  "bullish": 0.05
}
```

The full probability vector will be submitted directly through the Headline Arena probability-vector endpoint.

The legacy `direction + confidence` representation will not be used as the primary submission format because it forces the remaining probability mass to be distributed uniformly across the non-selected outcomes.

Direct vector submission preserves the model's complete probabilistic view.

---

## 5. Primary Evaluation Metric

The primary metric is multiclass Brier score.

For three classes:

`Brier = Σ(p_i - o_i)^2`

where:

- `p_i` is the predicted probability for class i,
- `o_i` is the one-hot encoded observed outcome.

Example:

Prediction:

- bearish = 0.60
- neutral = 0.35
- bullish = 0.05

Observed outcome:

- bearish = 1
- neutral = 0
- bullish = 0

Brier score:

```text
(0.60 - 1)^2
+ (0.35 - 0)^2
+ (0.05 - 0)^2
= 0.285
```

Lower Brier score is better.

The local implementation must reproduce Headline Arena's reported Brier score.

Any discrepancy between the local and Arena Brier calculations must be investigated before the experiment continues.

---

## 6. Secondary Evaluation Metrics

Secondary metrics may include:

- log loss
- accuracy
- macro F1
- per-class precision
- per-class recall
- confusion matrix
- calibration curves
- calibration error
- class prevalence

Accuracy will not be treated as the primary objective.

The experiment focuses on probability quality and calibration rather than only the argmax direction.

---

## 7. Baselines

At minimum, two probabilistic baselines will be evaluated.

### Uniform baseline

Each class receives probability:

- bearish = 1/3
- neutral = 1/3
- bullish = 1/3

### Historical class-prior baseline

Class probabilities are estimated using only the training observations available within each walk-forward training window.

Future observations must never be used to estimate historical class prevalence.

The model should be compared against both baselines.

---

## 8. Historical Data Policy

The ES daily experiment will use a new historical dataset.

The original Market Mood Forecasting v1.2.0 dataset remains frozen.

Historical data for this experiment should be updated through the final completed market session available before the final experiment model is frozen.

The historical cutoff will be recorded explicitly in the final experiment metadata.

Historical data sources must support reconstruction of the Headline Arena session definition as accurately as possible.

Before model training, historical ES session labels should be compared against a sample of already-settled Headline Arena ES challenges.

The following should be compared:

- historical session open
- Arena reference open
- historical session close
- Arena settlement close
- locally generated outcome
- Arena settled outcome

If the selected market-data source produces systematic disagreement with Arena settlement data, it should not be used as the experiment's primary target source.

---

## 9. Information Cutoff and Leakage Policy

Headline Arena submissions are due before the challenge deadline.

The model may only use information that was genuinely available before its predefined prediction cutoff.

The API-provided challenge deadline remains authoritative.

Potentially valid information includes:

- previous completed ES session information
- previous VIX close
- lagged ES returns
- lagged VIX changes
- rolling statistics based only on past observations
- previously released macroeconomic information
- previously known event information
- same-day pre-deadline market information only when historically reproducible

Examples of invalid information include:

- settlement-day 17:00 ES close
- settlement-day closing VIX
- economic releases published after the forecast cutoff
- news published after the forecast cutoff
- future or lead variables
- target-derived variables

If the historical availability timestamp of a feature cannot be reconstructed reliably, the feature should not be used in the first experiment version.

---

## 10. Event-Risk Information

Event-risk engineering may be evaluated as an extension of the market-only model.

Events must be recorded according to information that was publicly available at the time.

Where possible, each event record should include:

- first-known timestamp
- event date
- category
- severity
- description
- source

Event inclusion must not depend on knowing the subsequent ES return.

The procedure:

```text
market moved strongly
→ search afterward for explanatory news
→ label that news as a major event
```

is not permitted.

Event severity must also not be assigned based on the subsequent market outcome.

Potential daily event features may include:

- event_count_last_1d
- event_count_last_3d
- event_count_last_5d
- event_count_last_20d
- event_severity_last_3d
- event_severity_last_5d
- event_severity_last_20d
- major_event_last_1d
- major_event_last_5d
- days_since_last_event
- geopolitical_event_last_5d
- trade_event_last_5d
- energy_event_last_5d
- banking_event_last_5d
- macro_event_last_5d

The final feature set will be determined using leakage-safe historical evaluation.

---

## 11. Model Comparison

At minimum, two model specifications should be compared.

### Model A — Market-only model

Candidate inputs may include:

- lagged ES returns
- rolling ES volatility
- rolling ES trend/stability features
- previous VIX close
- lagged VIX changes
- rolling VIX statistics
- previous S&P-related returns
- calendar variables
- legitimately available macro variables

### Model B — Event-aware model

Model B uses the same modeling framework as Model A plus leakage-safe event-risk features.

The two models should be evaluated using the same walk-forward folds.

Event features will only be retained if they provide defensible improvement in probabilistic performance, calibration, robustness, or interpretation.

The experiment will not assume in advance that event features must improve predictive accuracy.

---

## 12. Initial Model Family

The initial primary model will be Multinomial Logistic Regression.

The motivation is:

- direct three-class probability output
- interpretability
- low model complexity
- strong probability baseline
- easier calibration analysis
- consistency with the broader Market Mood Forecasting philosophy

More complex models may be evaluated later, but complexity will only be added if it produces meaningful and stable improvement under leakage-safe walk-forward validation.

---

## 13. Class-Imbalance Policy

The primary objective is probabilistic forecasting.

Therefore the initial experiment should not automatically apply:

- SMOTE
- random oversampling
- random undersampling
- shuffled cross-validation

Class weighting will not automatically be enabled.

Any class-balancing method must be evaluated for its effect on probability calibration before adoption.

---

## 14. Walk-Forward Validation

Random train/test splitting will not be used as the primary validation strategy.

The experiment will use chronological expanding-window walk-forward validation.

Conceptually:

1. Train on historical observations available up to time T.
2. Predict the next unseen evaluation period.
3. Expand the training window.
4. Repeat.

Every out-of-sample historical probability should therefore be generated by a model trained only on earlier data.

The resulting out-of-sample probability vectors will be used for:

- Brier evaluation
- log-loss evaluation
- calibration analysis
- class-level metrics
- baseline comparison
- model comparison

---

## 15. Calibration

Probability calibration is a first-class objective.

Calibration will be examined separately for:

- bearish
- neutral
- bullish

Predicted probabilities should be compared with observed class frequencies.

The raw Multinomial Logistic Regression model should be evaluated before adding an explicit calibration layer.

Calibration methods should only be added if the historical out-of-sample evidence demonstrates a meaningful calibration problem.

---

## 16. Experiment Freeze

Before the first scored live Headline Arena prediction, the experiment will freeze and record:

- target definition
- historical data source
- historical cutoff
- feature list
- information cutoff
- event inclusion rules
- model architecture
- hyperparameters
- preprocessing pipeline
- calibration method, if any
- Brier implementation
- submission policy
- revision policy
- model artifact
- model metadata
- Git commit

Live results will not be used retrospectively to redesign the same frozen experiment while continuing to report it as unchanged.

Any material methodological change after live evaluation begins should create a new experiment version.

---

## 17. Live Submission Policy

The first live forecasts will be manually supervised.

Initial workflow:

1. retrieve the current Arena challenge
2. inspect challenge ID
3. inspect instrument
4. inspect deadline
5. inspect dead-zone percentage
6. build current feature vector
7. confirm feature timestamps
8. load the frozen model artifact
9. calculate three-class probabilities
10. verify probability sum
11. submit the full probability vector
12. save the API response
13. wait for settlement
14. reproduce the Arena Brier score locally

Automation should only begin after the manual workflow has been shown to operate reliably.

---

## 18. Revision Policy

The initial experiment uses:

> One model-generated forecast per challenge.

Discretionary revisions based on subsequent market movement are not allowed.

A forecast may only be corrected if there is a documented technical submission error and the correction remains valid under the Arena deadline rules.

Dynamic forecast revision may be studied separately in a future experiment version.

---

## 19. Forward Audit Trail

Every submitted forecast should eventually record:

- challenge_id
- instrument
- challenge_date
- deadline_utc
- dead_zone_pct
- experiment_version
- model_version
- Git commit
- feature_cutoff_utc
- p_bearish
- p_neutral
- p_bullish
- argmax_direction
- max_probability
- submitted_at_utc
- submission_id
- counts_for_score
- settled_outcome
- local_brier
- arena_brier

Historical live forecast records should be append-only whenever practical.

---

## 20. Secrets and Credentials

Headline Arena credentials must never be committed to GitHub.

The repository must not contain:

- API keys
- access tokens
- client secrets
- private authentication material

Credentials should be stored through environment variables or another local secret-storage mechanism excluded by `.gitignore`.

---

## 21. Experiment Milestones

### Integration milestone

1 successful scored forecast.

### Pipeline sanity milestone

Approximately 5 settled forecasts.

### Initial live-record milestone

Approximately 10 technically clean forecasts.

### First meaningful public milestone

Approximately 30 settled forecasts.

At this point, historical walk-forward results and live forward results may be compared in a dedicated analysis.

A small number of forecasts must not be presented as strong statistical evidence.

---

## 22. Promotion to Main Project

This experiment should not immediately replace Market Mood Forecasting v1.2.0.

A future v1.3.0 should only be considered after the prospective pipeline is technically and scientifically mature.

Potential readiness criteria include:

- daily ES target fully validated
- leakage audit passed
- historical walk-forward analysis completed
- Brier implementation verified
- probability calibration evaluated
- Headline Arena API integration validated
- multiple technically clean live submissions completed
- externally settled forecast record established
- historical and prospective results clearly documented

Until then, the ES daily system remains an experimental extension.

---

## 23. Core Scientific Principle

The goal of this experiment is not to maximize a leaderboard score retrospectively.

The goal is to determine whether a forecasting methodology defined before live evaluation can produce useful, calibrated probabilities on genuinely unseen future market sessions.

The integrity of the evaluation is more important than obtaining an impressive headline metric.
