# Market Mood Forecasting — Technical Documentation (v1.2.0)

---

# 1. Project Overview

**Project Name:** Market Mood Forecasting
**Version:** v1.2.0
**Primary Objective:** Build a leakage-safe, interpretable, event-aware forecasting pipeline capable of estimating next-week downside market risk using financial, macroeconomic, sentiment, and contextual event-risk features.

Version v1.2.0 significantly extends the earlier leakage-safe rebuild by introducing:

* contextual event-risk engineering
* walk-forward validation
* refreshed 2026 data alignment
* event-aware explainability analysis
* expanded SHAP diagnostics
* reproducible walk-forward evaluation outputs

This document is intended to live inside the `docs/` folder. Therefore, all image paths use relative references such as:

```text id="r3ajxh"
../images/...
```

---

# 2. v1.2.0 Project Evolution

Earlier versions of the project focused primarily on:

* leakage removal
* interpretable modeling
* sentiment and volatility forecasting

Version v1.2.0 introduces a broader macro-context forecasting layer through:

```text id="k5s1mn"
historical event-risk contextualization
```

The upgraded pipeline now incorporates:

* geopolitical events
* banking crises
* macroeconomic shocks
* tariff/trade disruptions
* monetary policy stress periods
* volatility shock windows

These events are converted into:

* recency windows
* severity aggregates
* event density indicators
* category-level contextual features

without exposing raw event identity to the model.

---

# 3. Repository Structure

```text id="i0k34d"
Market-Mood-Forecasting/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── feature_engineered/
│
├── docs/
│   ├── architecture.md
│   ├── technical_documentation.md
│   ├── model_card.md
│   ├── testing_instructions.md
│   └── presentation.pdf
│
├── images/
│   ├── eda/
│   ├── feature_engineering/
│   ├── modeling/
│   └── model_explain/
│
├── models/
│   ├── logreg_pipeline_v1_2_0_*.joblib
│   ├── logreg_pipeline_v1_2_0_*.json
│   ├── walk_forward_summary_v1_2_0.csv
│   ├── walk_forward_folds_v1_2_0.csv
│   ├── tscv_auc_summary_v1_2_0.csv
│   ├── tscv_auc_folds_v1_2_0.csv
│   ├── event_feature_comparison_v1_2_0.csv
│   ├── logreg_coeff_importance_v1_2_0.csv
│   ├── permutation_importance_v1_2_0.csv
│   └── shap_importance_v1_2_0.csv
│
├── notebooks/
│   ├── 01_load_data.ipynb
│   ├── 02_clean_data.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   ├── 06_model_explain.ipynb
│   └── 07_final_notebook.ipynb
│
├── utils/
│   └── market_events.py
│
├── app.py
├── README.md
├── requirements.txt
└── runtime.txt
```

---

# 4. End-to-End Workflow

The project follows a seven-stage workflow:

1. Load raw market and macroeconomic data
2. Clean and align datasets chronologically
3. Explore market, volatility, and sentiment behavior
4. Engineer leakage-safe features
5. Train and validate forecasting models
6. Explain model behavior and event influence
7. Serve the final artifact through Gradio

---

# 5. Notebook 01 — Data Loading

**File:** `01_load_data.ipynb`

Purpose:

* load all source datasets
* inspect date ranges
* standardize schema
* verify data completeness

Main data domains:

* S&P 500
* VIX
* Google Trends sentiment
* unemployment rate

Version v1.2.0 refreshes all datasets through approximately:

```text id="u0iz3g"
May 2026
```

with unemployment alignment serving as the limiting synchronization boundary.

---

# 6. Notebook 02 — Data Cleaning

**File:** `02_clean_data.ipynb`

Purpose:

* clean missing values
* align frequencies
* normalize dates
* enforce chronological integrity
* remove duplicates
* ensure numeric consistency

Key operations:

* datetime conversion
* resampling
* forward alignment
* chronological sorting
* safe type coercion

This stage remains leakage-safe because no future observations are introduced.

---

# 7. Notebook 03 — Exploratory Analysis

**File:** `03_exploratory_analysis.ipynb`

Purpose:

* explore market behavior
* visualize volatility dynamics
* inspect sentiment interactions
* identify regime changes
* create reviewer-friendly EDA visuals

Generated EDA visuals include:

```text id="h0o5an"
../images/eda/google_trends_sentiment.png
../images/eda/mood_vs_sp500.png
../images/eda/mood_vs_sp500_annotated.png
../images/eda/sp500_vs_vix.png
../images/eda/vix_over_time.png
```

This notebook remains descriptive and does not create final artifacts.

---

# 8. Notebook 04 — Feature Engineering

**File:** `04_feature_engineering.ipynb`

This notebook is the core engineering layer of v1.2.0.

---

## 8.1 Objectives

The notebook:

* creates leakage-safe forecasting features
* engineers rolling financial statistics
* creates contextual event-risk features
* prepares the final modeling matrix

---

## 8.2 Standard Financial Features

Examples include:

```text id="2v1pmt"
sp500_returns_lag1
sp500_returns_lag2
vix_change_lag1
vix_change_lag2
google_sentiment_7d_mean
google_sentiment_7d_std
```

---

## 8.3 Rolling Stability Features

Examples:

```text id="xhjlwm"
vix_change_roll4_stability
vix_change_roll8_lag_mean
sp500_ret_roll4_stability
sp500_ret_roll8_lag_mean
```

---

# 9. Event-Risk Engineering (v1.2.0)

Version v1.2.0 introduces a dedicated event-risk layer.

Historical market events are stored in:

```text id="ibw73w"
utils/market_events.py
```

---

## 9.1 Event Categories

Examples include:

* banking crises
* geopolitical shocks
* tariff/trade disruptions
* monetary policy events
* volatility spikes
* macroeconomic stress periods

---

## 9.2 Leakage-Safe Event Design

Events are transformed into:

* rolling event counts
* severity aggregates
* recency indicators
* category activity windows

Raw event identities are NEVER passed directly into the model.

Only events satisfying:

```text id="xbpjyu"
event_date <= current_row_date
```

are visible during feature construction.

This prevents future event leakage.

---

## 9.3 Event-Risk Features

Examples:

```text id="qlr5d5"
event_count_last_4w
event_count_last_8w
event_severity_last_4w
major_event_last_4w
days_since_last_event
tariff_trade_event_last_4w
banking_event_last_8w
```

---

## 9.4 Saved Feature Engineering Outputs

Generated artifacts include:

```text id="vtfskp"
../images/feature_engineering/feature_corr_heatmap_v1_2_0.png
```

and:

```text id="tdc54n"
data/feature_engineered/fe_dataset_v1_2_0.csv
```

---

# 10. Notebook 05 — Modeling

**File:** `05_modeling.ipynb`

This notebook trains and evaluates the final forecasting pipeline.

---

## 10.1 Final Model Choice

The final deployed model remains:

```text id="vzhb8d"
Logistic Regression
```

Alternative models including:

* Random Forest
* XGBoost

were evaluated but did not show sufficiently stable improvement under walk-forward validation.

Logistic Regression provided the strongest balance of:

```text id="4vm3lo"
interpretability
+ stability
+ reproducibility
+ deployment simplicity
```

---

## 10.2 Pipeline Structure

The final pipeline structure:

```text id="l09jgs"
Imputation → Scaling → Logistic Regression
```

---

# 11. Walk-Forward Validation (v1.2.0)

The largest modeling upgrade in v1.2.0 is the transition from:

```text id="h7u11v"
single chronological holdout
```

to:

```text id="g8l8an"
expanding-window walk-forward validation
```

---

## 11.1 Validation Process

For each fold:

1. train on older historical windows
2. validate on future unseen periods
3. expand training history
4. repeat sequentially

This better approximates real-world forecasting conditions.

---

## 11.2 Saved Validation Outputs

Artifacts include:

```text id="p9z65t"
walk_forward_summary_v1_2_0.csv
walk_forward_folds_v1_2_0.csv
tscv_auc_summary_v1_2_0.csv
tscv_auc_folds_v1_2_0.csv
```

---

## 11.3 Threshold Optimization

The deployed threshold is intentionally optimized for:

* downside sensitivity
* early warning behavior
* recall-oriented forecasting

Approximate deployed threshold:

```text id="zod6w0"
0.25
```

---

# 12. Event-Aware Comparison Analysis

Version v1.2.0 explicitly compares:

* baseline model without event features
* event-aware forecasting model

Results indicate:

* event-risk features slightly improve F1 behavior
* PR AUC remains relatively stable
* event features improve contextual awareness and interpretability

The event-aware model behaves more like:

```text id="v9v3q5"
contextual risk-alert forecasting system
```

than a precision-focused trading engine.

---

# 13. Notebook 06 — Model Explainability

**File:** `06_model_explain.ipynb`

Purpose:

* explain model behavior
* analyze feature influence
* evaluate event-risk interactions
* create reviewer-friendly SHAP diagnostics

---

## 13.1 Explainability Methods

Methods used:

* Logistic Regression coefficients
* permutation importance
* SHAP explainability
* dependence analysis

---

## 13.2 Generated Explainability Outputs

Artifacts include:

```text id="x4r0t5"
summary_v1_2_0.png
shap_top20_bar_v1_2_0.png
dependence_event_count_last_4w_v1_2_0.png
dependence_event_severity_last_4w_v1_2_0.png
dependence_major_event_last_4w_v1_2_0.png
```

---

## 13.3 Interpretation Notes

Several dependence plots appear highly linear because:

* the final model is Logistic Regression
* SHAP is consistent with linear behavior

This is expected and not a visualization error.

---

# 14. Notebook 07 — Final Notebook

**File:** `07_final_notebook.ipynb`

Purpose:

* consolidate final outputs
* summarize methodology
* present modeling conclusions
* provide reviewer-friendly narrative

The notebook combines:

* modeling outputs
* walk-forward results
* explainability outputs
* event-aware findings
* final conclusions

---

# 15. Final Metrics Summary

Approximate v1.2.0 metrics:

| Metric    | Value |
| --------- | ----: |
| ROC AUC   | ~0.53 |
| PR AUC    | ~0.44 |
| Best F1   | ~0.58 |
| Threshold | ~0.25 |

Interpretation:

* predictive power remains modest
* methodology quality is the primary achievement
* performance is intentionally conservative and leakage-safe

---

# 16. Application Layer (`app.py`)

The project includes a Gradio deployment layer.

---

## 16.1 Loaded Artifacts

The app loads:

```text id="3dk7tb"
models/logreg_pipeline_v1_2_0_*.joblib
models/logreg_pipeline_v1_2_0_*.json
```

---

## 16.2 App Responsibilities

The app:

* loads metadata
* validates feature order
* auto-fills hidden features
* prevents invalid inference scenarios
* generates local explanation outputs

---

## 16.3 Safety Guardrails

The app blocks:

* future-like features
* target-derived variables
* invalid all-zero scenarios

The app intentionally returns:

```text id="m4p0oz"
No prediction generated
```

for unrealistic baseline input.

---

## 16.4 Local Launch

Run locally:

```bash
python app.py
```

Open:

```text id="l2tr0n"
http://127.0.0.1:7860
```

---

# 17. Relative Image Paths from `docs/`

Correct image pattern:

```text id="zkkg2q"
../images/<folder>/<file>.png
```

Examples:

```text id="f0r1s7"
../images/eda/vix_over_time.png
../images/modeling/roc_curve_v1_2_0.png
../images/model_explain/summary_v1_2_0.png
```

---

# 18. Reproducibility

Recommended execution order:

1. `01_load_data.ipynb`
2. `02_clean_data.ipynb`
3. `03_exploratory_analysis.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_modeling.ipynb`
6. `06_model_explain.ipynb`
7. `07_final_notebook.ipynb`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

---

# 19. Limitations

Main limitations include:

* modest predictive performance
* simplified event aggregation
* Logistic Regression linearity
* absence of live market integration
* no production portfolio management layer
* financial regime instability

This remains:

```text id="2s98wf"
portfolio-grade forecasting workflow
```

not institutional trading infrastructure.

---

# 20. Future Improvements

Potential future improvements include:

* probability calibration
* transformer-based sentiment integration
* dynamic regime detection
* live API integration
* online retraining workflows
* advanced time-series ensemble comparison
* multilingual app polish
* UI/UX improvements for deployment layer

---

# 21. Final Assessment

Version v1.2.0 transforms the project into:

* event-aware financial forecasting workflow
* leakage-safe macro-risk modeling system
* walk-forward validated ML pipeline
* interpretable applied financial ML portfolio project

The strongest aspects are:

```text id="v7lfmg"
methodological discipline
+ temporal safety
+ contextual event engineering
+ explainability
+ deployment readiness
+ reproducibility
```

rather than raw predictive power alone.



