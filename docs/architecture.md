# Market Mood Forecasting — System Architecture (v1.2.0)

---

# 1. Purpose

This document explains the complete architecture of the Market Mood Forecasting project after the v1.2.0 event-risk and walk-forward upgrade.

It describes:

* how raw data moves through the notebook pipeline
* how leakage prevention is enforced
* how event-risk contextualization is implemented
* how walk-forward validation operates
* how explainability is generated
* how the Gradio application consumes the final artifact
* how repository components interact

This document is intended to live in:

```text id="n7v3kp"
docs/architecture.md
```

All image paths therefore use relative references such as:

```text id="1up69v"
../images/modeling/roc_curve_v1_2_0.png
```

---

# 2. Architectural Philosophy

The v1.2.0 architecture is intentionally built around six principles:

1. leakage prevention
2. reproducibility
3. interpretability
4. temporal integrity
5. contextual forecasting
6. lightweight deployment

The project intentionally prefers:

```text id="1ol5f4"
trustworthiness > inflated accuracy
interpretability > unnecessary complexity
temporal realism > random shuffling
reproducibility > hidden notebook behavior
```

This is why the final deployed model remains:

```text id="iv2i1g"
Logistic Regression
```

instead of a more complex nonlinear system.

---

# 3. Repository-Level Architecture

```text id="r8p31s"
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
│   ├── shap_importance_v1_2_0.csv
│   ├── permutation_importance_v1_2_0.csv
│   └── logreg_coeff_importance_v1_2_0.csv
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

Ignored local execution folders:

```text id="v32cd7"
.venv/
__pycache__/
.ipynb_checkpoints/
```

are intentionally excluded.

---

# 4. End-to-End System Flow

```text id="jlwm0q"
Raw Market + Volatility + Sentiment + Macro Data
                           │
                           ▼
                 01_load_data.ipynb
                           │
                           ▼
                 02_clean_data.ipynb
                           │
                           ▼
            03_exploratory_analysis.ipynb
                           │
                           ▼
             04_feature_engineering.ipynb
                           │
                           ▼
                  fe_dataset_v1_2_0.csv
                           │
                           ▼
                    05_modeling.ipynb
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
 walk-forward evaluation          saved artifacts + CSVs
        │
        ▼
               06_model_explain.ipynb
                           │
                           ▼
               07_final_notebook.ipynb
                           │
                           ▼
                         app.py
```

The architecture is intentionally sequential and reproducible.

Each notebook consumes only outputs from previous stages.

---

# 5. Data Architecture

---

## 5.1 Data Domains

The system combines four information domains:

```text id="l1d9gg"
1. S&P 500 market behavior
2. VIX volatility behavior
3. Google sentiment indicators
4. Macroeconomic indicators
```

Version v1.2.0 additionally introduces:

```text id="i99cv3"
5. historical event-risk context
```

---

## 5.2 Frequency Alignment

The data sources use different frequencies:

* daily market data
* weekly sentiment data
* monthly macroeconomic data

The architecture normalizes all domains into:

```text id="8mdzfc"
unified weekly forecasting timeline
```

The unemployment series serves as the limiting alignment boundary.

Current synchronized coverage extends approximately through:

```text id="0o0xyg"
May 2026
```

---

## 5.3 Data Layer Responsibilities

| Layer              | Folder                     | Responsibility                |
| ------------------ | -------------------------- | ----------------------------- |
| Raw                | `data/raw/`                | Original downloaded data      |
| Cleaned            | `data/cleaned/`            | Aligned chronological dataset |
| Feature Engineered | `data/feature_engineered/` | Final modeling matrix         |

Main modeling dataset:

```text id="7c0x7t"
data/feature_engineered/fe_dataset_v1_2_0.csv
```

---

# 6. Notebook Architecture

---

# 6.1 Notebook 01 — Data Loading

**File:** `01_load_data.ipynb`

Purpose:

* load raw datasets
* inspect schema
* normalize date columns
* establish unified market dataframe

Flow:

```text id="pv7r7l"
Raw Files
    │
    ▼
Schema Inspection + Date Parsing
    │
    ▼
Combined Raw Dataset
```

This notebook intentionally avoids feature engineering.

---

# 6.2 Notebook 02 — Data Cleaning

**File:** `02_clean_data.ipynb`

Purpose:

* align frequencies
* remove duplicates
* clean missing values
* sort chronologically
* prepare safe weekly dataset

Flow:

```text id="w9fgpl"
Combined Dataset
        │
        ▼
Missing Value Handling
        │
        ▼
Date Alignment + Sorting
        │
        ▼
Cleaned Dataset
```

Output saved into:

```text id="tbpb6t"
data/cleaned/
```

---

# 6.3 Notebook 03 — Exploratory Analysis

**File:** `03_exploratory_analysis.ipynb`

Purpose:

* explore volatility structure
* analyze sentiment behavior
* inspect regime changes
* visualize event overlays
* create reviewer-friendly visuals

Generated EDA images:

```text id="e1k5ks"
../images/eda/google_trends_sentiment.png
../images/eda/mood_vs_sp500.png
../images/eda/mood_vs_sp500_annotated.png
../images/eda/sp500_vs_vix.png
../images/eda/vix_over_time.png
```

Important architectural rule:

```text id="8h6nsv"
Notebook 03 is descriptive only.
```

It never creates final model artifacts.

---

# 6.4 Notebook 04 — Feature Engineering

**File:** `04_feature_engineering.ipynb`

This notebook is the core engineering layer of v1.2.0.

---

## 6.4.1 Standard Financial Features

Examples:

```text id="4r6v0m"
sp500_returns_lag1
sp500_returns_lag2
vix_change_lag1
vix_change_lag2
```

---

## 6.4.2 Rolling Features

Examples:

```text id="2nrybj"
google_sentiment_7d_mean
google_sentiment_7d_std
vix_change_roll4_lag_std
vix_change_roll8_lag_mean
```

---

## 6.4.3 Stability Features

Examples:

```text id="oxtjz9"
vix_change_roll4_stability
sp500_ret_roll4_stability
```

---

# 7. Event-Risk Architecture (v1.2.0)

Version v1.2.0 introduces a dedicated contextual event-risk layer.

Historical events are maintained inside:

```text id="zv8p1q"
utils/market_events.py
```

---

## 7.1 Event Categories

Supported categories include:

* geopolitical
* banking
* volatility
* tariff_trade
* macro
* monetary_policy
* global_market
* sovereign_credit

---

## 7.2 Event Engineering Flow

```text id="q6gikj"
Historical Events
        │
        ▼
Event Aggregation
        │
        ▼
Rolling Event Windows
        │
        ▼
Severity Aggregation
        │
        ▼
Category Indicators
        │
        ▼
Final Event-Risk Features
```

---

## 7.3 Event-Risk Features

Examples:

```text id="tvj25n"
event_count_last_4w
event_count_last_8w
event_severity_last_4w
major_event_last_4w
days_since_last_event
tariff_trade_event_last_4w
banking_event_last_8w
```

---

## 7.4 Leakage Prevention Rules

The architecture enforces:

```text id="7v8h0q"
event_date <= current_row_date
```

Raw event identities are never passed into the model.

Blocked examples:

```text id="k25b5v"
COVID Crash
Lehman Brothers Bankruptcy
Trump Black Monday
```

Only aggregate contextual information is allowed.

---

# 8. Leakage Prevention Architecture

Leakage prevention is enforced at multiple layers.

Blocked variables include:

```text id="9i3bg4"
Target_NextWeekDrop
Mood_Zone
Mood_Zone_Cat
future-like variables
lead-like variables
next-week derived columns
```

Architectural principle:

```text id="wgn4r1"
Only historical information may enter the final modeling matrix.
```

Defense-in-depth exists at:

* notebook level
* feature-engineering level
* app validation layer

---

# 9. Modeling Architecture

---

## 9.1 Final Pipeline

```text id="9t56f5"
Feature Matrix
      │
      ▼
Median Imputation
      │
      ▼
Feature Scaling
      │
      ▼
Balanced Logistic Regression
      │
      ▼
Predicted Probability
      │
      ▼
Threshold Optimization
      │
      ▼
Final Prediction
```

Equivalent pipeline:

```text id="6hkqg6"
Imputer → Scaler → LogisticRegression(class_weight="balanced")
```

---

## 9.2 Model Selection Logic

Models evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Logistic Regression was retained because it provided:

```text id="xw3paz"
stability
+ interpretability
+ deployment simplicity
+ reproducibility
```

under walk-forward validation.

---

# 10. Walk-Forward Validation Architecture

The major v1.2.0 upgrade replaces:

```text id="vwddsx"
single chronological holdout
```

with:

```text id="0kkvfr"
expanding-window walk-forward validation
```

---

## 10.1 Walk-Forward Flow

```text id="xrzf61"
Train Older Window
        │
        ▼
Validate Future Window
        │
        ▼
Expand Training Window
        │
        ▼
Repeat Across Folds
```

This architecture better approximates real forecasting conditions.

---

## 10.2 Validation Outputs

Generated artifacts:

```text id="wlt18m"
walk_forward_summary_v1_2_0.csv
walk_forward_folds_v1_2_0.csv
tscv_auc_summary_v1_2_0.csv
tscv_auc_folds_v1_2_0.csv
```

---

# 11. Threshold Architecture

The deployed threshold is intentionally optimized for:

* downside sensitivity
* recall-oriented forecasting
* conservative risk-alert behavior

Approximate deployed threshold:

```text id="12svcc"
0.25
```

Threshold optimization is derived from:

```text id="v3t29n"
../images/modeling/f1_vs_threshold_v1_2_0.png
```

---

# 12. Artifact Architecture

The deployment architecture separates:

```text id="b0cjlwm"
trained pipeline
```

from:

```text id="y3jbdh"
deployment metadata
```

Structure:

```text id="d33hgm"
models/
├── logreg_pipeline_v1_2_0_*.joblib
└── logreg_pipeline_v1_2_0_*.json
```

---

## `.joblib` Responsibilities

Contains:

* preprocessing pipeline
* imputer
* scaler
* Logistic Regression weights

---

## `.json` Responsibilities

Contains:

* feature order
* visible features
* medians
* metadata
* blocked features
* artifact version

Architectural principle:

```text id="ez7nb2"
weights and metadata are separated
```

for safer deployment.

---

# 13. Explainability Architecture

`06_model_explain.ipynb` loads the final saved artifact instead of retraining the model.

Flow:

```text id="0o6p7d"
Saved Artifact
      │
      ▼
Coefficient Importance
      │
      ├── permutation importance
      ├── SHAP summary
      ├── SHAP dependence plots
      └── exported explanation CSVs
```

---

## 13.1 Event-Aware Explainability

Version v1.2.0 introduces event-aware SHAP diagnostics.

Examples:

```text id="y7s5fs"
dependence_event_count_last_4w_v1_2_0.png
dependence_event_severity_last_4w_v1_2_0.png
dependence_major_event_last_4w_v1_2_0.png
```

---

## 13.2 Architectural Interpretation

Main architectural conclusion:

```text id="pnm7d8"
The final model relies more heavily on volatility structure and contextual instability than on raw sentiment alone.
```

---

# 14. Application Architecture

`app.py` is the final consumer of the artifact pair.

```text id="dnd3l7"
.joblib + .json
        │
        ▼
      app.py
        │
        ▼
  Gradio Interface
```

---

## 14.1 App Responsibilities

The app:

* validates user inputs
* reconstructs hidden features
* loads metadata
* performs prediction
* generates local explanations

---

## 14.2 App Data Flow

```text id="qb6tbv"
User Inputs
      │
      ▼
Visible Feature Validation
      │
      ▼
Median Reconstruction
      │
      ▼
Run Saved Pipeline
      │
      ├── predicted probability
      ├── binary prediction
      └── local explanation
```

Visible features include:

```text id="1zn0r6"
vix_change_roll4_stability
sp500_ret_roll4_stability
Google_Sentiment_Index
vix_change_lag1
vix_change_lag2
google_sentiment_7d_mean
Unemployment
Mood_Index
```

---

## 14.3 App Guardrails

Blocked patterns:

```text id="c40ghn"
Target_NextWeekDrop
Mood_Zone
Mood_Zone_Cat
next
future
lead
t+
```

This creates defense-in-depth.

---

## 14.4 App UI Structure

The Gradio app contains:

```text id="60b3ji"
Predict
Explain & Docs
Diagnostics
```

Run locally:

```bash id="g0sp48"
python app.py
```

Open:

```text id="vq3u9e"
http://127.0.0.1:7860
```

---

# 15. Documentation Architecture

| File                         | Purpose                        |
| ---------------------------- | ------------------------------ |
| `README.md`                  | Portfolio-friendly overview    |
| `architecture.md`            | System design                  |
| `technical_documentation.md` | Detailed methodology           |
| `model_card.md`              | Intended use and limitations   |
| `testing_instructions.md`    | Reproducibility checklist      |
| `presentation.pdf`           | Reviewer-friendly presentation |

Recommended reading order:

```text id="r6pn9g"
README.md
→ architecture.md
→ technical_documentation.md
→ model_card.md
→ testing_instructions.md
→ presentation.pdf
```

---

# 16. Environment Architecture

Pinned environment:

```text id="c0m3lc"
Python 3.10.11
pandas 2.3.1
numpy 2.2.6
scikit-learn 1.7.1
shap 0.48.0
```

`runtime.txt`:

```text id="bkmvht"
python-3.10.11
```

Version pinning prevents artifact incompatibility.

---

# 17. Architectural Strengths

Strongest architectural qualities:

* leakage prevention
* walk-forward validation
* event-aware contextualization
* explainability based on deployed artifact
* separation of artifact and metadata
* lightweight deployment
* reproducible notebook sequencing

---

# 18. Architectural Limitations

Current limitations:

* single deployed model
* moderate predictive performance
* no live retraining
* no online inference layer
* no probability calibration
* simplified event aggregation
* no live streaming market data

This remains:

```text id="vq4bm0"
portfolio-grade forecasting workflow
```

not institutional trading infrastructure.

---

# 19. Future Architectural Evolution

Potential future upgrades:

```text id="ej9mzz"
Current Architecture
        │
        ▼
Multilingual App Layer
        │
        ▼
Probability Calibration
        │
        ▼
Live API Integration
        │
        ▼
Advanced Event Simulation
        │
        ▼
Alternative Time-Series Ensembles
```

---

# 20. Final Architectural Assessment

The v1.2.0 architecture demonstrates:

* leakage-safe forecasting design
* event-aware contextual modeling
* walk-forward validation discipline
* explainable financial ML workflow
* deployment-ready portfolio engineering

The strongest architectural achievement is:

```text id="hqfj7m"
temporal integrity + contextual event engineering + reproducible deployment
```

rather than raw predictive power alone.
