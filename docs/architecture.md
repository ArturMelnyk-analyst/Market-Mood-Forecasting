# Market Mood Forecasting — System Architecture (Hotfix v1.1.2)

## 1. Purpose

This document explains the complete architecture of the Market Mood Forecasting project after the v1.1.2 leakage hotfix.

It describes:

- how raw data moves through the notebook pipeline
- where leakage prevention is enforced
- how the final model artifact is built
- how explainability is generated
- how the Gradio application consumes the final saved artifact
- how every major folder in the repository interacts

The document is intended to live in:

```text
docs/architecture.md
```

Because of that, all image references below use relative paths such as:

```text
../images/modeling/roc_curve_v1_1.png
```

---

## 2. Architectural Philosophy

The architecture of v1.1.2 is intentionally built around four principles:

1. leakage prevention
2. reproducibility
3. interpretability
4. lightweight deployment

The project deliberately prefers:

```text
trustworthiness > artificially high accuracy
interpretability > unnecessary model complexity
reproducibility > hidden notebook behavior
```

This is why the final deployed model is a Logistic Regression baseline instead of a more complex model.

---

## 3. Repository-Level Architecture

```text
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
│   ├── logreg_pipeline_v1_1_1775664292.joblib
│   ├── logreg_pipeline_v1_1_1775664292.json
│   ├── logreg_coeff_importance_v1_1.csv
│   ├── permutation_importance_v1_1.csv
│   ├── shap_importance_v1_1.csv
│   ├── shap_top10_v1_1.csv
│   ├── model_compare_v1_1.csv
│   ├── tscv_auc_summary_v1_1.csv
│   └── tscv_auc_folds_v1_1.csv
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
│
├── app.py
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
├── runtime.txt
└── README.md
```


The following local execution folders are intentionally excluded from the documented architecture because they are ignored by `.gitignore`:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
```

---

## 4. End-to-End System Flow

```text
Raw Market + Sentiment + Macro Data
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
           fe_dataset_v1_1.csv
                    │
                    ▼
             05_modeling.ipynb
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
 saved model artifact     saved figures + CSVs
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

The architecture is intentionally linear. Each notebook depends only on the outputs of earlier notebooks and never bypasses an earlier stage.

---

## 5. Data Architecture

### 5.1 Raw Data Domains

The project combines four information domains:

```text
1. S&P 500 market behavior
2. VIX volatility behavior
3. Google sentiment / mood indicators
4. Macroeconomic variables
```
The final v1.1.2 model-ready dataset covers approximately 2004–2025 at weekly frequency. After cleaning, rolling-window feature engineering, lag alignment, and row filtering, the final modeling matrix contains approximately 950 observations and 32 engineered features.
Typical macro variables include unemployment and related slow-moving economic signals.

These data sources have different frequencies and date ranges, so the first architectural requirement is to normalize everything onto a common weekly timeline.

```text
S&P500 + VIX + Sentiment + Macro
                │
                ▼
      Unified Weekly Dataset
```

---

### 5.2 Data Layer Responsibilities

| Layer | Folder | Responsibility |
|------|------|------|
| Raw | `data/raw/` | Original downloaded files |
| Cleaned | `data/cleaned/` | Aligned and cleaned weekly dataset |
| Feature Engineered | `data/feature_engineered/` | Final model-ready feature matrix |

The most important intermediate output is:

```text
data/feature_engineered/fe_dataset_v1_1.csv
```

This file is the direct input into `05_modeling.ipynb`.

---

## 6. Notebook Architecture

### 6.1 Notebook 01 — Data Loading

**File:** `notebooks/01_load_data.ipynb`

Purpose:

- load source files
- inspect schema
- normalize date columns
- create the first unified dataframe

```text
Raw Files
    │
    ▼
Schema Inspection + Date Parsing
    │
    ▼
Combined Raw Dataset
```

This notebook is intentionally lightweight and should not contain cleaning or feature engineering logic.

---

### 6.2 Notebook 02 — Cleaning

**File:** `notebooks/02_clean_data.ipynb`

Purpose:

- remove duplicate rows
- align time frequencies
- fill missing values
- coerce data types
- prepare a clean chronological dataset

```text
Combined Raw Dataset
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

The cleaned dataset is saved into:

```text
data/cleaned/
```

---

### 6.3 Notebook 03 — Exploratory Analysis

**File:** `notebooks/03_exploratory_analysis.ipynb`

Purpose:

- understand relationships before modeling
- inspect sentiment vs market movement
- inspect volatility vs market movement
- identify possible leakage risks before the model stage

Generated images:

```text
../images/eda/google_trends_sentiment.png
../images/eda/mood_vs_sp500.png
../images/eda/mood_vs_sp500_annotated.png
../images/eda/sp500_vs_vix.png
```

Important architectural note:

```text
Notebook 03 is descriptive only.
It never creates the final feature matrix or model artifact.
```

---

### 6.4 Notebook 04 — Feature Engineering

**File:** `notebooks/04_feature_engineering.ipynb`

This notebook is the core of the v1.1.2 redesign.

Its job is to transform the cleaned weekly dataset into a safe modeling dataset.

#### Main Feature Families

```text
Lag Features
    sp500_returns_lag1
    sp500_returns_lag2
    vix_change_lag1
    vix_change_lag2

Rolling Features
    google_sentiment_7d_mean
    google_sentiment_7d_std
    vix_change_roll4_lag_std
    vix_change_roll8_lag_mean

Stability Features
    vix_change_roll4_stability
    sp500_ret_roll4_stability
```

#### Feature Engineering Flow

```text
Cleaned Dataset
      │
      ▼
Lag Creation
      │
      ▼
Rolling Means + Rolling Std
      │
      ▼
Stability / Interaction Features
      │
      ▼
Feature Filtering
      │
      ▼
fe_dataset_v1_1.csv
```

#### Leakage Prevention Architecture

The v1.1.2 architecture explicitly blocks any feature that could indirectly reveal future information.

Blocked columns:

```text
Target_NextWeekDrop
Mood_Zone
Mood_Zone_Cat
future-like variables
lead-like variables
next-week derived columns
```

Critical architectural rule:

```text
Only historical information may enter the final model matrix.
```

The feature engineering notebook also exports:

```text
../images/feature_engineering/feature_corr_heatmap_v1_1.png
```

---

## 7. Modeling Architecture

### 7.1 Final Pipeline

The final v1.1.2 pipeline is:

```text
Model-Ready Features
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
Threshold = 0.41
        │
        ▼
Final Class Prediction
```

Equivalent scikit-learn representation:

```text
Imputer → Scaler → LogisticRegression(class_weight="balanced")
```

---

### 7.2 Model Selection Logic

Several models were compared:

* Logistic Regression
* Random Forest
* XGBoost

The final architecture selects Logistic Regression because tree-based alternatives did not show stable improvement under time-aware validation.

The decision was not based only on raw metric comparison. It was based on the full project goal:

```text
robustness + interpretability + deployment simplicity
```

Logistic Regression was therefore more appropriate for the final v1.1.2 artifact because it is:

- easier to explain
- easier to deploy
- less fragile after leakage removal
- more transparent for portfolio and interview review

---

### 7.3 Validation Architecture

The project intentionally avoids random shuffling.

Instead, it uses:

```text
Oldest 80% of rows → training
Newest 20% of rows → validation
```

This preserves chronological order and avoids future contamination.

This is the current validation layer. A future architecture upgrade should add rolling / walk-forward time-series cross-validation to test robustness across multiple historical windows.

The architecture also exports time-series cross-validation diagnostics:

```text
models/tscv_auc_summary_v1_1.csv
models/tscv_auc_folds_v1_1.csv
```

---

### 7.4 Threshold Architecture

The final probability threshold is not 0.50.

The correct final threshold selected from the F1 optimization curve is:

```text
0.41
```

This value is supported by:

```text
../images/modeling/f1_vs_threshold_v1_1.png
```

The threshold architecture is important because the class distribution is imbalanced.

---

## 8. Artifact Architecture

The deployment architecture intentionally separates the saved model from the saved metadata.

```text
models/
├── logreg_pipeline_v1_1_1775664292.joblib
└── logreg_pipeline_v1_1_1775664292.json
```

### `.joblib` Responsibilities

Contains:

- trained preprocessing pipeline
- imputer
- scaler
- final Logistic Regression model

### `.json` Responsibilities

Contains:

- feature order
- visible features shown in the app
- stored training medians
- artifact name
- blocked features
- model version

Architectural principle:

```text
Weights and deployment metadata are stored separately.
```

This makes the app safer and easier to maintain.

---

## 9. Explainability Architecture

`06_model_explain.ipynb` reads the saved final artifact instead of retraining the model.

```text
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

Saved explanation images:

```text
../images/model_explain/shap_top20_bar_v1_1.png
../images/model_explain/summary_v1_1.png
../images/model_explain/dependence_Google_Sentiment_Index_v1_1.png
../images/model_explain/dependence_sp500_ret_roll4_stability_v1_1.png
../images/model_explain/dependence_vix_change_lag1_v1_1.png
../images/model_explain/dependence_vix_change_roll4_lag_std_v1_1.png
../images/model_explain/dependence_vix_change_roll4_stability_v1_1.png
```

Most important architectural conclusion:

```text
The final model relies primarily on VIX stability structure rather than raw sentiment alone.
```

---

## 10. Application Architecture

`app.py` is the final consumer of the saved artifact pair.

```text
.joblib + .json
        │
        ▼
     app.py
        │
        ▼
 Gradio Interface
```

The app follows a separation-of-concerns design:

* `.joblib` stores the trained preprocessing and model pipeline
* `.json` stores feature order, medians, visible inputs, blocked patterns, and metadata
* `app.py` handles validation, UI behavior, prediction, and local explanations

This prevents the UI from becoming tightly coupled to hidden notebook state.

### 10.1 App Data Flow

```text
User Inputs
      │
      ▼
Validate Visible Features
      │
      ▼
Fill Missing Features With Stored Medians
      │
      ▼
Run Saved Pipeline
      │
      ├── predicted probability
      ├── final binary class
      └── local explanation
```

Visible user inputs:

```text
vix_change_roll4_stability
sp500_ret_roll4_stability
Google_Sentiment_Index
vix_change_lag1
vix_change_lag2
google_sentiment_7d_mean
Unemployment
Mood_Index
```

All other required features are automatically reconstructed from stored medians in the metadata file.

---

### 10.2 App Guardrails

The app contains a second protection layer beyond the notebook pipeline.

Blocked feature names include:

```text
Target_NextWeekDrop
Mood_Zone
Mood_Zone_Cat
next
future
lead
t+
```

This creates defense-in-depth:

```text
Even if a leaky feature accidentally survives the notebook layer,
the application still rejects it.
```

---

### 10.3 App UI Architecture

The Gradio application is divided into three tabs:

```text
Predict
Explain & Docs
Diagnostics
```

| Tab | Responsibility |
|------|------|
| Predict | Inputs, probability, classification, contribution chart |
| Explain & Docs | Image gallery and documentation references |
| Diagnostics | Metadata verification and loaded artifact details |

The app is launched locally through:

```text
python app.py
```

and runs at:

```text
http://127.0.0.1:7860
```

---

## 11. Documentation Architecture

The project intentionally separates documentation into multiple layers:

| File | Purpose |
|------|------|
| `README.md` | Short portfolio-friendly overview |
| `architecture.md` | End-to-end system design |
| `technical_documentation.md` | Detailed methodology and notebook logic |
| `model_card.md` | Intended use, limitations, metrics, assumptions |
| `testing_instructions.md` | Reproducible rerun and validation checklist |
| `presentation.pdf` | Interview-friendly summary slides |

Recommended reading order:

```text
README.md
→ architecture.md
→ technical_documentation.md
→ model_card.md
→ testing_instructions.md
→ presentation.pdf
```

---

## 12. Environment Architecture

Final pinned environment:

```text
Python 3.10.11
pandas 2.3.1
numpy 2.2.6
scikit-learn 1.7.1
shap 0.48.0
```

The environment is pinned because earlier versions of the project experienced incompatibility between saved artifacts and newer library versions.

`runtime.txt`:

```text
python-3.10.11
```

---

## 13. Architectural Strengths

The strongest parts of the final architecture are:

- explicit leakage prevention
- clean separation between artifact and metadata
- explainability based on the final deployed model
- reproducible notebook sequence
- lightweight but realistic deployment layer
- clear separation between technical documentation types

---

## 14. Architectural Limitations and Future Evolution

Current limitations:

- only one forecasting horizon
- only one deployed model
- modest predictive performance
- no probability calibration yet
- no deployed Hugging Face architecture yet

Future architecture upgrades could include:

```text
Current Architecture
        │
        ▼
Hugging Face Deployment
        │
        ▼
Multiple Forecast Horizons
        │
        ▼
Probability Calibration
        │
        ▼
Alternative Leakage-Safe Models
```

Even with these limitations, the current v1.1.2 architecture is intentionally appropriate for a portfolio project because it demonstrates:

```text
realistic modeling discipline
+
strong leakage prevention
+
clear deployment design
+
professional documentation structure
```
