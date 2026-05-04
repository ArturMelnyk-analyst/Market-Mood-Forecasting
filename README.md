# Market Mood Forecasting

Leakage-safe market sentiment forecasting project that predicts whether the S&P 500 is likely to decline in the following week.

The project combines:

* macroeconomic indicators
* Google sentiment data
* VIX behavior
* engineered stability and volatility features
* an interpretable Logistic Regression baseline

Version v1.1 introduces a full leakage hotfix, a reproducible modeling pipeline, refreshed explainability outputs, and a Gradio application aligned with the final model artifact.

---

# Project Goal

The goal of the project is to estimate the probability that the market will experience a meaningful drop during the next week.

The target variable is:

`Target_NextWeekDrop = 1` if the following week's S&P 500 return is negative enough to qualify as a drop.

## Dataset Scope

The final v1.1 dataset uses weekly observations from approximately 2004–2025.

After cleaning, feature engineering, lag alignment, and removal of rows without sufficient historical context, the final model uses approximately 950 weekly observations and 32 leakage-safe engineered features across four signal domains:

* S&P 500 market behavior
* VIX volatility behavior
* Google sentiment / Google Trends indicators
* macroeconomic context such as unemployment

The project is intentionally designed as:

* a classification problem
* time-aware and leakage-safe
* interpretable rather than black-box
* suitable for presentation, documentation, and deployment

---

# Final v1.1 Results

| Metric / Item               | Value |
| --------------------------- | ----: |
| ROC AUC                     | ~0.53 |
| Average Precision / PR AUC  | ~0.45 |
| Best F1 Score               | ~0.57 |
| Best Decision Threshold     | ~0.41 |
| Final Model                 | Logistic Regression |
| Final Feature Count         | 32 engineered features |
| Visible App Inputs          | 8 |
| Auto-filled Hidden Features | 24 |
| Validation Strategy         | Chronological holdout |
| Alternative Models Tested   | Random Forest, XGBoost |
| Explainability              | SHAP LinearExplainer + coefficients + permutation importance |

Key conclusion:

The model is only modestly predictive, but after the v1.1 leakage fix it becomes significantly more trustworthy, reproducible, and interpretable. Alternative models such as Random Forest and XGBoost were tested, but they did not show stable improvement under time-aware validation. Logistic Regression was selected because it provided the best balance of robustness, interpretability, and deployment simplicity.

---

# Repository Structure

```text
Market-Mood-Forecasting/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── feature_engineered/
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
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# End-to-End Workflow

The project is organized into seven notebooks.

## 01_load_data.ipynb

Purpose:

* load all source datasets
* inspect structure and date coverage
* standardize column names and date handling

Inputs include:

* S&P 500 prices
* VIX data
* Google sentiment / mood indicators
* macroeconomic variables such as unemployment

Output:

* unified raw dataset prepared for cleaning

---

## 02_clean_data.ipynb

Purpose:

* clean and align all datasets
* handle missing values
* standardize frequency and date index
* remove duplicates and impossible values

Typical operations:

* convert dates to weekly frequency
* forward-fill or interpolate macro features
* remove invalid rows at the start/end of the series

Output:

* cleaned dataset saved under `data/cleaned/`

---

## 03_exploratory_analysis.ipynb

Purpose:

* understand relationships in the data before modeling
* inspect trends, distributions, and correlations
* identify possible leakage risks

Exploration includes:

* target balance
* feature distributions
* pairwise correlations
* trend plots for sentiment, VIX, and market returns
* early warning that some original features could leak future information

Outputs:

* EDA figures under `images/eda/`

---

## 04_feature_engineering.ipynb

Purpose:

* build leakage-safe engineered features using only past information

v1.1 introduced a major redesign of this notebook.

Final engineered features include:

* lagged S&P 500 returns
* lagged VIX changes
* rolling means
* rolling standard deviations
* stability ratios
* sentiment-volatility interaction terms

Examples:

```text
sp500_returns_lag1
sp500_returns_lag2
vix_change_lag1
vix_change_lag2
google_sentiment_7d_mean
google_sentiment_7d_std
sp500_returns_5d_volatility
sentiment_vol_interaction
```

Important leakage guardrails:

* `Mood_Zone` is kept for EDA only
* `Mood_Zone_Cat` is never created
* no future, next-week, or target-derived feature is allowed in the model matrix
* only past observations are used for every engineered feature

Outputs:

* feature-engineered dataset
* correlation heatmap
* diagnostic feature plots

Saved dataset:

```text
data/feature_engineered/fe_dataset_v1_1.csv
```

---

## 05_modeling.ipynb

Purpose:

* train and evaluate the final leakage-safe model

The final v1.1 pipeline:

1. removes all leaky columns
2. keeps only numeric features
3. uses a chronological 80/20 train-validation split
4. trains a Logistic Regression baseline
5. evaluates ROC, PR, threshold behavior, and interpretability

Final modeling decisions:

* primary model: balanced Logistic Regression
* alternative models tested: Random Forest and XGBoost
* final selected model: Logistic Regression
* reason for selection: tree-based alternatives did not show stable improvement under time-aware validation, while Logistic Regression remained easier to interpret, debug, document, and deploy

Columns explicitly excluded from modeling:

```text
Date
Target_NextWeekDrop
Mood_Zone
Mood_Zone_Cat
raw contemporaneous SP500/VIX target-related columns
```

Model artifacts generated:

```text
models/logreg_pipeline_v1_1_1775664292.joblib
models/logreg_pipeline_v1_1_1775664292.json
```

Additional exported outputs:

* coefficient importance CSV
* permutation importance CSV
* time-series cross-validation summaries
* ROC curve
* PR curve
* F1-vs-threshold curve

Main findings:

* best threshold is approximately 0.41 rather than 0.50
* model performance drops after leakage removal, which confirms the hotfix worked
* VIX-based stability features become the strongest predictors

Top coefficient features:

1. `vix_change_roll4_lag_std`
2. `vix_change_lag1`
3. `vix_change_roll8_lag_mean`
4. `vix_change_roll12_lag_std`
5. `vix_change_roll4_stability`

---

## 06_model_explain.ipynb

Purpose:

* explain the final Logistic Regression model in detail

Explainability methods:

* Logistic Regression coefficients
* permutation importance
* SHAP LinearExplainer
* SHAP dependence plots

Most important features according to SHAP:

1. `vix_change_roll4_stability`
2. `sp500_ret_roll4_stability`
3. `Google_Sentiment_Index`
4. `vix_change_lag1`
5. `vix_change_roll4_lag_std`

Main explainability insight:

* unstable and rapidly rising VIX behavior increases the probability of a market drop
* higher Google sentiment generally reduces predicted downside risk
* the model relies more on volatility structure than on raw sentiment alone

Outputs saved under:

```text
images/model_explain/
```

Including:

* SHAP summary plot
* SHAP top-20 bar chart
* dependence plots for the top features

---

## 07_final_notebook.ipynb

Purpose:

* create a reviewer-friendly final report notebook
* consolidate results from the entire pipeline
* provide one clean notebook for presentation and grading

The final notebook includes:

* table of contents
* model provenance panel
* feature list and threshold used in production
* modeling plots from notebook 05
* explainability plots from notebook 06
* conclusions and next steps

---

# Leakage Hotfix (v1.1)

The most important change in version v1.1 is the removal of information leakage.

Before the hotfix, some variables could indirectly reveal future market movement.

The final project now enforces the following rules:

* no future-looking variables
* no target-derived categories
* no zone-based features in the model
* all rolling features use only historical values
* application-level checks reject forbidden features

The application also contains explicit guardrails that block features containing words such as:

```text
next
future
lead
t+
Target
Mood_Zone
Mood_Zone_Cat
```

This makes the final deployed model much safer and more realistic.

---

# Interactive App

The project includes a Gradio app built in `app.py`.

The app:

* loads the final Logistic Regression artifact and metadata
* exposes only a small set of interpretable visible features
* auto-fills all remaining features using training medians
* produces a probability and local explanation
* rejects unrealistic all-zero input scenarios

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

The app also:

* shows local feature contributions
* falls back to global coefficient importance when the user stays at baseline values
* includes a documentation gallery using the saved project images
* exposes diagnostics for the loaded model and metadata

To run the app locally:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

---

# Environment

Final environment used for v1.1:

```text
Python 3.10.11
pandas 2.3.1
numpy 2.2.6
scikit-learn 1.7.1
shap 0.48.0
```

`runtime.txt`:

```text
python-3.10.11
```

---

# Installation

```bash
git clone <your-repository-url>
cd Market-Mood-Forecasting

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

# Additional Documentation

This repository is accompanied by additional project documents:

* `technical_documentation.md` — detailed methodology, notebook logic, feature definitions, evaluation process, and artifact provenance
* `architecture.md` — end-to-end system and deployment architecture, including how the notebooks, saved artifacts, and app interact
* `model_card.md` — intended use, limitations, assumptions, feature list, metrics, and ethical considerations for the final v1.1 model
* `testing_instructions.md` — reproducible setup and validation checklist for rerunning notebooks, launching the app, and confirming expected outputs
* `presentation.pdf` — concise presentation version of the project suitable for interviews or portfolio review

Recommended reading order:

1. `README.md`
2. `architecture.md`
3. `technical_documentation.md`
4. `model_card.md`
5. `testing_instructions.md`
6. `presentation.pdf`

This structure keeps the README concise and portfolio-friendly, while the deeper technical details live in dedicated supporting files.

---

# Disclaimer

This project is for educational and portfolio purposes only.

It is not financial advice, investment advice, or a recommendation to buy, sell, or trade any financial instrument.

The model has modest predictive performance and is intended to demonstrate:

* leakage-safe time-series modeling
* feature engineering
* explainability
* reproducible deployment

It should not be used as the sole basis for real trading or investment decisions.

---

# Future Improvements

Potential next steps:

* add rolling / walk-forward time-series cross-validation for stronger robustness checks
* test alternative forecasting horizons
* calibrate predicted probabilities
* evaluate richer macroeconomic indicators
* compare with a small gradient boosting model under the same leakage-safe rules
* deploy the final app publicly using Hugging Face Spaces

---

# Author

Created by Artur Melnyk as a portfolio project demonstrating:

* end-to-end data science workflow
* time-series feature engineering
* leakage-safe modeling
* model explainability
* lightweight deployment with Gradio
