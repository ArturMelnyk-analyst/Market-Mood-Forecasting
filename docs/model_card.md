# Model Card — Market Mood Forecasting (Hotfix v1.1)

## 1. Model Details

**Project:** Market Mood Forecasting  
**Version:** Hotfix v1.1  
**Model type:** Binary classification  
**Final model:** Logistic Regression  
**Pipeline structure:** Imputation → Scaling → Logistic Regression  
**Primary artifact:** `models/logreg_pipeline_v1_1_1775664292.joblib`  
**Metadata artifact:** `models/logreg_pipeline_v1_1_1775664292.json`

### Prediction Task

The model estimates the probability that the market will experience a negative next-week movement significant enough to be labeled as a drop.

Target definition:

```text
Target_NextWeekDrop = 1
```

when the following week's S&P 500 return meets the project-defined drop condition.

### Why Logistic Regression Was Selected

Hotfix v1.1 intentionally uses Logistic Regression as the final deployed model because it is:

* more interpretable than tree-ensemble alternatives
* easier to debug after leakage removal
* easier to document and explain in a portfolio setting
* lighter to deploy inside a Gradio app
* more stable under the project’s time-aware validation setup

Alternative models, including Random Forest and XGBoost, were evaluated. They did not show stable improvement under time-aware validation, so Logistic Regression was retained as the final baseline.

---

## 2. Intended Use

This model is intended for:

- educational demonstration
- portfolio presentation
- documentation of a leakage-safe time-aware classification pipeline
- explainability and deployment examples
- technical discussion around market sentiment, volatility, and macro features

This model is **not** intended for:

- real-money trading
- investment advice
- automated execution systems
- risk management in production financial systems
- use as the sole basis for any financial decision

---

## 3. Users

Intended readers and users of this model include:

- recruiters
- hiring managers
- portfolio reviewers
- data science interviewers
- learners interested in interpretable market prediction systems

This model card is written for technical and semi-technical readers who want to understand what the model does, what it does not do, and how it should be interpreted.

---

## 4. Data Overview

The project combines four main data domains:

- S&P 500 market behavior
- VIX volatility behavior
- Google sentiment / mood indicators
- macroeconomic indicators such as unemployment

The final v1.1 model-ready dataset uses weekly observations from approximately 2004–2025.

After cleaning, lag alignment, rolling-window feature engineering, and removal of rows without sufficient historical context, the final dataset contains approximately 950 observations and 32 leakage-safe engineered features.

The data is processed through a notebook pipeline:

```text
01_load_data.ipynb
02_clean_data.ipynb
03_exploratory_analysis.ipynb
04_feature_engineering.ipynb
05_modeling.ipynb
06_model_explain.ipynb
07_final_notebook.ipynb
```

The final model-ready dataset is produced in:

```text
data/feature_engineered/fe_dataset_v1_1.csv
```

---

## 5. Input Features

The final model uses leakage-safe engineered features only.

Examples of feature groups include:

### Lag Features
- `sp500_returns_lag1`
- `sp500_returns_lag2`
- `vix_change_lag1`
- `vix_change_lag2`

### Rolling Features
- `google_sentiment_7d_mean`
- `google_sentiment_7d_std`
- `vix_change_roll4_lag_std`
- `vix_change_roll8_lag_mean`

### Stability / Volatility Features
- `vix_change_roll4_stability`
- `sp500_ret_roll4_stability`
- additional rolling stability features used in notebook 05

### Macro / Context Features
- `Google_Sentiment_Index`
- `Mood_Index`
- `Unemployment`

The full feature order is stored in the metadata JSON used by the app.

---

## 6. Leakage Controls

Hotfix v1.1 exists because the earlier project version required stronger leakage protection.

The final modeling system explicitly blocks or removes:

- `Target_NextWeekDrop`
- `Mood_Zone`
- `Mood_Zone_Cat`
- future-looking columns
- lead-like features
- target-derived categories
- raw contemporaneous variables that would make the setup unrealistic at inference time

The application layer also performs an additional guardrail check and blocks features containing patterns such as:

```text
next
future
lead
t+
Target
Mood_Zone
Mood_Zone_Cat
```

This means leakage prevention is enforced in both:
- the notebook pipeline
- the deployed application

---

## 7. Training and Validation

### Training Strategy
The model is trained on a chronological split rather than a random split.

```text
Oldest 80% of observations → training
Newest 20% of observations → validation
```

This validation setup is intentionally leakage-safe, but it remains a single chronological holdout split. A future improvement is rolling / walk-forward time-series cross-validation to evaluate robustness across multiple historical windows.

This preserves time order and reduces the risk of unrealistic evaluation caused by temporal leakage.

### Thresholding
The deployed classification threshold is not the default 0.50.

The selected threshold from the final v1.1 run is approximately:

```text
0.41
```

This threshold was chosen based on F1 behavior in the saved threshold analysis.

---

## 8. Performance Summary

Final v1.1 metrics are approximately:

| Metric | Value |
|---|---:|
| ROC AUC | ~0.53 |
| Average Precision (PR AUC) | ~0.45 |
| Best F1 Score | ~0.57 |
| Best Decision Threshold | ~0.41 |

The target is moderately imbalanced, with the positive drop class representing roughly 40% of observations. For this reason, Average Precision / PR AUC and F1-based threshold selection are emphasized alongside ROC AUC.

The selected threshold of approximately 0.41 is an early-warning boundary, not a trading signal.


### Interpretation

These metrics indicate **modest predictive power**.

That is important.

The main achievement of v1.1 is **not** maximizing accuracy.  
The main achievement is producing a more trustworthy, interpretable, leakage-safe system.

The performance is therefore best interpreted as:

- realistic rather than inflated
- useful as a methodological case study
- appropriate for a portfolio project focused on modeling discipline

---

## 9. Explainability Summary

The final model is explained using three complementary methods:

- Logistic Regression coefficient analysis
- permutation importance
- SHAP LinearExplainer

### Top SHAP Features
The most important features according to the v1.1 explainability run are:

1. `vix_change_roll4_stability`
2. `sp500_ret_roll4_stability`
3. `Google_Sentiment_Index`
4. `vix_change_lag1`
5. `vix_change_roll4_lag_std`

### Top Coefficient Features
Top coefficient-based drivers include:

1. `vix_change_roll4_lag_std`
2. `vix_change_lag1`
3. `vix_change_roll8_lag_mean`
4. `vix_change_roll12_lag_std`
5. `vix_change_roll4_stability`

### Main Modeling Insight
The model relies more heavily on volatility structure and stability behavior than on raw sentiment alone.

In plain language:

- unstable or rapidly worsening VIX patterns tend to increase downside risk
- stronger Google sentiment tends to reduce predicted downside risk
- the model is interpretable enough to inspect the directional role of major features

---

## 10. Application Integration

The model is deployed locally through `app.py` using Gradio.

### Visible User Inputs
The app exposes a small subset of interpretable visible inputs:

- `vix_change_roll4_stability`
- `sp500_ret_roll4_stability`
- `Google_Sentiment_Index`
- `vix_change_lag1`
- `vix_change_lag2`
- `google_sentiment_7d_mean`
- `Unemployment`
- `Mood_Index`

The app exposes 8 visible inputs. The remaining 24 required features are auto-filled using stored training medians, giving 32 total model features.

All remaining required model features are auto-filled using stored training medians from the metadata artifact.

### Special App Behavior
The app intentionally treats an all-zero visible-input scenario as invalid and returns:

```text
No prediction generated
```

This prevents obviously unrealistic user input from being treated as a meaningful forecast.

---

## 11. Limitations

This model has important limitations.

### Predictive Limitations
- performance is modest
- the model is only one baseline, not a production forecasting system
- results may not generalize across all market regimes
- the model is sensitive to feature engineering choices and historical context

### Scope Limitations
- only one forecasting horizon is modeled
- no probability calibration is applied yet
- no production portfolio management logic is attached
- no live market data integration is included

### Modeling Limitations
- Logistic Regression may underfit nonlinear relationships
- some economically meaningful interactions may not be fully captured by a linear model
- performance is intentionally conservative after leakage removal

---

## 12. Risks and Ethical Considerations

This project touches financial forecasting, so misuse risk matters.

### Main Risks
- a reader may over-interpret the probability as trading advice
- a user may mistake a portfolio model for a production-ready financial system
- modest predictive performance may be ignored if only the interface is viewed

### Ethical Position
This repository should be used to evaluate:

- data science process
- leakage handling
- explainability
- reproducible deployment

It should **not** be used as the sole basis for financial action.

---

## 13. Monitoring and Maintenance Considerations

If this model were ever extended beyond portfolio use, the following would need monitoring:

- data drift
- volatility regime changes
- feature distribution shifts
- threshold stability over time
- recalibration of probabilities
- artifact and environment compatibility

For the current repository, the most important maintenance practice is:

```text
keep notebooks, saved artifact, metadata JSON, and app.py aligned
```

---

## 14. Recommended Companion Documents

This model card should be read alongside:

- `README.md`
- `architecture.md`
- `technical_documentation.md`
- `testing_instructions.md`
- `presentation.pdf`

Suggested order:

```text
README.md
→ architecture.md
→ technical_documentation.md
→ model_card.md
→ testing_instructions.md
→ presentation.pdf
```

---

## 15. Final Assessment

The final v1.1 model should be viewed as:

- a leakage-safe forecasting baseline
- an interpretable market classification project
- a strong portfolio artifact for applied data science
- a demonstration of disciplined model rebuilding after identifying methodological risk

Its value lies more in:

```text
correct process
+ transparent reasoning
+ clear artifact flow
+ safe deployment behavior
```

than in raw predictive power alone.
