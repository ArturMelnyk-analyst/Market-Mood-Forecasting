# Model Card — Market Mood Forecasting (v1.2.0)

---

# 1. Model Details

**Project:** Market Mood Forecasting
**Version:** v1.2.0
**Model Type:** Binary Classification
**Final Model:** Logistic Regression
**Pipeline Structure:** Imputation → Scaling → Logistic Regression
**Validation Strategy:** Walk-Forward Validation
**Primary Artifact:** `models/logreg_pipeline_v1_2_0_*.joblib`
**Metadata Artifact:** `models/logreg_pipeline_v1_2_0_*.json`

---

## Prediction Task

The model estimates the probability that the S&P 500 will experience a downside movement during the following week.

Target definition:

```text id="tf6quj"
Target_NextWeekDrop = 1
```

when the following week's return is negative enough to qualify as a meaningful market decline.

---

## Why Logistic Regression Was Selected

Version v1.2.0 intentionally retains Logistic Regression as the final deployed model because it provides:

* strong interpretability
* transparent coefficient behavior
* lightweight deployment
* stable walk-forward performance
* easier debugging under leakage-safe constraints
* clearer explainability for financial context

Alternative models including:

* Random Forest
* XGBoost

were evaluated, but they did not demonstrate sufficiently stable improvement under time-aware validation conditions.

Logistic Regression remained the strongest balance of:

```text id="nvyl7l"
interpretability
+ stability
+ reproducibility
+ deployment simplicity
```

---

# 2. Intended Use

This project is intended for:

* educational demonstration
* portfolio presentation
* applied data science interviews
* explainability demonstrations
* leakage-safe financial ML examples
* macro-event feature engineering examples
* lightweight deployment demonstrations

This project is NOT intended for:

* live trading
* investment advice
* automated financial execution
* institutional forecasting systems
* production portfolio management

---

# 3. Intended Users

This project is designed primarily for:

* recruiters
* hiring managers
* portfolio reviewers
* technical interviewers
* applied data science learners

The documentation is written for technical and semi-technical audiences interested in:

* time-series ML
* leakage prevention
* event-aware feature engineering
* interpretable modeling
* walk-forward validation
* lightweight ML deployment

---

# 4. Dataset Overview

The project combines four primary data domains:

* S&P 500 market behavior
* VIX volatility behavior
* Google sentiment / Google Trends indicators
* macroeconomic indicators such as unemployment

Version v1.2.0 additionally introduces:

* historical event-risk context
* geopolitical event aggregation
* tariff-event engineering
* macro-event recency features
* event severity windows

---

## Dataset Time Range

The refreshed dataset spans approximately:

```text id="x4hvva"
2004 — May 2026
```

The final processed dataset contains approximately:

* ~950 weekly observations
* ~40+ leakage-safe engineered features

---

## Notebook Pipeline

The workflow is organized into seven notebooks:

```text id="6zbxpq"
01_load_data.ipynb
02_clean_data.ipynb
03_exploratory_analysis.ipynb
04_feature_engineering.ipynb
05_modeling.ipynb
06_model_explain.ipynb
07_final_notebook.ipynb
```

---

# 5. Feature Engineering

The final model uses only leakage-safe engineered features.

Feature groups include:

---

## Lag Features

Examples:

```text id="v8s9p8"
sp500_returns_lag1
sp500_returns_lag2
vix_change_lag1
vix_change_lag2
```

---

## Rolling Features

Examples:

```text id="f1ttkh"
vix_change_roll4_lag_std
vix_change_roll8_lag_mean
google_sentiment_7d_mean
google_sentiment_7d_std
```

---

## Stability Features

Examples:

```text id="q4ym4g"
vix_change_roll4_stability
sp500_ret_roll4_stability
```

---

## Macro / Sentiment Features

Examples:

```text id="8ytd7j"
Google_Sentiment_Index
Mood_Index
Unemployment
```

---

# 6. Event-Risk Features (v1.2.0)

Version v1.2.0 introduces leakage-safe event-risk engineering.

The model does NOT receive raw event identities such as:

* COVID Crash
* Lehman Brothers Bankruptcy
* Trump Black Monday

Instead, historical events are transformed into aggregate contextual features.

---

## Event-Risk Feature Types

Examples include:

```text id="eqx2zm"
event_count_last_4w
event_count_last_8w
event_severity_last_4w
major_event_last_4w
days_since_last_event
tariff_trade_event_last_4w
geopolitical_event_last_8w
banking_event_last_4w
```

---

## Leakage Safety Rules

Event features are constructed using strict time-aware rules:

* only past events are visible to each row
* event_date must be <= current row date
* raw event names are excluded
* no future events are visible
* no target-derived event features are allowed

This preserves contextual signal while preventing future leakage.

---

# 7. Leakage Controls

Leakage prevention is a central design principle of the project.

The pipeline explicitly blocks or removes:

* `Target_NextWeekDrop`
* `Mood_Zone`
* `Mood_Zone_Cat`
* future-looking variables
* lead features
* target-derived categories
* unrealistic contemporaneous variables

The deployed app additionally blocks suspicious feature patterns such as:

```text id="6xkrg8"
next
future
lead
target
t+
Mood_Zone
Mood_Zone_Cat
```

Leakage prevention is enforced at:

* notebook level
* feature-engineering level
* application level
* deployment artifact level

---

# 8. Training and Validation

---

## Walk-Forward Validation (v1.2.0)

Version v1.2.0 replaces the earlier single holdout setup with:

```text id="dn83te"
expanding-window walk-forward validation
```

The process:

1. trains on earlier historical windows
2. validates on future unseen windows
3. expands the training window chronologically
4. repeats across folds

This better reflects realistic financial forecasting conditions.

---

## Validation Artifacts

Saved outputs include:

```text id="fvw6r6"
walk_forward_summary_v1_2_0.csv
walk_forward_folds_v1_2_0.csv
tscv_auc_summary_v1_2_0.csv
tscv_auc_folds_v1_2_0.csv
```

---

## Threshold Selection

The deployed threshold is intentionally NOT the default 0.50.

The selected threshold is approximately:

```text id="zjlwmx"
0.25
```

This threshold prioritizes:

* downside-risk sensitivity
* early-warning behavior
* recall-oriented forecasting

rather than conservative precision.

---

# 9. Performance Summary

Approximate final v1.2.0 metrics:

| Metric         | Value |
| -------------- | ----: |
| ROC AUC        | ~0.53 |
| PR AUC         | ~0.44 |
| Best F1 Score  | ~0.58 |
| Best Threshold | ~0.25 |

---

## Interpretation

Performance should be interpreted carefully.

The primary achievement of v1.2.0 is NOT maximizing raw predictive accuracy.

The primary achievement is demonstrating:

* leakage-safe financial ML
* contextual event engineering
* walk-forward validation
* interpretable modeling
* reproducible experimentation
* deployment synchronization

The project behaves more like:

```text id="r5o1if"
contextual downside-risk alert system
```

than:

```text id="z8mk8u"
precision-focused trading engine
```

---

# 10. Baseline vs Event-Aware Comparison

Version v1.2.0 explicitly compares:

* baseline model without event-risk features
* refreshed event-aware model

Findings:

* event-risk features slightly improve F1-oriented behavior
* PR AUC remains relatively stable
* event features improve contextual awareness and interpretability
* volatility structure remains the strongest signal family

This indicates event features contribute more to:

```text id="e6z2yf"
contextual robustness
+ interpretability
```

than raw ranking performance.

---

# 11. Explainability Summary

The final model is explained using:

* Logistic Regression coefficients
* permutation importance
* SHAP explainability
* dependence plots

---

## Main Explainability Findings

The model relies primarily on:

* volatility instability
* rolling VIX structure
* market stability behavior
* event density context
* sentiment conditions

Key observations:

* worsening VIX behavior increases downside probability
* stable sentiment reduces downside probability
* dense recent macro-event activity slightly increases downside risk

---

## Explainability Outputs

Generated artifacts include:

```text id="99wb4k"
summary_v1_2_0.png
shap_top20_bar_v1_2_0.png
dependence_event_count_last_4w_v1_2_0.png
dependence_event_severity_last_4w_v1_2_0.png
```

---

# 12. Application Integration

The model is deployed using:

```text id="uyduw5"
Gradio + Hugging Face Spaces
```

---

## Deployment Characteristics

The deployment is intentionally designed as:

* lightweight
* interpretable
* portfolio-focused
* reproducible
* responsive

rather than enterprise-scale.

Large visualization artifacts remain inside the GitHub repository to keep the deployed Space lightweight and easier to maintain.

---

## Bilingual Interface (v1.2.0)

The deployed application includes:

* English interface support
* German interface support
* bilingual explanatory text
* bilingual guidance messaging

Technical feature names remain stable internally for modeling consistency.

---

## Visible User Inputs

The app exposes a small set of interpretable features, including:

* VIX behavior
* market stability
* Google sentiment
* unemployment
* Mood Index

Hidden engineered features are auto-filled internally using stored training medians.

---

## Diagnostics Layer

The deployment includes a diagnostics panel that exposes:

* model metadata
* threshold information
* artifact versioning
* pipeline details
* validation strategy summary

This helps improve transparency for reviewers and interviewers.

---

## App Safety Behavior

The application intentionally rejects unrealistic all-zero input scenarios and returns:

```text id="2i3e7g"
No prediction generated
```

This prevents meaningless forecasts.

---

# 13. Limitations

This project has important limitations.

---

## Predictive Limitations

* predictive performance remains modest
* financial markets are noisy and regime-dependent
* results may not generalize across all future environments
* event engineering remains experimental

---

## Modeling Limitations

* Logistic Regression may underfit nonlinear interactions
* event effects are simplified into aggregate windows
* macroeconomic relationships may evolve over time
* probability calibration is not implemented

---

## Scope Limitations

* only one forecasting horizon is modeled
* no live market integration
* no production portfolio management system
* no transaction cost simulation

---

# 14. Risks and Ethical Considerations

Main misuse risks include:

* treating probabilities as trading advice
* overestimating modest predictive power
* confusing portfolio research with production finance systems

This repository should instead be interpreted as:

* financial ML methodology demonstration
* leakage-safe modeling example
* explainability portfolio project
* applied data science workflow showcase

---

# 15. Maintenance Considerations

To keep the system aligned, the following artifacts must remain synchronized:

* notebooks
* saved model artifacts
* metadata JSON
* app.py
* deployment configuration
* documentation
* Hugging Face deployment state

The most important maintenance rule is:

```text id="5fh3mt"
keep notebooks, artifacts, app, deployment, and documentation aligned
```

---

# 16. Recommended Companion Documents

This model card should be read together with:

* `README.md`
* `architecture.md`
* `technical_documentation.md`
* `testing_instructions.md`
* `presentation.pdf`

Recommended order:

```text id="29m7ph"
README.md
→ architecture.md
→ technical_documentation.md
→ model_card.md
→ testing_instructions.md
→ presentation.pdf
```

---

# 17. Final Assessment

The final v1.2.0 pipeline should be interpreted as:

* leakage-safe market forecasting workflow
* event-aware macro-risk system
* interpretable financial ML experiment
* walk-forward validated forecasting baseline
* lightweight deployment-ready portfolio system

Its strongest qualities are:

```text id="3b4kcg"
correct methodology
+ temporal discipline
+ contextual event engineering
+ explainability
+ reproducibility
+ deployment readiness
+ bilingual deployment polish
```

rather than raw predictive power alone.
