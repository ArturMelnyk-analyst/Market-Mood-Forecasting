# Testing Instructions — Market Mood Forecasting (v1.2.0)

---

# 1. Purpose

This document explains how to verify that the Market Mood Forecasting project is functioning correctly after the v1.2.0 event-risk and walk-forward upgrade.

It is intended for:

* the project author
* technical reviewers
* recruiters and interviewers
* future maintenance and redeployment work

The goal is not only to run the code successfully, but also to verify:

* the correct v1.2.0 artifact pair is loaded
* notebooks 05–07 are aligned with saved outputs
* walk-forward validation outputs exist
* event-risk engineering is functioning
* leakage prevention remains active
* the app loads the intended artifacts correctly
* explainability outputs match the deployed model
* the environment is reproducible

This document is intended to live in:

```text id="b8zq6n"
docs/testing_instructions.md
```

---

# 2. What Should Be Verified

A successful validation should confirm:

1. the Python environment installs correctly
2. notebooks run in the intended order
3. the final v1.2.0 artifact pair exists
4. walk-forward outputs exist
5. event-risk features exist
6. SHAP outputs exist
7. the app launches correctly
8. the app loads the correct v1.2.0 artifacts
9. the app rejects invalid all-zero scenarios
10. non-zero prediction works correctly
11. diagnostics reflect the current artifact pair
12. leakage guardrails remain active
13. event identities are NOT passed into the model

---

# 3. Expected Final Files

---

# 3.1 Final Model Artifacts

Expected inside `models/`:

```text id="ng2gvl"
logreg_pipeline_v1_2_0_*.joblib
logreg_pipeline_v1_2_0_*.json
walk_forward_summary_v1_2_0.csv
walk_forward_folds_v1_2_0.csv
tscv_auc_summary_v1_2_0.csv
tscv_auc_folds_v1_2_0.csv
event_feature_comparison_v1_2_0.csv
logreg_coeff_importance_v1_2_0.csv
permutation_importance_v1_2_0.csv
shap_importance_v1_2_0.csv
```

---

# 3.2 Expected Modeling Plots

Expected inside `images/modeling/`:

```text id="99lg03"
f1_vs_threshold_v1_2_0.png
logreg_coeff_importance_v1_2_0.png
permutation_importance_v1_2_0.png
pr_curve_v1_2_0.png
roc_curve_v1_2_0.png
walk_forward_auc_v1_2_0.png
```

---

# 3.3 Expected Explainability Plots

Expected inside `images/model_explain/`:

```text id="bq9fyl"
summary_v1_2_0.png
shap_top20_bar_v1_2_0.png
dependence_event_count_last_4w_v1_2_0.png
dependence_event_severity_last_4w_v1_2_0.png
dependence_major_event_last_4w_v1_2_0.png
dependence_vix_change_lag1_v1_2_0.png
dependence_vix_change_roll4_stability_v1_2_0.png
```

---

# 3.4 Expected Feature Engineering Outputs

Expected inside `images/feature_engineering/`:

```text id="gmv0bw"
feature_corr_heatmap_v1_2_0.png
```

Expected inside `data/feature_engineered/`:

```text id="7y4x1g"
fe_dataset_v1_2_0.csv
```

---

# 3.5 Expected EDA Outputs

Expected inside `images/eda/`:

```text id="r4cf4u"
google_trends_sentiment.png
mood_vs_sp500.png
mood_vs_sp500_annotated.png
sp500_vs_vix.png
vix_over_time.png
```

---

# 4. Environment Setup Test

---

# 4.1 Create Virtual Environment

From the project root:

```bash id="c0i66o"
python -m venv .venv
```

Activate on Windows:

```bash id="c3zcfu"
.venv\Scripts\activate
```

---

# 4.2 Install Dependencies

```bash id="hhrjtb"
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4.3 Confirm Core Versions

Expected environment:

```text id="n89e58"
Python 3.10.11
pandas 2.3.1
numpy 2.2.6
scikit-learn 1.7.1
shap 0.48.0
```

Quick verification:

```bash id="szsk0t"
python -c "import platform, pandas, numpy, sklearn, shap; print(platform.python_version()); print(pandas.__version__); print(numpy.__version__); print(sklearn.__version__); print(shap.__version__)"
```

---

# 5. Notebook Execution Order

Run notebooks in this exact order:

```text id="cw3z1r"
01_load_data.ipynb
02_clean_data.ipynb
03_exploratory_analysis.ipynb
04_feature_engineering.ipynb
05_modeling.ipynb
06_model_explain.ipynb
07_final_notebook.ipynb
```

Every notebook should be executed using:

```text id="nk9bhi"
Restart Kernel and Run All Cells
```

---

# 6. Notebook 04 Validation Checklist

After running `04_feature_engineering.ipynb`, verify:

* event-risk features are generated
* no event-name leakage columns exist
* feature-engineered dataset is refreshed
* event categories appear correctly

Run:

```python id="e3mfs5"
event_feature_cols = [
    c for c in fe_df.columns
    if "event" in c.lower()
    or "days_since_last_event" in c.lower()
]

print(event_feature_cols)
```

Expected:

* multiple event-risk features appear
* no raw event identities appear

---

## Leakage Validation

Run:

```python id="8mbvmj"
forbidden_event_columns = [
    c for c in fe_df.columns
    if "event_name" in c.lower()
    or "covid_crash" in c.lower()
    or "lehman" in c.lower()
    or "black_monday" in c.lower()
]

print(forbidden_event_columns)
```

Expected:

```python id="xij7rw"
[]
```

---

# 7. Notebook 05 Validation Checklist

After running `05_modeling.ipynb`, verify:

* final artifact pair is saved
* walk-forward outputs exist
* comparison table exists
* event features remain inside model matrix
* walk-forward validation executes correctly

---

## 7.1 Verify Event Features Exist

Run:

```python id="yhzq1l"
print(event_model_features)
```

Expected:

* multiple event-risk features appear

---

## 7.2 Verify Leakage Block

Run:

```python id="lf0lhf"
blocked_event_features
```

Expected:

```python id="25m3vc"
[]
```

---

## 7.3 Verify Walk-Forward Outputs

Expected files:

```text id="l2b35o"
walk_forward_summary_v1_2_0.csv
walk_forward_folds_v1_2_0.csv
```

Expected behavior:

* fold metrics generated
* chronological folds preserved
* no random shuffling

---

## 7.4 Verify Event Comparison Table

Expected comparison:

```text id="wl7e5k"
without_event_features
with_event_features
```

Expected interpretation:

* event-risk features slightly improve F1 behavior
* PR AUC remains relatively stable
* event features improve contextual awareness

---

## 7.5 Verify Final Artifact Pair

Expected files:

```text id="pxwq2l"
logreg_pipeline_v1_2_0_*.joblib
logreg_pipeline_v1_2_0_*.json
```

---

## 7.6 Expected Final Metrics

Approximate expected values:

| Metric    | Approximate Value |
| --------- | ----------------: |
| ROC AUC   |             ~0.53 |
| PR AUC    |             ~0.44 |
| Best F1   |             ~0.58 |
| Threshold |             ~0.25 |

Interpretation:

* modest predictive performance
* strong methodological structure
* conservative risk-alert behavior

---

# 8. Notebook 06 Validation Checklist

After running `06_model_explain.ipynb`, verify:

* SHAP outputs exist
* event-risk dependence plots exist
* summary plots are generated
* outputs are saved correctly

---

## Required Outputs

```text id="p6y2r7"
summary_v1_2_0.png
shap_top20_bar_v1_2_0.png
dependence_event_count_last_4w_v1_2_0.png
dependence_event_severity_last_4w_v1_2_0.png
dependence_major_event_last_4w_v1_2_0.png
```

---

## Interpretation Check

Several SHAP dependence plots should appear relatively linear.

This is expected because:

* the final model is Logistic Regression
* SHAP LinearExplainer is used
* the architecture intentionally prioritizes interpretability

This is not a bug.

---

# 9. Notebook 07 Validation Checklist

After running `07_final_notebook.ipynb`, verify:

* notebook reflects v1.2.0 outputs
* walk-forward metrics appear correctly
* event-aware interpretation appears
* references match notebook 05 and 06 outputs

---

## Consistency Check

Notebook 07 should remain internally consistent with:

* notebook 05 metrics
* notebook 06 explainability outputs
* current saved artifacts
* README
* model_card.md

---

# 10. App Launch Test

Run from project root:

```bash id="kysb4j"
python app.py
```

Expected local address:

```text id="8r7mrj"
http://127.0.0.1:7860
```

---

# 11. App Startup Validation

Correct startup should indicate loading of:

```text id="lyx7y9"
./models/logreg_pipeline_v1_2_0_*.joblib
./models/logreg_pipeline_v1_2_0_*.json
```

No model-loading errors should appear.

---

# 12. App Functional Tests

---

# Test Case 1 — All-Zero Guardrail

## Steps

* open Predict tab
* set all visible inputs to `0`
* click Predict

## Expected Result

The app should:

* reject prediction
* display:

```text id="7lr1a0"
No prediction generated
```

This confirms baseline-abuse prevention.

---

# Test Case 2 — Non-Zero Prediction

## Example Inputs

```text id="zfg6f0"
vix_change_roll4_stability = 1
sp500_ret_roll4_stability = 1
Google_Sentiment_Index = 1
vix_change_lag1 = 1
vix_change_lag2 = 1
google_sentiment_7d_mean = 1
Unemployment = 1
Mood_Index = 1
```

## Expected Result

* probability generated
* classification generated
* contribution plot generated
* no artifact mismatch errors

---

# Test Case 3 — Diagnostics Tab

Expected diagnostics:

* model version
* feature counts
* visible feature list
* artifact paths
* metadata information

Expected approximate counts:

| Type             |              Expected |
| ---------------- | --------------------: |
| Visible Features |                     8 |
| Total Features   |                  ~40+ |
| Hidden Features  | remaining auto-filled |

---

# 13. Leakage Safety Test

A reviewer should verify:

* raw event names are excluded
* future-like features are excluded
* `Target_NextWeekDrop` is excluded
* event identities never enter final model matrix
* app guardrails remain active

Central principle:

```text id="2xw7pr"
Only historical information may enter the model.
```

---

# 14. Failure Signs

Testing should be considered failed if:

* app cannot load artifacts
* notebook 05 metrics differ dramatically
* walk-forward outputs missing
* event-risk outputs missing
* SHAP outputs missing
* forbidden leakage features appear
* all-zero guardrail fails
* notebook references are inconsistent
* environment versions mismatch saved artifacts

---

# 15. Quick Reviewer Checklist

* [ ] requirements install successfully
* [ ] environment versions match expected runtime
* [ ] v1.2.0 artifact pair exists
* [ ] walk-forward CSV outputs exist
* [ ] event-risk features exist
* [ ] SHAP outputs exist
* [ ] app launches successfully
* [ ] all-zero guardrail works
* [ ] non-zero prediction works
* [ ] Diagnostics tab loads correctly
* [ ] no forbidden leakage features appear

---

# 16. Recommended Validation Flow

```text id="t2d2k9"
1. Environment setup
2. Confirm final files exist
3. Run notebook 04
4. Run notebook 05
5. Run notebook 06
6. Run notebook 07
7. Launch app.py
8. Test all-zero guardrail
9. Test non-zero prediction
10. Check diagnostics
```

---

# 17. Final Interpretation of Successful Testing

If all checks pass, the repository can be considered:

* reproducible
* leakage-safe
* internally consistent
* walk-forward validated
* event-aware
* deployment-ready for portfolio demonstration

A successful test does NOT imply institutional trading readiness.

It confirms that the project demonstrates:

```text id="xkcx48"
correct methodology
+ reproducible artifacts
+ explainable outputs
+ contextual event engineering
+ deployment alignment
```

---

# 18. Live Demo Smoke Test

The Hugging Face Spaces deployment can be checked here:

```text id="yzf5ys"
https://huggingface.co/spaces/Artur-Melnyk/Market-Mood-Forecasting
```

Minimum checks:

* app loads successfully
* v1.2.0 artifacts load correctly
* all-zero prediction rejected
* non-zero prediction works
* diagnostics reflect current artifact pair
