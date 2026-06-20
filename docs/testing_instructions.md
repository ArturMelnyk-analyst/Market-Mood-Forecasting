# Testing Instructions — Market Mood Forecasting (v1.2.0)

---

# 1. Purpose

This document explains how to verify that the Market Mood Forecasting project is functioning correctly after the v1.2.0 event-risk, walk-forward, and bilingual deployment upgrade.

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
* bilingual EN/DE UI works correctly
* Hugging Face deployment is synchronized
* the environment is reproducible

This document is intended to live in:

```text id="cl02na"
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
14. English/German UI toggle works
15. Hugging Face deployment reflects the newest app version
16. old v1.1.2 app text is no longer visible

---

# 3. Expected Final Files

---

## 3.1 Final Model Artifacts

Expected inside `models/`:

```text id="v4jzmd"
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

## 3.2 Expected Modeling Plots

Expected inside `images/modeling/`:

```text id="fvk0l1"
f1_vs_threshold_v1_2_0.png
logreg_coeff_importance_v1_2_0.png
permutation_importance_v1_2_0.png
pr_curve_v1_2_0.png
roc_curve_v1_2_0.png
walk_forward_auc_v1_2_0.png
```

---

## 3.3 Expected Explainability Plots

Expected inside `images/model_explain/`:

```text id="xw9jq8"
summary_v1_2_0.png
shap_top20_bar_v1_2_0.png
dependence_event_count_last_4w_v1_2_0.png
dependence_event_severity_last_4w_v1_2_0.png
dependence_major_event_last_4w_v1_2_0.png
dependence_vix_change_lag1_v1_2_0.png
dependence_vix_change_roll4_stability_v1_2_0.png
```

---

## 3.4 Expected Feature Engineering Outputs

Expected inside `images/feature_engineering/`:

```text id="17yyv5"
feature_corr_heatmap_v1_2_0.png
```

Expected inside `data/feature_engineered/`:

```text id="ldrhik"
fe_dataset_v1_2_0.csv
```

---

## 3.5 Expected EDA Outputs

Expected inside `images/eda/`:

```text id="53o8w9"
google_trends_sentiment.png
mood_vs_sp500.png
mood_vs_sp500_annotated.png
sp500_vs_vix.png
vix_over_time.png
```

---

## 3.6 Expected Demo Asset

After PR #2 deployment testing, the project may include:

```text id="iaa4tq"
images/demo/demo_walkthrough.gif
```

This GIF should be recorded after the Hugging Face Space is successfully updated.

---

# 4. Environment Setup Test

---

## 4.1 Create Virtual Environment

From the project root:

```bash id="rzr69x"
python -m venv .venv
```

Activate on Windows:

```bash id="c89m8q"
.venv\Scripts\activate
```

---

## 4.2 Install Dependencies

```bash id="kj6czf"
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4.3 Confirm Core Versions

Expected environment:

```text id="pxz9a0"
Python 3.10.11
pandas 2.3.1
numpy 2.2.6
scikit-learn 1.7.1
shap 0.48.0
```

Quick verification:

```bash id="n6fdo8"
python -c "import platform, pandas, numpy, sklearn, shap; print(platform.python_version()); print(pandas.__version__); print(numpy.__version__); print(sklearn.__version__); print(shap.__version__)"
```

---

# 5. Notebook Execution Order

Run notebooks in this exact order:

```text id="kzocuo"
01_load_data.ipynb
02_clean_data.ipynb
03_exploratory_analysis.ipynb
04_feature_engineering.ipynb
05_modeling.ipynb
06_model_explain.ipynb
07_final_notebook.ipynb
```

Every notebook should be executed using:

```text id="0ybf1e"
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

```python id="w3qu26"
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

```python id="cmw1v7"
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

```python id="pe8j58"
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

```python id="ie1bc0"
print(event_model_features)
```

Expected:

* multiple event-risk features appear

---

## 7.2 Verify Leakage Block

Run:

```python id="icny9s"
blocked_event_features
```

Expected:

```python id="ijfgja"
[]
```

---

## 7.3 Verify Walk-Forward Outputs

Expected files:

```text id="f9m8j9"
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

```text id="dmv5fe"
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

```text id="j4hjkt"
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

```text id="k5i2m4"
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

# 10. Local App Launch Test

Run from project root:

```bash id="t20j44"
python app.py
```

Expected local address:

```text id="1drill"
http://127.0.0.1:7860
```

---

# 11. App Startup Validation

Correct startup should indicate loading of:

```text id="l0xadq"
./models/logreg_pipeline_v1_2_0_*.joblib
./models/logreg_pipeline_v1_2_0_*.json
```

No model-loading errors should appear.

---

# 12. App Functional Tests

---

## Test Case 1 — All-Zero Guardrail

Steps:

* open Predict tab
* set all visible inputs to `0`
* click Predict

Expected result:

* app rejects prediction
* app displays:

```text id="g72voz"
No prediction generated
```

This confirms baseline-abuse prevention.

---

## Test Case 2 — Non-Zero Prediction

Example inputs:

```text id="p4l1av"
vix_change_roll4_stability = 1
sp500_ret_roll4_stability = 1
Google_Sentiment_Index = 1
vix_change_lag1 = 1
vix_change_lag2 = 1
google_sentiment_7d_mean = 1
Unemployment = 1
Mood_Index = 1
```

Expected result:

* probability generated
* contribution plot generated
* no artifact mismatch errors

---

## Test Case 3 — Demo Nudge

Steps:

* keep default baseline values
* click demo button

Expected result:

* app modifies one visible input internally
* prediction appears
* contribution plot appears
* status message explains demo adjustment

---

## Test Case 4 — Diagnostics Tab

Expected diagnostics:

* model version
* feature counts
* visible feature list
* artifact paths
* metadata information
* Hugging Face environment indicator

Expected approximate counts:

| Type             |              Expected |
| ---------------- | --------------------: |
| Visible Features |                     8 |
| Total Features   |                  ~40+ |
| Hidden Features  | remaining auto-filled |

---

# 13. Bilingual UI Test

The PR #2 app includes an English/German interface layer.

---

## Test Case 5 — English UI

Steps:

* select English
* inspect hero text
* inspect app guidance
* inspect Explain & Docs tab
* inspect Diagnostics tab

Expected result:

* English text displays cleanly
* technical feature names remain unchanged
* prediction behavior is unchanged

---

## Test Case 6 — German UI

Steps:

* select Deutsch
* inspect hero text
* inspect app guidance
* inspect Explain & Docs tab
* inspect Diagnostics tab
* run prediction or demo nudge

Expected result:

* German text displays cleanly
* technical feature names remain unchanged
* prediction behavior is unchanged
* no layout break occurs

Important rule:

```text id="6u3tr9"
language changes presentation text only, not model behavior
```

---

# 14. Hugging Face Deployment Test

The live deployment can be checked here:

```text id="f4eo7r"
https://huggingface.co/spaces/Artur-Melnyk/Market-Mood-Forecasting
```

Minimum checks:

* app loads successfully
* v1.2.0 artifacts load correctly
* no old v1.1.2 text remains
* English/German toggle works
* all-zero prediction is rejected
* non-zero prediction works
* demo nudge works
* diagnostics reflect current artifact pair

---

## Hugging Face README Metadata Check

Before pushing to Hugging Face, confirm the Space README front matter is valid YAML.

Expected pattern:

```yaml id="3zwc8k"
---
title: Market Mood Forecasting
emoji: 📈
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10.11"
app_file: app.py
pinned: false
license: mit
short_description: Event-aware S&P 500 risk demo
---
```

Important:

* `short_description` must be 60 characters or fewer
* metadata block must end with exactly `---`
* do not use long dashed separators inside YAML

---

# 15. Leakage Safety Test

A reviewer should verify:

* raw event names are excluded
* future-like features are excluded
* `Target_NextWeekDrop` is excluded
* event identities never enter final model matrix
* app guardrails remain active

Central principle:

```text id="p0b1us"
Only historical information may enter the model.
```

---

# 16. Failure Signs

Testing should be considered failed if:

* app cannot load artifacts
* notebook 05 metrics differ dramatically
* walk-forward outputs are missing
* event-risk outputs are missing
* SHAP outputs are missing
* forbidden leakage features appear
* all-zero guardrail fails
* bilingual toggle changes model behavior
* German UI breaks layout
* old v1.1.2 deployment text remains
* Hugging Face README metadata fails validation
* environment versions mismatch saved artifacts

---

# 17. Quick Reviewer Checklist

* [ ] requirements install successfully
* [ ] environment versions match expected runtime
* [ ] v1.2.0 artifact pair exists
* [ ] walk-forward CSV outputs exist
* [ ] event-risk features exist
* [ ] SHAP outputs exist
* [ ] local app launches successfully
* [ ] Hugging Face app launches successfully
* [ ] all-zero guardrail works
* [ ] non-zero prediction works
* [ ] demo nudge works
* [ ] Diagnostics tab loads correctly
* [ ] English UI works
* [ ] German UI works
* [ ] no forbidden leakage features appear
* [ ] no old v1.1.2 text remains

---

# 18. Recommended Validation Flow

```text id="1bz26d"
1. Environment setup
2. Confirm final files exist
3. Run notebook 04
4. Run notebook 05
5. Run notebook 06
6. Run notebook 07
7. Launch app.py locally
8. Test all-zero guardrail
9. Test non-zero prediction
10. Test demo nudge
11. Test English UI
12. Test German UI
13. Check diagnostics
14. Push/test Hugging Face deployment
15. Record demo walkthrough GIF
```

---

# 19. Demo Walkthrough GIF Check

After the Hugging Face Space is updated and tested, record:

```text id="v0vu1z"
images/demo/demo_walkthrough.gif
```

Recommended GIF flow:

1. open the live app
2. show v1.2.0 title
3. switch from English to German
4. return to prediction
5. click demo nudge
6. show probability and explanation
7. briefly show diagnostics

Recommended duration:

```text id="is2h00"
20–35 seconds
```

The GIF should demonstrate usability, not every technical detail.

---

# 20. Final Interpretation of Successful Testing

If all checks pass, the repository can be considered:

* reproducible
* leakage-safe
* internally consistent
* walk-forward validated
* event-aware
* bilingual
* deployment-ready for portfolio demonstration

A successful test does NOT imply institutional trading readiness.

It confirms that the project demonstrates:

```text id="tcah48"
correct methodology
+ reproducible artifacts
+ explainable outputs
+ contextual event engineering
+ deployment alignment
+ bilingual UX polish
```
