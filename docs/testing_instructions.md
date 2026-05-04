# Testing Instructions — Market Mood Forecasting (Hotfix v1.1)

## 1. Purpose

This document explains how to verify that the Market Mood Forecasting project is functioning correctly after the v1.1 leakage hotfix.

It is designed for:

- the project author
- technical reviewers
- recruiters or interviewers who want to validate the project
- future maintenance or redeployment work

The goal is not only to run the code, but also to confirm that:

- the final model artifact is the correct one
- notebooks 05–07 are aligned with the saved outputs
- the application loads the intended artifact pair
- leakage guardrails are active
- the environment is reproducible

This file is intended to live in:

```text
docs/testing_instructions.md
```

---

## 2. What Should Be Verified

A successful validation of the project should confirm all of the following:

1. the Python environment installs correctly
2. the notebooks run in the intended order
3. the final artifact pair exists in `models/`
4. the main saved plots exist in `images/`
5. the app launches locally
6. the app loads the final metadata and model artifact without error
7. the app blocks unrealistic all-zero input behavior
8. the app produces a valid prediction for non-zero input
9. diagnostics reflect the loaded artifact correctly
10. the final model feature count is 32
11. the app exposes 8 visible inputs and auto-fills 24 hidden features

---

## 3. Expected Final Files

Before testing, confirm that the final key files exist.

## 3.1 Final Model Artifacts

Expected in `models/`:

```text
logreg_pipeline_v1_1_1775664292.joblib
logreg_pipeline_v1_1_1775664292.json
logreg_coeff_importance_v1_1.csv
model_compare_v1_1.csv
permutation_importance_v1_1.csv
shap_importance_v1_1.csv
shap_top10_v1_1.csv
tscv_auc_folds_v1_1.csv
tscv_auc_summary_v1_1.csv
```

## 3.2 Expected Modeling Plots

Expected in `images/modeling/`:

```text
f1_vs_threshold_v1_1.png
logreg_coeff_importance_v1_1.png
permutation_importance_v1_1.png
pr_curve_v1_1.png
roc_curve_v1_1.png
```

## 3.3 Expected Explainability Plots

Expected in `images/model_explain/`:

```text
dependence_Google_Sentiment_Index_v1_1.png
dependence_sp500_ret_roll4_stability_v1_1.png
dependence_vix_change_lag1_v1_1.png
dependence_vix_change_roll4_lag_std_v1_1.png
dependence_vix_change_roll4_stability_v1_1.png
shap_top20_bar_v1_1.png
summary_v1_1.png
```

## 3.4 Expected Feature Engineering Plot

Expected in `images/feature_engineering/`:

```text
feature_corr_heatmap_v1_1.png
```

## 3.5 Expected EDA Plots

Expected in `images/eda/`:

```text
google_trends_sentiment.png
mood_vs_sp500.png
mood_vs_sp500_annotated.png
sp500_vs_vix.png
```

---

## 4. Environment Setup Test

## 4.1 Create a Clean Virtual Environment

From the project root:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

## 4.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4.3 Confirm Core Versions

The final expected environment is:

```text
Python 3.10.11
pandas 2.3.1
numpy 2.2.6
scikit-learn 1.7.1
shap 0.48.0
```

You can verify quickly with:

```bash
python -c "import platform, pandas, numpy, sklearn, shap; print(platform.python_version()); print(pandas.__version__); print(numpy.__version__); print(sklearn.__version__); print(shap.__version__)"
```

Expected output should match or be intentionally aligned with:

```text
3.10.11
2.3.1
2.2.6
1.7.1
0.48.0
```

---

## 5. Notebook Execution Test

Run notebooks in this exact order:

```text
01_load_data.ipynb
02_clean_data.ipynb
03_exploratory_analysis.ipynb
04_feature_engineering.ipynb
05_modeling.ipynb
06_model_explain.ipynb
07_final_notebook.ipynb
```

This order matters because each stage depends on outputs from earlier notebooks.

## 5.1 Minimum Required Rerun for Final Validation

If a full rerun is not necessary, the minimum practical rerun for final v1.1 validation is:

```text
05_modeling.ipynb
06_model_explain.ipynb
07_final_notebook.ipynb
```

This is appropriate only when the earlier data and feature-engineering stages are already known to be correct.

---

## 6. Notebook 05 Validation Checklist

After running `05_modeling.ipynb`, verify that:

- the final artifact pair is saved in `models/`
- the modeling images are refreshed in `images/modeling/`
- the CSV outputs are refreshed in `models/`
- the selected threshold is approximately `0.41`
- the final model matrix contains 32 features
- the final app metadata exposes 8 visible inputs and stores medians for the remaining 24 features

### Key expected outcomes

- final model: Logistic Regression
- best threshold: approximately `0.41`
- ROC AUC: approximately `0.53`
- Average Precision: approximately `0.45`
- best F1: approximately `0.57`
- alternative models such as Random Forest and XGBoost should be reflected in the comparison outputs if `model_compare_v1_1.csv` is regenerated
- Logistic Regression remains the selected final model because alternatives do not show stable improvement under time-aware validation

### Required saved files

```text
models/logreg_pipeline_v1_1_1775664292.joblib
models/logreg_pipeline_v1_1_1775664292.json
images/modeling/f1_vs_threshold_v1_1.png
images/modeling/roc_curve_v1_1.png
images/modeling/pr_curve_v1_1.png
images/modeling/logreg_coeff_importance_v1_1.png
images/modeling/permutation_importance_v1_1.png
```

---

## 7. Notebook 06 Validation Checklist

After running `06_model_explain.ipynb`, verify that:

- SHAP explanation files are generated
- dependence plots are generated
- summary and top-20 SHAP plots are generated
- outputs are saved into `images/model_explain/`

### Required saved files

```text
images/model_explain/summary_v1_1.png
images/model_explain/shap_top20_bar_v1_1.png
images/model_explain/dependence_Google_Sentiment_Index_v1_1.png
images/model_explain/dependence_sp500_ret_roll4_stability_v1_1.png
images/model_explain/dependence_vix_change_lag1_v1_1.png
images/model_explain/dependence_vix_change_roll4_lag_std_v1_1.png
images/model_explain/dependence_vix_change_roll4_stability_v1_1.png
```

### Interpretation check

Several dependence plots should look highly linear.  
That is expected because:

- the final model is Logistic Regression
- SHAP LinearExplainer is used
- the final model is intentionally interpretable

This is not a bug.

---

## 8. Notebook 07 Validation Checklist

After running `07_final_notebook.ipynb`, verify that:

- the final notebook reflects the rebuilt v1.1 artifact
- the summary matches the current modeling and explainability outputs
- the notebook presents the final reviewer-facing narrative consistently

### Visual consistency check

Confirm that notebook 07 uses the same final metrics and artifact references as:

- notebook 05
- notebook 06
- the saved model files in `models/`

---

## 9. App Launch Test

## 9.1 Start the App

From the project root:

```bash
python app.py
```

The intended local address is:

```text
http://127.0.0.1:7860
```

## 9.2 Successful Launch Output

A correct startup should indicate that the app is loading:

```text
./models/logreg_pipeline_v1_1_1775664292.joblib
./models/logreg_pipeline_v1_1_1775664292.json
```

and should complete without model-loading or environment errors.

---

## 10. App Functional Test Cases

## Test Case 1 — All-Zero Input Guardrail

### Steps
- open the Predict tab
- set all visible inputs to `0`
- click **Predict**

### Expected result
- the app should **not** produce a normal prediction
- it should return:

```text
No prediction generated
```

- the explanation area should indicate that all visible inputs are zero
- the behavior should clearly signal that this is an unrealistic or empty scenario

### Reason
This guardrail exists intentionally and confirms that the app rejects meaningless baseline abuse.

---

## Test Case 2 — Non-Zero Manual Prediction

### Steps
Enter non-zero values into the visible fields, for example:

```text
vix_change_roll4_stability = 1
sp500_ret_roll4_stability = 1
Google_Sentiment_Index = 1
vix_change_lag1 = 1
vix_change_lag2 = 1
google_sentiment_7d_mean = 1
Unemployment = 1
Mood_Index = 1
```

Then click **Predict**.

### Expected result
- a prediction should be returned
- probability text should appear
- a contribution plot should be displayed
- no leakage or artifact errors should appear

---

## Test Case 3 — Demo Nudge

### Steps
- leave the app at baseline
- click **Demo: nudge off baseline**

### Expected result
- the app should generate a non-baseline scenario automatically
- a prediction should appear
- the explanation plot should update
- the status text should mention that a demo nudge was applied

### Reason
This confirms that the model can move away from the baseline and the explanation system is functioning.

---

## Test Case 4 — Diagnostics Tab

### Steps
- open the **Diagnostics** tab
- click **Get diagnostics**

### Expected result
Diagnostics should show values such as:

- model path
- meta path
- model version
- task type
- visible feature list
- feature counts
- median source information
- visible feature count should be 8
- total model feature count should be 32
- hidden auto-filled feature count should be 24

The values should be internally consistent with the current final artifact pair.

---

## 11. Leakage Safety Test

A reviewer should confirm that the final project behavior is consistent with the leakage hotfix.

### What to check
- `Mood_Zone` is not used as a model feature
- `Mood_Zone_Cat` is not created or consumed by the final deployment
- future-like names do not appear in the final model feature list
- `Target_NextWeekDrop` is excluded from the final training matrix
- the app blocks forbidden names

### Why this matters
The central value of v1.1 is methodological correctness after removing leakage.

---

## 12. Failure Signs to Watch For

The test should be considered failed if any of the following occur:

- `app.py` cannot load the `.joblib` or `.json`
- the app launches but crashes on prediction
- the environment versions are inconsistent with the saved artifact
- notebook 05 produces a different final artifact name unexpectedly
- notebook 06 fails to create SHAP outputs
- notebook 07 reports metrics inconsistent with notebook 05
- all-zero inputs return a normal probability instead of a guardrail message
- forbidden leakage features appear in the final feature list

---

## 13. Quick Reviewer Checklist

A fast technical reviewer can validate the project with this checklist:

- [ ] `requirements.txt` installs successfully
- [ ] Python runtime matches expected v1.1 environment
- [ ] final `.joblib` and `.json` exist in `models/`
- [ ] notebook 05 modeling images exist
- [ ] notebook 06 SHAP images exist
- [ ] `python app.py` launches successfully
- [ ] app opens at `127.0.0.1:7860`
- [ ] all-zero prediction is blocked
- [ ] non-zero prediction works
- [ ] Diagnostics tab loads correctly
- [ ] no forbidden leakage features appear in deployment

---

## 14. Recommended Testing Order

For a full practical check, use this order:

```text
1. Environment setup
2. Confirm final files exist
3. Run notebook 05
4. Run notebook 06
5. Run notebook 07
6. Launch app.py
7. Test all-zero guardrail
8. Test manual non-zero prediction
9. Test Demo nudge
10. Check Diagnostics
```

This is the most efficient way to confirm that the project is fully aligned.

---

## 15. Final Interpretation of a Successful Test

If all checks above pass, then the repository can be considered:

- reproducible
- internally consistent
- leakage-safe at the deployed level
- properly aligned between notebooks, artifacts, and app
- ready for portfolio presentation and documentation review

A successful test does **not** mean the model is production-ready for finance.

It means the project is strong as a portfolio-grade applied data science system with:

```text
correct methodology
+ reproducible artifacts
+ interpretable outputs
+ safe demo deployment
```
