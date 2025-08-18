#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py
import os, io, socket, warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import shap
import gradio as gr

warnings.filterwarnings("ignore", category=UserWarning)


# In[2]:


# =========================
# CONFIG
# =========================
MODEL_PATH = "./models/XGBoost_model.pkl"
TRAIN_DF_PATH = "./data/cleaned/cleaned_data.csv"
THRESHOLD = float(os.getenv("THRESHOLD", "0.24"))
ZERO_EPS = float(os.getenv("ZERO_EPS", "1e-9"))  # for all-zero safeguard

VISIBLE_FEATURES: List[str] = [
    "VIX_Change_Lag1",
    "VIX_Change",
    "SP500_Returns",
    "Mood_Index",
    "Mood_Index_Lag1",
    "SP500_Returns_Lag1",
]

HIDDEN_FEATURES: List[str] = [
    "Close", "Volume", "VIX_Close", "Unemployment", "Google_Sentiment_Index",
    "VIX_Norm", "Google_Norm", "Unemp_Norm", "Google_Trend_Lag1",
    "Unemployment_Lag1", "Mood_Zone_Cat",
]

mood_mapping = {"Calm": 0, "Cautious": 1, "Panic": 2}


# In[3]:


# =========================
# LOADS
# =========================
model = joblib.load(MODEL_PATH)
df = pd.read_csv(TRAIN_DF_PATH)

def get_feature_order(m, fallback_cols: List[str]) -> List[str]:
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    try:
        booster = m.get_booster()
        if booster and booster.feature_names:
            return list(booster.feature_names)
    except Exception:
        pass
    return list(fallback_cols)

FEATURE_ORDER = list(dict.fromkeys(
    [c for c in get_feature_order(model, list(df.columns)) if isinstance(c, str)]
))

def _median_for(col: str, base: str | None = None) -> float:
    if col in df.columns:
        return float(df[col].median(skipna=True))
    if base and base in df.columns:
        return float(df[base].median(skipna=True))
    return 0.0

VISIBLE_FEATURE_MEDIANS: Dict[str, float] = {
    "VIX_Change_Lag1": _median_for("VIX_Change_Lag1", base="VIX_Change"),
    "VIX_Change": _median_for("VIX_Change"),
    "SP500_Returns": _median_for("SP500_Returns"),
    "Mood_Index": _median_for("Mood_Index"),
    "Mood_Index_Lag1": _median_for("Mood_Index_Lag1", base="Mood_Index"),
    "SP500_Returns_Lag1": _median_for("SP500_Returns_Lag1", base="SP500_Returns"),
}

hidden_feature_defaults: Dict[str, float | int] = {
    "Close": float(df["Close"].median(skipna=True)),
    "Volume": float(df["Volume"].median(skipna=True)),
    "VIX_Close": float(df["VIX_Close"].median(skipna=True)),
    "Unemployment": float(df["Unemployment"].median(skipna=True)),
    "Google_Sentiment_Index": float(df["Google_Sentiment_Index"].median(skipna=True)),
    "VIX_Norm": float(df["VIX_Norm"].median(skipna=True)),
    "Google_Norm": float(df["Google_Norm"].median(skipna=True)),
    "Unemp_Norm": float(df["Unemp_Norm"].median(skipna=True)),
    "Google_Trend_Lag1": float(df["Google_Trend_Lag1"].median(skipna=True)) if "Google_Trend_Lag1" in df.columns
                          else float(df["Google_Norm"].median(skipna=True)),
    "Unemployment_Lag1": float(df["Unemployment_Lag1"].median(skipna=True)) if "Unemployment_Lag1" in df.columns
                          else float(df["Unemployment"].median(skipna=True)),
    "Mood_Zone_Cat": int(mood_mapping.get(df["Mood_Zone"].mode(dropna=True)[0], 1)) if "Mood_Zone" in df.columns else 1,
}


# In[4]:


# =========================
# BUILD ONE-ROW INPUT
# =========================
def _coerce_num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def _is_blank_visible(payload: Dict[str, float]) -> bool:
    return all(abs(float(payload.get(f, 0.0))) <= ZERO_EPS for f in VISIBLE_FEATURES)

def sanitize_payload(payload: Dict[str, float]) -> tuple[Dict[str, float], str]:
    if _is_blank_visible(payload):
        return ({f: VISIBLE_FEATURE_MEDIANS[f] for f in VISIBLE_FEATURES},
                "ℹ️ Inputs detected as blank (all zeros). Using typical market medians instead.")
    return payload, ""

def build_row_from_inputs(visible_payload: Dict[str, float]) -> pd.DataFrame:
    row: Dict[str, float | int] = {}

    # visible (fill NaNs with medians)
    for f in VISIBLE_FEATURES:
        v = _coerce_num(visible_payload.get(f))
        if np.isnan(v):
            v = VISIBLE_FEATURE_MEDIANS.get(f, 0.0)
        row[f] = v

    # if all six are ~0 → replace with medians
    if _is_blank_visible(row):
        for f in VISIBLE_FEATURES:
            row[f] = VISIBLE_FEATURE_MEDIANS[f]
            
# hidden defaults
    for f in HIDDEN_FEATURES:
        if f not in row:
            row[f] = hidden_feature_defaults[f]

    # ensure every model feature exists
    for f in FEATURE_ORDER:
        if f not in row:
            row[f] = float(df[f].median(skipna=True)) if f in df.columns else 0.0

    # create dataframe AFTER all fills
    row_df = pd.DataFrame([row])

    # dtype guard
    if "Mood_Zone_Cat" in row_df.columns:
        row_df["Mood_Zone_Cat"] = row_df["Mood_Zone_Cat"].astype(int)

    # final reorder & check
    missing = [c for c in FEATURE_ORDER if c not in row_df.columns]
    if missing:
        raise ValueError(f"Missing features for model: {missing}")

    return row_df[FEATUREER_ORDER] if (FEATUREER_ORDER := FEATURE_ORDER) else row_df  # keep name stable


# In[5]:


# =========================
# SHAP HELPERS
# =========================
explainer = shap.TreeExplainer(model)

def _normalize_base_values(bv):
    try:
        arr = np.asarray(bv)
        if arr.shape == ():
            return float(arr)
        return arr
    except Exception:
        return bv

def _slice_visible_explanation(exp_row: shap.Explanation, visible_names: List[str]) -> shap.Explanation:
    full_names = [str(n) for n in exp_row.feature_names]
    idx = [i for i, n in enumerate(full_names) if n in visible_names]
    missing = [n for n in visible_names if n not in full_names]
    if missing:
        raise ValueError(
            f"Visible features missing in SHAP Explanation: {missing}. "
            f"Explanation has (sample): {full_names[:10]}{'...' if len(full_names)>10 else ''}"
        )
    vals = np.asarray(exp_row.values, dtype=float)[idx]
    data = (np.asarray(exp_row.data)[idx] if exp_row.data is not None else None)
    sel_names = [full_names[i] for i in idx]
    order = np.argsort(np.abs(vals))[::-1]
    vals = vals[order]
    data = data[order] if data is not None else None
    sel_names = [sel_names[i] for i in order]
    base = _normalize_base_values(exp_row.base_values)
    return shap.Explanation(values=vals, base_values=base, data=data, feature_names=sel_names)

def shap_waterfall_visible_only(df_row: pd.DataFrame) -> Image.Image:
    exp_row = explainer(df_row)[0]
    exp_vis = _slice_visible_explanation(exp_row, VISIBLE_FEATURES)
    fig = plt.figure(figsize=(7.8, 5.6), dpi=150)
    shap.plots.waterfall(exp_vis, max_display=min(6, len(exp_vis.values)), show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")

def shap_bar_visible(df_row: pd.DataFrame) -> Image.Image:
    exp_row = explainer(df_row)[0]
    exp_vis = _slice_visible_explanation(exp_row, VISIBLE_FEATURES)
    fig = plt.figure(figsize=(7.8, 4.6), dpi=150)
    ax = plt.gca()
    ax.barh(exp_vis.feature_names, exp_vis.values)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Contribution"); ax.set_title("Top SHAP Contributions (Visible Inputs)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")

def shap_text_insight(df_row: pd.DataFrame) -> str:
    exp_row = explainer(df_row)[0]
    exp_vis = _slice_visible_explanation(exp_row, VISIBLE_FEATURES)
    pairs = list(zip(exp_vis.feature_names, exp_vis.values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:2] if len(pairs) >= 2 else pairs
    if not top:
        return "### SHAP Insights\nNo visible-feature contributions available."
    lines = []
    for feat, impact in top:
        direction = "increases risk" if impact > 0 else "decreases risk"
        lines.append(f"- **{feat}** {direction} (impact {impact:+.3f})")
    return "### 💡 SHAP Insights\n" + "\n".join(lines)


# In[6]:


# =========================
# INFERENCE
# =========================
def predict_payload(payload: Dict[str, float]) -> Tuple[float, int]:
    row = build_row_from_inputs(payload)
    proba = float(model.predict_proba(row)[:, 1][0])
    return proba, int(proba >= THRESHOLD)


# In[7]:


# =========================
# UI
# =========================
def build_visible_inputs():
    MARKET = ["VIX_Change_Lag1", "VIX_Change", "SP500_Returns"]
    MOOD   = ["Mood_Index", "Mood_Index_Lag1", "SP500_Returns_Lag1"]
    comps = []
    with gr.Accordion("📈 Market Indicators", open=True):
        for f in MARKET:
            comps.append(gr.Number(label=f, value=VISIBLE_FEATURE_MEDIANS[f], precision=None))
    with gr.Accordion("🧠 Mood Indicators", open=True):
        for f in MOOD:
            comps.append(gr.Number(label=f, value=VISIBLE_FEATURE_MEDIANS[f], precision=None))
    return comps

with gr.Blocks(title="Market Mood — Gradio App",
               css=".gradio-container {max-width: 1000px !important;}") as app:

    gr.Markdown("""
# 📈 Market Mood Forecasting

This app predicts the probability of a market drop next week using a model trained on **17 engineered financial & sentiment features**.

👉 **Why only 6 inputs?**  
The app exposes only the **6 most impactful features** based on SHAP explainability:
- **VIX_Change_Lag1**  
- **VIX_Change**  
- **SP500_Returns**  
- **Mood_Index**  
- **Mood_Index_Lag1**  
- **SP500_Returns_Lag1**  

The remaining 11 features are automatically filled with **typical market values** (medians) from recent data.  
This keeps the app **simple and fast for demo purposes** while preserving full model accuracy.
""")

    inputs = build_visible_inputs()

    with gr.Tab("Predict"):
        btn = gr.Button("Predict")
        prob_md = gr.Markdown("")
        bar_img = gr.Image(label="Top SHAP (Visible Inputs)", type="pil")

        def on_predict(*vals):
            try:
                raw = {f: vals[i] for i, f in enumerate(VISIBLE_FEATURES)}
                payload, note = sanitize_payload(raw)
                proba, _ = predict_payload(payload)
                row = build_row_from_inputs(payload)
                msg = f"**Probability of market drop next week: {proba*100:.2f}%**"
                if note:
                    msg = f"{note}\n\n" + msg
                return msg, shap_bar_visible(row)
            except Exception as e:
                return f"⚠️ Prediction failed: {str(e)}", None

        btn.click(on_predict, inputs=inputs, outputs=[prob_md, bar_img])

    with gr.Tab("Explain"):
        exp_btn = gr.Button("Explain")
        wf_img = gr.Image(label="SHAP Waterfall (Visible 6)", type="pil")
        bar_img2 = gr.Image(label="Top SHAP (Visible Inputs)", type="pil")
        exp_text = gr.Markdown("")

        def on_explain(*vals):
            try:
                raw = {f: vals[i] for i, f in enumerate(VISIBLE_FEATURES)}
                payload, note = sanitize_payload(raw)
                row = build_row_from_inputs(payload)
                wf = shap_waterfall_visible_only(row)
                bar = shap_bar_visible(row)
                txt = shap_text_insight(row)
                if note:
                    txt = f"{note}\n\n" + txt
                return wf, bar, txt
            except Exception as e:
                return None, None, f"⚠️ Explanation failed: {str(e)}"

        exp_btn.click(on_explain, inputs=inputs, outputs=[wf_img, bar_img2, exp_text])

    with gr.Accordion("Diagnostics", open=False):
        def diag_text():
            try:
                row = build_row_from_inputs({})  # medians path
                exp = explainer(row)[0]
                names = [str(n) for n in exp.feature_names]
                missing = [f for f in VISIBLE_FEATURES if f not in names]
                return (
                    f"- Model features (n={len(FEATURE_ORDER)}): {FEATURE_ORDER[:8]}{' ...' if len(FEATURE_ORDER)>8 else ''}\n"
                    f"- Visible (6): {VISIBLE_FEATURES}\n"
                    f"- Missing visible in SHAP Explanation: {missing}\n"
                    f"- Threshold: {THRESHOLD:.2f}\n"
                    f"- ℹ️ Safeguard: if all visible inputs are zero, app auto-replaces them with medians"
                )
            except Exception as e:
                return f"Diagnostics error: {e}"

        gr.Markdown(value=diag_text())

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":
    def first_free_port(start=7860, end=7870):
        for p in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        raise OSError("No free port available")
    app.launch(server_name="0.0.0.0", server_port=first_free_port())

