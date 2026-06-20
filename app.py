# app.py — Market Mood Forecasting (v1.2.0)
# PR #2 UX/UI polish + bilingual EN/DE interface
# Stable Gradio version aligned to final v1.2.0 Logistic Regression artifact
# All-zero visible input is treated as invalid/empty scenario.

import os
import re
import json
import glob
import socket
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import gradio as gr

try:
    from sklearn.pipeline import Pipeline
except Exception:
    Pipeline = None


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Market Mood — Next-Week Outlook (v1.2.0)"
MODEL_PATH = os.getenv("MMF_MODEL_PATH", "./models/logreg_pipeline_v1_2_0_1781959836.joblib")
META_PATH = os.getenv("MMF_META_PATH", "./models/logreg_pipeline_v1_2_0_1781959836.json")

LEAKY_PATTERNS = [r"next", r"lead", r"future", r"t\+"]
BLOCKLIST_HARD = {
    "Target_NextWeekDrop", "Target", "Label", "y", "y_true",
    "Date", "Close", "Volume", "VIX_Close",
    "SP500_Returns", "VIX_Change",
    "Mood_Zone", "Mood_Zone_Cat",
}

DEFAULT_VISIBLE = [
    "vix_change_roll4_stability",
    "sp500_ret_roll4_stability",
    "Google_Sentiment_Index",
    "vix_change_lag1",
    "vix_change_lag2",
    "google_sentiment_7d_mean",
    "Unemployment",
    "Mood_Index",
]

ZERO_EPS = 1e-12
warnings.filterwarnings("ignore", category=UserWarning)

IS_HF_SPACE = os.getenv("SPACE_ID") is not None


# =========================================================
# TRANSLATION / UX COPY
# =========================================================
COPY = {
    "English": {
        "hero": """
# Market Mood — Next-Week Outlook

**Leakage-safe, event-aware market-risk demo.**  
This app estimates next-week S&P 500 downside risk using the final v1.2.0 Logistic Regression pipeline.

The visible inputs are intentionally small and interpretable. Hidden engineered features are auto-filled from training medians.
""",
        "model_badges": "**Model:** `{model_version}`  \n**Task:** `{task}`  \n**Visible inputs:** `{n_visible}`  \n**Hidden auto-filled features:** `{n_hidden}`",
        "method_note": """
### How to use this app

1. Keep the prefilled baseline values for a typical training-median scenario.
2. Change one or more inputs to test a custom scenario.
3. Use **Demo: nudge off baseline** to confirm that the model responds to meaningful changes.

**Important:** This is a portfolio forecasting demo, not financial advice.
""",
        "predict_tab": "Predict",
        "docs_tab": "Explain & Docs",
        "diag_tab": "Diagnostics",
        "inputs_title": "### Inputs",
        "input_note": """
Feature names are kept in their original technical form for reproducibility.

- all zeros = invalid / empty scenario
- non-zero values = custom scenario
- event-risk features are hidden and filled from training medians
""",
        "predict_btn": "Predict",
        "demo_btn": "Demo: nudge off baseline",
        "pred_label": "Prediction",
        "range_label": "Approx. 95% range",
        "plot_label": "Explanation",
        "zero_pred": "No prediction generated",
        "zero_plot": "All visible inputs are zero.\nNo explanation generated.",
        "zero_status": "All visible inputs are zero. This is treated as an empty or unrealistic scenario. Please enter at least one meaningful non-zero value or use Demo.",
        "baseline_status": "Inputs match the training-median baseline. Showing global |coefficients|.",
        "local_status": "Inputs differ from baseline. Showing local feature contributions.",
        "demo_prefix": "Demo nudge applied.",
        "docs_intro": """
## Explainability and project context

Local contributions compare your inputs to a training-median baseline.  
At baseline, the app shows global coefficient strength.

The v1.2.0 pipeline adds leakage-safe event-risk features, walk-forward validation, and event-aware explainability.  
Raw event names are not used as model features.
""",
        "docs_empty": "_Project visuals are available in the full GitHub repository. This Hugging Face Space is kept lightweight for live deployment._",
        "guardrail": "— **Leakage guardrails active**: future / lead / target / zone features are blocked.",
        "feature_guide": """
## Visible Feature Guide

| Feature | Plain-English meaning |
|---|---|
| `vix_change_roll4_stability` | recent stability of VIX changes |
| `sp500_ret_roll4_stability` | recent stability of S&P 500 returns |
| `Google_Sentiment_Index` | Google sentiment signal |
| `vix_change_lag1` | previous-period VIX change |
| `vix_change_lag2` | two-period lagged VIX change |
| `google_sentiment_7d_mean` | short-window sentiment average |
| `Unemployment` | macroeconomic labor-market context |
| `Mood_Index` | combined market mood index |

Hidden event-risk features remain inside the model but are not manually edited in the app.
""",
        "links": """
## Project Links

- [Full GitHub Repository](https://github.com/ArturMelnyk-analyst/Market-Mood-Forecasting)
- [Live Hugging Face Space](https://huggingface.co/spaces/Artur-Melnyk/Market-Mood-Forecasting)
""",
        "diagnostics_note": "Use diagnostics to verify the loaded artifact, feature counts, visible inputs, and metadata.",
        "diag_btn": "Get diagnostics",
        "footer": "Educational portfolio project only. Not investment advice."
    },
    "Deutsch": {
        "hero": """
# Market Mood — Ausblick für die nächste Woche

**Leakage-sichere, ereignisbewusste Marktrisiko-Demo.**  
Diese App schätzt das Risiko eines möglichen S&P-500-Rückgangs in der nächsten Woche mit der finalen v1.2.0 Logistic-Regression-Pipeline.

Die sichtbaren Eingaben bleiben bewusst klein und interpretierbar. Versteckte technische Merkmale werden automatisch mit Trainingsmedianen gefüllt.
""",
        "model_badges": "**Modell:** `{model_version}`  \n**Aufgabe:** `{task}`  \n**Sichtbare Eingaben:** `{n_visible}`  \n**Automatisch gefüllte versteckte Merkmale:** `{n_hidden}`",
        "method_note": """
### So benutzt man die App

1. Die vorgefüllten Werte stehen für ein typisches Trainingsmedian-Szenario.
2. Ändere eine oder mehrere Eingaben, um ein eigenes Szenario zu testen.
3. Nutze **Demo: vom Basiswert abweichen**, um zu sehen, dass das Modell auf sinnvolle Änderungen reagiert.

**Wichtig:** Dies ist eine Portfolio-Demo, keine Finanzberatung.
""",
        "predict_tab": "Prognose",
        "docs_tab": "Erklärung & Doku",
        "diag_tab": "Diagnostik",
        "inputs_title": "### Eingaben",
        "input_note": """
Technische Feature-Namen bleiben unverändert, damit das Modell reproduzierbar bleibt.

- alle Werte null = ungültiges / leeres Szenario
- nicht-null Werte = eigenes Szenario
- Event-Risk-Features sind versteckt und werden mit Trainingsmedianen gefüllt
""",
        "predict_btn": "Prognose erstellen",
        "demo_btn": "Demo: vom Basiswert abweichen",
        "pred_label": "Prognose",
        "range_label": "Ca. 95%-Bereich",
        "plot_label": "Erklärung",
        "zero_pred": "Keine Prognose erstellt",
        "zero_plot": "Alle sichtbaren Eingaben sind null.\nKeine Erklärung erstellt.",
        "zero_status": "Alle sichtbaren Eingaben sind null. Das wird als leeres oder unrealistisches Szenario behandelt. Bitte mindestens einen sinnvollen Nicht-Null-Wert eingeben oder Demo benutzen.",
        "baseline_status": "Die Eingaben entsprechen dem Trainingsmedian-Basiswert. Es werden globale |Koeffizienten| angezeigt.",
        "local_status": "Die Eingaben unterscheiden sich vom Basiswert. Es werden lokale Feature-Beiträge angezeigt.",
        "demo_prefix": "Demo-Anpassung angewendet.",
        "docs_intro": """
## Erklärbarkeit und Projektkontext

Lokale Beiträge vergleichen deine Eingaben mit einem Trainingsmedian-Basiswert.  
Beim Basiswert zeigt die App die globale Koeffizientenstärke.

Die v1.2.0-Pipeline ergänzt leakage-sichere Event-Risk-Features, Walk-Forward-Validierung und ereignisbewusste Erklärbarkeit.  
Rohdaten wie konkrete Event-Namen werden nicht als Modellfeatures verwendet.
""",
        "docs_empty": "_Projektvisualisierungen sind im vollständigen GitHub-Repository verfügbar. Diese Hugging-Face-Space-Version bleibt leichtgewichtig für die Live-Demo._",
        "guardrail": "— **Leakage-Schutz aktiv**: Future-, Lead-, Target- und Zone-Features werden blockiert.",
        "feature_guide": """
## Leitfaden zu sichtbaren Features

| Feature | Bedeutung |
|---|---|
| `vix_change_roll4_stability` | aktuelle Stabilität von VIX-Veränderungen |
| `sp500_ret_roll4_stability` | aktuelle Stabilität der S&P-500-Renditen |
| `Google_Sentiment_Index` | Google-Sentiment-Signal |
| `vix_change_lag1` | VIX-Veränderung der vorherigen Periode |
| `vix_change_lag2` | VIX-Veränderung mit zwei Perioden Verzögerung |
| `google_sentiment_7d_mean` | kurzfristiger Sentiment-Durchschnitt |
| `Unemployment` | makroökonomischer Arbeitsmarktkontext |
| `Mood_Index` | kombinierter Market-Mood-Index |

Versteckte Event-Risk-Features bleiben im Modell, werden aber in der App nicht manuell bearbeitet.
""",
        "links": """
## Projektlinks

- [Vollständiges GitHub-Repository](https://github.com/ArturMelnyk-analyst/Market-Mood-Forecasting)
- [Live Hugging Face Space](https://huggingface.co/spaces/Artur-Melnyk/Market-Mood-Forecasting)
""",
        "diagnostics_note": "Die Diagnostik prüft geladenes Artefakt, Feature-Anzahl, sichtbare Eingaben und Metadaten.",
        "diag_btn": "Diagnostik anzeigen",
        "footer": "Nur ein Bildungs- und Portfolio-Projekt. Keine Anlageberatung."
    }
}

CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
}
.mmf-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 16px 18px;
    background: rgba(128,128,128,0.06);
}
.mmf-small {
    font-size: 0.92rem;
    opacity: 0.88;
}
.mmf-hero h1 {
    margin-bottom: 0.2rem;
}
"""


def _copy(lang: str, key: str) -> str:
    return COPY.get(lang, COPY["English"]).get(key, COPY["English"][key])


# =========================================================
# LOAD MODEL + META
# =========================================================
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Meta JSON not found: {META_PATH}")

print("Loading model from:", MODEL_PATH)
print("Loading meta from:", META_PATH)

model = joblib.load(MODEL_PATH)
meta = _read_json(META_PATH)

if not meta:
    raise ValueError(f"Meta JSON missing or empty at {META_PATH}.")

print("Loaded model successfully.")
print("Meta model_version:", meta.get("model_version"))
print("Meta artifact:", meta.get("artifact"))


# =========================================================
# FEATURE ORDER / TASK
# =========================================================
def _split_pipeline(m):
    if Pipeline and isinstance(m, Pipeline):
        try:
            return m[:-1], m[-1]
        except Exception:
            pass
    return None, m

def _get_feature_order(m, meta_obj: dict) -> List[str]:
    if meta_obj.get("feature_names_in_"):
        return list(meta_obj["feature_names_in_"])
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if Pipeline and isinstance(m, Pipeline):
        last = m.steps[-1][1]
        if hasattr(last, "feature_names_in_"):
            return list(last.feature_names_in_)
    raise ValueError("feature_names_in_ missing in model/meta.")

FEATURE_ORDER = _get_feature_order(model, meta)
VISIBLE_FEATURES = [f for f in (meta.get("visible_features") or DEFAULT_VISIBLE) if f in FEATURE_ORDER]
INVISIBLE_FEATURES = [f for f in FEATURE_ORDER if f not in VISIBLE_FEATURES]

if not VISIBLE_FEATURES:
    raise ValueError("No valid visible features resolved from meta/default list.")

preproc, est = _split_pipeline(model)
TASK = "classification" if (hasattr(est, "predict_proba") or hasattr(est, "decision_function")) else "regression"
Y_UNIT = "score" if TASK == "classification" else meta.get("y_unit", "pct_return")
MODEL_VERSION = meta.get("model_version", "v1.2.0")
RESIDUAL_STD = meta.get("residual_std", None)


# =========================================================
# LEAKAGE CHECKS
# =========================================================
def _scan_leakage(names: List[str]) -> List[str]:
    offenders = set()
    for n in names:
        if n in BLOCKLIST_HARD:
            offenders.add(n)
            continue
        low = n.lower()
        for pat in LEAKY_PATTERNS:
            if re.search(pat, low):
                offenders.add(n)
                break
    return sorted(offenders)

offenders = _scan_leakage(FEATURE_ORDER)
if offenders:
    raise RuntimeError(
        "Leakage policy violation. These features are not allowed:\n"
        + "\n".join(f"- {x}" for x in offenders)
    )


# =========================================================
# TRAINING MEDIANS
# =========================================================
def _ensure_training_medians(meta_obj: dict) -> Tuple[Dict[str, float], str, List[str]]:
    med = dict(meta_obj.get("training_medians", {}))
    missing = [f for f in FEATURE_ORDER if f not in med]

    if not missing:
        return med, "meta", []

    scaler_means = None
    if preproc is not None and hasattr(preproc, "steps"):
        for _, step in preproc.steps:
            if hasattr(step, "mean_"):
                arr = np.asarray(step.mean_, dtype=float)
                if arr.size == len(FEATURE_ORDER):
                    scaler_means = arr
                    break

    if scaler_means is not None:
        med = {f: float(v) for f, v in zip(FEATURE_ORDER, scaler_means)}
        source = "scaler_mean"
    else:
        for f in missing:
            med[f] = 0.0
        source = "zeros"

    meta_obj["training_medians"] = med
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, indent=2)
    except Exception:
        pass

    return med, source, missing

TRAINING_MEDIANS, MEDIANS_SOURCE, MEDIANS_MISSING = _ensure_training_medians(meta)


# =========================================================
# HELPERS
# =========================================================
def _fmt_float(x: float) -> str:
    try:
        return f"{x:.6f}"
    except Exception:
        return "n/a"

def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "n/a"

def _pack_visible(values: List[float]) -> pd.DataFrame:
    row = {f: float(TRAINING_MEDIANS[f]) for f in INVISIBLE_FEATURES}
    row.update({f: float(v) for f, v in zip(VISIBLE_FEATURES, values)})
    return pd.DataFrame([row], dtype="float64")[FEATURE_ORDER]

def _predict(df_row: pd.DataFrame) -> dict:
    X = preproc.transform(df_row) if preproc is not None else df_row
    out = {"raw": None, "prob": None}

    if hasattr(est, "decision_function"):
        raw = float(np.ravel(est.decision_function(X))[0])
        out["raw"] = raw
        try:
            from scipy.special import expit
            out["prob"] = float(expit(raw))
        except Exception:
            if hasattr(est, "predict_proba"):
                out["prob"] = float(est.predict_proba(X)[:, 1][0])
        return out

    if hasattr(est, "predict_proba"):
        p = float(est.predict_proba(X)[:, 1][0])
        out["raw"] = p
        out["prob"] = p
        return out

    out["raw"] = float(np.ravel(est.predict(X))[0])
    return out

def _coef_contributions(df_row: pd.DataFrame, baseline_row: pd.DataFrame) -> pd.Series:
    coef = getattr(est, "coef_", None)
    if coef is None:
        return pd.Series(0.0, index=FEATURE_ORDER)

    coef = np.asarray(coef, dtype=float).reshape(-1)

    if preproc is not None:
        X_now = preproc.transform(df_row)
        X_base = preproc.transform(baseline_row)
    else:
        X_now = df_row.values
        X_base = baseline_row.values

    delta = (X_now - X_base).reshape(-1)
    contrib = coef * delta
    return pd.Series(contrib, index=FEATURE_ORDER, name="contribution")

def _plot_contribs_with_fallback(contrib: pd.Series, top_k: int = 8):
    if not contrib.empty and not (contrib.abs() < 1e-12).all():
        top = contrib.sort_values(key=lambda v: v.abs()).iloc[-top_k:]
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(top))))
        ax.barh(top.index, top.values)
        ax.axvline(0.0, linewidth=1)
        ax.set_title("Top Feature Contributions")
        ax.set_xlabel("Contribution vs baseline")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        return fig

    coef = getattr(est, "coef_", None)
    if coef is None:
        return _empty_plot("Inputs match baseline (medians); no local contributions.")

    coef = np.asarray(coef, dtype=float).reshape(-1)
    s = pd.Series(np.abs(coef), index=FEATURE_ORDER, name="|coef|")
    top = s.sort_values().iloc[-top_k:]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(top))))
    ax.barh(top.index, top.values)
    ax.set_title("At baseline — global |coefficients|")
    ax.set_xlabel("|Coefficient|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig

def _empty_plot(message: str = "No explanation generated.") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    return fig

def _all_zero_visible(vals: List[float]) -> bool:
    return all(abs(float(v)) <= ZERO_EPS for v in vals)


# =========================================================
# ROBUST NUDGE
# =========================================================
def _extract_coeff_and_scaler():
    coef = getattr(est, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef, dtype=float).reshape(-1)
    else:
        coef = np.zeros(len(FEATURE_ORDER), dtype=float)

    scale_arr = None
    if preproc is not None and hasattr(preproc, "steps"):
        for _, step in preproc.steps:
            if hasattr(step, "scale_"):
                s = np.asarray(step.scale_, dtype=float)
                if s.size == len(FEATURE_ORDER):
                    scale_arr = s
                    break

    return coef, scale_arr

def _choose_visible_feature_with_largest_coef(coef: np.ndarray) -> Optional[str]:
    vis_idx = [i for i, f in enumerate(FEATURE_ORDER) if f in VISIBLE_FEATURES]
    if not vis_idx:
        return None
    best_i, best_mag = max(((i, abs(float(coef[i]))) for i in vis_idx), key=lambda t: t[1])
    if best_mag <= 1e-12:
        return VISIBLE_FEATURES[0]
    return FEATURE_ORDER[best_i]

def _nudge_to_hit_logit_delta(base_visible_vals: List[float], target_logit_delta: float = 0.25) -> Tuple[List[float], str]:
    coef, scale_arr = _extract_coeff_and_scaler()
    feat = _choose_visible_feature_with_largest_coef(coef)
    if feat is None:
        return base_visible_vals, "No visible feature available to nudge."

    j = FEATURE_ORDER.index(feat)
    coef_j = float(coef[j])
    vals = base_visible_vals[:]
    idx_vis = VISIBLE_FEATURES.index(feat)

    if abs(coef_j) <= 1e-12:
        vals[idx_vis] = vals[idx_vis] + 1.0
        return vals, f"{feat} +1.0"

    delta_z = target_logit_delta / coef_j

    if scale_arr is not None:
        delta_x = float(delta_z) * float(scale_arr[j])
        vals[idx_vis] = vals[idx_vis] + delta_x
        return vals, f"{feat} {delta_x:+.6f}"

    vals[idx_vis] = vals[idx_vis] + float(delta_z)
    return vals, f"{feat} {delta_z:+.6f}"


# =========================================================
# DOC IMAGES
# =========================================================
def _find_doc_images() -> List[str]:
    override = os.getenv("MMF_DOC_GLOB")
    if override:
        return sorted(glob.glob(override, recursive=True))[:24]

    priority_patterns = [
        "./images/modeling/*v1_2_0.png",
        "./images/model_explain/*v1_2_0.png",
        "./images/feature_engineering/*v1_2_0.png",
        "./images/eda/*.png",
    ]

    candidates = []
    for pattern in priority_patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    seen, out = set(), []
    for p in candidates:
        if os.path.exists(p) and p not in seen:
            out.append(p)
            seen.add(p)
    return out[:24]


# =========================================================
# UI LOGIC
# =========================================================
def _predict_from_values(lang: str, vals: List[float], status_prefix: str = ""):
    baseline_vals = [float(TRAINING_MEDIANS[f]) for f in VISIBLE_FEATURES]

    if _all_zero_visible(vals):
        fig = _empty_plot(_copy(lang, "zero_plot"))
        status = (f"{status_prefix}\n\n" if status_prefix else "") + _copy(lang, "zero_status")
        return _copy(lang, "zero_pred"), "n/a", fig, status

    at_baseline = all(abs(a - b) <= ZERO_EPS for a, b in zip(vals, baseline_vals))

    row = _pack_visible(vals)
    baseline_row = _pack_visible(baseline_vals)

    pred = _predict(row)
    contrib = _coef_contributions(row, baseline_row)
    fig = _plot_contribs_with_fallback(contrib)

    if TASK == "classification" and pred["prob"] is not None:
        pred_text = f"score={_fmt_float(pred['raw'])}, prob={_fmt_pct(pred['prob'])}"
    else:
        pred_text = _fmt_pct(pred["raw"]) if Y_UNIT == "pct_return" else _fmt_float(pred["raw"])

    interval_txt = "± n/a"
    if TASK == "regression" and RESIDUAL_STD is not None and np.isfinite(RESIDUAL_STD):
        lo = pred["raw"] - 1.96 * RESIDUAL_STD
        hi = pred["raw"] + 1.96 * RESIDUAL_STD
        interval_txt = f"[{_fmt_float(lo)}, {_fmt_float(hi)}]"

    core_status = _copy(lang, "baseline_status") if at_baseline else _copy(lang, "local_status")
    status = f"{status_prefix}\n\n{core_status}".strip()
    return pred_text, interval_txt, fig, status

def ui_predict(lang, *args):
    vals = []
    for i, f in enumerate(VISIBLE_FEATURES):
        v = args[i]
        vals.append(float(TRAINING_MEDIANS[f]) if v in ("", None) else float(v))
    return _predict_from_values(lang, vals)

def ui_demo_predict(lang):
    base_vals = [float(TRAINING_MEDIANS.get(f, 0.0)) for f in VISIBLE_FEATURES]
    nudged_vals, note = _nudge_to_hit_logit_delta(base_vals, target_logit_delta=0.25)
    return _predict_from_values(lang, nudged_vals, status_prefix=f"{_copy(lang, 'demo_prefix')} {note}")

def ui_diagnostics():
    return pd.DataFrame([{
        "model_path": MODEL_PATH,
        "meta_path": META_PATH,
        "model_version": MODEL_VERSION,
        "task": TASK,
        "y_unit": Y_UNIT,
        "n_features": len(FEATURE_ORDER),
        "n_visible": len(VISIBLE_FEATURES),
        "n_invisible": len(INVISIBLE_FEATURES),
        "visible_features": ",".join(VISIBLE_FEATURES),
        "medians_source": MEDIANS_SOURCE,
        "missing_medians": ",".join(MEDIANS_MISSING),
        "hf_space": IS_HF_SPACE,
    }])

def ui_text(lang):
    badges = _copy(lang, "model_badges").format(
        model_version=MODEL_VERSION,
        task=TASK,
        n_visible=len(VISIBLE_FEATURES),
        n_hidden=len(INVISIBLE_FEATURES),
    )
    return (
        _copy(lang, "hero"),
        badges,
        _copy(lang, "method_note"),
        _copy(lang, "input_note"),
        _copy(lang, "docs_intro"),
        _copy(lang, "feature_guide"),
        _copy(lang, "links"),
        _copy(lang, "diagnostics_note"),
        _copy(lang, "guardrail"),
        _copy(lang, "footer"),
    )


# =========================================================
# GRADIO APP
# =========================================================
def build_interface():
    with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
        language = gr.Radio(
            ["English", "Deutsch"],
            value="English",
            label="Language / Sprache",
            interactive=True,
        )

        hero_md = gr.Markdown(_copy("English", "hero"), elem_classes=["mmf-hero"])
        badges_md = gr.Markdown(
            _copy("English", "model_badges").format(
                model_version=MODEL_VERSION,
                task=TASK,
                n_visible=len(VISIBLE_FEATURES),
                n_hidden=len(INVISIBLE_FEATURES),
            ),
            elem_classes=["mmf-card"],
        )
        method_md = gr.Markdown(_copy("English", "method_note"), elem_classes=["mmf-small"])

        if MEDIANS_SOURCE != "meta":
            gr.Markdown(
                f"⚠️ Training medians came from `{MEDIANS_SOURCE}` because some were missing in meta."
            )

        with gr.Tab("Predict / Prognose"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Inputs / Eingaben")
                    input_note_md = gr.Markdown(_copy("English", "input_note"), elem_classes=["mmf-small"])

                    inputs = []
                    for f in VISIBLE_FEATURES:
                        comp = gr.Number(
                            label=f,
                            value=float(TRAINING_MEDIANS.get(f, 0.0)),
                            precision=6
                        )
                        inputs.append(comp)

                    with gr.Row():
                        predict_btn = gr.Button("Predict / Prognose", variant="primary")
                        demo_btn = gr.Button("Demo: nudge / Demo", variant="secondary")

                with gr.Column(scale=2):
                    pred_out = gr.Textbox(
                        label="Prediction / Prognose",
                        interactive=False
                    )
                    range_out = gr.Textbox(
                        label="Approx. range / Bereich",
                        interactive=False
                    )
                    plot_out = gr.Plot(label="Explanation / Erklärung")
                    status_out = gr.Markdown(elem_classes=["mmf-card"])

            predict_btn.click(
                fn=ui_predict,
                inputs=[language] + inputs,
                outputs=[pred_out, range_out, plot_out, status_out]
            )

            demo_btn.click(
                fn=ui_demo_predict,
                inputs=[language],
                outputs=[pred_out, range_out, plot_out, status_out]
            )

        with gr.Tab("Explain & Docs / Erklärung & Doku"):
            docs_intro_md = gr.Markdown(_copy("English", "docs_intro"))
            feature_guide_md = gr.Markdown(_copy("English", "feature_guide"))

            files = _find_doc_images()
            if files:
                gr.Gallery(
                    files,
                    label="Model documentation / Modelldokumentation",
                    columns=2,
                    preview=True,
                    allow_preview=True
                )
            else:
                gr.Markdown(_copy("English", "docs_empty"))

            links_md = gr.Markdown(_copy("English", "links"))

        with gr.Tab("Diagnostics / Diagnostik"):
            diag_note_md = gr.Markdown(_copy("English", "diagnostics_note"))
            diag_btn = gr.Button("Get diagnostics / Diagnostik anzeigen")
            diag_df = gr.Dataframe(wrap=True, interactive=False)
            diag_btn.click(fn=ui_diagnostics, inputs=None, outputs=diag_df)

        guardrail_md = gr.Markdown(_copy("English", "guardrail"))
        footer_md = gr.Markdown(_copy("English", "footer"), elem_classes=["mmf-small"])

        language.change(
            fn=ui_text,
            inputs=[language],
            outputs=[
                hero_md,
                badges_md,
                method_md,
                input_note_md,
                docs_intro_md,
                feature_guide_md,
                links_md,
                diag_note_md,
                guardrail_md,
                footer_md,
            ],
        )

    return demo


# =========================================================
# LAUNCH
# =========================================================
def first_free_port(start=7860, end=7879):
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return None

if __name__ == "__main__":
    app = build_interface()

    app.launch(
        server_name="0.0.0.0" if IS_HF_SPACE else "127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
