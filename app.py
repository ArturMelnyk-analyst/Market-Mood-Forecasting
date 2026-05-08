# app.py — Market Mood Forecasting (Hotfix v1.1.2)
# Stable Gradio version aligned to v1.1.2 Logistic Regression baseline
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
APP_TITLE = "Market Mood — Next-Week Outlook (Hotfix v1.1.2)"
APP_DESC = (
    "Leakage-safe interface. Visible inputs are a small, interpretable subset; "
    "the rest are auto-filled from training medians. Explanations compare your inputs "
    "to a typical median baseline."
)

MODEL_PATH = os.getenv("MMF_MODEL_PATH", "./models/logreg_pipeline_v1_1_1775664292.joblib")
META_PATH = os.getenv("MMF_META_PATH", "./models/logreg_pipeline_v1_1_1775664292.json")

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
MODEL_VERSION = meta.get("model_version", "v1.1.2-hotfix")
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
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Inputs match baseline (medians); no local contributions.", ha="center", va="center")
        ax.axis("off")
        return fig

    coef = np.asarray(coef, dtype=float).reshape(-1)
    s = pd.Series(np.abs(coef), index=FEATURE_ORDER, name="|coef|")
    top = s.sort_values().iloc[-top_k:]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(top))))
    ax.barh(top.index, top.values)
    ax.set_title("At baseline — showing global |coefficients|")
    ax.set_xlabel("|Coefficient| (model scale)")
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

    mean_arr, scale_arr = None, None
    if preproc is not None and hasattr(preproc, "steps"):
        for _, step in preproc.steps:
            if hasattr(step, "mean_") and hasattr(step, "scale_"):
                m = np.asarray(step.mean_, dtype=float)
                s = np.asarray(step.scale_, dtype=float)
                if m.size == len(FEATURE_ORDER) and s.size == len(FEATURE_ORDER):
                    mean_arr, scale_arr = m, s
                    break

    return coef, mean_arr, scale_arr

def _choose_visible_feature_with_largest_coef(coef: np.ndarray) -> Optional[str]:
    vis_idx = [i for i, f in enumerate(FEATURE_ORDER) if f in VISIBLE_FEATURES]
    if not vis_idx:
        return None
    best_i, best_mag = max(((i, abs(float(coef[i]))) for i in vis_idx), key=lambda t: t[1])
    if best_mag <= 1e-12:
        return VISIBLE_FEATURES[0]
    return FEATURE_ORDER[best_i]

def _nudge_to_hit_logit_delta(base_visible_vals: List[float], target_logit_delta: float = 0.25) -> Tuple[List[float], str]:
    coef, _, scale_arr = _extract_coeff_and_scaler()
    feat = _choose_visible_feature_with_largest_coef(coef)
    if feat is None:
        return base_visible_vals, "No visible feature available to nudge."

    j = FEATURE_ORDER.index(feat)
    coef_j = float(coef[j])
    vals = base_visible_vals[:]
    idx_vis = VISIBLE_FEATURES.index(feat)

    if abs(coef_j) <= 1e-12:
        vals[idx_vis] = vals[idx_vis] + 1.0
        return vals, f"Nudged {feat} by +1.0 (fallback; coefficient near zero)."

    delta_z = target_logit_delta / coef_j

    if scale_arr is not None:
        delta_x = float(delta_z) * float(scale_arr[j])
        vals[idx_vis] = vals[idx_vis] + delta_x
        return vals, f"Nudged {feat} by {delta_x:+.6f} in original units (target Δlogit≈{target_logit_delta})."

    vals[idx_vis] = vals[idx_vis] + float(delta_z)
    return vals, f"Nudged {feat} by {delta_z:+.6f} (no scaler found; fallback in original units)."


# =========================================================
# DOC IMAGES
# =========================================================
def _find_doc_images() -> List[str]:
    override = os.getenv("MMF_DOC_GLOB")
    if override:
        return sorted(glob.glob(override, recursive=True))[:24]

    roots = ["./images", "./models", "./notebooks", "."]
    candidates = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
            if os.path.exists(p):
                candidates.append(p)

    seen, out = set(), []
    for p in candidates:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:24]


# =========================================================
# UI LOGIC
# =========================================================
def _predict_from_values(vals: List[float], status_prefix: str = ""):
    baseline_vals = [float(TRAINING_MEDIANS[f]) for f in VISIBLE_FEATURES]

    # Best-practice behavior: all-zero visible input is invalid / empty scenario
    if _all_zero_visible(vals):
        pred_text = "No prediction generated"
        interval_txt = "n/a"
        fig = _empty_plot("All visible inputs are zero.\nNo explanation generated.")
        status = (
            f"{status_prefix}\n\n" if status_prefix else ""
        ) + (
            "All visible inputs are zero. This is treated as an empty or unrealistic scenario. "
            "Please enter at least one meaningful non-zero value or use 'Demo: nudge off baseline'."
        )
        return pred_text, interval_txt, fig, status

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

    core_status = (
        "Inputs match baseline. Showing global |coefficients|."
        if at_baseline else
        "Inputs differ from baseline. Showing local contributions."
    )
    status = f"{status_prefix}\n\n{core_status}".strip()
    return pred_text, interval_txt, fig, status

def ui_predict(*args):
    vals = []
    for i, f in enumerate(VISIBLE_FEATURES):
        v = args[i]
        vals.append(float(TRAINING_MEDIANS[f]) if v in ("", None) else float(v))
    return _predict_from_values(vals)

def ui_demo_predict():
    base_vals = [float(TRAINING_MEDIANS.get(f, 0.0)) for f in VISIBLE_FEATURES]
    nudged_vals, note = _nudge_to_hit_logit_delta(base_vals, target_logit_delta=0.25)
    return _predict_from_values(nudged_vals, status_prefix=f"Demo nudge applied. {note}")

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
    }])


# =========================================================
# GRADIO APP
# =========================================================
def build_interface():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"""# {APP_TITLE}

{APP_DESC}

**Model:** `{MODEL_VERSION}`  
**Task:** `{TASK}`  
**Visible inputs:** `{len(VISIBLE_FEATURES)}`  
**Invisible auto-filled features:** `{len(INVISIBLE_FEATURES)}`
""")

        if MEDIANS_SOURCE != "meta":
            gr.Markdown(
                f"⚠️ Training medians came from `{MEDIANS_SOURCE}` because some were missing in meta."
            )

        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""### Inputs (visible features)
Tip: baseline values are prefilled from training medians.

Best practice:
- all zeros = invalid / empty scenario
- use real values for custom scenarios
- use **Demo: nudge off baseline** to prove the model moves""")

                    inputs = []
                    for f in VISIBLE_FEATURES:
                        comp = gr.Number(
                            label=f,
                            value=float(TRAINING_MEDIANS.get(f, 0.0)),
                            precision=6
                        )
                        inputs.append(comp)

                    predict_btn = gr.Button("Predict", variant="primary")
                    demo_btn = gr.Button("Demo: nudge off baseline", variant="secondary")

                with gr.Column(scale=2):
                    pred_out = gr.Textbox(
                        label=("Prediction (score & prob)" if TASK == "classification" else "Prediction"),
                        interactive=False
                    )
                    range_out = gr.Textbox(
                        label="Approx. 95% range (regression only)",
                        interactive=False
                    )
                    plot_out = gr.Plot(label="Explanations")
                    status_out = gr.Markdown()

            predict_btn.click(
                fn=ui_predict,
                inputs=inputs,
                outputs=[pred_out, range_out, plot_out, status_out]
            )

            demo_btn.click(
                fn=ui_demo_predict,
                inputs=None,
                outputs=[pred_out, range_out, plot_out, status_out]
            )

        with gr.Tab("Explain & Docs"):
            gr.Markdown(
                "Local contributions compare your inputs to a training-median baseline. "
                "At baseline, the app shows global |coefficients|."
            )
            files = _find_doc_images()
            if files:
                gr.Gallery(
                    files,
                    label="Model documentation",
                    columns=2,
                    preview=True,
                    allow_preview=True
                )
            else:
                gr.Markdown(
                    "_Project visuals are available in the full GitHub repository. This Hugging Face Space is kept lightweight for live app deployment._"
                )

        with gr.Tab("Diagnostics"):
            diag_btn = gr.Button("Get diagnostics")
            diag_df = gr.Dataframe(wrap=True, interactive=False)
            diag_btn.click(fn=ui_diagnostics, inputs=None, outputs=diag_df)

        gr.Markdown("— **Leakage guardrails active**: future/lead/target/zone features are blocked.")

    return demo


# =========================================================
# LAUNCH
# =========================================================
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