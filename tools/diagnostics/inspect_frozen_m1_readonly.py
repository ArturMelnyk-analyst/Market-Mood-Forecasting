#!/usr/bin/env python3
"""Read-only frozen M1 inspection for the Market Mood Forecasting ES experiment.

Never modifies a model, source snapshot, forecast, receipt, or notebook.
Only load the already-trusted local joblib produced by your own project;
joblib files can execute Python code during deserialization.

Usage from the repository root (Windows CMD):
    python inspect_frozen_m1_readonly.py

This inspection is diagnostic, NOT proof of forecast/model execution and
NOT an analysis of the 16 historical feature vectors.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "data/processed/es_arena_nb10_v0_1/"
    "deployment_ab4d5c14423352c7/es_daily_lr_deployment_v0_1.joblib"
)
EXPECTED_SHA256 = "5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d"
EXPECTED_FEATURES = (
    "es_return_lag1", "es_return_lag2", "es_return_lag3", "es_return_lag5",
    "es_range_pct_lag1", "es_log_volume_lag1", "es_return_mean_5",
    "es_return_mean_20", "es_vol_5", "es_vol_20",
    "es_abs_return_mean_5", "es_abs_return_mean_20",
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(x):
    return type(x).__module__ + "." + type(x).__qualname__


def is_pipeline(x):
    return hasattr(x, "steps") and hasattr(x, "named_steps") and hasattr(x, "predict_proba")


def is_lr(x):
    return all(hasattr(x, a) for a in ("coef_", "intercept_", "classes_", "predict_proba"))


def find_candidate(obj):
    """Return an identifiable trained estimator; no arbitrary object recursion."""
    if is_pipeline(obj) or is_lr(obj):
        return obj, "root"
    if isinstance(obj, dict):
        preferred = (
            "pipeline", "model_pipeline", "fitted_pipeline", "estimator", "model",
            "classifier", "clf", "logreg_pipeline", "logreg_model",
        )
        for key in preferred:
            val = obj.get(key)
            if is_pipeline(val) or is_lr(val):
                return val, f"dict[{key!r}]"
        for key, val in obj.items():
            if is_pipeline(val) or is_lr(val):
                return val, f"dict[{key!r}]"
    return None, None


def get_feature_names(obj, pipe_or_model, n):
    candidates = []
    for x, label in ((pipe_or_model, "fitted pipeline/model"), (obj, "joblib root")):
        if hasattr(x, "feature_names_in_"):
            candidates.append((getattr(x, "feature_names_in_"), label + ".feature_names_in_"))
        if isinstance(x, dict):
            for key in ("feature_columns", "feature_names", "features", "FEATURE_COLUMNS", "FEATURE_NAMES"):
                if key in x:
                    candidates.append((x[key], "joblib dict " + key))
            meta = x.get("metadata")
            if isinstance(meta, dict):
                for key in ("feature_columns", "feature_names", "features"):
                    if key in meta:
                        candidates.append((meta[key], "joblib metadata " + key))
    for vals, source in candidates:
        if isinstance(vals, (list, tuple)) or type(vals).__name__ == "ndarray":
            vals = list(vals)
            if len(vals) == n and all(isinstance(v, str) for v in vals):
                return vals, source
    return [f"FEATURE_INDEX_{i:02d}_ORDER_UNVERIFIED" for i in range(n)], None


def main():
    try:
        import numpy as np
        import sklearn
        import joblib
    except ImportError as exc:
        print("Missing installed dependency:", exc, file=sys.stderr)
        return 2

    path = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_PATH
    if len(sys.argv) > 2:
        print("Usage: python inspect_frozen_m1_readonly.py [optional_model_path]", file=sys.stderr)
        return 2
    print("FROZEN M1 READ-ONLY INSPECTION")
    print("Path:", path)
    if not path.is_file():
        print("STOP: Model file not found. Do not guess a replacement artifact.")
        return 2
    sha_before = digest_file(path)
    print("Artifact bytes:", path.stat().st_size)
    print("Artifact SHA256:", sha_before)
    if sha_before.lower() != EXPECTED_SHA256:
        print("STOP: SHA256 DOES NOT MATCH the frozen M1 fingerprint. Model NOT loaded.")
        return 3
    print("Frozen artifact hash check: PASS")
    print("Python version:", sys.version.split()[0])
    print("scikit-learn version:", sklearn.__version__)
    print("joblib version:", joblib.__version__)
    print("WARNING: Only load trusted locally produced joblib files.")

    obj = joblib.load(path)
    print("Loaded root object type:", describe(obj))
    if isinstance(obj, dict):
        print("Root dictionary keys (no values):", [str(k) for k in obj.keys()])

    candidate, where = find_candidate(obj)
    if candidate is None:
        print("STOP: Unrecognized deployment container. No claims about coefficients.")
        print("Share the root object type and dictionary key names (no values),")
        print("then inspect the actual deployment format in Notebook 10.")
        return 4
    print("Estimator location:", where)
    print("Estimator type:", describe(candidate))

    if is_pipeline(candidate):
        print("Pipeline steps:", [(str(name), describe(step)) for name, step in candidate.steps])
        fitted_lrs = [(name, step) for name, step in candidate.steps if is_lr(step)]
        if len(fitted_lrs) != 1:
            print("STOP: Expected exactly one fitted LR step, found", len(fitted_lrs))
            return 4
        lr_name, lr = fitted_lrs[0]
        print("LogisticRegression step:", lr_name)
        scaler_steps = [(name, step) for name, step in candidate.steps if hasattr(step, "mean_") and hasattr(step, "scale_")]
        print("Standardizing scaler step(s):", [name for name, _ in scaler_steps])
    else:
        lr = candidate
        print("IMPORTANT: no surrounding scaler found in the identified estimator; inspect deployment metadata.")

    coef = np.asarray(lr.coef_, dtype=float)
    intercept = np.asarray(lr.intercept_, dtype=float)
    labels = list(lr.classes_)
    print("Estimator classes_ (actual column order):", [str(v) for v in labels])
    print("Coefficient shape:", tuple(coef.shape))
    print("Intercept shape:", tuple(intercept.shape))
    print("Configured C:", getattr(lr, "C", "unavailable"))
    print("Configured solver:", getattr(lr, "solver", "unavailable"))
    if coef.shape != (3, 12) or intercept.shape != (3,) or len(labels) != 3:
        print("STOP: unexpected M1 class or feature shape; don't interpret using 3x12 assumption.")
        return 4

    names, source = get_feature_names(obj, candidate, coef.shape[1])
    print("Feature-order source:", source or "NOT AVAILABLE IN LOADED OBJECT")
    if source:
        if set(names) == set(EXPECTED_FEATURES):
            print("Feature names match expected ES-only twelve: YES")
        else:
            print("WARNING: feature-name set differs from Kopei's expected twelve; inspect before interpreting.")
    else:
        print("WARNING: Do NOT attribute coefficients to particular feature names until order verified in Notebook 10.")

    # At a standardized zero-feature vector, only the class intercepts contribute.
    shifted = intercept - np.max(intercept)
    p_at_zero = np.exp(shifted) / np.exp(shifted).sum()
    print("\nINTERCEPT-ONLY DIAGNOSTIC: at a zero standardized feature vector")
    for label, bias, pp in zip(labels, intercept, p_at_zero):
        print(f"class={str(label):>9}  intercept={bias:+.8f}  p_at_zero={pp:.8f}")
    print("NOTE: p_at_zero is NOT necessarily the historical class frequency or the actual daily forecast.")

    print("\nCOEFFICIENTS BY CLASS (indices unless named order verified)")
    for i, label in enumerate(labels):
        order = np.argsort(-np.abs(coef[i]))
        top = order[:5]
        print("Class", str(label), "5 largest absolute standardized coefficients:")
        for j in top:
            print(f"  {names[int(j)]}: {coef[i, int(j)]:+.8f}")

    print("\nCHECKPOINT")
    sha_after = digest_file(path)
    print("Model SHA256 unchanged after inspection:", sha_before == sha_after)
    if sha_after != sha_before:
        print("STOP: original model bytes changed unexpectedly")
        return 5
    print("This script did not call fit(), predict_proba() on historical data, or write any file.")
    print("Actual forecast feature contributions require separately verified saved feature vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
