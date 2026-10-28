
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------
# 1. Locate the existing project artifacts
# -----------------------------------------------------

ROOT = Path(
    r"C:\Users\melny\Market-Mood-Forecasting"
)

DEPLOYMENT_DIR = (
    ROOT
    / "data/processed/es_arena_nb10_v0_1"
    / "deployment_ab4d5c14423352c7"
)

MODEL_PATH = (
    DEPLOYMENT_DIR
    / "es_daily_lr_deployment_v0_1.joblib"
)

METADATA_PATH = (
    DEPLOYMENT_DIR
    / "es_daily_lr_deployment_v0_1_metadata.json"
)

MANIFEST_PATH = (
    DEPLOYMENT_DIR
    / "es_daily_lr_deployment_v0_1_manifest.json"
)

DATA_PATH = (
    ROOT
    / "data/processed/es_daily_modeling_v0_1.parquet"
)

for path in [
    MODEL_PATH,
    METADATA_PATH,
    MANIFEST_PATH,
    DATA_PATH,
]:
    assert path.is_file(), f"Missing file: {path}"

# -----------------------------------------------------
# 2. Verify frozen model identity
# -----------------------------------------------------

metadata = json.loads(
    METADATA_PATH.read_text(encoding="utf-8")
)

manifest = json.loads(
    MANIFEST_PATH.read_text(encoding="utf-8")
)

model_hash = hashlib.sha256(
    MODEL_PATH.read_bytes()
).hexdigest()

assert model_hash == manifest["model_sha256"], (
    "STOP: Frozen model SHA-256 mismatch"
)

assert metadata["model_training_rows"] == 3835

features = metadata[
    "deployment_identity"
]["feature_columns"]

assert len(features) == 12

print("FROZEN MODEL INTEGRITY: PASS")
print("Model SHA-256:", model_hash)

# -----------------------------------------------------
# 3. Load the historical fitting data
# -----------------------------------------------------

historical = pd.read_parquet(DATA_PATH)

X = historical.loc[:, features].apply(
    pd.to_numeric,
    errors="raise"
)

valid_mask = np.isfinite(
    X.to_numpy(dtype=float)
).all(axis=1)

X = X.loc[valid_mask].copy()

assert len(X) == 3835, (
    f"Expected 3835 fitting rows; got {len(X)}"
)

print("\nHISTORICAL DATA")
print("Original rows:", len(historical))
print("Feature-complete rows:", len(X))
print("Features:", len(features))

# -----------------------------------------------------
# 4. Load the EXISTING frozen model
# -----------------------------------------------------

# Only load our own previously verified local artifact.
model = joblib.load(MODEL_PATH)

assert model.named_steps["scale"].n_samples_seen_ == 3835

# No fit() or retraining operation is performed.
probabilities = model.predict_proba(X)

assert probabilities.shape == (3835, 3)

assert np.isfinite(probabilities).all()

assert np.allclose(
    probabilities.sum(axis=1),
    1.0,
    atol=1e-10
)

# -----------------------------------------------------
# 5. Calculate model confidence
# -----------------------------------------------------

confidence = probabilities.max(axis=1)

summary = pd.Series(
    confidence,
    name="top_class_probability"
)

print("\n" + "=" * 55)
print("HISTORICAL FITTED-SAMPLE CONFIDENCE")
print("=" * 55)

print(
    summary.describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

# -----------------------------------------------------
# 6. Check whether confidence exceeds 42%
# -----------------------------------------------------

print("\n" + "=" * 55)
print("CONFIDENCE THRESHOLDS")
print("=" * 55)

for threshold in [
    0.40,
    0.42,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
]:
    count = int(
        (confidence > threshold).sum()
    )

    percentage = (
        count / len(confidence) * 100
    )

    print(
        f"Above {threshold:.0%}: "
        f"{count} observations "
        f"({percentage:.2f}%)"
    )

# -----------------------------------------------------
# 7. Inspect the strongest historical predictions
# -----------------------------------------------------

native_classes = np.asarray(
    model.named_steps["logreg"].classes_
)

top_indices = np.argsort(
    confidence
)[-10:][::-1]

print("\nTOP 10 HIGHEST-CONFIDENCE OBSERVATIONS")

for idx in top_indices:
    predicted_class = native_classes[
        np.argmax(probabilities[idx])
    ]

    print(
        f"Row {X.index[idx]} | "
        f"Class: {predicted_class} | "
        f"Confidence: {confidence[idx]:.4%}"
    )

print("\nMAXIMUM HISTORICAL CONFIDENCE:")
print(f"{confidence.max():.4%}")

print("\nREAD-ONLY DIAGNOSTIC: COMPLETED")
print("Frozen model unchanged.")
print("No forecasts submitted.")
print("No data files modified.")
