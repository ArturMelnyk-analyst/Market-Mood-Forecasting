"""Read-only audit of historical ES label counts and frozen deployment provenance.

Run from C:\\Users\\melny\\Market-Mood-Forecasting in the existing .venv:
    python inspect_es_training_prior_readonly.py

Never fits, submits, purchases, or writes any file. The two reported historical
populations are *candidates*, not an automatically authorized baseline.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_PARQUET_SHA256 = (
    "6787e64fe1c0c3508d68cd2d98cc554c6e5da19f320c7080ad7e34831b9b5db5"
)
EXPECTED_MODEL_SHA256 = (
    "5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d"
)
CLASSES = ("bearish", "neutral", "bullish")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("STOP — " + msg)


def show_counts(label: str, table: pd.DataFrame) -> None:
    vals = table["target"].astype(str)
    counts = vals.value_counts()
    require(set(vals.unique()).issubset(set(CLASSES)), "Unexpected class labels")
    print(f"\n{label}")
    print("Rows:", len(vals))
    print("First date:", table.session_date.iloc[0].date())
    print("Last date:", table.session_date.iloc[-1].date())
    for c in CLASSES:
        n = int(counts.get(c, 0))
        print(f"  {c:<8} count={n:>5}, prior={n/len(vals):.8f}")
    require(sum(int(counts.get(c, 0)) for c in CLASSES) == len(vals), "Count mismatch")


def main() -> None:
    root = Path.cwd().resolve()
    require((root / "es_arena_daily.py").is_file(),
            "Open Command Prompt in your Market-Mood-Forecasting project root")
    parquet = root / "data/processed/es_daily_modeling_v0_1.parquet"
    metadata_path = root / "data/processed/es_daily_modeling_v0_1_metadata.json"
    require(parquet.is_file() and metadata_path.is_file(), "Original Notebook 08 artifacts missing")
    parquet_hash = sha256_file(parquet)
    require(parquet_hash == EXPECTED_PARQUET_SHA256, "Historical parquet hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("model_table_sha256") == parquet_hash, "Parquet/metadata mismatch")
    features = metadata["feature_contract"]["feature_columns"]
    require(len(features) == 12 and len(set(features)) == 12, "Unexpected feature schema")
    df = pd.read_parquet(parquet)
    require(df.shape == (3855, 16), f"Unexpected historical shape: {df.shape}")
    df = df.copy()
    df["session_date"] = pd.to_datetime(df.session_date)
    require(df.session_date.is_monotonic_increasing and not df.session_date.duplicated().any(),
            "Date order or uniqueness mismatch")
    require(str(df.session_date.iloc[-1].date()) == "2026-09-30", "Cutoff mismatch")
    require(set(df.target.astype(str).unique()) == set(CLASSES), "Unexpected historical classes")
    X = df[features].apply(pd.to_numeric, errors="raise")
    numbers = X.to_numpy(dtype=float, na_value=np.nan)
    require(not np.isinf(numbers).any(), "Infinite feature values")
    complete = X.notna().all(axis=1)
    require(int((~complete).sum()) == 20 and (~complete).iloc[:20].all()
            and complete.iloc[20:].all(), "Warm-up eligibility differs from frozen Notebook 09")
    valid = df.loc[complete].copy().reset_index(drop=True)
    require(len(valid) == 3835, "Complete-feature population mismatch")
    initial = valid.loc[valid.session_date.le(pd.Timestamp("2018-12-31"))]
    require(len(initial) == 1934, "Initial train population mismatch")
    print("HISTORICAL PARQUET SHA-256: VERIFIED", parquet_hash)
    show_counts("CANDIDATE A — NB09 initial training sample (through 2018-12-31)", initial)
    show_counts("CANDIDATE B — entire pre-Arena feature-complete history (through 2026-09-30)", valid)

    deployment = root / "data/processed/es_arena_nb10_v0_1/deployment_ab4d5c14423352c7"
    model_path = deployment / "es_daily_lr_deployment_v0_1.joblib"
    md_path = deployment / "es_daily_lr_deployment_v0_1_metadata.json"
    manifest_path = deployment / "es_daily_lr_deployment_v0_1_manifest.json"
    require(all(p.is_file() for p in (model_path, md_path, manifest_path)),
            "Frozen Notebook 10 deployment files missing")
    mh = sha256_file(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ident = json.loads(md_path.read_text(encoding="utf-8")).get("deployment_identity", {})
    require(mh == EXPECTED_MODEL_SHA256 and manifest.get("model_sha256") == mh,
            "Frozen deployment SHA-256 mismatch")
    require(ident.get("train_through") == "2026-09-30", "Deployment cutoff mismatch")
    require(ident.get("class_order") == list(CLASSES), "Deployment class order mismatch")
    print("\nFROZEN DEPLOYMENT MODEL HASH: VERIFIED", mh)
    print("Final deployment train_through:", ident.get("train_through"))
    for field in ("train_n", "training_n", "n_train", "fit_rows", "training_rows", "n_samples_fit"):
        if field in ident:
            print("Deployment fit-row count metadata:", field, "=", ident[field])

    notebook10 = root / "notebooks/10_es_arena_prospective_inference_v0_1.ipynb"
    if notebook10.is_file():
        nb = json.loads(notebook10.read_text(encoding="utf-8-sig"))
        print("\nNOTEBOOK 10 — code snippets around fitting calls (inspect context carefully):")
        n = 0
        for cellno, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            lines = "".join(cell.get("source", [])).splitlines()
            matching = [i for i, line in enumerate(lines)
                        if re.search(r"\bfit\s*\(|\.fit\s*\(", line) and not line.lstrip().startswith("#")]
            for i in matching:
                n += 1
                print(f"\n  Cell {cellno}, line {i+1}:")
                for k in range(max(0, i-5), min(len(lines), i+4)):
                    print(f"    {k+1}: {lines[k][:180]}")
        if n == 0:
            print("  No obvious fitting call found; inspect source manually.")
    else:
        print("\nNOTEBOOK 10 source unavailable locally: exact fitting sample CANNOT be confirmed.")

    print("\nREAD-ONLY AUDIT COMPLETE — NO CSV CREATED OR MODIFIED.")
    print("Before filling training_class_counts.csv, verify which candidate population")
    print("actually fitted the frozen Notebook 10 deployment, and document that source.")


if __name__ == "__main__":
    main()
