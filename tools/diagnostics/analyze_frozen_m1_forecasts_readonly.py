"""Read-only consistency and bullish-preference diagnostics for frozen ES M1.

Only loads a trusted local joblib and saved forecast JSONs. Does not submit,
retrain, recalculate official records, modify input files, or use outcomes.
Writes private diagnostic reports to an explicitly provided output directory.

Usage (Windows CMD):
 python analyze_frozen_m1_forecasts_readonly.py --model "...joblib" \
   --forecasts "...\\forecasts" --output "...\\diagnostics"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_SHA256 = "5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d"
CLASSES = ("bearish", "bullish", "neutral")
TOLERANCE = 1e-8


def hashfile(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_num(x, field):
    if isinstance(x, bool) or not isinstance(x, (float, int)):
        raise ValueError(f"{field}: expected numeric JSON value (not a string/bool)")
    f = float(x)
    if not np.isfinite(f):
        raise ValueError(f"{field}: nonfinite value")
    return f


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()

    model_path = args.model
    files = sorted(args.forecasts.glob("forecast_*.json"))
    if not model_path.is_file():
        print(f"STOP: frozen artifact not found: {model_path}", file=sys.stderr)
        return 2
    if not files:
        print(f"STOP: no forecast_*.json in {args.forecasts}", file=sys.stderr)
        return 2
    expected = args.expected_sha256.lower()
    actual = hashfile(model_path)
    if actual != expected:
        print(f"STOP: artifact SHA256 mismatch. Expected {expected}; observed {actual}", file=sys.stderr)
        return 2

    # joblib relies on pickle. Only load previously generated, trusted files.
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        print("STOP: expected a scikit-learn Pipeline", file=sys.stderr)
        return 2
    if len(pipeline.steps) != 2 or not isinstance(pipeline.steps[0][1], StandardScaler) or not isinstance(pipeline.steps[1][1], LogisticRegression):
        print("STOP: unexpected fitted estimator structure", file=sys.stderr)
        return 2
    scaler = pipeline.steps[0][1]
    lr = pipeline.steps[1][1]
    classnames = [str(x) for x in pipeline.classes_]
    if set(classnames) != set(CLASSES) or len(classnames) != 3:
        print(f"STOP: unrecognized class identities {classnames}", file=sys.stderr)
        return 2
    if not hasattr(pipeline, "feature_names_in_"):
        print("STOP: frozen pipeline does not expose feature_names_in_", file=sys.stderr)
        return 2
    order = [str(x) for x in pipeline.feature_names_in_]
    if len(order) != 12 or len(set(order)) != 12 or lr.coef_.shape != (3, 12):
        print("STOP: unexpected feature count or coefficients", file=sys.stderr)
        return 2
    if [str(x) for x in lr.classes_] != classnames:
        print("STOP: inner classifier class order differs from pipeline", file=sys.stderr)
        return 2

    idx = {k: classnames.index(k) for k in CLASSES}
    rows = []
    contributions = []
    errors = []
    for fp in files:
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                raise ValueError("forecast JSON root must be an object")
            if str(doc.get("model_sha256", "")).lower() != expected:
                raise ValueError("declared model SHA256 differs from checked artifact")
            recorded_order = doc.get("feature_order")
            if not isinstance(recorded_order, list) or recorded_order != order:
                raise ValueError("saved feature_order is missing or not identical to frozen model order")
            raw = doc.get("features")
            if not isinstance(raw, dict) or set(raw) != set(order):
                raise ValueError("saved features do not have exactly the frozen 12 names")
            raw_values = [finite_num(raw[name], f"features.{name}") for name in order]
            x = pd.DataFrame([raw_values], columns=order)
            std = scaler.transform(x)[0]
            pred = pipeline.predict_proba(x)[0]
            logits = lr.decision_function(scaler.transform(x))[0]
            if not np.isfinite(pred).all() or not np.isfinite(std).all():
                raise ValueError("nonfinite computed probabilities/features")
            if not np.allclose(logits, lr.intercept_ + lr.coef_ @ std, rtol=0, atol=1e-10):
                raise ValueError("per-feature contributions do not reconstruct class scores")
            saved_raw = doc.get("probabilities")
            if not isinstance(saved_raw, dict) or set(saved_raw) != set(CLASSES):
                raise ValueError("saved probabilities missing or contain extra class keys")
            saved = np.array([finite_num(saved_raw[k], f"probabilities.{k}") for k in classnames])
            if (saved < 0).any() or abs(float(saved.sum()) - 1.0) > TOLERANCE:
                raise ValueError("saved probabilities negative or do not sum to one")
            err = float(np.max(np.abs(pred - saved)))
            if err > TOLERANCE:
                raise ValueError(f"frozen inference vs original saved probabilities mismatch: max absolute error {err:.12g}")
            winner = classnames[int(np.argmax(pred))]
            if doc.get("argmax_direction") != winner:
                raise ValueError("saved argmax_direction conflicts with reproduced probability argmax")
            ranked = sorted((float(v), c) for c, v in zip(classnames, pred))
            row = {
                "forecast_file": fp.name,
                "forecast_sha256": hashfile(fp),
                "target_session_date": doc.get("target_session_date", ""),
                "last_eligible_session": doc.get("last_eligible_session", ""),
                "challenge_id": doc.get("challenge_id", ""),
                "predicted_class": winner,
                "winning_margin_probability": ranked[-1][0] - ranked[-2][0],
                "max_abs_probability_reproduction_error": err,
                "p_bearish": float(pred[idx["bearish"]]),
                "p_bullish": float(pred[idx["bullish"]]),
                "p_neutral": float(pred[idx["neutral"]]),
            }
            for other in ("bearish", "neutral"):
                key = f"bullish_minus_{other}"
                difference = lr.coef_[idx["bullish"]] - lr.coef_[idx[other]]
                base = float(lr.intercept_[idx["bullish"]] - lr.intercept_[idx[other]])
                effects = difference * std
                total = float(base + effects.sum())
                if abs(total - float(logits[idx["bullish"]] - logits[idx[other]])) > 1e-10:
                    raise ValueError(f"{key}: component scores do not sum correctly")
                row[f"{key}_intercept"] = base
                row[f"{key}_feature_total"] = float(effects.sum())
                row[f"{key}_total_score"] = total
                for j, name in enumerate(order):
                    contributions.append({
                        "forecast_file": fp.name,
                        "target_session_date": row["target_session_date"],
                        "comparison": key,
                        "feature": name,
                        "original_feature_value": raw_values[j],
                        "standardized_feature_value": float(std[j]),
                        "coefficient_difference": float(difference[j]),
                        "score_contribution": float(effects[j]),
                    })
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append((fp.name, str(exc)))

    if hashfile(model_path) != expected:
        print("STOP: frozen model file changed during this inspection", file=sys.stderr)
        return 2
    if errors:
        print(f"STOP: {len(errors)} / {len(files)} forecasts failed strict validation. No reports have been written.", file=sys.stderr)
        for filename, reason in errors:
            print(f"  {filename}: {reason}", file=sys.stderr)
        return 2
    if len(rows) != len(files):
        print("STOP: record coverage inconsistent", file=sys.stderr)
        return 2

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    detail_cols = list(rows[0].keys())
    contribution_cols = list(contributions[0].keys())
    write_csv(output / "m1_forecast_comparison_PRIVATE.csv", rows, detail_cols)
    write_csv(output / "m1_feature_contributions_PRIVATE.csv", contributions, contribution_cols)

    counts = Counter(r["predicted_class"] for r in rows)
    def avg(key):
        return sum(float(r[key]) for r in rows) / len(rows)
    text = [
        "# M1 frozen bullish-preference diagnostic — PRIVATE",
        "",
        "Exploratory, read-only reconstruction of saved native forecasts. Not an official settled-sample evaluation.",
        f"Model SHA-256: `{expected}`",
        f"Native forecast files inspected: **{len(files)} / {len(files)}**",
        f"All reproduced original probability vectors within max absolute tolerance: **{TOLERANCE:g}**",
        f"Worst original-vs-recomputed probability difference: **{max(r['max_abs_probability_reproduction_error'] for r in rows):.3g}**",
        "",
        "## Reproduced predicted-class counts (native forecast files only)",
        "",
        f"- Bullish: **{counts['bullish']}**",
        f"- Neutral: **{counts['neutral']}**",
        f"- Bearish: **{counts['bearish']}**",
        "",
        "## Average original/reproduced probabilities",
        "",
        f"- P(bullish): **{avg('p_bullish'):.6f}**",
        f"- P(neutral): **{avg('p_neutral'):.6f}**",
        f"- P(bearish): **{avg('p_bearish'):.6f}**",
        f"- Mean argmax winning margin: **{avg('winning_margin_probability'):.6f}**",
        f"- Median argmax winning margin: **{float(np.median([r['winning_margin_probability'] for r in rows])):.6f}**",
        "",
        "## Pairwise class-score decomposition (means across native forecasts)",
        "",
        "Positive values favor bullish. These are LOGIT differences, not probability points.",
    ]
    for other in ("bearish", "neutral"):
        key = f"bullish_minus_{other}"
        text.extend([
            "",
            f"### Bullish vs {other}",
            f"- Intercept score advantage: **{avg(key+'_intercept'):+.6f}**",
            f"- Mean total feature effect: **{avg(key+'_feature_total'):+.6f}**",
            f"- Mean final score difference: **{avg(key+'_total_score'):+.6f}**",
        ])
        pairs = [(feat, sum(c['score_contribution'] for c in contributions if c['comparison'] == key and c['feature'] == feat) / len(rows)) for feat in order]
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        text.extend(["", "Largest absolute *mean* feature effects (positive supports bullish):", ""])
        for feat, value in pairs[:8]:
            text.append(f"- `{feat}`: **{value:+.6f}**")
    text.extend([
        "", "## Important limits", "",
        "- These are the native saved forecasts, including any native record not yet officially settled; legacy October 20 is not recreated.",
        "- This analysis does not determine official forecasting accuracy or statistical skill.",
        "- Score contributions are exact for this linear classifier, conditional on the stored feature vector; they are not causal market explanations.",
        "- Input feature JSONs and local joblib were trusted as historical artifacts, not independently time-certified.",
        "- Original receipts, market data, and immutable forecasts were not modified.",
        "- Outputs and forecast/challenge identifiers are private and should not be staged to GitHub without sanitization.",
    ])
    (output / "m1_bias_summary_PRIVATE.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("PASS: checked frozen M1 SHA256, all saved feature orders, probabilities and predicted classes")
    print(f"Native forecasts validated: {len(rows)}/{len(files)}")
    print(f"Reproduction max absolute probability error: {max(r['max_abs_probability_reproduction_error'] for r in rows):.3g}")
    print(f"Predicted counts: bullish={counts['bullish']}, neutral={counts['neutral']}, bearish={counts['bearish']}")
    print("PRIVATE REPORTS:")
    for name in ("m1_bias_summary_PRIVATE.md", "m1_forecast_comparison_PRIVATE.csv", "m1_feature_contributions_PRIVATE.csv"):
        print("  " + str(output / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
