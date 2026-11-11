#!/usr/bin/env python3
"""Synthetic, local-only tests for frozen M1 diagnostic tools.

No original ES data, credentials, Arena receipts, or private joblib artifacts
are needed or loaded. Tests write exclusively into temporary directories.

From repository root (Windows CMD):
    python -m unittest -v tools/diagnostics/test_m1_diagnostics.py
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "analyze_frozen_m1_forecasts_readonly.py"
INSPECTOR = ROOT / "inspect_frozen_m1_readonly.py"
FEATURES = (
    "es_return_lag1", "es_return_lag2", "es_return_lag3", "es_return_lag5",
    "es_range_pct_lag1", "es_log_volume_lag1", "es_return_mean_5",
    "es_return_mean_20", "es_vol_5", "es_vol_20",
    "es_abs_return_mean_5", "es_abs_return_mean_20",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_es_m1_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.forecasts = self.base / "forecasts"
        self.forecasts.mkdir()
        self.output = self.base / "output"
        self.model = self.base / "synthetic_m1.joblib"

        rng = np.random.default_rng(2042)
        values = rng.normal(size=(150, len(FEATURES)))
        x = pd.DataFrame(values, columns=FEATURES)
        score = 1.1 * values[:, 0] - .7 * values[:, 1] + .5 * values[:, 6]
        y = np.where(score < -.45, "bearish", np.where(score > .45, "bullish", "neutral"))
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(C=.01, solver="lbfgs", max_iter=2000)),
        ])
        self.pipeline.fit(x, y)
        joblib.dump(self.pipeline, self.model)
        self.sha = digest(self.model)
        self.data_files = []
        for i in range(3):
            row = x.iloc[i]
            probs = self.pipeline.predict_proba(x.iloc[[i]])[0]
            labels = [str(v) for v in self.pipeline.classes_]
            record = {
                "model_sha256": self.sha,
                "feature_order": list(FEATURES),
                "features": {name: float(row[name]) for name in FEATURES},
                "probabilities": {label: float(probs[j]) for j, label in enumerate(labels)},
                "argmax_direction": labels[int(np.argmax(probs))],
                "target_session_date": f"2026-10-{21+i:02d}",
                "last_eligible_session": f"2026-10-{20+i:02d}",
                "challenge_id": f"synthetic-challenge-{i}",
            }
            dest = self.forecasts / f"forecast_synthetic_{i}.json"
            dest.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            self.data_files.append(dest)

    def run_analyzer(self, *, sha=None):
        return subprocess.run([
            sys.executable, str(ANALYZER),
            "--model", str(self.model),
            "--forecasts", str(self.forecasts),
            "--output", str(self.output),
            "--expected-sha256", sha if sha is not None else self.sha,
        ], cwd=self.base, capture_output=True, text=True, timeout=60)

    def change_record(self, func):
        first = self.data_files[0]
        doc = json.loads(first.read_text(encoding="utf-8"))
        func(doc)
        first.write_text(json.dumps(doc), encoding="utf-8")

    def assert_rejected_without_reports(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("STOP:", result.stderr)
        self.assertFalse(self.output.exists(), "Failed validation must not create reports")

    def test_valid_synthetic_forecasts_exactly_reproduce(self):
        model_before = digest(self.model)
        forecast_before = {f.name: digest(f) for f in self.data_files}
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Native forecasts validated: 3/3", result.stdout)
        self.assertIn("Reproduction max absolute probability error: 0", result.stdout)
        self.assertEqual(model_before, digest(self.model))
        self.assertEqual(forecast_before, {f.name: digest(f) for f in self.data_files})
        summary = self.output / "m1_bias_summary_PRIVATE.md"
        details = self.output / "m1_forecast_comparison_PRIVATE.csv"
        features = self.output / "m1_feature_contributions_PRIVATE.csv"
        self.assertTrue(summary.is_file())
        with details.open(newline="", encoding="utf-8") as h:
            self.assertEqual(len(list(csv.DictReader(h))), 3)
        with features.open(newline="", encoding="utf-8") as h:
            contributions = list(csv.DictReader(h))
        self.assertEqual(len(contributions), 3 * 12 * 2)
        self.assertEqual({r["comparison"] for r in contributions}, {"bullish_minus_bearish", "bullish_minus_neutral"})

    def test_wrong_model_hash_fails_before_deserialization(self):
        result = self.run_analyzer(sha="0" * 64)
        self.assert_rejected_without_reports(result)
        self.assertIn("artifact SHA256 mismatch", result.stderr)

    def test_missing_feature_is_rejected(self):
        self.change_record(lambda d: d["features"].pop(FEATURES[0]))
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("saved features", result.stderr)

    def test_wrong_feature_order_is_rejected(self):
        self.change_record(lambda d: d["feature_order"].reverse())
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("feature_order", result.stderr)

    def test_changed_probability_is_rejected(self):
        def mutate(d):
            d["probabilities"]["bullish"] += .02
            d["probabilities"]["neutral"] -= .02
        self.change_record(mutate)
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("probabilities mismatch", result.stderr)

    def test_changed_feature_is_rejected_by_prediction_reproduction(self):
        def mutate(d):
            d["features"][FEATURES[0]] += 3.0
        self.change_record(mutate)
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("probabilities mismatch", result.stderr)

    def test_boolean_feature_is_rejected(self):
        self.change_record(lambda d: d["features"].__setitem__(FEATURES[0], True))
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("numeric JSON value", result.stderr)

    def test_wrong_forecast_model_hash_is_rejected(self):
        self.change_record(lambda d: d.__setitem__("model_sha256", "f" * 64))
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("declared model SHA256 differs", result.stderr)

    def test_mismatched_saved_argmax_is_rejected(self):
        def mutate(d):
            d["argmax_direction"] = next(c for c in ("bearish", "bullish", "neutral") if c != d["argmax_direction"])
        self.change_record(mutate)
        result = self.run_analyzer()
        self.assert_rejected_without_reports(result)
        self.assertIn("argmax_direction", result.stderr)

    def test_inspector_reports_fitted_classes_without_modification(self):
        # Inspect through its public main function, but mock the deployment hash
        # constant to match this strictly synthetic test artifact.
        spec = importlib.util.spec_from_file_location("inspect_synthetic_m1", INSPECTOR)
        inspector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspector)
        model_before = digest(self.model)
        capture = io.StringIO()
        with patch.object(inspector, "EXPECTED_SHA256", self.sha), \
             patch.object(sys, "argv", [str(INSPECTOR), str(self.model)]), \
             contextlib.redirect_stdout(capture):
            result = inspector.main()
        self.assertEqual(result, 0, capture.getvalue())
        self.assertIn("Feature names match expected ES-only twelve: YES", capture.getvalue())
        self.assertIn("Model SHA256 unchanged after inspection: True", capture.getvalue())
        self.assertEqual(model_before, digest(self.model))


if __name__ == "__main__":
    unittest.main(verbosity=2)
