"""Mocked control-flow tests for the existing Arena ES submission client.

Exercises the actual run_prepare/run_submit functions with all network access and
private writes replaced by mocks. Does not contact Arena, load M1, or change
historical forecasts/receipts.

Run from the repository root:
  python -m unittest test_arena_submission_no_post_integration_v0_1.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

import es_arena_daily as client
from arena_deadline_guard_v0_1 import require_safe_submission_time


class SubmissionNoPostIntegrationTests(unittest.TestCase):
    DEADLINE = datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)
    CID = "test-es-challenge-id"

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="arena_guard_test_")
        self.addCleanup(temp.cleanup)
        self.private_dir = Path(temp.name).resolve()
        self.forecast_path = self.private_dir / "forecast_test.json"
        self.forecast_path.write_text(
            json.dumps({
                "challenge_id": self.CID,
                "target_session_date": "2026-11-13",
                "contract_symbol": "ESZ6",
                "deadline_utc": self.DEADLINE.isoformat(),
                "status": "PREVIEW_NOT_SUBMITTED",
                "probabilities": {"bearish": 0.30, "neutral": 0.30, "bullish": 0.40},
                "source_csv": "test-only-source.csv",
                "source_sha256": "synthetic-hash",
                "source_manifest_sha256": "synthetic-hash",
                "model_dir": "test-only-model-dir",
                "model_sha256": "synthetic-model-hash",
                "prepared_at_utc": "2026-11-13T13:00:00+00:00",
                "note": "Synthetic test only; never submit",
            }), encoding="utf-8"
        )
        self.submit_args = SimpleNamespace(
            forecast=str(self.forecast_path),
            confirm_id=self.CID,
            approve="I_APPROVE_ONE_SCORED_SUBMISSION",
        )
        self.prepare_args = SimpleNamespace(
            challenge_id=self.CID,
            contract="ESZ6",
            date="2026-11-13",
            last_session="2026-11-12",
            sessions="synthetic-source.csv",
        )
        self.writes = []
        self.post_mock = Mock(side_effect=AssertionError("A real or unexpected POST is forbidden"))
        self._patches = [
            patch.object(client, "PRIVATE_DIR", self.private_dir),
            patch.object(client, "safe_path", side_effect=lambda p: Path(p).resolve()),
            patch.object(client, "checksum", return_value="synthetic-hash"),
            patch.object(client, "model_files", return_value=(
                Path("test-only.joblib"), {"model_sha256": "synthetic-model-hash"}, {}
            )),
            patch.object(client, "authenticate", return_value={"Authorization": "synthetic-test-only"}),
            patch.object(client, "fetch_one", return_value={"id": self.CID}),
            patch.object(client, "check_live_contract", return_value=None),
            patch.object(client, "deadline_for", return_value=self.DEADLINE),
            patch.object(client, "write_private_once", side_effect=self._record_write),
            patch.object(client.requests, "post", self.post_mock),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _record_write(self, path, payload):
        self.writes.append((Path(path).name, payload))
        return path

    def _clock(self, *times):
        # Use the *real* guard math, but supply deterministic synthetic UTC times.
        timeline = iter(times)

        def guard(deadline, *, now=None):
            self.assertEqual(deadline, self.DEADLINE)
            return require_safe_submission_time(deadline, now=next(timeline))

        return patch.object(client, "require_safe_submission_time", side_effect=guard)

    def test_prepare_inside_buffer_stops_before_data_processing(self):
        with patch.object(client, "validate_sessions") as process, self._clock(
            self.DEADLINE - timedelta(minutes=4)
        ):
            with self.assertRaisesRegex(RuntimeError, "Five-minute submission safety cutoff"):
                client.run_prepare(self.prepare_args)
        process.assert_not_called()
        self.post_mock.assert_not_called()
        self.assertEqual(self.writes, [])

    def test_submit_at_exact_cutoff_creates_no_attempt_or_post(self):
        with self._clock(self.DEADLINE - timedelta(minutes=5)):
            with self.assertRaisesRegex(RuntimeError, "Five-minute submission safety cutoff"):
                client.run_submit(self.submit_args)
        self.post_mock.assert_not_called()
        self.assertEqual(self.writes, [])

    def test_submit_crosses_cutoff_after_attempt_but_before_post(self):
        with self._clock(
            self.DEADLINE - timedelta(minutes=6),
            self.DEADLINE - timedelta(minutes=4),
        ):
            with self.assertRaisesRegex(RuntimeError, "Five-minute submission safety cutoff"):
                client.run_submit(self.submit_args)
        self.post_mock.assert_not_called()
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.writes[0][0], f"attempt_{self.CID}.json")
        self.assertTrue(all(not name.startswith("receipt_") for name, _ in self.writes))

    def test_safe_submission_reaches_exactly_one_mocked_post(self):
        response = Mock()
        response.status_code = 201
        response.json.return_value = {
            "challenge_id": self.CID,
            "prediction_id": "synthetic-prediction",
            "counts_for_score": True,
        }
        self.post_mock.side_effect = None
        self.post_mock.return_value = response
        with self._clock(
            self.DEADLINE - timedelta(minutes=7),
            self.DEADLINE - timedelta(minutes=6),
        ):
            client.run_submit(self.submit_args)
        self.post_mock.assert_called_once()
        self.assertEqual([name for name, _ in self.writes], [
            f"attempt_{self.CID}.json", f"receipt_{self.CID}.json",
        ])

    def test_mocked_post_timeout_does_not_retry(self):
        self.post_mock.side_effect = requests.Timeout("synthetic timeout")
        with self._clock(
            self.DEADLINE - timedelta(minutes=7),
            self.DEADLINE - timedelta(minutes=6),
        ):
            with self.assertRaisesRegex(RuntimeError, "POST result ambiguous"):
                client.run_submit(self.submit_args)
        self.post_mock.assert_called_once()
        self.assertEqual([name for name, _ in self.writes], [
            f"attempt_{self.CID}.json"
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
