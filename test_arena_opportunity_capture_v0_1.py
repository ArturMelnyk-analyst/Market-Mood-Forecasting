
"""Offline tests for the Arena discovery logger.

Uses mocked API responses and temporary output directories.
No real authentication, network calls, or forecast submissions.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import arena_opportunity_capture_v0_1 as logger


class CaptureTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.archive = Path(self.temp.name) / "discovery"

        patcher_archive = patch.object(
            logger, "ARCHIVE", self.archive
        )
        patcher_auth = patch.object(
            logger, "authenticate",
            return_value={"Authorization": "test-only"}
        )
        patcher_post = patch.object(
            requests, "post",
            side_effect=AssertionError("POST forbidden")
        )

        for patcher in (
            patcher_archive,
            patcher_auth,
            patcher_post,
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def records(self):
        files = list(self.archive.glob("*.json"))
        self.assertEqual(len(files), 1)

        path = files[0]
        content = path.read_bytes()

        return (
            json.loads(content),
            hashlib.sha256(content).hexdigest(),
        )

    def mock_response(self, body):
        response = Mock()
        response.status_code = 200
        response.headers = {
            "Date": "Thu, 12 Nov 2026 15:41:30 GMT"
        }
        response.content = json.dumps(body).encode("utf-8")
        response.json.return_value = body
        response.raise_for_status.return_value = None
        return response

    def test_mixed_market_feed(self):
        body = {
            "challenges": [
                {
                    "challenge": {
                        "id": "test-es-id",
                        "asset": "ES",
                        "challenge_type": "daily",
                        "contract_symbol": "ESZ6",
                        "deadline": "2026-11-13T15:00:00Z",
                    }
                },
                {
                    "challenge": {
                        "id": "test-palm-id",
                        "asset": "PALM",
                        "challenge_type": "daily",
                    }
                },
            ]
        }

        with patch.object(
            logger.requests,
            "get",
            return_value=self.mock_response(body),
        ):
            logger.capture()

        record, digest = self.records()

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["total_challenges"], 2)
        self.assertEqual(len(record["es_daily_challenges"]), 1)
        self.assertEqual(
            record["es_daily_challenges"][0]["challenge_id"],
            "test-es-id",
        )
        self.assertEqual(len(digest), 64)

    def test_empty_feed(self):
        with patch.object(
            logger.requests,
            "get",
            return_value=self.mock_response({
                "challenges": []
            }),
        ):
            logger.capture()

        record, _ = self.records()

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["total_challenges"], 0)
        self.assertEqual(record["es_daily_challenges"], [])

    def test_malformed_feed(self):
        with patch.object(
            logger.requests,
            "get",
            return_value=self.mock_response({
                "unexpected_field": []
            }),
        ):
            with self.assertRaises(SystemExit):
                logger.capture()

        record, _ = self.records()

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "ValueError")

    def test_network_timeout(self):
        with patch.object(
            logger.requests,
            "get",
            side_effect=requests.Timeout("simulated"),
        ):
            with self.assertRaises(SystemExit):
                logger.capture()

        record, _ = self.records()

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "Timeout")

    def test_duplicate_id_rejected(self):
        body = {
            "challenges": [
                {"id": "duplicate-id", "asset": "ES"},
                {"id": "duplicate-id", "asset": "ES"},
            ]
        }

        with patch.object(
            logger.requests,
            "get",
            return_value=self.mock_response(body),
        ):
            with self.assertRaises(SystemExit):
                logger.capture()

        record, _ = self.records()

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "ValueError")



    def test_missing_es_deadline(self):
        body = {
            "challenges": [
                {
                    "id": "es-no-deadline",
                    "asset": "ES",
                    "challenge_type": "daily",
                    "contract_symbol": "ESZ6",
                    "deadline": None,
                }
            ]
        }

        with patch.object(
            logger.requests,
            "get",
            return_value=self.mock_response(body),
        ):
            with self.assertRaises(SystemExit):
                logger.capture()

        record, _ = self.records()

        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["error_type"], "ESMetadataError")
        self.assertEqual(record["response_json"], body)

    def test_es_deadline_without_timezone(self):
        body = {
            "challenges": [
                {
                    "id": "es-no-timezone",
                    "asset": "ES",
                    "challenge_type": "daily",
                    "contract_symbol": "ESZ6",
                    "deadline": "2026-11-13T15:00:00",
                }
            ]
        }

        with patch.object(
            logger.requests,
            "get",
            return_value=self.mock_response(body),
        ):
            with self.assertRaises(SystemExit):
                logger.capture()

        record, _ = self.records()

        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["error_type"], "ESMetadataError")
        self.assertEqual(record["response_json"], body)



if __name__ == "__main__":
    unittest.main(verbosity=2)
