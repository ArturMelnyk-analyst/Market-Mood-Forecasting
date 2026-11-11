"""Synthetic unit tests. Creates only temporary fake artifacts, never real Arena POSTs."""
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MOD = Path(__file__).with_name("verify_es_forecast_provenance.py")
spec = importlib.util.spec_from_file_location("verify_es_forecast_provenance", MOD)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
CID = "ccd1eb17-7dba-47c9-a199-eb6e638b2b3b"


def js(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def mkroot(base):
    root = base / "project"
    root.mkdir()
    (root / "es_arena_daily.py").write_text("# inert fixture\n")
    mdir = root / "data/processed/es_arena_nb10_v0_1" / ("deployment_" + v.DEPLOYMENT_ID)
    mdir.mkdir(parents=True)
    binary = mdir / "es_daily_lr_deployment_v0_1.joblib"
    binary.write_bytes(b"fake synthetic joblib bytes; do not execute")
    expected_sha = v.sha256_file(binary)
    meta = mdir / "es_daily_lr_deployment_v0_1_metadata.json"
    js(meta, {"deployment_identity": {"train_through": "2026-09-30", "class_order": list(v.CLASSES), "feature_columns": list(v.FEATURES)}})
    js(mdir / "es_daily_lr_deployment_v0_1_manifest.json", {
        "deployment_id": v.DEPLOYMENT_ID, "model_sha256": expected_sha,
        "metadata_sha256": v.sha256_file(meta), "source_sha256": "source-test-sha"})
    snapdir = root / "data/processed/es_arena_nb11_v0_1/snapshots"
    snapdir.mkdir(parents=True)
    auditdir = root / "data/processed/es_arena_nb11_v0_1/audits"
    auditdir.mkdir(parents=True)
    reviewed = auditdir / "reviewed_example.csv"
    reviewed.write_bytes(b"fake-reviewed-audit\n")
    # Snapshot needs hash-derived name. Reconcile with original data & manifest.
    content = ("session_date,session_start_utc,session_end_utc,target_eligible,coverage_reviewed,raw_symbol,open,close\n"
               "2026-11-10,2026-11-09T23:00:00+00:00,2026-11-10T22:00:00+00:00,True,True,ESZ6,7580.75,7555.00\n")
    ssha = hashlib.sha256(content.encode()).hexdigest()
    snapshot = snapdir / f"es_sessions_20261110_{ssha[:16]}.csv"
    # Write the *exact bytes* that were hashed. Text-mode write_text() converts
    # LF to CRLF on Windows, invalidating the SHA-256 and cascading into 6 failures.
    snapshot.write_bytes(content.encode("utf-8"))
    assert v.sha256_file(snapshot) == ssha, "Fixture snapshot hash mismatch"
    mf = snapshot.with_suffix(".manifest.json")
    js(mf, {"snapshot_sha256": ssha,
            "snapshot_path": snapshot.relative_to(root).as_posix(),
            "candidate_audit_id": "example",
            "reviewed_audit_sha256": v.sha256_file(reviewed)})
    forecast = root / "data/processed/es_arena_live/forecasts" / f"forecast_20261111_{CID}.json"
    obj = {
        "challenge_id": CID, "target_session_date": "2026-11-11", "last_eligible_session": "2026-11-10",
        "contract_symbol": "ESZ6", "model_sha256": expected_sha,
        "model_dir": mdir.relative_to(root).as_posix(),
        "frozen_model_source_sha256": "source-test-sha",
        "source_csv": snapshot.relative_to(root).as_posix(),
        "source_sha256": ssha, "source_manifest_sha256": v.sha256_file(mf),
        "feature_order": list(v.FEATURES), "features": dict.fromkeys(v.FEATURES, 0.1),
        "probabilities": {"bearish": 0.25, "neutral": 0.35, "bullish": 0.40},
        "argmax_direction": "bullish", "status": "PREVIEW_NOT_SUBMITTED",
        "data_available_utc": "2026-11-11T09:30:00+00:00",
        "prepared_at_utc": "2026-11-11T10:00:00+00:00",
        "deadline_utc": "2026-11-11T15:00:00+00:00",
    }
    js(forecast, obj)
    receipts = root / "data/processed/es_arena_live/receipts"
    js(receipts / f"attempt_{CID}.json", {"challenge_id": CID, "forecast_sha256": v.sha256_file(forecast)})
    js(receipts / f"receipt_{CID}.json", {"http_status": 201,
        "forecast_sha256": v.sha256_file(forecast),
        "response": {"challenge_id": CID, "prediction_id": "pred-fake", "counts_for_score": True,
                     "created_at_utc": "2026-11-11T10:01:00+00:00"}})
    return root, forecast, snapshot


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root, self.forecast, self.snapshot = mkroot(Path(self.temp.name))
        self.model_hash = v.sha256_file(self.root / "data/processed/es_arena_nb10_v0_1" /
            ("deployment_"+v.DEPLOYMENT_ID) / "es_daily_lr_deployment_v0_1.joblib")
        self.mock = patch.object(v, "EXPECTED_MODEL_SHA", self.model_hash)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    def one(self):
        a = v.audit_repository(self.root)
        rec = next(r for r in a["records"] if r["challenge_id"] == CID)
        return rec

    def test_all_local_chain_verified(self):
        r = self.one()
        self.assertEqual(r["provenance_status"], "VERIFIED_LOCAL_CHAIN", r["issues"])
        self.assertTrue(r["model_hash_verified"])
        self.assertTrue(r["snapshot_manifest_verified"])
        self.assertTrue(r["reviewed_audit_verified"])
        self.assertTrue(r["accepted_by_receipt"])
        self.assertTrue(r["deadline_verified"])

    def test_missing_canonical_date_fails(self):
        obj = v.read_json(self.forecast)
        obj.pop("last_eligible_session")
        js(self.forecast, obj)
        r = self.one()
        self.assertEqual(r["provenance_status"], "FAIL")
        self.assertTrue(any("CANONICAL_DATE" in x for x in r["issues"]))

    def test_changed_probabilities_invalidates_receipt(self):
        obj = v.read_json(self.forecast)
        obj["probabilities"]["bullish"] = 0.30
        js(self.forecast, obj)
        r = self.one()
        self.assertEqual(r["provenance_status"], "FAIL")
        self.assertFalse(r["receipt_forecast_hash_verified"])

    def test_corrupt_source_snapshot(self):
        self.snapshot.write_text(self.snapshot.read_text()+"corruption\n")
        r = self.one()
        self.assertEqual(r["provenance_status"], "FAIL")
        self.assertIn("SOURCE_SNAPSHOT_BYTES_MISMATCH", r["issues"])

    def test_missing_model_file_no_verified_model(self):
        target = (self.root / "data/processed/es_arena_nb10_v0_1" /
            ("deployment_"+v.DEPLOYMENT_ID) / "es_daily_lr_deployment_v0_1.joblib")
        target.unlink()
        r = self.one()
        self.assertFalse(r["model_hash_verified"])
        self.assertEqual(r["provenance_status"], "INCOMPLETE")

    def test_no_receipt_is_prepared_not_submitted(self):
        (self.root / "data/processed/es_arena_live/receipts" / f"receipt_{CID}.json").unlink()
        (self.root / "data/processed/es_arena_live/receipts" / f"attempt_{CID}.json").unlink()
        r = self.one()
        self.assertTrue(r["artifact_integrity_verified"])
        self.assertEqual(r["provenance_status"], "PREPARED_NOT_SUBMITTED")
        self.assertFalse(r["accepted_by_receipt"])

    def test_missing_attempt_evidence_cannot_be_full_verified(self):
        (self.root / "data/processed/es_arena_live/receipts" / f"attempt_{CID}.json").unlink()
        r = self.one()
        self.assertEqual(r["provenance_status"], "INCOMPLETE")
        self.assertTrue(r["accepted_by_receipt"])

    def test_timezone_naive_server_time_does_not_verify_deadline(self):
        rp = self.root / "data/processed/es_arena_live/receipts" / f"receipt_{CID}.json"
        obj = v.read_json(rp)
        obj["response"]["created_at_utc"] = "2026-11-11T10:01:00"
        js(rp, obj)
        r = self.one()
        self.assertTrue(r["accepted_by_receipt"])
        self.assertFalse(r["deadline_verified"])
        self.assertEqual(r["provenance_status"], "INCOMPLETE")

    def test_wrong_manifest_hash_fails(self):
        mf = self.snapshot.with_suffix(".manifest.json")
        mf.write_text(mf.read_text()+"\n")
        r = self.one()
        self.assertIn("FORECAST_SOURCE_MANIFEST_HASH_MISMATCH", r["issues"])
        self.assertEqual(r["provenance_status"], "FAIL")

    def test_original_files_not_modified_by_audit(self):
        before = {p: v.sha256_file(p) for p in self.root.rglob("*") if p.is_file()}
        v.audit_repository(self.root)
        after = {p: v.sha256_file(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_receipt_and_existing_attempt_is_not_safe(self):
        (self.root / "data/processed/es_arena_live/receipts" / f"receipt_{CID}.json").unlink()
        r = self.one()
        self.assertTrue(r["attempt_present"])
        self.assertEqual(r["provenance_status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
