"""Offline synthetic tests of M11 Stage 7 local inventory; no live endpoints."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phase_b_m11_stage7_source_inventory_v0_1 as s7


def sha(b):
    return hashlib.sha256(b).hexdigest()


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        parent = Path(self.tmp.name)
        self.repo = parent / "repo"
        self.private = parent / "private"
        (self.repo / ".git").mkdir(parents=True)
        self.private.mkdir()
        pins, privatepins = {}, {}
        for rel in s7.PINNED:
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((rel + " pinned fixture").encode())
            pins[rel] = s7.digest(p)
        for rel in s7.PRIVATE_PINS:
            p = self.private / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((rel + " pinned fixture").encode())
            privatepins[rel] = s7.digest(p)
        self.patchpins = patch.object(s7, "PINNED", pins)
        self.patchpriv = patch.object(s7, "PRIVATE_PINS", privatepins)
        self.patchpins.start()
        self.patchpriv.start()
        self.addCleanup(self.patchpins.stop)
        self.addCleanup(self.patchpriv.stop)

    def add(self, rel, data=b"sample", owner="repo"):
        p = (self.repo if owner == "repo" else self.private) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def scan(self, **kw):
        return s7.scan_scope(self.repo, self.private, **kw)

    def test_pins_validate(self):
        s7.validate_environment(self.repo, self.private, git_check=False)

    def test_modified_registry_pin_rejected(self):
        self.add("phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json", b"changed")
        with self.assertRaisesRegex(ValueError, "upstream pin"):
            s7.validate_environment(self.repo, self.private, git_check=False)

    def test_modified_adoption_rejected(self):
        self.add("m11_future/adoption/owner_adoption_decision_20261114_v1_0.json", b"changed", "private")
        with self.assertRaisesRegex(ValueError, "private evidence"):
            s7.validate_environment(self.repo, self.private, git_check=False)

    def test_modified_sealed_settlements_rejected(self):
        self.add("m11_future/historical_baseline_20261114/arena_official_settlements_19rows_20261114.csv", b"changed", "private")
        with self.assertRaisesRegex(ValueError, "private evidence"):
            s7.validate_environment(self.repo, self.private, git_check=False)

    def test_empty_source_roots_not_certified(self):
        r = self.scan()
        self.assertEqual(r["discovered_file_count"], 1)  # sealed baseline only
        self.assertEqual(r["verified_external_data_sources"], 0)
        self.assertEqual(r["validated_training_dataset_rows"], 0)
        self.assertFalse(r["historical_asof_availability_certified"])

    def test_raw_es_file_found_but_not_validated(self):
        p = self.add("data/raw/databento/incremental_v0_1/ES_fake.dbn.zst")
        r = self.scan()
        row = next(x for x in r["file_metadata"] if x["path"] == str(p))
        self.assertEqual(row["source_bucket_by_directory_only"], "ES")
        self.assertFalse(row["market_data_contents_validated"])
        self.assertEqual(row["sha256"], sha(b"sample"))

    def test_external_source_buckets_only_directory(self):
        self.add("data/raw/gdelt/example.parquet")
        self.add("data/raw/vix/example.csv")
        self.add("data/raw/google_trends/example.csv")
        self.add("data/raw/fred/example.csv")
        r = self.scan()
        for src in ("GDELT", "VIX", "GOOGLE_TRENDS", "UNEMPLOYMENT"):
            self.assertEqual(r["candidate_file_counts_by_directory_name_only"][src], 1)
        self.assertEqual(r["verified_external_data_sources"], 0)

    def test_unrecognized_suffix_not_discovered(self):
        self.add("data/raw/vix/unrelated.exe")
        self.assertEqual(self.scan()["candidate_file_counts_by_directory_name_only"]["VIX"], 0)

    def test_large_file_not_hashed(self):
        p = self.add("data/raw/databento/incremental_v0_1/big.parquet", b"abcd")
        row = next(x for x in self.scan(hash_max_bytes=2)["file_metadata"] if x["path"] == str(p))
        self.assertIsNone(row["sha256"])
        self.assertEqual(row["sha256_status"], "SKIPPED_SIZE_LIMIT")

    def test_zero_hash_threshold(self):
        r = self.scan(hash_max_bytes=0)
        self.assertTrue(all(x["sha256"] is None for x in r["file_metadata"]))

    def test_negative_hash_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self.scan(hash_max_bytes=-1)

    def test_file_cap_prevents_unbounded_scan(self):
        self.add("data/raw/databento/incremental_v0_1/cap.csv")
        with self.assertRaisesRegex(ValueError, "file cap"):
            self.scan(max_files=1)

    def test_sibling_directory_not_scanned(self):
        self.add("secret/private_data/do_not_scan.csv")
        r = self.scan()
        self.assertFalse(any("do_not_scan" in f["path"] for f in r["file_metadata"]))

    def test_symlink_file_skipped(self):
        target = self.add("data/raw/vix/a.csv")
        try:
            (target.parent / "b.csv").symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation disallowed")
        self.assertEqual(self.scan()["candidate_file_counts_by_directory_name_only"]["VIX"], 1)

    def test_symlink_root_rejected(self):
        outside = self.repo / "safe"
        outside.mkdir()
        d = self.repo / "data/raw/vix"
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            d.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation disallowed")
        with self.assertRaisesRegex(ValueError, "Unsafe candidate directory"):
            self.scan()

    def test_snapshot_private_only(self):
        # patch the git-dependent validation call to check local CLI file writing only
        p = self.private / "snapshot.json"
        with patch.object(s7, "validate_environment"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                status = s7.run(["scan", "--repo", str(self.repo), "--private-root", str(self.private), "--snapshot-file", str(p)])
        self.assertEqual(status, 0)
        self.assertTrue(p.exists())
        self.assertEqual(json.loads(p.read_text())["trained_new_models"], 0)

    def test_snapshot_no_overwrite(self):
        p = self.private / "snapshot.json"
        p.write_text("original")
        with patch.object(s7, "validate_environment"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = s7.run(["scan", "--repo", str(self.repo), "--private-root", str(self.private), "--snapshot-file", str(p)])
        self.assertEqual(code, 2)
        self.assertEqual(p.read_text(), "original")

    def test_snapshot_refuses_repo_path(self):
        p = self.repo / "oops.json"
        with patch.object(s7, "validate_environment"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = s7.run(["scan", "--repo", str(self.repo), "--private-root", str(self.private), "--snapshot-file", str(p)])
        self.assertEqual(code, 2)
        self.assertFalse(p.exists())

    def test_read_only_scan_creates_no_files(self):
        before = sorted(str(x) for x in self.repo.rglob("*") if x.is_file())
        _ = self.scan()
        after = sorted(str(x) for x in self.repo.rglob("*") if x.is_file())
        self.assertEqual(before, after)

    def test_command_scan_without_snapshot_writes_none(self):
        with patch.object(s7, "validate_environment"):
            out = StringIO()
            with redirect_stdout(out):
                status = s7.run(["scan", "--repo", str(self.repo), "--private-root", str(self.private)])
        self.assertEqual(status, 0)
        self.assertIn("FILES_WRITTEN: 0", out.getvalue())
        self.assertIn("MODELS_TRAINED: 0", out.getvalue())

    def test_snapshot_parent_must_exist(self):
        p = self.private / "missing" / "new.json"
        with patch.object(s7, "validate_environment"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = s7.run(["scan", "--repo", str(self.repo), "--private-root", str(self.private), "--snapshot-file", str(p)])
        self.assertEqual(code, 2)
        self.assertFalse(p.exists())

    def test_source_file_mtime_not_reported_as_publication(self):
        self.add("data/raw/vix/file.csv")
        r = self.scan()
        self.assertTrue(all("mtime_utc_local_filesystem_not_publication_time" in f for f in r["file_metadata"]))

    def test_all_source_files_missing_does_not_claim_no_opportunities(self):
        r = self.scan()
        self.assertNotIn("ARENA_NO_OPPORTUNITIES", json.dumps(r))
        self.assertEqual(r["forecast_submissions"], 0)


if __name__ == "__main__":
    unittest.main()
