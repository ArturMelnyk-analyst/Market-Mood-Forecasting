"""Synthetic-only M11 Stage 3 tests. Never access real private evidence or network."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import phase_a_m11_readiness_v0_1 as legacy
import phase_a_m11_stage3_immutable_baseline_v0_1 as m


def csv_bytes(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(sorted(m.COLUMNS)), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


class BaselineFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.repo = self.base / "repo"
        self.root = self.base / "private"
        self.repo.mkdir()
        self.root.mkdir()
        self.integrated = self.root / "m10c_integrated_post_nov13_20261114_A.json"
        self.settlement = self.root / "m10c_settlement_post_nov13_20261114_A.json"
        self.frozen = self.root / "m11_future/historical_baseline_20261114/arena_official_settlements_19rows_20261114.csv"
        self.adoption = self.root / "m11_future/adoption/owner_adoption_decision_20261114_v1_0.json"
        self.policy = self.repo / "docs/M11_PROSPECTIVE_OPPORTUNITY_PROTOCOL_v1_0.md"
        self.live = self.repo / "data/processed/es_arena_research_v0_1/arena_official_settlements.csv"
        for p in [self.frozen, self.adoption, self.policy, self.live]:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.cids = [str(UUID(int=i + 1)) for i in range(19)]
        self.rows = []
        for i, cid in enumerate(self.cids):
            record = {k: "" for k in m.COLUMNS}
            record.update(challenge_id=cid, asset="ES", contract_symbol="ESZ6",
                          official_label="bearish" if i < 8 else "bullish" if i < 16 else "neutral",
                          official_deadline_utc="2026-11-13T19:00:00Z",
                          settlement_date="2026-11-13", source_url="https://example.invalid/synthetic")
            self.rows.append(record)
        csvraw = csv_bytes(self.rows)
        self.frozen.write_bytes(csvraw)
        self.live.write_bytes(csvraw)
        integrated = {
            "retained_challenges": 19, "revision_records": 19, "review_flags": [],
            "records": [{"challenge_id": i} for i in self.cids],
            "model_execution_certified": False, "final_scored_revision_certified": False,
            "historical_eligible_opportunity_census_certified": False, "phase_a_protocol_adopted": False,
        }
        self.integrated.write_text(json.dumps(integrated), encoding="utf-8")
        intsha = m.sha256(self.integrated.read_bytes())
        settlement = {
            "challenge_count": 19, "settlement_rows_in_observed_cohort": 19,
            "settlement_rows_outside_observed_cohort": 0, "review_flags": [],
            "counts": {"deadline_DEADLINES_MATCH_AS_RECORDED": 19},
            "records": [{"challenge_id": i, "official_settlement_row_present": True} for i in self.cids],
            "sources": {"m10c_integrated_physical_sha256": intsha,
                        "local_official_settlements_csv_physical_sha256": m.sha256(csvraw)},
            "final_scored_revision_certified": False, "model_execution_certified": False,
            "historical_opportunity_census_certified": False,
            "phase_a_150_opportunity_protocol_adopted": False,
        }
        self.settlement.write_text(json.dumps(settlement), encoding="utf-8")
        self.policy.write_bytes(b"synthetic final protocol")
        owner = {
            "record_type": "M11_OWNER_ADOPTION_DECISION_V0_1",
            "decision": "AFFIRMATIVELY_ADOPTED_BY_OWNER", "protocol_git_commit": m.COMMIT,
            "study_design": "ORIGINAL_150_TOTAL_DESCRIPTIVE",
            "protocol_sha256": m.sha256(self.policy.read_bytes()),
            "approved_effective_at_utc": "2026-11-15T12:00:00Z",
            "total_accepted_forecast_target": 150,
            "historical_accepted_evidenced_checkpoint": 19,
            "additional_accepted_forecasts_at_checkpoint": 131,
            "accepted_forecast_phase_lengths": [30, 60, 60],
            "historical_19_count_in_descriptive_target": True,
            "historical_19_prospectively_enrolled": False,
            "prospective_opportunities_enrolled_at_recording": 0,
        }
        self.adoption.write_text(json.dumps(owner), encoding="utf-8")
        self.pins = dict(integrated=intsha,
                         settlement=m.sha256(self.settlement.read_bytes()),
                         snapshot=m.sha256(csvraw), adoption=m.sha256(self.adoption.read_bytes()),
                         protocol=m.sha256(self.policy.read_bytes()))

    def audit(self, *, baseline_only=False):
        with patch.object(legacy, "PIN", dict(integrated=self.pins["integrated"],
                                               csv=self.pins["snapshot"],
                                               settlement=self.pins["settlement"],
                                               m10_commit="synthetic")):
            return m.audit(self.repo, self.root, baseline_only=baseline_only, pins=self.pins)

    def error(self, prefix):
        with self.assertRaises(m.BaselineError) as exc:
            self.audit()
        self.assertIn(prefix, str(exc.exception))

    def test_clean_sealed_baseline_and_live_19(self):
        r = self.audit()
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["baseline_settlement_rows"], 19)
        self.assertEqual(r["live_csv"]["total_official_settlement_rows"], 19)
        self.assertEqual(r["live_csv"]["additional_live_settlement_rows_unclassified"], 0)
        self.assertFalse(r["historical_eligible_opportunity_census_certified"])

    def test_new_twentieth_row_is_allowed_not_automatically_accepted(self):
        extra = dict(self.rows[0], challenge_id=str(UUID(int=23)), official_label="neutral")
        self.live.write_bytes(csv_bytes(self.rows + [extra]))
        r = self.audit()
        self.assertEqual(r["live_csv"]["total_official_settlement_rows"], 20)
        self.assertEqual(r["live_csv"]["additional_live_settlement_rows_unclassified"], 1)
        self.assertEqual(r["additional_accepted_forecasts_verified_by_this_reader"], 0)
        self.assertEqual(r["prospective_opportunities_enrolled_by_this_reader"], 0)

    def test_live_extra_column_allowed_with_historical_values_intact(self):
        newrows = [dict(r, extra_source_metadata="x") for r in self.rows]
        out = io.StringIO(newline="")
        w = csv.DictWriter(out, fieldnames=list(sorted(m.COLUMNS)) + ["extra_source_metadata"], lineterminator="\n")
        w.writeheader(); w.writerows(newrows)
        self.live.write_bytes(out.getvalue().encode())
        self.assertEqual(self.audit()["live_csv"]["total_official_settlement_rows"], 19)

    def test_mutated_historical_label_rejected(self):
        rows = [dict(r) for r in self.rows]
        rows[0]["official_label"] = "neutral"
        self.live.write_bytes(csv_bytes(rows))
        self.error("HISTORICAL_CELL_MUTATED_IN_LIVE")

    def test_mutated_historical_deadline_rejected(self):
        rows = [dict(r) for r in self.rows]
        rows[1]["official_deadline_utc"] = "2026-11-13T19:01:00Z"
        self.live.write_bytes(csv_bytes(rows))
        self.error("HISTORICAL_CELL_MUTATED_IN_LIVE")

    def test_missing_historical_id_rejected(self):
        self.live.write_bytes(csv_bytes(self.rows[1:]))
        self.error("HISTORICAL_ID_REMOVED_FROM_LIVE")

    def test_duplicate_live_id_rejected(self):
        self.live.write_bytes(csv_bytes(self.rows + [self.rows[0]]))
        self.error("LIVE_DUPLICATE_UUID")

    def test_live_malformed_uuid_rejected(self):
        rows = [dict(r) for r in self.rows]
        rows.append(dict(rows[0], challenge_id="not-uuid"))
        self.live.write_bytes(csv_bytes(rows))
        self.error("LIVE_INVALID_UUID")

    def test_baseline_byte_tamper_rejected(self):
        self.frozen.write_bytes(self.frozen.read_bytes() + b"\n")
        self.error("PINNED_FILE_HASH_MISMATCH")

    def test_integrated_source_tamper_rejected(self):
        self.integrated.write_bytes(self.integrated.read_bytes() + b" ")
        self.error("PINNED_FILE_HASH_MISMATCH")

    def test_settlement_source_tamper_rejected(self):
        self.settlement.write_bytes(self.settlement.read_bytes() + b" ")
        self.error("PINNED_FILE_HASH_MISMATCH")

    def test_adoption_source_tamper_rejected(self):
        self.adoption.write_bytes(self.adoption.read_bytes() + b" ")
        self.error("PINNED_FILE_HASH_MISMATCH")

    def test_protocol_source_tamper_rejected(self):
        self.policy.write_bytes(self.policy.read_bytes() + b" ")
        self.error("PINNED_FILE_HASH_MISMATCH")

    def test_false_historical_certification_rejected(self):
        obj = json.loads(self.integrated.read_text())
        obj["final_scored_revision_certified"] = True
        self.integrated.write_text(json.dumps(obj))
        self.pins["integrated"] = m.sha256(self.integrated.read_bytes())
        stmt = json.loads(self.settlement.read_text())
        stmt["sources"]["m10c_integrated_physical_sha256"] = self.pins["integrated"]
        self.settlement.write_text(json.dumps(stmt))
        self.pins["settlement"] = m.sha256(self.settlement.read_bytes())
        self.error("HISTORICAL_REVIEW_FLAGS")

    def test_wrong_owner_design_rejected_even_with_rehashed_synthetic(self):
        owner = json.loads(self.adoption.read_text())
        owner["study_design"] = "FRESH_150_OPPORTUNITIES"
        self.adoption.write_text(json.dumps(owner))
        self.pins["adoption"] = m.sha256(self.adoption.read_bytes())
        self.error("ADOPTION_FIELD_MISMATCH")

    def test_baseline_only_does_not_read_live_csv(self):
        self.live.unlink()
        result = self.audit(baseline_only=True)
        self.assertIsNone(result["live_csv"])
        self.assertEqual(result["baseline_settlement_rows"], 19)

    def test_live_missing_rejected_when_live_check_requested(self):
        self.live.unlink()
        self.error("LIVE_CSV_MISSING")

    def test_read_only_audit_does_not_write_or_modify_files(self):
        files = [self.frozen, self.integrated, self.settlement, self.adoption, self.policy, self.live]
        before = [(p, p.read_bytes(), p.stat().st_mtime_ns) for p in files]
        self.audit()
        self.assertEqual([(p, p.read_bytes(), p.stat().st_mtime_ns) for p in files], before)

    def test_labels_distribution_is_checked(self):
        r = self.audit()
        self.assertEqual(r["baseline_label_counts"], {"bearish": 8, "bullish": 8, "neutral": 3})


if __name__ == "__main__":
    unittest.main()
