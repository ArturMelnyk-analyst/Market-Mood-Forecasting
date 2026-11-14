#!/usr/bin/env python3
"""M11 Stage 3 — offline READ-ONLY immutable 19-row historical ES baseline.

The November 14 snapshot is byte pinned; the mutable live CSV is NOT pinned to
its original 19-row hash, but historical rows must remain identical. New live
rows are counted as unclassified additions, never assumed accepted or enrolled.
Existing M10/M11 files, model, notebook and private source evidence are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import phase_a_m11_readiness_v0_1 as legacy

SCHEMA = "phase_a_m11_stage3_immutable_baseline_v0_1"
PIN = {
    "integrated": "033b513f662524dc84579d9934a6593b79fd3b70eccd601116a19d82c8d1f920",
    "settlement": "3f819eb2b422e02a95df5eb271b7dadf8770ee3f52223dc0709d854dce335be9",
    "snapshot": "093bc2aa57c7e87f073c2ff86b9fbe7ebd3cbb0b59d0491ee7d95b8c3aee68c4",
    "adoption": "90d797e773e50a023b59ff6b30cd9637fd2dcf8f5d432066b138d6b927111c25",
    "protocol": "a0eec9b6929397338ee69ee9a17e48cd4807e006c27463f8bd6508af576dd63d",
}
COMMIT = "6f089df837e5dcd1aa5fc198490e287a98fa7fd8"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SHA = re.compile(r"^[a-f0-9]{64}$")
COLUMNS = {"challenge_id", "settlement_date", "asset", "contract_symbol", "official_label",
           "settlement_return_pct", "official_deadline_utc", "official_settled_at_utc",
           "arena_score", "source_url", "verified_by", "verified_at_utc"}
LABELS = {"bearish", "neutral", "bullish"}


class BaselineError(ValueError):
    pass


def ensure(cond: bool, why: str) -> None:
    if not cond:
        raise BaselineError(why)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pinned_file(path: Path, expected: str) -> bytes:
    ensure(bool(SHA.fullmatch(expected)), "PIN_INVALID")
    ensure(path.is_file(), "PINNED_FILE_MISSING: " + path.name)
    raw = path.read_bytes()
    ensure(sha256(raw) == expected, "PINNED_FILE_HASH_MISMATCH: " + path.name)
    return raw


def read_csv(raw: bytes, kind: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    try:
        content = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content, newline=""))
        cols = reader.fieldnames
        ensure(bool(cols) and len(set(cols)) == len(cols), kind + "_BAD_COLUMNS")
        ensure(COLUMNS.issubset(set(cols)), kind + "_REQUIRED_COLUMNS_MISSING")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BaselineError(kind + "_INVALID_UTF8_OR_CSV") from exc
    result = {}
    for row in rows:
        ensure(None not in row and None not in row.values(), kind + "_MALFORMED_ROW")
        cid = row["challenge_id"]
        ensure(bool(UUID.fullmatch(cid)), kind + "_INVALID_UUID")
        ensure(cid not in result, kind + "_DUPLICATE_UUID")
        ensure(row["asset"] == "ES", kind + "_NON_ES_ROW")
        ensure(row["official_label"].lower() in LABELS, kind + "_INVALID_LABEL")
        result[cid] = row
    return cols, result


def load_object(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BaselineError(label + "_INVALID_JSON") from exc
    ensure(isinstance(value, dict), label + "_NOT_OBJECT")
    return value


def guarded_file(root: Path, relative: str) -> Path:
    # resolve() guards symlink/junction escape to a directory outside private root.
    path = (root / relative).resolve()
    ensure(path.is_relative_to(root), "PRIVATE_PATH_ESCAPE: " + relative)
    return path


def audit(repo: Path, private_root: Path, *, baseline_only: bool = False,
          pins: dict[str, str] | None = None) -> dict:
    pins = PIN if pins is None else pins
    repo, private_root = repo.resolve(), private_root.resolve()
    ensure(repo.is_dir(), "REPO_MISSING")
    ensure(private_root.is_dir(), "PRIVATE_ROOT_MISSING")
    ensure(not private_root.is_relative_to(repo), "PRIVATE_ROOT_INSIDE_REPO")
    ensure(not repo.is_relative_to(private_root), "REPO_INSIDE_PRIVATE_ROOT")

    integrated_path = guarded_file(private_root, "m10c_integrated_post_nov13_20261114_A.json")
    settlement_path = guarded_file(private_root, "m10c_settlement_post_nov13_20261114_A.json")
    frozen_path = guarded_file(private_root, "m11_future/historical_baseline_20261114/arena_official_settlements_19rows_20261114.csv")
    adoption_path = guarded_file(private_root, "m11_future/adoption/owner_adoption_decision_20261114_v1_0.json")
    policy_path = (repo / "docs/M11_PROSPECTIVE_OPPORTUNITY_PROTOCOL_v1_0.md").resolve()
    ensure(policy_path.is_relative_to(repo), "PROTOCOL_PATH_ESCAPE")

    integrated = load_object(pinned_file(integrated_path, pins["integrated"]), "INTEGRATED")
    settlement = load_object(pinned_file(settlement_path, pins["settlement"]), "SETTLEMENT")
    frozen_raw = pinned_file(frozen_path, pins["snapshot"])
    frozen_cols, frozen = read_csv(frozen_raw, "BASELINE")
    ensure(len(frozen) == 19, "BASELINE_NOT_19_ROWS")
    ensure(Counter(x["official_label"].lower() for x in frozen.values()) ==
           Counter({"bearish": 8, "bullish": 8, "neutral": 3}), "BASELINE_LABEL_DISTRIBUTION_MISMATCH")

    # The original readiness audit is preserved and reused for its 19-record
    # historical report checks, not for its now-obsolete pinned LIVE CSV path.
    # Its fixed PIN["csv"] is the sealed historical snapshot hash.
    ensure(pins["integrated"] == legacy.PIN["integrated"] and
           pins["snapshot"] == legacy.PIN["csv"], "LEGACY_PIN_LINKAGE_MISMATCH")
    history = legacy.evaluate_historical(integrated, settlement, frozen_raw.decode("utf-8-sig"))
    ensure(history["review_flags"] == [], "HISTORICAL_REVIEW_FLAGS: " + repr(history["review_flags"]))

    owner = load_object(pinned_file(adoption_path, pins["adoption"]), "ADOPTION")
    expected_owner = {
        "record_type": "M11_OWNER_ADOPTION_DECISION_V0_1",
        "decision": "AFFIRMATIVELY_ADOPTED_BY_OWNER",
        "protocol_git_commit": COMMIT,
        "study_design": "ORIGINAL_150_TOTAL_DESCRIPTIVE",
        "protocol_sha256": pins["protocol"],
        "approved_effective_at_utc": "2026-11-15T12:00:00Z",
        "total_accepted_forecast_target": 150,
        "historical_accepted_evidenced_checkpoint": 19,
        "additional_accepted_forecasts_at_checkpoint": 131,
        "accepted_forecast_phase_lengths": [30, 60, 60],
        "historical_19_count_in_descriptive_target": True,
        "historical_19_prospectively_enrolled": False,
        "prospective_opportunities_enrolled_at_recording": 0,
    }
    for key, val in expected_owner.items():
        ensure(key in owner and type(owner[key]) is type(val) and owner[key] == val,
               "ADOPTION_FIELD_MISMATCH: " + key)
    pinned_file(policy_path, pins["protocol"])

    live = None
    if not baseline_only:
        live_path = (repo / "data/processed/es_arena_research_v0_1/arena_official_settlements.csv").resolve()
        ensure(live_path.is_relative_to(repo), "LIVE_PATH_ESCAPE")
        ensure(live_path.is_file(), "LIVE_CSV_MISSING")
        live_raw = live_path.read_bytes()
        live_cols, live_rows = read_csv(live_raw, "LIVE")
        ensure(set(frozen_cols).issubset(set(live_cols)), "LIVE_REMOVED_BASELINE_COLUMN")
        for cid, base in frozen.items():
            ensure(cid in live_rows, "HISTORICAL_ID_REMOVED_FROM_LIVE: " + cid)
            for column in frozen_cols:
                ensure(live_rows[cid][column] == base[column],
                       "HISTORICAL_CELL_MUTATED_IN_LIVE: " + cid + "/" + column)
        additions = set(live_rows) - set(frozen)
        live = {
            "physical_sha256": sha256(live_raw),
            "total_official_settlement_rows": len(live_rows),
            "historical_19_rows_present_unchanged": True,
            "additional_live_settlement_rows_unclassified": len(additions),
            "additional_live_ids_sha256_sorted": sha256(("\n".join(sorted(additions)) + "\n").encode("utf-8")) if additions else None,
            "additional_rows_are_not_automatically_accepted_forecasts": True,
            "additional_rows_are_not_automatically_prospective_opportunities": True,
        }

    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "read_only": True,
        "baseline_snapshot_physical_sha256": pins["snapshot"],
        "baseline_settlement_rows": 19,
        "baseline_label_counts": dict(sorted(Counter(x["official_label"].lower() for x in frozen.values()).items())),
        "historical_cross_source_review_flags": [],
        "historical_accepted_evidenced_descriptive_checkpoint": 19,
        "historical_eligible_opportunity_census_certified": False,
        "historical_final_scored_revision_certified": False,
        "original_notebook12_first_30_accepted_unchanged": True,
        "adoption_record_physical_sha256": pins["adoption"],
        "owner_decision": "AFFIRMATIVELY_ADOPTED_BY_OWNER",
        "approved_protocol_effective_utc": owner["approved_effective_at_utc"],
        "independent_adoption_timestamp_certified": False,
        "research_design": owner["study_design"],
        "accepted_forecast_target": 150,
        "additional_accepted_forecasts_verified_by_this_reader": 0,
        "prospective_opportunities_enrolled_by_this_reader": 0,
        "live_csv": live,
        "files_written": 0,
        "network_calls": 0,
        "forecast_submissions": 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--private-root", required=True, type=Path)
    p.add_argument("--baseline-only", action="store_true", help="Do not inspect mutable live CSV")
    p.add_argument("--json", action="store_true", help="Print full report to stdout")
    a = p.parse_args(argv)
    try:
        report = audit(a.repo, a.private_root, baseline_only=a.baseline_only)
    except (BaselineError, OSError, UnicodeError, ValueError, KeyError) as exc:
        print("M11_STAGE3_BASELINE: ERROR", type(exc).__name__, str(exc), file=sys.stderr)
        print("FILES_WRITTEN: 0 | NETWORK_CALLS: 0 | FORECAST_SUBMISSIONS: 0", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("M11_STAGE3_IMMUTABLE_BASELINE: PASS")
        print("HISTORICAL_19_SETTLEMENTS: VERIFIED")
        print("BASELINE_SHA256:", report["baseline_snapshot_physical_sha256"])
        print("ADOPTION_DECISION: AFFIRMATIVELY_ADOPTED_BY_OWNER")
        print("EFFECTIVE_UTC:", report["approved_protocol_effective_utc"])
        print("ACCEPTED_TOTAL_CHECKPOINT: 19/150 | NEW_ACCEPTED_CERTIFIED_BY_READER: 0")
        if report["live_csv"] is None:
            print("LIVE_CSV: NOT_CHECKED (baseline-only)")
        else:
            print("LIVE_CSV_HISTORICAL_ROWS: 19 UNCHANGED")
            print("LIVE_TOTAL_SETTLEMENT_ROWS:", report["live_csv"]["total_official_settlement_rows"])
            print("ADDITIONAL_LIVE_ROWS_UNCLASSIFIED:", report["live_csv"]["additional_live_settlement_rows_unclassified"])
        print("HISTORICAL_CENSUS_CERTIFIED: NO | FINAL_SCORED_REVISION_CERTIFIED: NO")
        print("FILES_WRITTEN: 0 | NETWORK_CALLS: 0 | FORECAST_SUBMISSIONS: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
