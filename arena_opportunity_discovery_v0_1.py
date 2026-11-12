
"""Offline validation of preserved Arena challenge-discovery evidence.

No API requests, forecast submissions, model execution, or file writes.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INSPECTION = (
    ROOT
    / "data/processed/es_arena_live"
    / "active_challenges_inspection_v0_1.json"
)

LEDGER = (
    ROOT
    / "data/processed/es_arena_research_v0_1"
    / "arena_opportunity_candidate_20261112.csv"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamp(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: missing timestamp")

    parsed = datetime.fromisoformat(
        value.strip().replace("Z", "+00:00")
    )

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field}: timezone missing")

    return parsed.astimezone(timezone.utc)


def extract_challenges(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")

    entries = payload.get("challenges")

    if not isinstance(entries, list):
        raise ValueError("Expected challenges list")

    records = []
    seen = set()

    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {index}: not an object")

        challenge = item.get("challenge", item)

        if not isinstance(challenge, dict):
            raise ValueError(f"Entry {index}: invalid challenge")

        challenge_id = challenge.get("id")

        if not isinstance(challenge_id, str) or not challenge_id:
            raise ValueError(f"Entry {index}: missing ID")

        if challenge_id in seen:
            raise ValueError(
                f"Duplicate challenge ID: {challenge_id}"
            )

        seen.add(challenge_id)

        created = parse_timestamp(
            challenge.get("created_at"), "created_at"
        )
        deadline = parse_timestamp(
            challenge.get("deadline"), "deadline"
        )

        if deadline <= created:
            raise ValueError(
                f"{challenge_id}: invalid deadline ordering"
            )

        asset = challenge.get("asset")
        contract = challenge.get("contract_symbol")

        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(
                f"{challenge_id}: missing asset"
            )

        if not isinstance(contract, str) or not contract.strip():
            raise ValueError(
                f"{challenge_id}: missing contract symbol"
            )

        records.append({
            "challenge_id": challenge_id,
            "asset": asset.upper(),
            "contract_symbol": contract,
            "challenge_type": challenge.get("challenge_type"),
            "status": challenge.get("status"),
            "created_at_utc": created.isoformat(),
            "deadline_utc": deadline.isoformat(),
        })

    return records


def run_offline():
    assert INSPECTION.is_file(), "Inspection file missing"
    assert LEDGER.is_file(), "Candidate ledger missing"

    expected_inspection_hash = (
        "d53ebca9bff1125a269f62a0965efb839f3833bc3d73"
        "edc0927f281f4d5afd47"
    )

    expected_ledger_hash = (
        "620816c8161312508e7220c85a064776f60e6af4ae4"
        "acbeb1fd5a846b851a575"
    )

    assert sha256(INSPECTION) == expected_inspection_hash, (
        "Historical inspection hash changed"
    )
    assert sha256(LEDGER) == expected_ledger_hash, (
        "Candidate ledger hash changed"
    )

    payload = json.loads(
        INSPECTION.read_text(encoding="utf-8")
    )

    records = extract_challenges(payload)

    with LEDGER.open(
        newline="", encoding="utf-8"
    ) as stream:
        ledger_rows = list(csv.DictReader(stream))

    known_ids = {
        row["challenge_id"] for row in ledger_rows
    }

    es_records = [
        record for record in records
        if record["asset"] == "ES"
        and record["challenge_type"] == "daily"
    ]

    print("OFFLINE DISCOVERY VALIDATION")
    print("----------------------------")
    print("Inspection SHA256:", sha256(INSPECTION))
    print("Candidate ledger SHA256:", sha256(LEDGER))
    print("Total challenges:", len(records))
    print(
        "Assets:",
        dict(Counter(r["asset"] for r in records))
    )
    print("ES daily challenges:", len(es_records))

    for record in es_records:
        print()
        print("ES challenge:", record["challenge_id"])
        print("Contract:", record["contract_symbol"])
        print("Status:", record["status"])
        print("Created:", record["created_at_utc"])
        print("Deadline:", record["deadline_utc"])
        print(
            "MATCHED ACCEPTED LEDGER:",
            record["challenge_id"] in known_ids
        )

    # Regression checks against the preserved October 20 source.
    assert len(records) == 12, "Unexpected historical count"
    assert len(es_records) == 1, "Unexpected ES count"
    assert es_records[0]["challenge_id"] in known_ids
    assert es_records[0]["contract_symbol"] == "ESZ6"

    # Test duplicate rejection without touching the source file.
    duplicate_payload = {
        "challenges": [
            payload["challenges"][0],
            payload["challenges"][0],
        ]
    }

    try:
        extract_challenges(duplicate_payload)
    except ValueError as exc:
        assert "Duplicate challenge ID" in str(exc)
        print("Duplicate-ID rejection: PASS")
    else:
        raise AssertionError(
            "Duplicate IDs were not rejected"
        )

    # Test timezone rejection.
    try:
        parse_timestamp(
            "2026-10-20T14:00:00", "test_deadline"
        )
    except ValueError as exc:
        assert "timezone missing" in str(exc)
        print("Timezone rejection: PASS")
    else:
        raise AssertionError(
            "Timezone-free deadline was accepted"
        )

    print()
    print("HISTORICAL OFFLINE VALIDATION: PASS")
    print("FILES WRITTEN: 0")
    print("NETWORK REQUESTS: 0")
    print("FORECAST SUBMISSIONS: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["offline"],
        help="Validate preserved evidence without network access",
    )
    args = parser.parse_args()

    if args.mode == "offline":
        run_offline()


if __name__ == "__main__":
    main()
