"""Read-only Phase A opportunity evidence audit for Headline Arena ES daily.

Inputs are local, Git-ignored research evidence. No network, model execution,
submissions, or modification of inputs. Never infer challenges from weekdays.

Commands:
  python phase_a_opportunity_audit_v0_1.py audit
  python phase_a_opportunity_audit_v0_1.py audit --json-summary

Original Notebook 12 first-30-ACCEPTED definition is NOT changed by this tool.
The separately proposed eligible-opportunity protocol requires a dated amendment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "data/processed/es_arena_research_v0_1"
LIVE = ROOT / "data/processed/es_arena_live"
CANDIDATE = RESEARCH / "arena_opportunity_candidate_20261112.csv"
DISCOVERY = LIVE / "discovery"
HISTORICAL = LIVE / "active_challenges_inspection_v0_1.json"
OFFICIAL = RESEARCH / "arena_official_settlements.csv"
ORIGINAL_FIRST_ID = "118cefc4-96a8-4e47-b617-73af53c67d03"
CANDIDATE_SHA256 = "620816c8161312508e7220c85a064776f60e6af4ae4acbeb1fd5a846b851a575"
HISTORICAL_SHA256 = "d53ebca9bff1125a269f62a0965efb839f3833bc3d73edc0927f281f4d5afd47"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        t = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None or t.utcoffset() is None:
        return None
    return t.astimezone(timezone.utc)


def header_date(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        t = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if t.tzinfo is None or t.utcoffset() is None:
        return None
    return t.astimezone(timezone.utc)


def unpack_challenges(body: object) -> list[dict]:
    if isinstance(body, list):
        entries = body
    elif isinstance(body, dict):
        entries = body.get("challenges", body.get("items"))
    else:
        raise ValueError("Challenge feed root must be list or object")
    if not isinstance(entries, list):
        raise ValueError("Challenge list missing or invalid")
    challenges = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Non-object feed entry")
        c = entry.get("challenge", entry)
        if not isinstance(c, dict):
            raise ValueError("Non-object nested challenge")
        challenges.append(c)
    return challenges


def is_es_daily(c: dict) -> bool:
    scope = c.get("asset") or c.get("scope_key") or c.get("instrument_root")
    return str(scope).upper() == "ES" and str(c.get("challenge_type", "")).lower() == "daily"


def get_id(c: dict) -> str | None:
    raw = c.get("id") or c.get("challenge_id")
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def load_accepted(path: Path) -> dict[str, dict]:
    if sha256(path) != CANDIDATE_SHA256:
        raise ValueError("Immutable November 12 candidate hash differs; do not replace silently")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18:
        raise ValueError("Expected exactly 18 November 12 candidate rows")
    result = {}
    for row in rows:
        cid = row.get("challenge_id")
        day = row.get("settlement_date")
        if not cid or cid in result or not day or row.get("opportunity_status") != "confirmed_accepted":
            raise ValueError("Invalid or duplicate accepted candidate record")
        result[cid] = row
    if ORIGINAL_FIRST_ID not in result:
        raise ValueError("Original October 20 challenge missing")
    return result


def load_official(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    official = {}
    for row in rows:
        cid = row.get("challenge_id")
        if not cid or cid in official:
            raise ValueError("Official settlement has missing or duplicate challenge ID")
        official[cid] = row
    return official


def discovery_observations(*, discovery_dir: Path, historical_path: Path, root: Path = ROOT) -> tuple[list[dict], list[dict]]:
    """Return ES observation summaries and file-level integrity/errors.

    A challenge seen in raw JSON is reported even when the logger marked its
    metadata incomplete, but such an observation is not upgraded to confirmed.
    A postdeadline response is NOT positive predeadline opportunity evidence.
    """
    observations = []
    archive_checks = []
    if historical_path.is_file():
        h = sha256(historical_path)
        if h != HISTORICAL_SHA256:
            raise ValueError("Historical October 20 inspection hash differs")
        body = json.loads(historical_path.read_text(encoding="utf-8"))
        for c in unpack_challenges(body):
            if is_es_daily(c):
                observations.append({
                    "challenge_id": get_id(c),
                    "source_file": str(historical_path.relative_to(root)).replace("\\", "/"),
                    "evidence_kind": "preserved_historical_active_inspection",
                    "collection_status": "historical_snapshot_unknown_capture_time",
                    "server_date_at_capture": None,
                    "deadline": utc_timestamp(c.get("deadline_utc") or c.get("deadline_at") or c.get("deadline")),
                    "created": utc_timestamp(c.get("created_at")),
                    "contract": c.get("contract_symbol"),
                    "challenge_status": c.get("status"),
                    "predeadline_server_date": False,
                    "timing_note": "historical source has no independently observed retrieval-time header",
                })
        archive_checks.append({"source": historical_path.name, "status": "HASH_VERIFIED", "sha256": h})

    for path in sorted(discovery_dir.glob("arena_discovery_*.json")) if discovery_dir.is_dir() else []:
        relative = str(path.relative_to(root)).replace("\\", "/")
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            status = blob.get("status")
            http = blob.get("http_status")
            date = header_date(blob.get("server_date_header"))
            completed = utc_timestamp(blob.get("record_completed_at_utc"))
            raw_body = blob.get("response_json")
            if status == "success" and (http != 200 or raw_body is None):
                raise ValueError("Success record lacking HTTP 200 or response_json")
            if raw_body is None:
                archive_checks.append({"source":relative, "status":f"NO_DECODED_FEED_{status}", "sha256":sha256(path)})
                continue
            challenges = unpack_challenges(raw_body)
            ids = [get_id(c) for c in challenges]
            if not all(ids) or len(ids) != len(set(ids)):
                raise ValueError("Duplicate or invalid challenge IDs")
            archive_checks.append({"source":relative, "status":f"PARSED_{status}", "sha256":sha256(path), "feed_count":len(challenges)})
            for c in challenges:
                if not is_es_daily(c):
                    continue
                dl = utc_timestamp(c.get("deadline_utc") or c.get("deadline_at") or c.get("deadline"))
                valid = status == "success" and http == 200 and dl is not None
                in_time = bool(valid and date is not None and date < dl and completed is not None)
                observations.append({
                    "challenge_id": get_id(c),
                    "source_file": relative,
                    "evidence_kind": "authenticated_active_feed",
                    "collection_status": status,
                    "server_date_at_capture": date,
                    "deadline": dl,
                    "created": utc_timestamp(c.get("created_at")),
                    "contract": c.get("contract_symbol"),
                    "challenge_status": c.get("status"),
                    "predeadline_server_date": in_time,
                    "timing_note": ("server Date before stated deadline; not a signed proof" if in_time
                                    else "not independently demonstrated predeadline by this capture"),
                })
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            archive_checks.append({"source":relative, "status":"REVIEW_ERROR", "error_type":type(exc).__name__, "sha256":sha256(path)})
    return observations, archive_checks


def build_report(root: Path = ROOT) -> dict:
    """Build read-only evidence summary. `root` supports offline fixtures."""
    candidate = root / "data/processed/es_arena_research_v0_1/arena_opportunity_candidate_20261112.csv"
    live = root / "data/processed/es_arena_live"
    # load_accepted uses pinned hash by design. For synthetic tests, use
    # mocked module hash, not a changed source file in real operation.
    accepted = load_accepted(candidate)
    official = load_official(root / "data/processed/es_arena_research_v0_1/arena_official_settlements.csv")
    observations, archive_checks = discovery_observations(
        discovery_dir=live / "discovery", historical_path=live / "active_challenges_inspection_v0_1.json", root=root
    )
    by_id = defaultdict(list)
    for obs in observations:
        if obs["challenge_id"]:
            by_id[obs["challenge_id"]].append(obs)
    records = []
    for cid, row in sorted(accepted.items(), key=lambda kv: (kv[1]["settlement_date"],kv[0])):
        ev = by_id.get(cid, [])
        records.append({
            "challenge_id":cid,
            "settlement_date":row["settlement_date"],
            "record_status":"ACCEPTED_DOCUMENTED",
            "official_settled_in_local_csv":cid in official,
            "discovery_observations":len(ev),
            "historical_discovery_evidenced":any(x["evidence_kind"] == "preserved_historical_active_inspection" for x in ev),
            "predeadline_server_date_discovery_evidenced":any(x["predeadline_server_date"] for x in ev),
            "discovery_paths":[x["source_file"] for x in ev],
        })
    for cid, ev in sorted(by_id.items()):
        if cid in accepted:
            continue
        records.append({
            "challenge_id":cid,
            "settlement_date":None,  # never infer session date from creation/deadline
            "record_status":"DISCOVERED_NOT_IN_ACCEPTED_CANDIDATE_REVIEW_REQUIRED",
            "official_settled_in_local_csv":cid in official,
            "discovery_observations":len(ev),
            "historical_discovery_evidenced":any(x["evidence_kind"] == "preserved_historical_active_inspection" for x in ev),
            "predeadline_server_date_discovery_evidenced":any(x["predeadline_server_date"] for x in ev),
            "discovery_paths":[x["source_file"] for x in ev],
        })
    unmatched_official = sorted(set(official)-{x["challenge_id"] for x in records})
    statuses = Counter(x["record_status"] for x in records)
    return {
        "protocol_version":"phase_a_opportunity_audit_v0_1",
        "original_nb12_cohort":"first 30 accepted forecasts; not changed",
        "proposed_opportunity_cohort":"first 30 eligible ES challenge opportunities; needs dated documented amendment",
        "candidate_sha256":sha256(candidate),
        "accepted_candidate_count":len(accepted),
        "official_local_settlement_count_in_candidate":sum(r["official_settled_in_local_csv"] for r in records if r["record_status"]=="ACCEPTED_DOCUMENTED"),
        "record_status_counts":dict(statuses),
        "positive_discovery_with_server_predeadline_count":sum(r["predeadline_server_date_discovery_evidenced"] for r in records),
        "historical_inspection_matched_accepted_count":sum(r["historical_discovery_evidenced"] and r["record_status"]=="ACCEPTED_DOCUMENTED" for r in records),
        "unmatched_official_challenge_ids":unmatched_official,
        "archive_checks":archive_checks,
        "records":records,
        "limitations":[
            "No complete historical active-challenge inventory established.",
            "Absence of a challenge in a snapshot is not proof no opportunity existed.",
            "No weekdays are inferred as automatically eligible, missed, cancelled, or successful.",
            "Discovery server Date is an unsigned header; it does not certify acceptance time.",
            "A discovered ID lacking accepted evidence is unresolved, not automatically a failed or missed submission.",
            "The November 12 candidate is a frozen accepted-record snapshot, not an expanding opportunity ledger.",
        ],
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["audit"])
    p.add_argument("--json-summary",action="store_true")
    args=p.parse_args()
    report=build_report()
    if args.json_summary:
        print(json.dumps(report,indent=2,ensure_ascii=False))
        return
    print("PHASE A OPPORTUNITY EVIDENCE AUDIT (READ ONLY)")
    print("Candidate SHA256:",report["candidate_sha256"])
    print("Accepted documented:",report["accepted_candidate_count"])
    print("Official settlements joined by challenge ID:",report["official_local_settlement_count_in_candidate"])
    print("ES challenges discovered but not in accepted candidate:",report["record_status_counts"].get("DISCOVERED_NOT_IN_ACCEPTED_CANDIDATE_REVIEW_REQUIRED",0))
    print("Accepted IDs matched to historical inspection:",report["historical_inspection_matched_accepted_count"])
    print("IDs with server-Date predeadline discovery evidence:",report["positive_discovery_with_server_predeadline_count"])
    print("Discovery archive checks:")
    for item in report["archive_checks"]:
        print(" ",item["status"],item["source"])
    if report["unmatched_official_challenge_ids"]:
        print("REVIEW unmatched official IDs:",report["unmatched_official_challenge_ids"])
    for row in report["records"]:
        if row["record_status"] != "ACCEPTED_DOCUMENTED":
            print("REVIEW discovered unmatched ID:",row["challenge_id"],row["discovery_paths"])
    print("OPPORTUNITY-CENSUS COMPLETE: NOT ESTABLISHED")
    print("ORIGINAL NB12 FIRST-30-ACCEPTED COHORT: UNCHANGED")
    print("FILES WRITTEN: 0 | NETWORK REQUESTS: 0 | PREDICTION SUBMISSIONS: 0")


if __name__=="__main__":
    main()
