"""M11 Stage 2: offline, private, append-only CANDIDATE opportunity journal.

No forecasts, network access, Git operations, automatic adoption or enrollment.
All write actions require --apply. The journal records revisions, not one line per
opportunity. The separately generated candidate snapshot has one row per ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import phase_a_m11_readiness_v0_1 as m11

SCHEMA = "phase_a_m11_stage2_event_v0_1"
LOCKED = (
    "challenge_id", "asset", "challenge_type", "contract_symbol", "open_at_utc",
    "deadline_utc", "opportunity_first_evidence_path", "opportunity_first_evidence_sha256",
)

class JournalError(ValueError):
    pass


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def ensure_under_private(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise JournalError("File path must be inside the designated private evidence root")
    return path


def historical_ids(private_root: Path) -> set[str]:
    path = private_root / "m10c_integrated_post_nov13_20261114_A.json"
    data = m11.load_json_object(m11.pinned_bytes(path, m11.PIN["integrated"]))
    if data.get("retained_challenges") != 19 or data.get("review_flags") != []:
        raise JournalError("Historical M10C baseline is not valid")
    ids = {r["challenge_id"] for r in data["records"]}
    if len(ids) != 19:
        raise JournalError("Historical M10C identities incomplete")
    return ids


def validate_row(row: dict, root: Path, excluded: set[str]):
    if not isinstance(row, dict):
        raise JournalError("Each event row must be a complete JSON object")
    output = m11.validate_ledger(canonical(row) + b"\n", root)
    if output["review_issues"] or output["candidate_rows"] != 1:
        raise JournalError("M11 candidate row invalid: " + json.dumps(output["review_issues"], sort_keys=True))
    if row["challenge_id"] in excluded:
        raise JournalError("Historical M10C challenge cannot enter prospective candidate journal")
    first = m11.utc(row["first_observed_at_utc"])
    opening = m11.utc(row["open_at_utc"])
    # Seeing a challenge before it opened is logically inconsistent; no observation inferred.
    if first < opening:
        raise JournalError("First observation precedes recorded opening")
    # A predeadline discovery claim must really precede the deadline (already M11 validated).
    if row["availability_status"] == "CONFIRMED_OPEN" and row["provenance_level"] == "SOURCE_DISCOVERY_PREDEADLINE" and first >= m11.utc(row["deadline_utc"]):
        raise JournalError("Confirmed predeadline discovery must precede cutoff")
    return row


def parse_journal(raw: bytes, root: Path, excluded: set[str]):
    events = []
    state = {}
    parent = "GENESIS"
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise JournalError(f"Blank journal line {lineno}")
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise JournalError(f"Invalid journal JSON line {lineno}") from exc
        if not isinstance(event, dict):
            raise JournalError(f"Malformed event line {lineno}")
        recorded = event.get("event_sha256")
        core = {k: v for k, v in event.items() if k != "event_sha256"}
        if (event.get("schema") != SCHEMA or event.get("sequence") != lineno or
                event.get("previous_sha256") != parent or not isinstance(recorded, str) or
                sha(canonical(core)) != recorded):
            raise JournalError(f"Journal hash chain failed at line {lineno}")
        try:
            m11.utc(event.get("event_recorded_at_utc"))
        except m11.EvidenceError as exc:
            raise JournalError(f"Invalid journal time at line {lineno}") from exc
        row = validate_row(event.get("candidate_row"), root, excluded)
        cid = row["challenge_id"]
        op = event.get("operation")
        if op == "CREATE":
            if cid in state:
                raise JournalError(f"Duplicate CREATE for challenge {cid}")
            if event.get("reason") not in (None, ""):
                raise JournalError("CREATE reason must be omitted")
        elif op == "UPDATE":
            if cid not in state:
                raise JournalError("UPDATE without initial CREATE")
            if not isinstance(event.get("reason"), str) or len(event["reason"].strip()) < 12:
                raise JournalError("UPDATE requires a substantive change reason")
            previous = state[cid]
            if any(previous.get(k) != row.get(k) for k in LOCKED):
                raise JournalError("An UPDATE attempted to change immutable source/challenge identity")
            if previous == row:
                raise JournalError("Redundant UPDATE with no changed fields")
        else:
            raise JournalError("Unknown journal operation")
        state[cid] = row
        events.append(event)
        parent = recorded
    return state, events, parent


def check_journal_pin(raw, expected):
    if expected == "NONE":
        if raw:
            raise JournalError("Expected new journal but existing content found")
    elif not m11.SHA.fullmatch(str(expected)) or sha(raw) != expected:
        raise JournalError("Journal physical SHA-256 does not match expected value")


def load_root_journal(root, journal):
    journal = ensure_under_private(journal, root)
    raw = journal.read_bytes() if journal.exists() else b""
    return journal, raw


def append_event(root, journal, event_file, expected_event_hash, expected_journal_hash, apply=False):
    root = root.resolve()
    journal, old = load_root_journal(root, journal)
    event_file = ensure_under_private(event_file, root)
    source = event_file.read_bytes()
    if sha(source) != expected_event_hash:
        raise JournalError("Input event file SHA-256 mismatch")
    check_journal_pin(old, expected_journal_hash)
    excluded = historical_ids(root)
    state, events, last = parse_journal(old, root, excluded)
    info = json.loads(source.decode("utf-8-sig"))
    if not isinstance(info, dict) or set(info) - {"operation", "candidate_row", "reason"}:
        raise JournalError("Input event JSON must contain only operation, candidate_row, reason")
    op = info.get("operation")
    row = validate_row(info.get("candidate_row"), root, excluded)
    cid = row["challenge_id"]
    if op == "CREATE":
        if cid in state:
            raise JournalError("Challenge already exists; submit an UPDATE event with reason")
    elif op == "UPDATE":
        if cid not in state:
            raise JournalError("Cannot UPDATE a challenge that does not exist")
        if any(state[cid].get(k) != row.get(k) for k in LOCKED):
            raise JournalError("UPDATE cannot modify immutable challenge identity or first evidence")
        if row == state[cid]:
            raise JournalError("UPDATE makes no change")
        if not isinstance(info.get("reason"), str) or len(info["reason"].strip()) < 12:
            raise JournalError("UPDATE needs a reason of at least 12 characters")
    else:
        raise JournalError("Event operation must be CREATE or UPDATE")
    core = {"schema": SCHEMA, "sequence": len(events)+1, "previous_sha256": last,
            "event_recorded_at_utc": now_utc(), "operation": op, "candidate_row": row,
            "reason": info.get("reason", "")}
    new_event = dict(core, event_sha256=sha(canonical(core)))
    new_raw = old + canonical(new_event) + b"\n"
    parse_journal(new_raw, root, excluded)
    result = {"status": "CANDIDATE_ONLY_NOT_ADOPTED", "event_number": len(events)+1,
              "challenge_id": cid, "operation": op, "prior_journal_sha256": sha(old),
              "new_journal_sha256": sha(new_raw), "files_written": 0}
    if apply:
        journal.parent.mkdir(parents=True, exist_ok=True)
        # Fail closed on concurrent writers using an exclusive short-lived lock.
        lock = journal.with_name(journal.name + ".lock")
        fd = None
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            fd = None
            current = journal.read_bytes() if journal.exists() else b""
            if current != old:
                raise JournalError("Journal changed during operation; refuse append")
            with journal.open("ab") as writer:
                written = writer.write(canonical(new_event) + b"\n")
                writer.flush()
                os.fsync(writer.fileno())
            if written != len(canonical(new_event)) + 1 or sha(journal.read_bytes()) != sha(new_raw):
                raise JournalError("Journal write verification failed; preserve file for review")
            result["files_written"] = 1
        finally:
            if fd is not None:
                os.close(fd)
            if lock.exists():
                lock.unlink()
    return result


def audit(root, journal, expected):
    root = root.resolve()
    journal, raw = load_root_journal(root, journal)
    check_journal_pin(raw, expected)
    excluded = historical_ids(root)
    state, events, last = parse_journal(raw, root, excluded)
    return {"status": "CANDIDATE_ONLY_NOT_ADOPTED", "events": len(events),
            "distinct_challenges": len(state), "head_event_sha256": last,
            "journal_sha256": sha(raw), "candidate_rows": len(state),
            "enrolled_to_150": 0, "phase_ordinals_assigned": 0,
            "rows": [state[cid] for cid in sorted(state)], "files_written": 0}


def snapshot(root, journal, expected, destination, apply=False):
    result = audit(root, journal, expected)
    path = ensure_under_private(destination, root)
    if path.exists():
        raise JournalError("Refuse to overwrite existing snapshot")
    content = b"".join(canonical(row)+b"\n" for row in result["rows"])
    check = m11.validate_ledger(content, root)
    if check["review_issues"]:
        raise JournalError("Snapshot validator found issues")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if sha(path.read_bytes()) != sha(content):
            raise JournalError("Snapshot write verification failed")
    return {"status": "CANDIDATE_ONLY_NOT_ADOPTED", "rows": len(result["rows"]),
            "snapshot_sha256": sha(content), "files_written": int(apply),
            "original_journal_sha256": result["journal_sha256"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("audit", "append", "snapshot"))
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--expected-journal-sha256", required=True, help="Current exact physical hash or NONE for new journal")
    p.add_argument("--event-file", type=Path)
    p.add_argument("--expected-event-sha256")
    p.add_argument("--snapshot-path", type=Path)
    p.add_argument("--apply", action="store_true", help="Explicit opt-in to append a journal event or create a new snapshot")
    args = p.parse_args(argv)
    try:
        if args.mode == "audit":
            if args.apply:
                raise JournalError("Audit is always read-only")
            result = audit(args.private_root, args.journal, args.expected_journal_sha256)
            result.pop("rows")
        elif args.mode == "append":
            if not args.event_file or not args.expected_event_sha256:
                raise JournalError("append needs --event-file and --expected-event-sha256")
            result = append_event(args.private_root, args.journal, args.event_file,
                                  args.expected_event_sha256, args.expected_journal_sha256, args.apply)
        else:
            if not args.snapshot_path:
                raise JournalError("snapshot needs --snapshot-path")
            result = snapshot(args.private_root, args.journal,
                              args.expected_journal_sha256, args.snapshot_path, args.apply)
    except (OSError, ValueError, KeyError, UnicodeDecodeError) as exc:
        print("M11_STAGE2_ERROR:", type(exc).__name__, str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
