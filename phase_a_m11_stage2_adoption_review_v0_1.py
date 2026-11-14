"""M11 Stage 2: OFFLINE candidate adoption-record review, NEVER automatic adoption.

An adoption record or published Git commit is evidence for manual scrutiny, not
independent proof of UTC publication or programme eligibility on this machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import phase_a_m11_readiness_v0_1 as m11

ALLOWED_STUDY = {"FRESH_150_OPPORTUNITIES", "ORIGINAL_150_TOTAL_DESCRIPTIVE"}


def evaluate(record: dict, protocol_file: Path, protocol_hash: str, root: Path):
    problems = []
    if not isinstance(record, dict):
        return ["record_not_object"]
    if record.get("record_type") != "M11_CANDIDATE_ADOPTION_REVIEW_V0_1":
        problems.append("incorrect_record_type")
    if record.get("status") != "PROPOSED_NOT_ADOPTED":
        problems.append("automatic_or_false_adoption_claim")
    if record.get("chosen_study_design") not in ALLOWED_STUDY:
        problems.append("study_design_not_explicitly_selected")
    if record.get("chosen_study_design") == "FRESH_150_OPPORTUNITIES" and record.get("planned_phase_lengths") != [30,60,60]:
        problems.append("unapproved_phase_lengths")
    if record.get("historical_19_enrolled") is not False:
        problems.append("historical_cohort_must_not_be_enrolled")
    if record.get("original_notebook12_preserved") is not True:
        problems.append("notebook12_preservation_not_confirmed")
    if not m11.SHA.fullmatch(str(protocol_hash)) or record.get("protocol_sha256") != protocol_hash:
        problems.append("protocol_hash_pin_mismatch")
    if not protocol_file.is_file() or m11.digest(protocol_file.read_bytes()) != protocol_hash:
        problems.append("protocol_file_not_pinned")
    try:
        approval = m11.utc(record.get("owner_reviewed_at_utc"))
        effective = m11.utc(record.get("proposed_effective_at_utc"))
        if effective <= approval:
            problems.append("effective_time_not_after_owner_review")
    except m11.EvidenceError:
        approval = effective = None
        problems.append("missing_or_invalid_utc_times")
    if record.get("first_eligible_challenge_id") not in (None, ""):
        problems.append("first_future_challenge_cannot_be_claimed_in_candidate_record")
    if record.get("publication_evidence_path") or record.get("publication_evidence_sha256"):
        source = record.get("publication_evidence_path")
        pin = record.get("publication_evidence_sha256")
        if not isinstance(source, str) or not m11.SHA.fullmatch(str(pin)):
            problems.append("invalid_publication_reference")
        else:
            filepath = (root / source).resolve() if not Path(source).is_absolute() else Path(source).resolve()
            if not filepath.is_relative_to(root.resolve()) or not filepath.is_file() or m11.digest(filepath.read_bytes()) != pin:
                problems.append("publication_evidence_missing_or_hash_mismatch")
    # No boolean/screenshot/hash supplied by caller can assert independent certification.
    if record.get("independent_publication_time_certified") is not False:
        problems.append("independent_timestamp_claim_must_be_manual_separate_review")
    return sorted(set(problems))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--private-root", required=True, type=Path)
    p.add_argument("--expected-candidate-sha256", required=True)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--expected-protocol-sha256", required=True)
    a = p.parse_args(argv)
    try:
        path = a.candidate.resolve()
        root = a.private_root.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Candidate adoption record must be in private root")
        raw = m11.pinned_bytes(path, a.expected_candidate_sha256)
        record = m11.load_json_object(raw)
        problems = evaluate(record, a.protocol.resolve(), a.expected_protocol_sha256, root)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print("ADOPTION_REVIEW_ERROR:", type(exc).__name__, str(exc), file=sys.stderr)
        return 2
    print("CANDIDATE_ADOPTION_REVIEW:", "PASS" if not problems else "REVIEW")
    print("REVIEW_FLAGS:", json.dumps(problems))
    print("RESEARCH_STATUS: NOT_ADOPTED | ENROLLED: 0 | EXTERNAL_TIME_CERTIFIED: NO")
    print("NETWORK_CALLS: 0 | FILES_WRITTEN: 0 | FORECASTS_SUBMITTED: 0")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
