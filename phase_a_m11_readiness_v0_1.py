"""Milestone 11: OFFLINE, READ-ONLY Phase A prospective-program readiness.

Pins the Nov 14 post-Nov-13 historical M10C report and official settlement CSV
and optionally validates a prospective *candidate* ledger, never adopting a
protocol or counting historical challenges in a new 150-opportunity trial.

No network, data writes, submissions, Git staging, notebook/model changes.
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
from datetime import datetime, timezone
from pathlib import Path

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
CONTRACT = re.compile(r"^ES[FGHJKMNQUVXZ][0-9]$")
PIN = {
    "integrated": "033b513f662524dc84579d9934a6593b79fd3b70eccd601116a19d82c8d1f920",
    "settlement": "3f819eb2b422e02a95df5eb271b7dadf8770ee3f52223dc0709d854dce335be9",
    "csv": "093bc2aa57c7e87f073c2ff86b9fbe7ebd3cbb0b59d0491ee7d95b8c3aee68c4",
    "m10_commit": "3c35a3a13e017b87f1723ce50eccb858eeca41f3",
}
AVAILABILITY = {"CONFIRMED_OPEN", "OBSERVED_BUT_TERMS_INCOMPLETE", "CANCELLED_AFTER_OPEN", "NEEDS_REVIEW"}
SUBMISSION = {"ACCEPTED_SCORED", "REJECTED_CONFIRMED", "ATTEMPT_AMBIGUOUS", "NO_ACCEPTED_RECORD_YET", "MISSED_VERIFIED", "UNKNOWN"}
OUTCOME = {"OFFICIALLY_SETTLED", "PENDING", "CANCELLED_NO_OUTCOME", "UNAVAILABLE", "UNRESOLVED"}
PROVENANCE = {"SOURCE_DISCOVERY_PREDEADLINE", "OFFICIAL_ACCEPTED_WITH_TIME_LIMITATION", "LEGACY_ACCEPTED", "INCOMPLETE", "UNRESOLVED"}
LABELS = {"bearish", "neutral", "bullish"}

class EvidenceError(ValueError):
    pass

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def pinned_bytes(path: Path, expected: str) -> bytes:
    if not SHA.fullmatch(expected):
        raise EvidenceError("Invalid SHA-256 pin")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise EvidenceError("Pinned source bytes changed: " + path.name)
    return raw

def utc(value):
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("UTC timestamp missing")
    try:
        v = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("Invalid ISO datetime") from exc
    if v.tzinfo is None or v.utcoffset() is None or v.utcoffset().total_seconds() != 0:
        raise EvidenceError("Timestamp must have explicit UTC offset")
    return v.astimezone(timezone.utc)

def load_json_object(raw: bytes):
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise EvidenceError("Expected JSON object")
    return data

def evaluate_historical(integrated: dict, settlement: dict, csv_text: str):
    problems = []
    if integrated.get("review_flags") != [] or settlement.get("review_flags") != []:
        problems.append("M10C source has review flags")
    if integrated.get("retained_challenges") != 19 or integrated.get("revision_records") != 19:
        problems.append("Unexpected historical integrated cohort")
    if settlement.get("challenge_count") != 19 or settlement.get("settlement_rows_in_observed_cohort") != 19:
        problems.append("Unexpected 19-settlement count")
    if settlement.get("settlement_rows_outside_observed_cohort") != 0:
        problems.append("Unexpected settlement outside observed cohort")
    for name in ("model_execution_certified", "final_scored_revision_certified", "historical_eligible_opportunity_census_certified", "phase_a_protocol_adopted"):
        if integrated.get(name) is not False:
            problems.append("Integrated report incorrectly certifies " + name)
    for name in ("final_scored_revision_certified", "model_execution_certified", "historical_opportunity_census_certified", "phase_a_150_opportunity_protocol_adopted"):
        if settlement.get(name) is not False:
            problems.append("Settlement report incorrectly certifies " + name)
    if settlement.get("sources", {}).get("m10c_integrated_physical_sha256") != PIN["integrated"]:
        problems.append("Integrated source SHA mismatch")
    if settlement.get("sources", {}).get("local_official_settlements_csv_physical_sha256") != PIN["csv"]:
        problems.append("Settlement CSV source SHA mismatch")
    a=integrated.get("records", [])
    b=settlement.get("records", [])
    if not isinstance(a,list) or not isinstance(b,list) or len(a)!=19 or len(b)!=19:
        problems.append("M10C record cardinality mismatch")
        a=[]; b=[]
    ia=[r.get("challenge_id") for r in a if isinstance(r,dict)]
    ib=[r.get("challenge_id") for r in b if isinstance(r,dict)]
    if len(set(ia))!=19 or set(ia)!=set(ib):
        problems.append("M10C record identity mismatch")
    if any(r.get("official_settlement_row_present") is not True for r in b):
        problems.append("Some settlement rows missing in M10C report")
    if settlement.get("counts", {}).get("deadline_DEADLINES_MATCH_AS_RECORDED") != 19:
        problems.append("Not all recorded deadlines match")
    csv_rows=list(csv.DictReader(io.StringIO(csv_text,newline="")))
    ci=[r.get("challenge_id") for r in csv_rows]
    if len(ci)!=19 or len(set(ci))!=19 or set(ci)!=set(ia):
        problems.append("Official CSV identifiers/count disagree with integrated report")
    if any((r.get("official_label") or "").strip().lower() not in LABELS for r in csv_rows):
        problems.append("CSV has missing/invalid label")
    if csv_rows and Counter((r.get("official_label") or "").lower() for r in csv_rows) != Counter({"bearish":8,"bullish":8,"neutral":3}):
        problems.append("Recorded historical label distribution changed")
    return {"historical_observed_challenges":19,"historical_settlement_rows":len(csv_rows),"review_flags":sorted(set(problems)),
            "historical_eligible_census_certified":False, "historical_final_scored_revision_certified":False}

def validate_ledger(raw: bytes, source_root: Path):
    """Validate future *candidate* ledger only; never infer adoption/ordinals.

    Ledger is JSONL. Captured source files and hashes must be byte-identical.
    """
    entries=[]
    try:
        for index,line in enumerate(raw.decode("utf-8-sig").splitlines(),1):
            if not line.strip():
                continue
            x=json.loads(line)
            if not isinstance(x,dict):
                raise EvidenceError(f"Ledger row {index}: expected JSON object")
            entries.append(x)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise EvidenceError("Invalid JSONL candidate ledger") from exc
    issues=[]; seen=set(); accepted=0; settled=0; canceled=0; confirmed=0
    for n,x in enumerate(entries,1):
        err=[]
        cid=x.get("challenge_id")
        if not isinstance(cid,str) or not UUID.fullmatch(cid) or cid in seen:
            err.append("invalid_or_duplicate_challenge_id")
        elif cid:
            seen.add(cid)
        if x.get("asset")!="ES" or x.get("challenge_type")!="daily" or not CONTRACT.fullmatch(str(x.get("contract_symbol", ""))):
            err.append("invalid_market_scope")
        for field,allowed in (("availability_status",AVAILABILITY),("submission_status",SUBMISSION),
                              ("outcome_status",OUTCOME),("provenance_level",PROVENANCE)):
            if x.get(field) not in allowed:
                err.append("invalid_" + field)
        opening=deadline=observed=None
        for field in ("open_at_utc", "deadline_utc", "first_observed_at_utc"):
            try:
                value=utc(x.get(field))
                if field=="open_at_utc":opening=value
                elif field=="deadline_utc":deadline=value
                else:observed=value
            except EvidenceError:
                err.append("invalid_"+field)
        if opening and deadline and opening>=deadline:
            err.append("invalid_open_deadline_order")
        if observed and deadline and observed>=deadline and x.get("provenance_level")=="SOURCE_DISCOVERY_PREDEADLINE":
            err.append("postdeadline_evidence_claimed_predate")
        if x.get("availability_status")=="CONFIRMED_OPEN" and not opening:
            err.append("unsupported_confirmed_open")
        path=x.get("opportunity_first_evidence_path")
        claimed=x.get("opportunity_first_evidence_sha256")
        if not isinstance(path,str) or not path.strip() or not isinstance(claimed,str) or not SHA.fullmatch(claimed):
            err.append("missing_source_and_sha")
        else:
            file=(source_root/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
            # Prevent symlinked files or arbitrary paths outside designated evidence tree.
            if not file.is_relative_to(source_root.resolve()) or not file.is_file():
                err.append("source_not_under_private_evidence_root")
            elif digest(file.read_bytes()) != claimed:
                err.append("source_sha_mismatch")
        if x.get("submission_status")=="ACCEPTED_SCORED":
            if not SHA.fullmatch(str(x.get("accepted_receipt_sha256", ""))) or not x.get("prediction_id"):
                err.append("missing_scored_acceptance_reference")
            else:
                accepted+=1
        if x.get("submission_status")=="MISSED_VERIFIED":
            if not SHA.fullmatch(str(x.get("miss_evidence_sha256", ""))):
                err.append("missing_independent_miss_evidence")
            try:
                reviewed=utc(x.get("miss_reviewed_at_utc"))
                if deadline and reviewed<=deadline:
                    err.append("miss_declared_before_deadline")
            except EvidenceError:
                err.append("missing_utc_miss_review_time")
        if x.get("outcome_status")=="OFFICIALLY_SETTLED":
            if x.get("official_label") not in LABELS or not SHA.fullmatch(str(x.get("official_settlement_evidence_sha256", ""))):
                err.append("missing_official_settlement_reference")
            else:
                settled+=1
        if x.get("availability_status")=="CANCELLED_AFTER_OPEN" or x.get("outcome_status")=="CANCELLED_NO_OUTCOME":
            canceled+=1
            if x.get("outcome_status")=="OFFICIALLY_SETTLED":
                err.append("cancellation_outcome_conflict")
        if x.get("availability_status")=="CONFIRMED_OPEN":
            confirmed+=1
        if err:
            issues.append({"row":n,"challenge_id":cid if isinstance(cid,str) and UUID.fullmatch(cid) else None,"issues":sorted(set(err))})
    return {"candidate_rows":len(entries),"candidate_confirmed_open":confirmed,"candidate_accepted_scored":accepted,
            "candidate_officially_settled":settled,"candidate_cancelled":canceled,
            "review_issues":issues,"candidate_ledger_physical_sha256":digest(raw),
            "automatically_adopted":False,"verified_programme_ordinal_assignments":0,
            "new_150_opportunity_trial_enrolled_count":0}

def run(repo: Path, private_root: Path, ledger_path=None, ledger_sha=None):
    repo=repo.resolve(); private_root=private_root.resolve()
    files={"integrated":private_root/"m10c_integrated_post_nov13_20261114_A.json",
           "settlement":private_root/"m10c_settlement_post_nov13_20261114_A.json",
           "csv":repo/"data/processed/es_arena_research_v0_1/arena_official_settlements.csv"}
    for name,path in files.items():
        if not path.is_file():
            raise EvidenceError("Required historical checkpoint missing: " + path.name)
    integrated=load_json_object(pinned_bytes(files["integrated"],PIN["integrated"]))
    settlement=load_json_object(pinned_bytes(files["settlement"],PIN["settlement"]))
    csv_data=pinned_bytes(files["csv"],PIN["csv"]).decode("utf-8-sig")
    hist=evaluate_historical(integrated,settlement,csv_data)
    if ledger_path is None:
        ledger={"candidate_rows":0,"candidate_confirmed_open":0,"candidate_accepted_scored":0,
                "candidate_officially_settled":0,"candidate_cancelled":0,
                "review_issues":[],"candidate_ledger_physical_sha256":None,
                "automatically_adopted":False,"verified_programme_ordinal_assignments":0,
                "new_150_opportunity_trial_enrolled_count":0}
    else:
        path=Path(ledger_path).resolve()
        if not path.is_relative_to(private_root) or not path.is_file():
            raise EvidenceError("Candidate ledger must exist inside private root")
        if not ledger_sha:
            raise EvidenceError("Explicit --expected-ledger-sha256 required")
        ledger=validate_ledger(pinned_bytes(path,ledger_sha),private_root)
    return {"schema_version":"phase_a_m11_readiness_v0_1","read_only":True,
            "m10_reference_commit_from_prior_user_verification":PIN["m10_commit"],
            "m10_checkpoint_source_sha256":PIN,
            "historical":hist,"prospective_candidate_ledger":ledger,
            "original_notebook12_cohort":"FIRST_30_ACCEPTED_FORECASTS_UNCHANGED",
            "proposed_phase_lengths":[30,60,60],
            "candidate_protocol_status":"DRAFT_NOT_ADOPTED",
            "historical_phase_ordinal_certified":False,
            "official_challenge_opening_census_complete":False,
            "independent_prospective_publication_time_verified":False,
            "phase_b_boundary_certified":False,
            "no_network_calls":True,"no_files_written":True,"no_forecast_submissions":True}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--private-root",required=True,type=Path)
    p.add_argument("--ledger",type=Path,default=None,help="Optional PRIVATE proposed future JSONL ledger; never enrolls")
    p.add_argument("--expected-ledger-sha256",default=None)
    p.add_argument("--json",action="store_true")
    a=p.parse_args(argv)
    try:
        rep=run(a.repo,a.private_root,a.ledger,a.expected_ledger_sha256)
    except (OSError,ValueError,UnicodeDecodeError,json.JSONDecodeError) as exc:
        print("M11 READINESS: ERROR",type(exc).__name__,str(exc),file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(rep,sort_keys=True,indent=2,ensure_ascii=False))
    else:
        print("M11 PROSPECTIVE READINESS: " + ("OK" if not rep["historical"]["review_flags"] and not rep["prospective_candidate_ledger"]["review_issues"] else "REVIEW"))
        print("M10_HISTORICAL_OBSERVED_CHALLENGES:",rep["historical"]["historical_observed_challenges"])
        print("M10_HISTORICAL_SETTLEMENTS:",rep["historical"]["historical_settlement_rows"])
        print("CANDIDATE_LEDGER_ROWS:",rep["prospective_candidate_ledger"]["candidate_rows"])
        print("REVIEW_FLAGS:",rep["historical"]["review_flags"])
        print("LEDGER_REVIEW_ISSUES:",rep["prospective_candidate_ledger"]["review_issues"])
        print("150_OPPORTUNITY_PROTOCOL: DRAFT_NOT_ADOPTED")
        print("NEW_150_TRIAL_OPPORTUNITIES_ENROLLED: 0 | PHASE_BOUNDARY_CERTIFIED: NO")
        print("NETWORK_CALLS: 0 | FILES_WRITTEN: 0 | SUBMISSIONS: 0")
    return 0 if not rep["historical"]["review_flags"] and not rep["prospective_candidate_ledger"]["review_issues"] else 1

if __name__=="__main__":
    sys.exit(main())
