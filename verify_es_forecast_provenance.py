"""Read-only ES forecast provenance audit (v1.0, 2026-11-11).

No network, joblib loading, data purchases, mutation of evidence, or forecast submissions.
The report separates server acceptance from artifact integrity and on-time evidence.
Missing evidence is INCOMPLETE, contradictory evidence is FAIL; Oct-20 legacy is flagged.

CLI: python verify_es_forecast_provenance.py --root C:\\Users\\melny\\Market-Mood-Forecasting
     python verify_es_forecast_provenance.py --root . --preflight <challenge-uuid>
     python verify_es_forecast_provenance.py --root . --json-out data/processed/es_arena_research_v0_1/provenance_reports/audit_YYYYMMDD.json
Optional reports are created exclusively, never overwritten; default is read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

EXPECTED_MODEL_SHA = "5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d"
DEPLOYMENT_ID = "ab4d5c14423352c7"
CLASSES = ("bearish", "neutral", "bullish")
FEATURES = (
    "es_return_lag1", "es_return_lag2", "es_return_lag3", "es_return_lag5",
    "es_range_pct_lag1", "es_log_volume_lag1", "es_return_mean_5",
    "es_return_mean_20", "es_vol_5", "es_vol_20",
    "es_abs_return_mean_5", "es_abs_return_mean_20",
)
FORECAST_RX = re.compile(r"^forecast_(\d{8})_([a-fA-F0-9-]{36})\.json$")
RECEIPT_RX = re.compile(r"^receipt_([a-fA-F0-9-]{36})\.json$")
SNAPSHOT_RX = re.compile(r"^es_sessions_(\d{8})_([a-f0-9]{16})\.csv$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object: {path.name}")
    return obj


def parse_date(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} missing or not YYYY-MM-DD")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as e:
        raise ValueError(f"Invalid {field}: {value}") from e


def parse_aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        v = value.strip().replace("Z", "+00:00")
        t = datetime.fromisoformat(v)
        return t.astimezone(timezone.utc) if t.tzinfo is not None and t.utcoffset() is not None else None
    except ValueError:
        return None


def bool_true(value: Any) -> bool:
    return value is True or isinstance(value, str) and value.strip().lower() == "true"


def nested(obj: dict, path: str) -> Any:
    node: Any = obj
    for component in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(component)
    return node


def find_unique(obj: dict, paths: tuple[str, ...], name: str) -> Any:
    matches = [(p, nested(obj, p)) for p in paths if nested(obj, p) is not None]
    if len({str(v) for _, v in matches}) > 1:
        raise ValueError(f"Conflicting {name}: {','.join(p for p, _ in matches)}")
    return matches[0][1] if matches else None


def repo_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Missing project-relative source/model path")
    # Accept recorded Windows separators on Linux and vice versa.
    candidate = Path(raw.replace("\\", "/"))
    if candidate.is_absolute() or re.match(r"^[a-zA-Z]:", raw):
        raise ValueError("Absolute paths in evidence are not trusted; use a project-relative path")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Path escapes repository")
    return resolved


def base_record(cid: str = "", forecast_file: str = "") -> dict:
    return {
        "challenge_id": cid, "forecast_file": forecast_file, "target_session_date": "",
        "last_eligible_session": "", "contract_symbol": "", "forecast_sha256": "",
        "model_sha256": "", "model_hash_verified": False,
        "model_manifest_verified": False, "source_sha256": "", "snapshot_hash_verified": False,
        "snapshot_manifest_verified": False, "source_review_verified": False,
        "source_cutoff_verified": False, "reviewed_audit_verified": False, "forecast_schema_verified": False,
        "receipt_file": "", "receipt_sha256": "", "receipt_forecast_hash_verified": False,
        "receipt_server_accepted": False, "accepted_by_receipt": False,
        "receipt_http_status": "", "receipt_counts_for_score": "", "submitted_at_utc": "",
        "attempt_file": "", "attempt_present": False, "attempt_hash_verified": False,
        "deadline_utc": "", "deadline_verified": False, "deadline_buffer_minutes": None,
        "prepared_at_utc": "", "p_bearish": None, "p_neutral": None, "p_bullish": None,
        "predicted_label": "", "confidence": None, "provenance_status": "INCOMPLETE",
        "artifact_integrity_verified": False, "evidence_type": "NATIVE",
        "issues": [], "warnings": [],
    }


def audit_model(root: Path) -> dict:
    model_dir = root / "data/processed/es_arena_nb10_v0_1" / ("deployment_" + DEPLOYMENT_ID)
    artifact = model_dir / "es_daily_lr_deployment_v0_1.joblib"
    metadata = model_dir / "es_daily_lr_deployment_v0_1_metadata.json"
    manifest = model_dir / "es_daily_lr_deployment_v0_1_manifest.json"
    res = {"verified": False, "manifest_verified": False, "sha256": "", "manifest": {}, "issues": []}
    if not all(p.is_file() for p in (artifact, metadata, manifest)):
        res["issues"].append("MODEL_FILES_MISSING")
        return res
    try:
        man, meta = read_json(manifest), read_json(metadata)
        sha = sha256_file(artifact)
        res["sha256"] = sha
        identity = meta.get("deployment_identity", {})
        checks = {
            "model_bytes": sha == EXPECTED_MODEL_SHA == man.get("model_sha256"),
            "metadata_bytes": sha256_file(metadata) == man.get("metadata_sha256"),
            "deployment_id": man.get("deployment_id") == DEPLOYMENT_ID,
            "train_cutoff": identity.get("train_through") == "2026-09-30",
            "class_order": identity.get("class_order") == list(CLASSES),
            "feature_order": identity.get("feature_columns") == list(FEATURES),
        }
        for key, ok in checks.items():
            if not ok:
                res["issues"].append("MODEL_" + key.upper() + "_MISMATCH")
        res["manifest_verified"] = not res["issues"]
        res["verified"] = not res["issues"]
        res["manifest"] = man
    except (ValueError, OSError, KeyError) as exc:
        res["issues"].append("MODEL_ARTIFACT_PARSE_ERROR:" + str(exc))
    return res


def audit_forecast(root: Path, fp: Path, model: dict, require_attempt: bool = True) -> dict:
    match = FORECAST_RX.fullmatch(fp.name)
    if not match:
        rec = base_record(forecast_file=fp.name)
        rec["issues"].append("FORECAST_FILENAME_INVALID")
        rec["provenance_status"] = "FAIL"
        return rec
    cid = match.group(2).lower()
    rec = base_record(cid=cid, forecast_file=fp.name)
    evidence_errors: list[str] = rec["issues"]
    warnings: list[str] = rec["warnings"]

    def fail(code: str) -> None:
        if code not in evidence_errors:
            evidence_errors.append(code)

    def incomplete(code: str) -> None:
        if code not in warnings:
            warnings.append(code)

    try:
        if str(uuid.UUID(cid)) != cid:
            fail("INVALID_CHALLENGE_UUID")
        obj = read_json(fp)
        rec["forecast_sha256"] = sha256_file(fp)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        fail("FORECAST_JSON_INVALID:" + str(exc))
        rec["provenance_status"] = "FAIL"
        return rec

    expected_date = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
    if obj.get("challenge_id") != cid:
        fail("CHALLENGE_ID_MISMATCH_OR_MISSING")
    try:
        target = parse_date(obj.get("target_session_date"), "target_session_date")
        source = parse_date(obj.get("last_eligible_session"), "last_eligible_session")
        rec["target_session_date"] = target
        rec["last_eligible_session"] = source
        if target != expected_date:
            fail("TARGET_DATE_FILENAME_MISMATCH")
        if not source < target:
            fail("SOURCE_DATE_NOT_BEFORE_TARGET")
    except ValueError as exc:
        fail("CANONICAL_DATE_FIELDS_INVALID:" + str(exc))

    rec["contract_symbol"] = obj.get("contract_symbol", "")
    if not re.fullmatch(r"ES[A-Z]\d", str(rec["contract_symbol"])):
        fail("CONTRACT_MISSING_OR_INVALID")
    rawp = obj.get("probabilities")
    if not isinstance(rawp, dict) or set(rawp) != set(CLASSES):
        fail("CLASS_PROBABILITY_SCHEMA_INVALID")
    else:
        try:
            vals = [rawp[k] for k in CLASSES]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vals):
                raise ValueError("Probabilities must be numeric (not booleans or strings)")
            nums = [float(v) for v in vals]
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in nums) or abs(sum(nums) - 1.0) > 1e-10:
                raise ValueError("Nonfinite, out-of-range or nonnormalized probability")
            for k, v in zip(CLASSES, nums):
                rec["p_" + k] = v
            direction = CLASSES[max(range(3), key=lambda i: nums[i])]
            rec["predicted_label"] = direction
            rec["confidence"] = max(nums)
            if obj.get("argmax_direction") != direction:
                fail("ARGMAX_DIRECTION_MISMATCH")
        except (ValueError, TypeError) as exc:
            fail("PROBABILITY_VALUES_INVALID:" + str(exc))
    if obj.get("feature_order") != list(FEATURES):
        fail("FEATURE_ORDER_MISMATCH")
    features = obj.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        fail("FEATURE_VECTOR_SCHEMA_INVALID")
    elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in features.values()):
        fail("FEATURE_VECTOR_NUMBERS_INVALID")
    if obj.get("status") not in {"PREVIEW_NOT_SUBMITTED", "PREPARED_NOT_SUBMITTED"}:
        incomplete("FORECAST_STATUS_NOT_STANDARD_PREVIEW")
    prep = parse_aware_utc(obj.get("prepared_at_utc"))
    deadline = parse_aware_utc(obj.get("deadline_utc"))
    available = parse_aware_utc(obj.get("data_available_utc"))
    rec["prepared_at_utc"] = prep.isoformat() if prep else ""
    rec["deadline_utc"] = deadline.isoformat() if deadline else ""
    if not prep or not deadline:
        fail("PREPARED_OR_DEADLINE_TIMESTAMP_MISSING_OR_UNZONED")
    elif prep >= deadline:
        fail("FORECAST_PREPARED_AFTER_DEADLINE")
    if available is None:
        fail("DATA_AVAILABLE_TIMESTAMP_MISSING_OR_UNZONED")
    elif prep and available > prep:
        fail("INPUT_RETRIEVED_AFTER_PREPARATION")
    else:
        incomplete("DATA_AVAILABLE_UTC_IS_LOCAL_OBSERVATION_NOT_PROVIDER_FIRST_AVAILABLE_PROOF")
    rec["forecast_schema_verified"] = not evidence_errors

    # Model identity: verify actual bytes/manifest, not only declared SHA.
    rec["model_sha256"] = str(obj.get("model_sha256", ""))
    if rec["model_sha256"] != EXPECTED_MODEL_SHA:
        fail("FORECAST_MODEL_SHA_MISMATCH_OR_MISSING")
    if not model["verified"]:
        if any("MISMATCH" in issue or "PARSE_ERROR" in issue for issue in model["issues"]):
            fail("LOCAL_MODEL_INTEGRITY_MISMATCH")
        else:
            incomplete("LOCAL_MODEL_NOT_VERIFIED")
    else:
        model_path = obj.get("model_dir")
        if model_path is None:
            fail("FORECAST_MODEL_DIR_MISSING")
        else:
            try:
                expected = root / "data/processed/es_arena_nb10_v0_1" / ("deployment_" + DEPLOYMENT_ID)
                if repo_path(root, model_path) != expected.resolve():
                    fail("FORECAST_MODEL_DIR_MISMATCH")
            except ValueError as exc:
                fail("FORECAST_MODEL_DIR_INVALID:" + str(exc))
        orig_src = obj.get("frozen_model_source_sha256")
        expected_src = model["manifest"].get("source_sha256")
        if not orig_src or orig_src != expected_src:
            fail("FORECAST_FROZEN_MODEL_SOURCE_HASH_MISMATCH")
        rec["model_hash_verified"] = (
            rec["model_sha256"] == EXPECTED_MODEL_SHA
            and model_path is not None
            and not any(x.startswith("FORECAST_MODEL_DIR_") or x == "FORECAST_FROZEN_MODEL_SOURCE_HASH_MISMATCH" for x in evidence_errors)
        )
        rec["model_manifest_verified"] = rec["model_hash_verified"]

    rec["source_sha256"] = str(obj.get("source_sha256", ""))
    snapshot = None
    if not obj.get("source_csv"):
        fail("SOURCE_SNAPSHOT_PATH_MISSING")
    else:
        try:
            snapshot = repo_path(root, obj["source_csv"])
            if not snapshot.is_file():
                incomplete("SOURCE_SNAPSHOT_FILE_MISSING")
            else:
                if sha256_file(snapshot) != rec["source_sha256"]:
                    fail("SOURCE_SNAPSHOT_BYTES_MISMATCH")
                else:
                    rec["snapshot_hash_verified"] = True
                sn = SNAPSHOT_RX.fullmatch(snapshot.name)
                if not sn or (rec["last_eligible_session"] and sn.group(1) != rec["last_eligible_session"].replace("-", "")):
                    fail("SOURCE_SNAPSHOT_FILENAME_DATE_MISMATCH")
                if sn and rec["source_sha256"] and not rec["source_sha256"].startswith(sn.group(2)):
                    fail("SOURCE_SNAPSHOT_FILENAME_HASH_MISMATCH")
                mf = snapshot.with_suffix(".manifest.json")
                if not mf.is_file():
                    incomplete("SOURCE_MANIFEST_MISSING")
                else:
                    if sha256_file(mf) != obj.get("source_manifest_sha256"):
                        fail("FORECAST_SOURCE_MANIFEST_HASH_MISMATCH")
                    man = read_json(mf)
                    if man.get("snapshot_sha256") != rec["source_sha256"]:
                        fail("MANIFEST_SOURCE_SNAPSHOT_HASH_MISMATCH")
                    manifest_path = man.get("snapshot_path", "")
                    try:
                        if repo_path(root, manifest_path) != snapshot:
                            fail("MANIFEST_SOURCE_PATH_MISMATCH")
                    except ValueError:
                        fail("MANIFEST_SOURCE_PATH_INVALID")
                    rec["snapshot_manifest_verified"] = not any(
                        issue.startswith(("FORECAST_SOURCE_MANIFEST", "MANIFEST_SOURCE_", "SOURCE_MANIFEST"))
                        for issue in evidence_errors
                    )
                    # An optional reviewed CSV can be independently linked by candidate audit ID.
                    audit_id = man.get("candidate_audit_id")
                    if audit_id:
                        candidate_review = root / "data/processed/es_arena_nb11_v0_1/audits" / ("reviewed_" + str(audit_id) + ".csv")
                        if not candidate_review.is_file():
                            incomplete("REVIEWED_AUDIT_FILE_MISSING")
                        elif sha256_file(candidate_review) != man.get("reviewed_audit_sha256"):
                            fail("REVIEWED_AUDIT_HASH_MISMATCH")
                        else:
                            rec["reviewed_audit_verified"] = True
                    else:
                        incomplete("SOURCE_MANIFEST_AUDIT_ID_MISSING")
                with snapshot.open("r", encoding="utf-8-sig", newline="") as inp:
                    rows = list(csv.DictReader(inp))
                if not rows:
                    fail("SOURCE_SNAPSHOT_EMPTY")
                else:
                    dates = [r.get("session_date", "") for r in rows]
                    if len(set(dates)) != len(dates):
                        fail("SOURCE_SNAPSHOT_DUPLICATE_DATES")
                    eligible = [r for r in rows if bool_true(r.get("target_eligible"))]
                    if not eligible:
                        fail("NO_ELIGIBLE_SOURCE_SESSIONS")
                    else:
                        last = max(eligible, key=lambda r: r["session_date"])
                        if rec["last_eligible_session"] and last["session_date"] != rec["last_eligible_session"]:
                            fail("LATEST_ELIGIBLE_SOURCE_DATE_MISMATCH")
                        if str(last.get("raw_symbol")) != rec["contract_symbol"]:
                            fail("LATEST_SOURCE_CONTRACT_MISMATCH")
                        if not bool_true(last.get("coverage_reviewed")):
                            fail("LATEST_SOURCE_NOT_HUMAN_REVIEWED")
                        if any(not bool_true(r.get("coverage_reviewed")) for r in rows):
                            fail("SOURCE_SNAPSHOT_CONTAINS_UNREVIEWED_ROWS")
                        end = parse_aware_utc(last.get("session_end_utc"))
                        if end is None:
                            fail("LATEST_SOURCE_SESSION_END_INVALID")
                        elif prep is not None and end > prep:
                            fail("SOURCE_SESSION_END_AFTER_FORECAST")
                        rec["source_cutoff_verified"] = bool(end and prep and end <= prep and
                            rec["last_eligible_session"] and last["session_date"] == rec["last_eligible_session"])
                        rec["source_review_verified"] = not any(x.startswith(("LATEST_SOURCE_", "SOURCE_SNAPSHOT_CONTAINS_UNREVIEWED", "REVIEWED_AUDIT_HASH")) for x in evidence_errors)
        except (ValueError, KeyError, csv.Error, OSError) as exc:
            fail("SOURCE_ARTIFACT_PARSE_ERROR:" + str(exc))

    # Receipt may establish a scored acceptance even when complete provenance is unavailable.
    receipt = root / "data/processed/es_arena_live/receipts" / ("receipt_" + cid + ".json")
    rec["receipt_file"] = receipt.name
    if receipt.is_file():
        try:
            receipt_obj = read_json(receipt)
            rec["receipt_sha256"] = sha256_file(receipt)
            http = receipt_obj.get("http_status")
            response = receipt_obj.get("response", {})
            rec["receipt_http_status"] = str(http) if http is not None else ""
            rec["receipt_counts_for_score"] = str(response.get("counts_for_score")) if isinstance(response, dict) else ""
            if http != 201 or not isinstance(response, dict) or response.get("counts_for_score") is not True:
                fail("RECEIPT_NOT_HTTP201_SCORED")
            elif str(response.get("challenge_id", "")).lower() != cid:
                fail("RECEIPT_CHALLENGE_MISMATCH")
            elif not response.get("prediction_id"):
                fail("RECEIPT_PREDICTION_ID_MISSING")
            else:
                rec["receipt_server_accepted"] = True
            if receipt_obj.get("forecast_sha256") != rec["forecast_sha256"]:
                fail("RECEIPT_FORECAST_SHA_MISMATCH")
            else:
                rec["receipt_forecast_hash_verified"] = True
            rec["accepted_by_receipt"] = bool(rec["receipt_server_accepted"] and rec["receipt_forecast_hash_verified"])
            # Prefer timestamped SERVER fields. A client-side request timestamp is
            # not, on its own, independently verified server acceptance time.
            accepted_utc = None
            for path_key in ("accepted_at_utc", "submitted_at_utc", "created_at_utc", "created_at"):
                if isinstance(response, dict) and response.get(path_key) is not None:
                    accepted_utc = parse_aware_utc(response[path_key])
                    if accepted_utc is not None:
                        break
            if accepted_utc is None:
                submitted_local = find_unique(receipt_obj,
                    ("submitted_at_utc", "submitted_at"), "local receipt submission time")
                if parse_aware_utc(submitted_local) is not None:
                    incomplete("ONLY_CLIENT_SUBMISSION_TIME_PRESENT_NOT_SERVER_ACCEPTANCE_TIME")
            rec["submitted_at_utc"] = accepted_utc.isoformat() if accepted_utc else ""
            if accepted_utc and deadline:
                if accepted_utc <= deadline:
                    rec["deadline_verified"] = True
                    rec["deadline_buffer_minutes"] = round((deadline - accepted_utc).total_seconds() / 60.0, 6)
                else:
                    fail("SUBMISSION_ACCEPTED_AFTER_DECLARED_DEADLINE")
            else:
                incomplete("SERVER_ACCEPTANCE_TIME_NOT_INDEPENDENTLY_UTC_VERIFIED")
        except (ValueError, OSError, KeyError) as exc:
            fail("RECEIPT_JSON_INVALID:" + str(exc))
    else:
        incomplete("RECEIPT_MISSING_NOT_PROOF_OF_NO_SUBMISSION")

    attempt = root / "data/processed/es_arena_live/receipts" / ("attempt_" + cid + ".json")
    rec["attempt_file"] = attempt.name
    if attempt.is_file():
        rec["attempt_present"] = True
        try:
            ao = read_json(attempt)
            att_id = find_unique(ao, ("challenge_id", "request.challenge_id", "forecast.challenge_id"), "attempt challenge")
            if att_id is not None and str(att_id).lower() != cid:
                fail("ATTEMPT_CHALLENGE_ID_MISMATCH")
            att_hash = find_unique(ao, ("forecast_sha256", "request.forecast_sha256"), "attempt forecast hash")
            if att_hash is not None and att_hash != rec["forecast_sha256"]:
                fail("ATTEMPT_FORECAST_SHA_MISMATCH")
            rec["attempt_hash_verified"] = att_hash is not None and att_hash == rec["forecast_sha256"]
            if att_hash is None:
                incomplete("ATTEMPT_HAS_NO_FORECAST_HASH")
        except (ValueError, OSError) as exc:
            fail("ATTEMPT_JSON_INVALID:" + str(exc))
    elif require_attempt and rec["accepted_by_receipt"]:
        incomplete("SUBMISSION_ATTEMPT_MARKER_MISSING")

    # Separate artifact integrity from HTTP acceptance. Missing private files never become PASS.
    rec["artifact_integrity_verified"] = bool(
        rec["forecast_schema_verified"] and rec["model_hash_verified"]
        and rec["snapshot_hash_verified"] and rec["snapshot_manifest_verified"]
        and rec["source_review_verified"] and rec["source_cutoff_verified"]
        and rec["reviewed_audit_verified"] and not evidence_errors
    )
    if evidence_errors:
        rec["provenance_status"] = "FAIL"
    elif not rec["artifact_integrity_verified"]:
        rec["provenance_status"] = "INCOMPLETE"
    elif not rec["accepted_by_receipt"]:
        rec["provenance_status"] = "PREPARED_NOT_SUBMITTED" if not receipt.is_file() and not attempt.is_file() else "INCOMPLETE"
    elif not rec["deadline_verified"] or not rec["attempt_present"] or not rec["attempt_hash_verified"]:
        rec["provenance_status"] = "INCOMPLETE"
    else:
        rec["provenance_status"] = "VERIFIED_LOCAL_CHAIN"
    return rec


def audit_repository(root: str | Path) -> dict:
    root = Path(root).resolve()
    model = audit_model(root)
    directory = root / "data/processed/es_arena_live/forecasts"
    rows = [audit_forecast(root, path, model) for path in sorted(directory.glob("forecast_*.json"))] if directory.is_dir() else []
    known = {r["challenge_id"] for r in rows}
    receipt_dir = root / "data/processed/es_arena_live/receipts"
    if receipt_dir.is_dir():
        for path in sorted(receipt_dir.glob("receipt_*.json")):
            match = RECEIPT_RX.fullmatch(path.name)
            if match and match.group(1).lower() not in known:
                row = base_record(match.group(1).lower())
                row.update(receipt_file=path.name, receipt_sha256=sha256_file(path), provenance_status="INCOMPLETE")
                row["issues"].append("ORPHAN_RECEIPT_NO_FORECAST_JSON")
                rows.append(row)
    legacy = root / "data/processed/es_arena_research_v0_1/legacy_submissions"
    if legacy.is_dir():
        for fp in sorted(legacy.glob("legacy_*.json")):
            try:
                o = read_json(fp)
                cid = str(o.get("challenge_id", ""))
            except (ValueError, OSError):
                cid = ""
            if cid and cid not in known:
                row = base_record(cid)
                row.update(evidence_type="LEGACY_RECOVERED", provenance_status="LEGACY_LIMITATION")
                row["warnings"].append("ORIGINAL_NATIVE_FORECAST_OR_RECEIPT_UNAVAILABLE")
                rows.append(row)
    return {"schema_version": "es-provenance-audit-v1", "root": str(root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": {k: v for k, v in model.items() if k != "manifest"},
            "records": sorted(rows, key=lambda r: (r["target_session_date"], r["challenge_id"])),
            "summary": {s: sum(r["provenance_status"] == s for r in rows) for s in
                        ("VERIFIED_LOCAL_CHAIN", "PREPARED_NOT_SUBMITTED", "INCOMPLETE", "FAIL", "LEGACY_LIMITATION")}}


def preflight(root: Path, audit: dict, cid: str, minimum_minutes: int = 5) -> tuple[bool, list[str]]:
    """Local-only gate; never replaces live Arena status, authenticated submission checks."""
    records = [r for r in audit["records"] if r["challenge_id"] == cid]
    problems = []
    if len(records) != 1:
        return False, ["Exactly one local forecast required for challenge"]
    r = records[0]
    if not r["artifact_integrity_verified"] or r["provenance_status"] == "FAIL":
        problems.append("Forecast/model/snapshot provenance is not fully verified")
    if r["receipt_server_accepted"] or r["attempt_present"]:
        problems.append("Existing receipt or submission attempt; STOP and reconcile with Arena")
    deadline = parse_aware_utc(r["deadline_utc"])
    if not deadline:
        problems.append("Missing unambiguous UTC deadline")
    else:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds() / 60
        if remaining < minimum_minutes:
            problems.append(f"Deadline safety buffer not met ({remaining:.2f} minutes remaining)")
    if r["target_session_date"] and r["target_session_date"] != datetime.now(timezone.utc).date().isoformat():
        problems.append("Target date differs from current UTC date; investigate timezone/challenge")
    problems.append("Verify the official LIVE Arena challenge, contract, agent, deadline, and lack of earlier submission manually; this tool has NO network access")
    return len(problems) == 1, problems


def export_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(payload)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="Market-Mood-Forecasting checkout root")
    p.add_argument("--challenge-id", help="Show only one challenge")
    p.add_argument("--preflight", help="Read-only local check for one challenge before manual submission")
    p.add_argument("--minimum-buffer-minutes", type=int, default=5)
    p.add_argument("--json-out", help="Optional write-once JSON report path; no overwrite")
    p.add_argument("--csv-out", help="Optional write-once CSV report path; no overwrite")
    p.add_argument("--strict", action="store_true", help="Nonzero return if FAIL/INCOMPLETE found, excluding legacy/prepared")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    if not (root / "es_arena_daily.py").exists():
        p.error(f"Not a project root: {root}")
    audit = audit_repository(root)
    records = audit["records"]
    if args.challenge_id:
        records = [r for r in records if r["challenge_id"] == args.challenge_id]
    print("MODEL:", "VERIFIED" if audit["model"]["verified"] else "NOT_VERIFIED", audit["model"]["issues"])
    print("CHALLENGE | DATE | SOURCE | ARTIFACT | RECEIPT | DEADLINE | PROVENANCE")
    for r in records:
        print(" | ".join([r["challenge_id"], r["target_session_date"], r["last_eligible_session"],
              "PASS" if r["artifact_integrity_verified"] else "NOT_VERIFIED",
              "PASS" if r["accepted_by_receipt"] else "NOT_VERIFIED",
              "PASS" if r["deadline_verified"] else "NOT_VERIFIED", r["provenance_status"]]))
        if r["issues"] or r["warnings"]:
            print("  Issues:", "; ".join(r["issues"]) or "none")
            print("  Notes:", "; ".join(r["warnings"]) or "none")
    print("SUMMARY:", audit["summary"])
    if args.preflight:
        ok, notes = preflight(root, audit, args.preflight, args.minimum_buffer_minutes)
        print("LOCAL PREFLIGHT:", "PASS (LIVE ARENA VERIFICATION STILL REQUIRED)" if ok else "STOP")
        for line in notes:
            print(" ", line)
        if not ok:
            return 2
    for opt, out_type in ((args.json_out, "json"), (args.csv_out, "csv")):
        if opt:
            out = Path(opt)
            if not out.is_absolute():
                out = root / out
            if not out.resolve().is_relative_to(root):
                p.error("Report output must remain inside project root")
            # Files under the ignored research tree only: prevent public leak.
            if not out.resolve().is_relative_to((root / "data/processed/es_arena_research_v0_1").resolve()):
                p.error("Reports may only be saved inside private data/processed/es_arena_research_v0_1/")
            # Do not create a report that Git might publish accidentally.
            import subprocess
            ignore = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", str(out)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ignore.returncode != 0:
                p.error("Private output path is not Git-ignored; refuse report write")
            if out_type == "json":
                payload = (json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            else:
                from io import StringIO
                sio = StringIO()
                fields = [k for k in base_record() if k not in ("issues", "warnings")]
                writer = csv.DictWriter(sio, fieldnames=fields + ["issues", "warnings"], extrasaction="ignore")
                writer.writeheader()
                for row in audit["records"]:
                    writer.writerow({**row, "issues": ";".join(row["issues"]), "warnings": ";".join(row["warnings"])})
                payload = sio.getvalue().encode("utf-8")
            try:
                export_exclusive(out, payload)
            except FileExistsError:
                print(f"STOP: report exists; never overwrite: {out}", file=sys.stderr)
                return 2
            print("CREATED WRITE-ONCE PRIVATE REPORT:", out)
    if args.strict and any(r["provenance_status"] in {"FAIL", "INCOMPLETE"} for r in records):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
