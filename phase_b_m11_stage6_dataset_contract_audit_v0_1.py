#!/usr/bin/env python3
"""Offline M11 Stage 6: validate draft dataset contracts and candidate chronological split metadata.

No models fitted, source truth certified, challenge enrollment, external requests, or file writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from phase_b_m11_stage5_registry_audit_v0_1 import (
    EXPECTED_SOURCES, validate_registry, validate_evidence, parse_utc,
)

REGISTRY_SHA = "cf5f318f36fde97d6ea769667d184bb4ddb9ce25c3bea2515643a74a2dc7076d"
STAGE5_COMMIT = "52ed45faffc0cc932345aee9935d77ccadf49dad"
ORDER = ("train", "validation", "test")
HEX = re.compile(r"^[0-9a-f]{64}$")
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONTROL = "MATCHED_TRAINING_ES_ONLY_UNWEIGHTED"


class ContractError(ValueError):
    pass


def require(ok, description):
    if not ok:
        raise ContractError(description)


def load_json(path: Path):
    require(path.is_file(), f"File does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_sha(path: Path, expected: str, what: str):
    require(isinstance(expected, str) and HEX.fullmatch(expected) is not None,
            f"Invalid expected {what} SHA-256")
    require(digest(path) == expected, f"{what} SHA-256 differs from supplied pin")


def audit_contract(contract: dict, registry: dict):
    validate_registry(registry)
    require(isinstance(contract, dict), "Contract must be JSON object")
    fixed = {
        "record_type": "M11_PHASE_B_DATASET_SPLIT_CONTRACT_DRAFT_V0_1",
        "status": "DRAFT_NOT_FROZEN",
        "study_design": "ORIGINAL_150_TOTAL_DESCRIPTIVE",
        "upstream_stage5_commit": STAGE5_COMMIT,
        "upstream_registry_sha256": REGISTRY_SHA,
        "target": "ARENA_ES_DAILY_THREE_CLASS",
        "candidate_configurations": ["M2", "M3", "M4", "M5", "M6", "M7"],
        "matched_es_only_control": CONTROL,
        "frozen_m1_retraining_permitted": False,
        "split_order": list(ORDER),
        "split_key": "OFFICIAL_CHALLENGE_UUID",
        "time_key": "FORECAST_DECISION_CUTOFF_UTC",
        "same_target_session_across_splits_forbidden": True,
        "minimum_synthetic_fold_rows": {"train": 3, "validation": 2, "test": 2},
        "fit_time_rule": "AFTER_ALL_TRAIN_LABEL_PUBLICATIONS_BEFORE_FIRST_VALIDATION_DECISION",
        "selection_time_rule": "AFTER_ALL_VALIDATION_LABEL_PUBLICATIONS_BEFORE_FIRST_TEST_DECISION",
        "test_holdout_rule": "NEVER_USE_TEST_LABELS_OR_TEST_FEATURES_TO_TUNE_OR_SELECT",
        "allowed_evidence_modes": ["PROSPECTIVE_CAPTURED", "HISTORICAL_RECONSTRUCTED"],
        "historical_reconstruction_rule": "LOCAL_ASOF_METADATA_CONSISTENCY_ONLY_REQUIRES_EXTERNAL_VINTAGE_CORROBORATION",
        "external_data_rights_verified": False,
        "model_artifacts_trained_by_stage6": 0,
        "phase_b_model_freeze_certified": False,
        "fold_dates_or_sample_ids_frozen": False,
        "chronological_split_method": "EXPANDING_WINDOW_DRAFT_REQUIRES_LATER_TRAINING_PLAN_AND_PURGE_RULE",
        "primary_metric": "UNSCALED_MULTICLASS_BRIER",
        "forbidden_new_sources": ["SP500_INDEX"],
        "private_data_publication": "FORBIDDEN",
    }
    require(set(contract) == set(fixed), "Contract fields added/removed; requires versioned review")
    for name, expected in fixed.items():
        require(contract.get(name) == expected, f"Draft contract changed: {name}")
    return {"status": "DRAFT_CONTRACT_VALIDATED_NOT_FROZEN",
            "dataset_rows_validated": 0, "models_trained": 0,
            "phase_b_model_freeze_certified": False,
            "external_data_verified": False,
            "file_writes": 0, "network_requests": 0, "forecast_submissions": 0}


def validate_reconstruction_source(row: dict, decision: datetime, source: str):
    """Only test user-supplied local metadata; no claim of original vintage authenticity."""
    require(row.get("source") == source, "Reconstruction source mismatch")
    require(row.get("asof_evidence_status") == "LOCAL_HISTORICAL_VINTAGE_REVIEW_REQUIRED",
            f"Historical evidence must retain unverified status: {source}")
    for key in ("observation_end_utc", "source_publication_utc", "source_available_utc"):
        require(parse_utc(row.get(key)) <= decision, f"Historical lookahead for {source}.{key}")
    require(parse_utc(row["source_publication_utc"]) <= parse_utc(row["source_available_utc"]),
            f"Publication occurs after claimed availability: {source}")
    require(bool(row.get("source_revision_or_vintage")) and isinstance(row["source_revision_or_vintage"], str),
            f"Historical vintage absent: {source}")
    require(isinstance(row.get("provenance_sha256"), str) and HEX.fullmatch(row["provenance_sha256"]) is not None,
            f"Historical provenance hash absent: {source}")
    require(row.get("licensing_restriction") in
            ("PRIVATE_LICENSED", "PUBLIC_REUSE_REVIEW_REQUIRED", "PUBLIC_REUSE_CONFIRMED"),
            f"Missing rights status: {source}")
    require(row.get("includes_target_date_settlement") is False,
            f"Target settlement used as feature: {source}")
    # Retrospective retrieval may be later than the historical decision, but it is never proof
    # that the original vintage or features actually existed before the forecast decision.
    parse_utc(row.get("retrieved_at_utc"))


def audit_candidate_folds(manifest: dict, contract: dict, registry: dict, contract_sha: str):
    audit_contract(contract, registry)
    require(isinstance(manifest, dict), "Fold manifest must be a JSON object")
    require(manifest.get("record_type") == "M11_STAGE6_CHRONOLOGICAL_FOLD_MANIFEST_V0_1",
            "Wrong fold manifest record type")
    require(manifest.get("status") == "CANDIDATE_NOT_FROZEN", "Fold manifest cannot be called frozen")
    require(manifest.get("contract_sha256") == contract_sha, "Fold manifest contract pin mismatch")
    mid = manifest.get("model_id")
    require(mid in contract["candidate_configurations"] + [CONTROL], "M1 cannot be retrained by Stage 6")
    sources = ["ES"] if mid == CONTROL else EXPECTED_SOURCES[mid]
    mode = manifest.get("evidence_mode")
    require(mode in contract["allowed_evidence_modes"], "Unknown historical/prospective evidence mode")
    fit_at = parse_utc(manifest.get("fit_at_utc"))
    selection_at = parse_utc(manifest.get("selection_at_utc"))
    require(fit_at < selection_at, "Fit time must precede selection")
    rows = manifest.get("rows")
    require(isinstance(rows, list), "Fold rows must be a list")
    counters = {name: 0 for name in ORDER}
    times = {name: [] for name in ORDER}
    label_times = {name: [] for name in ORDER}
    ids = set()
    date_split = {}
    for r in rows:
        require(isinstance(r, dict), "Fold row must be an object")
        ident = r.get("challenge_id")
        try:
            uuid.UUID(str(ident))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ContractError("Missing or malformed official challenge UUID") from exc
        require(ident not in ids, "Repeated challenge ID across folds")
        ids.add(ident)
        split = r.get("split")
        require(split in ORDER, "Unrecognized train/validation/test fold")
        when = parse_utc(r.get("decision_cutoff_utc"))
        target = r.get("target_session")
        require(isinstance(target, str) and ISO_DAY.fullmatch(target) is not None,
                "Missing target session ISO date")
        try:
            datetime.strptime(target, "%Y-%m-%d")
        except ValueError as exc:
            raise ContractError("Invalid target-session date") from exc
        require(target not in date_split or date_split[target] == split,
                "Same target-session assigned across train/validation/test")
        date_split[target] = split
        label_at = r.get("label_available_at_utc")
        if label_at is not None:
            label_dt = parse_utc(label_at)
            require(label_dt > when, "Target outcome appears available before decision")
            label_times[split].append(label_dt)
        elif split != "test":
            raise ContractError("Training and validation labels must have verified availability times")
        input_rows = r.get("input_evidence")
        require(isinstance(input_rows, list) and len(input_rows) == len(sources),
                "Wrong number of source evidence records")
        names = [rec.get("source") if isinstance(rec, dict) else None for rec in input_rows]
        require(len(set(names)) == len(sources) and set(names) == set(sources),
                "Missing, duplicate or extra model source")
        if mode == "PROSPECTIVE_CAPTURED":
            evidence = {
                "record_type": "M11_STAGE5_CANDIDATE_INPUT_EVIDENCE_V0_1",
                "model_id": "M1" if mid == CONTROL else mid,
                "decision_cutoff_utc": r["decision_cutoff_utc"],
                "inputs": input_rows,
            }
            validate_evidence(registry, evidence, "M1" if mid == CONTROL else mid,
                              r["decision_cutoff_utc"])
        else:
            for rec in input_rows:
                validate_reconstruction_source(rec, when, rec["source"])
        counters[split] += 1
        times[split].append(when)
    for name, minimum in contract["minimum_synthetic_fold_rows"].items():
        require(counters[name] >= minimum, f"Too few candidate rows in {name} fold")
    require(max(times["train"]) < min(times["validation"]), "Train/validation chronology overlapped")
    require(max(times["validation"]) < min(times["test"]), "Validation/test chronology overlapped")
    require(max(label_times["train"]) <= fit_at < min(times["validation"]),
            "Training label availability crosses fit or first validation")
    require(max(label_times["validation"]) <= selection_at < min(times["test"]),
            "Validation label availability crosses selection or first test")
    require(all(t < fit_at for t in times["train"]), "Train decision happens after fit cutoff")
    require(all(t < selection_at for t in times["validation"]), "Validation decision happens after selection cutoff")
    # This validates the supplied metadata only. Model training/holdout use is not inspected.
    return {"status": "CANDIDATE_FOLD_METADATA_CONSISTENT_NOT_CERTIFIED",
            "model_id": mid, "evidence_mode": mode,
            "fold_row_counts": counters, "unique_challenges": len(ids),
            "historical_vintages_externally_certified": False,
            "model_training_or_holdout_nonuse_certified": False,
            "phase_b_model_freeze_certified": False,
            "file_writes": 0, "network_requests": 0, "forecast_submissions": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("audit-contract", "audit-folds"):
        sp = subs.add_parser(name)
        sp.add_argument("--registry", type=Path, required=True)
        sp.add_argument("--contract", type=Path, required=True)
        sp.add_argument("--expected-registry-sha256", default=REGISTRY_SHA)
        sp.add_argument("--expected-contract-sha256", required=(name == "audit-folds"))
        if name == "audit-folds":
            sp.add_argument("--fold-manifest", type=Path, required=True)
            sp.add_argument("--expected-fold-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        assert_sha(args.registry, args.expected_registry_sha256, "upstream registry")
        reg = load_json(args.registry)
        if args.expected_contract_sha256:
            assert_sha(args.contract, args.expected_contract_sha256, "contract")
        contract = load_json(args.contract)
        outcome = audit_contract(contract, reg)
        outcome["contract_sha256"] = digest(args.contract)
        outcome["upstream_registry_sha256"] = digest(args.registry)
        if args.command == "audit-folds":
            assert_sha(args.fold_manifest, args.expected_fold_manifest_sha256, "fold manifest")
            outcome = audit_candidate_folds(load_json(args.fold_manifest), contract, reg, digest(args.contract))
            outcome["fold_manifest_sha256"] = digest(args.fold_manifest)
            outcome["contract_sha256"] = digest(args.contract)
        print("M11_STAGE6_READ_ONLY_AUDIT: PASS")
        print(json.dumps(outcome, sort_keys=True, indent=2))
        return 0
    except (ContractError, ValueError, OSError, json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
        print("M11_STAGE6_AUDIT: FAIL - " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
