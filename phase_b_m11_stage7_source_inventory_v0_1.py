#!/usr/bin/env python3
"""Read-only, evidence-conservative Phase B local source inventory.

Does NOT parse market values, certify data rights/availability, enroll opportunities,
run predictions, train models, or call network. Optional exclusive private snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BRANCH = "experiment/headline-arena-es-daily"
STAGE6_COMMIT = "fad6162a314ba086b8843654c6f52e9613d7e8f2"
PINNED = {
    "docs/M11_PROSPECTIVE_OPPORTUNITY_PROTOCOL_v1_0.md": "a0eec9b6929397338ee69ee9a17e48cd4807e006c27463f8bd6508af576dd63d",
    "phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json": "cf5f318f36fde97d6ea769667d184bb4ddb9ce25c3bea2515643a74a2dc7076d",
    "phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json": "bf7cbd2339753efb6a82a42bcb8e08e56262f24d6d5a395974b8aaf77ad61c06",
}
PRIVATE_PINS = {
    "m11_future/adoption/owner_adoption_decision_20261114_v1_0.json": "90d797e773e50a023b59ff6b30cd9637fd2dcf8f5d432066b138d6b927111c25",
    "m11_future/historical_baseline_20261114/arena_official_settlements_19rows_20261114.csv": "093bc2aa57c7e87f073c2ff86b9fbe7ebd3cbb0b59d0491ee7d95b8c3aee68c4",
}
# No generic user-directory scanning. Only documented data/evidence directories.
SCOPE = (
    ("ES", "repo", "data/raw/databento/incremental_v0_1"),
    ("ES", "repo", "data/processed/es_arena_nb11_v0_1/snapshots"),
    ("ES", "repo", "data/processed/es_arena_nb11_v0_1/audits"),
    ("ES", "repo", "data/processed/es_arena_nb10_v0_1/deployment_ab4d5c14423352c7"),
    ("GDELT", "repo", "data/raw/gdelt"),
    ("VIX", "repo", "data/raw/vix"),
    ("GOOGLE_TRENDS", "repo", "data/raw/google_trends"),
    ("UNEMPLOYMENT", "repo", "data/raw/unemployment"),
    ("UNEMPLOYMENT", "repo", "data/raw/fred"),
    ("UNCLASSIFIED_PHASE_B", "repo", "data/processed/phase_b"),
    ("HISTORICAL_ES_CHECKPOINT", "private", "m11_future/historical_baseline_20261114"),
)
ALLOWED_SUFFIX = (".csv", ".parquet", ".json", ".jsonl", ".dbn", ".dbn.zst", ".arrow", ".feather", ".joblib", ".pkl")
PRIMARY = ("ES", "GDELT", "VIX", "GOOGLE_TRENDS", "UNEMPLOYMENT")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_environment(repo: Path, private: Path, *, git_check: bool = True) -> None:
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError("Repository missing")
    if not private.is_dir() or private.is_symlink():
        raise ValueError("Private root missing or symlinked")
    if git_check:
        def git(*args: str):
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        b = git("branch", "--show-current")
        if b.returncode or b.stdout.strip() != BRANCH:
            raise ValueError("Wrong Git branch")
        if git("merge-base", "--is-ancestor", STAGE6_COMMIT, "HEAD").returncode:
            raise ValueError("Published Stage 6 commit is not in current history")
    for rel, expected in PINNED.items():
        p = repo / rel
        if not p.is_file() or p.is_symlink() or digest(p) != expected:
            raise ValueError("Published upstream pin mismatch: " + rel)
    for rel, expected in PRIVATE_PINS.items():
        p = private / rel
        if not p.is_file() or p.is_symlink() or digest(p) != expected:
            raise ValueError("Sealed private evidence mismatch: " + rel)


def scan_scope(repo: Path, private: Path, *, hash_max_bytes: int = 8 * 1024 * 1024,
               max_files: int = 2500) -> dict:
    if hash_max_bytes < 0 or max_files <= 0:
        raise ValueError("Invalid scan limits")
    roots = []
    records = []
    seen = set()
    for source, owner, rel in SCOPE:
        root = (repo if owner == "repo" else private) / rel
        if not root.exists():
            roots.append({"source": source, "area": owner + "/" + rel, "status": "DIRECTORY_NOT_PRESENT", "files": 0})
            continue
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Unsafe candidate directory (not directory or symlink): " + rel)
        found = 0
        for walkbase, dirs, files in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not (Path(walkbase) / d).is_symlink())
            for filename in sorted(files):
                p = Path(walkbase) / filename
                if p.is_symlink() or not p.is_file():
                    continue
                if not filename.lower().endswith(ALLOWED_SUFFIX):
                    continue
                resolved = p.resolve(strict=True)
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError("Candidate escaped scan root")
                if resolved in seen:
                    continue
                seen.add(resolved)
                if len(records) >= max_files:
                    raise ValueError("Inventory file cap exceeded; use more targeted source roots")
                stat = p.stat()
                records.append({
                    "source_bucket_by_directory_only": source,
                    "path": str(p),  # private snapshot only; never Git commit
                    "size_bytes": stat.st_size,
                    "mtime_utc_local_filesystem_not_publication_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                    "sha256": digest(p) if stat.st_size <= hash_max_bytes else None,
                    "sha256_status": "COMPUTED" if stat.st_size <= hash_max_bytes else "SKIPPED_SIZE_LIMIT",
                    "market_data_contents_validated": False,
                    "publication_or_historical_vintage_verified": False,
                    "licensing_verified": False,
                })
                found += 1
        roots.append({"source": source, "area": owner + "/" + rel, "status": "DIRECTORY_SCANNED", "files": found})
    counts = {src: sum(r["source_bucket_by_directory_only"] == src for r in records) for src in PRIMARY}
    return {"record_type": "M11_STAGE7_LOCAL_SOURCE_INVENTORY_V0_1",
            "status": "LOCAL_INVENTORY_ONLY_NOT_DATASET_VALIDATION",
            "observed_at_utc_local_clock": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "upstream_stage6_commit": STAGE6_COMMIT,
            "upstream_registry_sha256": PINNED["phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json"],
            "upstream_contract_sha256": PINNED["phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json"],
            "sealed_19_settlement_snapshot_sha256": PRIVATE_PINS["m11_future/historical_baseline_20261114/arena_official_settlements_19rows_20261114.csv"],
            "scanned_directories": roots, "file_metadata": records,
            "candidate_file_counts_by_directory_name_only": counts,
            "discovered_file_count": len(records),
            "verified_external_data_sources": 0,
            "validated_training_dataset_rows": 0,
            "historical_asof_availability_certified": False,
            "license_rights_certified": False,
            "trained_new_models": 0,
            "forecast_submissions": 0,
            "external_network_requests": 0,
            "notes": "A file's presence/mtime/hash is NOT evidence of historical source availability, license rights, correct schema or usable training rows."}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan", nargs="?", choices=["scan"], default="scan")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--private-root", type=Path, required=True)
    ap.add_argument("--snapshot-file", type=Path, help="Optional NEW private JSON; exclusive creation, never overwrite")
    ap.add_argument("--max-files", type=int, default=2500)
    ap.add_argument("--hash-max-mib", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        repo, private = args.repo.resolve(), args.private_root.resolve()
        validate_environment(repo, private)
        snapshot = None
        if args.snapshot_file:
            snapshot = args.snapshot_file.absolute()
            if not snapshot.parent.resolve().is_relative_to(private) or snapshot.is_symlink():
                raise ValueError("Snapshot must be under private root")
            if not snapshot.parent.is_dir():
                raise ValueError("Snapshot parent directory does not exist; create it explicitly")
            if snapshot.exists():
                raise ValueError("Snapshot already exists; never overwrite")
        report = scan_scope(repo, private, max_files=args.max_files,
                            hash_max_bytes=args.hash_max_mib * 1024 * 1024)
        print("M11_STAGE7_LOCAL_INVENTORY: PASS")
        print("CANDIDATE_FILES_SEEN:", report["discovered_file_count"])
        for k, n in report["candidate_file_counts_by_directory_name_only"].items():
            print(k + "_CANDIDATE_FILES:", n)
        print("REAL_DATASETS_VALIDATED: 0 | SOURCE_LICENSES_CERTIFIED: 0 | MODELS_TRAINED: 0")
        print("NETWORK_REQUESTS: 0 | FORECAST_SUBMISSIONS: 0")
        if snapshot is not None:
            contents = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            with snapshot.open("xb") as fh:
                fh.write(contents)
            print("PRIVATE_SNAPSHOT_WRITTEN:", snapshot)
            print("PRIVATE_SNAPSHOT_SHA256:", hashlib.sha256(contents).hexdigest())
        else:
            print("FILES_WRITTEN: 0")
        return 0
    except (ValueError, OSError, OverflowError) as exc:
        print("M11_STAGE7_STOP:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(run())
