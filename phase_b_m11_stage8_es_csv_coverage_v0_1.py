#!/usr/bin/env python3
"""Offline read-only Stage 8 review of ES CSV candidates in a pinned private Stage 7 inventory.

This is a structural/coverage inspection, NOT licensed-data, point-in-time vintage,
market-calendar, label, dataset, model, or official-challenge certification.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import phase_b_m11_stage7_source_inventory_v0_1 as s7

TYPE = 'M11_STAGE8_ES_CSV_COVERAGE_AUDIT_V0_1'
CSV_SCOPES = (
    'data/raw/databento/incremental_v0_1/',
    'data/processed/es_arena_nb11_v0_1/snapshots/',
    'data/processed/es_arena_nb11_v0_1/audits/',
)
DATE_FIELDS = {'date','session','session_date','trading_date','target_date','ts_event','ts_recv','timestamp','datetime','session_start_utc','session_end_utc'}
TARGET_FIELDS = {'target','target_class','label','realized_class','settlement','settlement_class','outcome','y_true'}
HASH_RE = re.compile(r'^[a-f0-9]{64}$')


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(repo: Path, record: dict) -> tuple[Path, str]:
    if record.get('source_bucket_by_directory_only') != 'ES':
        raise ValueError('Not an ES bucket')
    raw = record.get('path')
    if not isinstance(raw, str) or not raw:
        raise ValueError('Inventory record has no path')
    p = Path(raw)
    if not p.is_absolute():
        raise ValueError('Inventory path is not absolute')
    root = repo.resolve(strict=True)
    resolved = p.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError('ES candidate path escapes repository')
    rel = resolved.relative_to(root).as_posix()
    if not any(rel.startswith(prefix) for prefix in CSV_SCOPES) and not rel.startswith('data/processed/es_arena_nb10_v0_1/deployment_ab4d5c14423352c7/'):
        raise ValueError('ES candidate path outside Stage 7 approved ES roots')
    # Verify all existing ancestors are ordinary directories; in particular no symlink escapes.
    q = p
    while q != root and q != q.parent:
        if q.is_symlink():
            raise ValueError('Symlinked ES candidate/path forbidden')
        q = q.parent
    return p, rel


def date_part(value: str) -> tuple[str | None, str]:
    """Return YYYY-MM-DD and time-classification; refuse inferring UTC for naive values."""
    val = value.strip()
    if not val:
        return None, 'EMPTY'
    if len(val) == 10:
        try:
            return date.fromisoformat(val).isoformat(), 'DATE_ONLY'
        except ValueError:
            return None, 'UNPARSEABLE'
    try:
        dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
        return dt.date().isoformat(), 'UTC_OFFSET_TIMESTAMP' if dt.tzinfo is not None else 'NAIVE_TIMESTAMP_UNVERIFIED_ZONE'
    except ValueError:
        return None, 'UNPARSEABLE'


def inspect_csv(path: Path, *, max_bytes: int, max_rows: int) -> dict:
    if max_bytes <= 0 or max_rows <= 0:
        raise ValueError('Invalid CSV inspection caps')
    if path.stat().st_size > max_bytes:
        return {'inspection_status': 'SKIPPED_FILE_SIZE_CAP', 'rows_inspected': 0, 'complete_csv_scanned': False}
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as fh:
            reader = csv.reader(fh, strict=True)
            header = next(reader, None)
            if header is None:
                return {'inspection_status': 'EMPTY_CSV', 'rows_inspected': 0, 'complete_csv_scanned': True}
            if not header or any(not name.strip() for name in header) or len(set(name.strip().casefold() for name in header)) != len(header):
                return {'inspection_status': 'INVALID_CSV_HEADER', 'rows_inspected': 0, 'complete_csv_scanned': False}
            parsed_header = [x.strip() for x in header]
            schema = sha_bytes(json.dumps(parsed_header, ensure_ascii=False, separators=(',', ':')).encode())
            normalized = [x.strip().casefold() for x in header]
            dt_indices = [i for i, x in enumerate(normalized) if x in DATE_FIELDS]
            targets = [parsed_header[i] for i, x in enumerate(normalized) if x in TARGET_FIELDS]
            range_by_field = {parsed_header[i]: {'min_date': None, 'max_date': None, 'parsed_count': 0, 'unparseable_count': 0, 'timestamp_kinds': {}} for i in dt_indices}
            counts = Counter()
            row_count = 0
            malformed = 0
            extra_present = False
            for row in reader:
                if row_count >= max_rows:
                    extra_present = True
                    break
                row_count += 1
                if len(row) != len(header):
                    malformed += 1
                    continue
                for idx, name in zip(dt_indices, (parsed_header[i] for i in dt_indices)):
                    part, kind = date_part(row[idx]); rng = range_by_field[name]
                    counts[(name, kind)] += 1
                    if part is not None:
                        rng['parsed_count'] += 1
                        rng['min_date'] = min(part, rng['min_date']) if rng['min_date'] else part
                        rng['max_date'] = max(part, rng['max_date']) if rng['max_date'] else part
                    elif kind != 'EMPTY':
                        rng['unparseable_count'] += 1
            for name, rng in range_by_field.items():
                rng['timestamp_kinds'] = {kind: count for (column, kind), count in sorted(counts.items()) if column == name}
            return {'inspection_status': 'ROW_WIDTH_MISMATCH' if malformed else ('TRUNCATED_ROW_CAP' if extra_present else 'CSV_STRUCTURAL_SCAN_COMPLETE'),
                    'header_fields': parsed_header, 'header_sha256': schema, 'rows_inspected': row_count,
                    'malformed_row_count': malformed, 'complete_csv_scanned': not extra_present and not malformed,
                    'date_column_observations': range_by_field, 'target_like_columns_review_required': targets,
                    'source_market_calendar_verified': False, 'source_target_label_semantics_verified': False,
                    'historical_point_in_time_availability_verified': False}
    except (UnicodeError, csv.Error, OSError, OverflowError) as exc:
        return {'inspection_status': 'CSV_PARSE_ERROR', 'reason_type': type(exc).__name__, 'rows_inspected': 0,
                'complete_csv_scanned': False}


def audit_inventory(repo: Path, private: Path, inventory_file: Path, expected_sha: str,
                    *, max_csv_bytes: int=16*1024*1024, max_rows: int=200000) -> dict:
    s7.validate_environment(repo, private)
    if not HASH_RE.fullmatch(expected_sha):
        raise ValueError('Expected inventory SHA-256 must be 64 lowercase hex characters')
    inventory_file = inventory_file.absolute()
    if inventory_file.is_symlink() or not inventory_file.is_file():
        raise ValueError('Stage 7 private inventory missing/symlink')
    if not inventory_file.resolve().is_relative_to((private/'m11_future/source_inventories').resolve()):
        raise ValueError('Inventory must remain under designated private source_inventories directory')
    if s7.digest(inventory_file) != expected_sha:
        raise ValueError('Stage 7 private inventory SHA-256 changed')
    obj = json.loads(inventory_file.read_text(encoding='utf-8'))
    if obj.get('record_type') != 'M11_STAGE7_LOCAL_SOURCE_INVENTORY_V0_1' or obj.get('status') != 'LOCAL_INVENTORY_ONLY_NOT_DATASET_VALIDATION':
        raise ValueError('Unexpected Stage 7 inventory schema/status')
    if obj.get('upstream_registry_sha256') != s7.PINNED['phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json'] or obj.get('upstream_contract_sha256') != s7.PINNED['phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json']:
        raise ValueError('Inventory upstream pins are inconsistent')
    entries = obj.get('file_metadata')
    if not isinstance(entries, list) or obj.get('discovered_file_count') != len(entries):
        raise ValueError('Inventory file count invalid')
    if max_csv_bytes <= 0 or max_rows <= 0:
        raise ValueError('Non-positive inspection caps')
    seen=set(); categories=Counter(); examined=[]; counts=Counter(); schema_groups=Counter(); total_rows=0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Malformed inventory entry')
        if entry.get('source_bucket_by_directory_only') != 'ES':
            continue
        p, rel = safe_path(repo, entry)
        if rel in seen:
            raise ValueError('Duplicate ES inventory path')
        seen.add(rel)
        name = p.name.lower()
        if name.startswith('purchase_attempt_') and name.endswith('.json'):
            categories['purchase_attempt_records_not_market_rows'] += 1
        elif name.endswith('.dbn.zst') or name.endswith('.dbn'):
            categories['raw_dbn_files_not_parsed'] += 1
        elif name.endswith('.csv'):
            categories['csv_candidates'] += 1
        else:
            categories['other_es_artifacts_not_parsed'] += 1
        if not name.endswith('.csv'):
            continue
        rec: dict[str, Any] = {'relative_path': rel, 'inventory_sha256': entry.get('sha256'), 'inventory_size': entry.get('size_bytes')}
        if not p.is_file() or p.is_symlink():
            rec['inspection_status'] = 'MISSING_OR_SYMLINK_AFTER_STAGE7'
        elif p.stat().st_size != entry.get('size_bytes'):
            rec['inspection_status'] = 'FILE_SIZE_CHANGED_AFTER_STAGE7'
        elif isinstance(entry.get('sha256'), str) and s7.digest(p) != entry.get('sha256'):
            rec['inspection_status'] = 'FILE_HASH_CHANGED_AFTER_STAGE7'
        else:
            rec.update(inspect_csv(p, max_bytes=max_csv_bytes, max_rows=max_rows))
            if entry.get('sha256') is None:
                rec['stage7_file_sha256_not_recorded'] = True
            if rec.get('inspection_status') == 'CSV_STRUCTURAL_SCAN_COMPLETE':
                schema_groups[rec['header_sha256']] += 1
            total_rows += rec.get('rows_inspected', 0)
        counts[rec['inspection_status']] += 1
        examined.append(rec)
    status = 'PASS_WITH_LIMITATIONS' if not any(k in counts for k in ('MISSING_OR_SYMLINK_AFTER_STAGE7','FILE_SIZE_CHANGED_AFTER_STAGE7','FILE_HASH_CHANGED_AFTER_STAGE7','ROW_WIDTH_MISMATCH','CSV_PARSE_ERROR','INVALID_CSV_HEADER')) else 'REVIEW_REQUIRED'
    return {'record_type': TYPE, 'status': status, 'source_inventory_sha256': expected_sha,
            'upstream_registry_sha256': s7.PINNED['phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json'],
            'upstream_contract_sha256': s7.PINNED['phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json'],
            'es_bucket_count_by_directory_only': sum(categories.values()),
            'category_counts': dict(sorted(categories.items())), 'csv_status_counts': dict(sorted(counts.items())),
            'csv_candidates_inspected_or_skipped': len(examined), 'rows_structurally_inspected_not_validated': total_rows,
            'full_scanned_csv_schema_fingerprints': dict(sorted(schema_groups.items())),
            'csv_details_private_only': examined,
            'training_rows_certified': 0, 'es_session_calendar_certified': False, 'point_in_time_vintages_certified': False,
            'model_artifacts_trained': 0, 'phase_b_freeze_certified': False, 'file_writes': 0,
            'network_requests': 0, 'forecast_submissions': 0,
            'notes': 'ES bucket includes purchase records/model artifacts, not 1 file = 1 training dataset. Dates only as found; timestamps with missing timezone stay unverified. CSV structure does not establish market-session coverage, labels, data rights or historical source availability.'}


def run(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['audit-inventory'])
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--private-root', type=Path, required=True)
    ap.add_argument('--inventory-file', type=Path, required=True)
    ap.add_argument('--expected-inventory-sha256', required=True)
    ap.add_argument('--max-csv-mib', type=int, default=16)
    ap.add_argument('--max-rows', type=int, default=200000)
    ap.add_argument('--snapshot-file', type=Path)
    args=ap.parse_args(argv)
    try:
        repo=args.repo.resolve(strict=True); private=args.private_root.resolve(strict=True)
        report_path=None
        if args.snapshot_file:
            report_path=args.snapshot_file.absolute()
            target_parent=(private/'m11_future/source_inventories').resolve()
            if report_path.is_symlink() or report_path.exists() or not report_path.parent.is_dir() or not report_path.parent.resolve().is_relative_to(target_parent):
                raise ValueError('Report must be NEW file in existing private source_inventories directory')
        report=audit_inventory(repo,private,args.inventory_file,args.expected_inventory_sha256,
                               max_csv_bytes=args.max_csv_mib*1024*1024,max_rows=args.max_rows)
        print('M11_STAGE8_ES_COVERAGE:',report['status'])
        print('ES_INVENTORY_BUCKET_FILES:',report['es_bucket_count_by_directory_only'])
        for k,v in report['category_counts'].items():
            print(k.upper()+':',v)
        print('CSV_INSPECTION_STATUS_COUNTS:',json.dumps(report['csv_status_counts'],sort_keys=True))
        print('ROWS_STRUCTURALLY_INSPECTED_NOT_VALIDATED:',report['rows_structurally_inspected_not_validated'])
        print('TRAINING_ROWS_CERTIFIED: 0 | VINTAGE_CERTIFIED: NO | MODELS_TRAINED: 0')
        print('NETWORK_REQUESTS: 0 | FORECAST_SUBMISSIONS: 0')
        if report_path:
            payload=(json.dumps(report, indent=2,ensure_ascii=False)+'\n').encode('utf-8')
            with report_path.open('xb') as f:
                f.write(payload)
            print('PRIVATE_REPORT_WRITTEN:',report_path)
            print('PRIVATE_REPORT_SHA256:',sha_bytes(payload))
        else:
            print('FILES_WRITTEN: 0')
        return 0 if report['status']=='PASS_WITH_LIMITATIONS' else 3
    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError, OverflowError) as ex:
        print('M11_STAGE8_STOP:',type(ex).__name__,str(ex),file=sys.stderr)
        return 2

if __name__=='__main__':
    sys.exit(run())
