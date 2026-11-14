#!/usr/bin/env python3
"""Read-only, offline verifier for Arena authenticated own-prediction revisions.

Input 1: captured own-prediction JSON (not a live request)
Input 2: captured public prediction-opportunity-audit JSON (not a live request)

Server submitted_at establishes timing of a retained revision, NOT model execution,
account-specific opportunity availability, or official scoring attribution.

No network, submissions, changes to original files, or reports by default.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

SCHEMA_VERSION = 'phase_a_m10_revision_timing_audit_v0_1'
AUDIT_CONTRACT = 'financial-opportunity-audit-v1'
CLASSES = ('bearish', 'neutral', 'bullish')


class AuditError(ValueError):
    pass


def read_json(path: Path, expected_sha256: str):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256.lower()):
        raise AuditError('expected SHA-256 must be a 64-character hexadecimal digest')
    if digest != expected_sha256.lower():
        raise AuditError(f'SHA256 mismatch for {path.name}; input left unchanged')
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuditError(f'invalid JSON in {path.name}') from exc
    if not isinstance(obj, dict):
        raise AuditError('JSON top level must be an object')
    return obj, digest


def utc_datetime(value, field: str):
    """Require a timezone-aware ISO timestamp; never infer UTC from a naive value."""
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f'{field}: missing timestamp')
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise AuditError(f'{field}: malformed timestamp') from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise AuditError(f'{field}: timezone is required')
    return dt.astimezone(timezone.utc)


def vector_check(row):
    v = row.get('prediction', {}).get('probabilities') if isinstance(row.get('prediction'), dict) else None
    if not isinstance(v, dict) or set(v) != set(CLASSES):
        raise AuditError('prediction.probabilities must have exactly bearish, neutral, bullish')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in v.values()):
        raise AuditError('probability values must be finite nonnegative numbers')
    total = sum(v.values())
    if all(x <= 1 for x in v.values()) and abs(total - 1) <= 0.002:
        representation = 'unit_interval'
    elif all(x <= 100 for x in v.values()) and abs(total - 100) <= 0.2:
        representation = 'percentage'
    else:
        raise AuditError('probabilities do not sum to 1 or 100 within known tolerances')
    canonical = json.dumps({k: v[k] for k in CLASSES}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return representation, hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def build_report(readback, audit, readback_sha, audit_sha, challenge_id):
    if readback.keys() == {'detail'}:
        raise AuditError('own-prediction response is an API error, not readback data')
    rows = readback.get('predictions')
    if not isinstance(rows, list) or not rows:
        raise AuditError('own-prediction response requires nonempty predictions array')
    if audit.get('api_contract_version') != AUDIT_CONTRACT or audit.get('asset') != 'ES':
        raise AuditError('unsupported public audit contract or asset')
    entries = audit.get('entries')
    if not isinstance(entries, list):
        raise AuditError('public audit entries missing')
    matches = [x for x in entries if isinstance(x, dict) and x.get('challenge_id') == challenge_id]
    if len(matches) != 1:
        raise AuditError(f'expected exactly one public audit challenge match; found {len(matches)}')
    entry = matches[0]
    deadline = utc_datetime(entry.get('deadline'), 'official deadline')
    window = audit.get('window') or {}
    start = utc_datetime(window.get('start'), 'window.start')
    end = utc_datetime(window.get('end'), 'window.end')
    as_of = utc_datetime(audit.get('as_of'), 'audit.as_of')
    snapshot_at = utc_datetime(audit.get('snapshot_at'), 'audit.snapshot_at')
    if start >= end or not(start <= deadline < end) or as_of < start:
        raise AuditError('invalid public audit deadline window/as_of')
    if snapshot_at < as_of:
        raise AuditError('audit snapshot must not precede requested evidence cutoff')
    if audit.get('next_cursor') is not None:
        raise AuditError('public audit pagination is incomplete; fetch all pages first')
    if entry.get('asset') != 'ES':
        raise AuditError('public challenge is not ES')
    coverage = audit.get('coverage')
    if not isinstance(coverage, dict):
        raise AuditError('missing coverage information')

    records = []
    revision_numbers = set()
    for row in rows:
        if not isinstance(row, dict):
            raise AuditError('invalid revision object')
        if row.get('challenge_id') != challenge_id:
            raise AuditError('unexpected challenge ID in authenticated readback')
        rev = row.get('revision_number')
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 1 or rev in revision_numbers:
            raise AuditError('missing, invalid or duplicate revision number')
        revision_numbers.add(rev)
        prediction_id = row.get('prediction_id')
        if not isinstance(prediction_id, str) or not prediction_id.strip():
            raise AuditError('missing prediction ID')
        repr_name, vector_sha = vector_check(row)
        submitted_raw = row.get('submitted_at')
        if submitted_raw is None:
            submitted = None
            timing = 'SUBMISSION_TIME_UNKNOWN'
            safety = 'UNKNOWN'
            margin_seconds = None
        else:
            try:
                submitted = utc_datetime(submitted_raw, f'revision {rev} submitted_at')
                margin_seconds = round((deadline - submitted).total_seconds(), 6)
                timing = 'SERVER_TIME_PREDEADLINE' if submitted < deadline else 'SERVER_TIME_AT_OR_AFTER_DEADLINE'
                safety = 'BEFORE_FIVE_MINUTE_CUTOFF' if submitted < deadline - timedelta(minutes=5) else 'AT_OR_AFTER_FIVE_MINUTE_CUTOFF'
            except AuditError:
                submitted = None
                margin_seconds = None
                timing = 'SUBMISSION_TIME_INVALID'
                safety = 'UNKNOWN'
        records.append({
            'challenge_id': challenge_id,
            'prediction_id': prediction_id,
            'revision_number': rev,
            'server_submitted_at_raw': submitted_raw,
            'server_submitted_at_utc': submitted.isoformat().replace('+00:00', 'Z') if submitted else None,
            'official_deadline_utc': deadline.isoformat().replace('+00:00', 'Z'),
            'timing_status': timing,
            'five_minute_safety_status': safety,
            'margin_seconds': margin_seconds,
            'probability_source': row.get('probability_source'),
            'probability_representation': repr_name,
            'server_readback_vector_sha256': vector_sha,
            'original_M1_vector_exactly_matched': False,
            'scored_revision_certified_by_this_readback': False,
        })
    records.sort(key=lambda x: x['revision_number'])
    counts = dict(Counter(x['timing_status'] for x in records))
    return {
        'schema_version': SCHEMA_VERSION,
        'status': 'OFFLINE_REVISION_TIMING_CHECK_COMPLETE',
        'challenge_id': challenge_id,
        'server_readback_sha256': readback_sha,
        'public_opportunity_audit_sha256': audit_sha,
        'revision_count': len(records),
        'timing_counts': counts,
        'records': records,
        'public_audit_coverage_complete': coverage.get('complete') is True,
        'public_audit_coverage_status': coverage.get('status'),
        'public_open_before_as_of': (entry.get('public_open_before_as_of') or {}).get('status'),
        'reader_delivery_before_as_of': (entry.get('reader_delivery_before_as_of') or {}).get('status'),
        'account_eligibility_status_at_as_of': entry.get('submission_eligibility_at_as_of'),
        'round_state_at_as_of': entry.get('state_at_as_of'),
        'official_deadline_evidence_basis': audit.get('time_basis'),
        'historical_deadline_immutability_certified': False,
        'model_execution_certified': False,
        'scored_revision_certified': False,
        'historical_eligible_opportunity_census_certified': False,
        'protocol_adopted': False,
        'network_requests': 0,
        'prediction_submissions': 0,
        'original_files_modified': 0,
    }


def outside_repo(path: Path):
    dest = path.resolve()
    if any((parent / '.git').exists() for parent in [dest.parent, *dest.parent.parents]):
        raise AuditError('private report must be outside a Git repository')


def write_private_report(path: Path, report):
    outside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as f:
        f.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readback', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--expected-readback-sha256', required=True)
    parser.add_argument('--expected-audit-sha256', required=True)
    parser.add_argument('--challenge-id', required=True)
    parser.add_argument('--private-report', type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        readback, rsha = read_json(args.readback, args.expected_readback_sha256)
        audit, asha = read_json(args.audit, args.expected_audit_sha256)
        report = build_report(readback, audit, rsha, asha, args.challenge_id)
        if args.private_report is not None:
            write_private_report(args.private_report, report)
        print('M10A OFFLINE REVISION AUDIT: OK')
        print('CHALLENGE_MATCH: True')
        print('READBACK_SHA256:', rsha)
        print('AUDIT_SHA256:', asha)
        print('REVISION_COUNT:', report['revision_count'])
        print('TIMING_COUNTS:', json.dumps(report['timing_counts'], sort_keys=True))
        for x in report['records']:
            print('REVISION:', x['revision_number'], 'TIMING:', x['timing_status'], 'SAFETY:', x['five_minute_safety_status'], 'MARGIN_SECONDS:', x['margin_seconds'])
        print('HISTORICAL_COVERAGE_COMPLETE:', report['public_audit_coverage_complete'])
        print('ACCOUNT_ELIGIBILITY_CERTIFIED_BY_PUBLIC_AUDIT: NO')
        print('MODEL_EXECUTION_CERTIFIED: NO | SCORED_REVISION_CERTIFIED: NO')
        print('NETWORK_REQUESTS: 0 | SUBMISSIONS: 0 | ORIGINAL_FILES_MODIFIED: 0')
        print('PRIVATE_REPORT_WRITTEN:', args.private_report is not None)
        return 0
    except (AuditError, OSError, TypeError, KeyError) as exc:
        print('M10A OFFLINE REVISION AUDIT: FAILED:', str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
