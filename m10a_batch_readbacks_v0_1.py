#!/usr/bin/env python3
"""Controlled, resumable own-only ES readback capture; offline M10A summary.

No POST, no repository edits, no token logging. 'plan' is default and offline.
'fetch' explicitly performs authenticated GETs for IDs in the pinned public
opportunity audit, except excluded IDs. All server bodies are kept privately
and written exclusively; incomplete captures are never silently repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

CONTRACT = 'financial-opportunity-audit-v1'
ROUTE = '/api/v1/eval/challenges/{challenge_id}/own-prediction'


class CaptureError(ValueError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_audit(path: Path, expected_sha: str, expected_count: int = 19):
    if len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha.lower()):
        raise CaptureError('expected audit SHA-256 must be 64 hexadecimal characters')
    raw = path.read_bytes()
    if sha256(raw) != expected_sha.lower():
        raise CaptureError('public audit SHA-256 mismatch; source not modified')
    try:
        audit = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise CaptureError('invalid public audit JSON') from exc
    if not isinstance(audit, dict) or audit.get('api_contract_version') != CONTRACT or audit.get('asset') != 'ES':
        raise CaptureError('unexpected public audit contract or asset')
    if audit.get('next_cursor') is not None:
        raise CaptureError('public audit pagination incomplete; do not infer a census')
    entries = audit.get('entries')
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise CaptureError('unexpected retained challenge count; review source before capture')
    identifiers = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('asset') != 'ES':
            raise CaptureError('non-ES or malformed audit entry')
        cid = entry.get('challenge_id')
        try:
            canonical = str(uuid.UUID(cid))
        except (TypeError, AttributeError, ValueError) as exc:
            raise CaptureError('malformed challenge UUID in public audit') from exc
        if cid != canonical:
            raise CaptureError('noncanonical challenge UUID in public audit')
        identifiers.append(cid)
    if len(set(identifiers)) != len(identifiers):
        raise CaptureError('duplicate challenge UUIDs in audit')
    return audit, sorted(identifiers)


def target_ids(identifiers, excluded):
    excluded = set(excluded)
    if not excluded.issubset(set(identifiers)):
        raise CaptureError('excluded challenge UUID was not found in the pinned audit')
    return [cid for cid in identifiers if cid not in excluded]


def require_private(path: Path):
    dest = path.resolve()
    if 'MarketMood_Private' not in dest.parts:
        raise CaptureError('destination must be within a MarketMood_Private directory')
    if any((parent / '.git').exists() for parent in [dest.parent, *dest.parent.parents]):
        raise CaptureError('private captures/reports must be outside Git repositories')


def validate_existing(folder: Path, cid: str):
    """Return preserved metadata or None; never overwrite/dismiss partial state."""
    path = folder / cid
    if not path.exists():
        return None
    body = path / 'body.json'
    meta = path / 'metadata.json'
    if not body.is_file() or not meta.is_file():
        raise CaptureError('partial capture detected; preserve and investigate before resuming')
    try:
        data = json.loads(meta.read_text(encoding='utf-8'))
    except (UnicodeError, ValueError) as exc:
        raise CaptureError('invalid existing capture metadata') from exc
    if (not isinstance(data, dict) or data.get('challenge_id') != cid
            or data.get('body_sha256') != sha256(body.read_bytes())
            or data.get('body_bytes') != body.stat().st_size):
        raise CaptureError('existing capture hash/metadata mismatch; preserve original')
    return data


def store_response(folder: Path, cid: str, status: int, content_type: str, body: bytes):
    """Save raw response for any status. No authorization headers or secrets."""
    path = folder / cid
    path.mkdir(exist_ok=False)
    with (path / 'body.json').open('xb') as f:
        f.write(body)
    meta = {
        'challenge_id': cid,
        'endpoint': ROUTE.format(challenge_id=cid),
        'http_status': status,
        'content_type': content_type,
        'body_sha256': sha256(body),
        'body_bytes': len(body),
        'capture_method': 'authenticated_GET_only',
        'server_submitted_at_certified_from_metadata': False,
    }
    with (path / 'metadata.json').open('x', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write('\n')
    return meta


def readback_shape(body: bytes, cid: str):
    try:
        data = json.loads(body)
    except (UnicodeError, ValueError):
        return 'INVALID_JSON', 0
    if not isinstance(data, dict) or not isinstance(data.get('predictions'), list):
        return 'UNEXPECTED_SHAPE', 0
    rows = data['predictions']
    if any(not isinstance(r, dict) or r.get('challenge_id') != cid for r in rows):
        return 'CHALLENGE_MISMATCH', len(rows)
    return ('REVISIONS_RETURNED' if rows else 'NO_REVISIONS'), len(rows)


def fetch_batch(audit, ids, output: Path, repo: Path, *, resume=False,
                max_requests=2, requester=None, headers=None, sleep_seconds=0.25):
    """Only GET. Dependencies injectable for offline tests."""
    require_private(output)
    if output.exists() and not resume:
        raise CaptureError('output already exists; use --resume only after inspection')
    if not output.exists() and resume:
        raise CaptureError('resume output does not exist; use initial --fetch first')
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
    # Verify all existing captures before the first new network request.
    existing = {cid: validate_existing(output, cid) for cid in ids}
    missing = [cid for cid in ids if existing[cid] is None]
    if requester is None or headers is None:
        sys.path.insert(0, str(repo.resolve()))
        from es_arena_daily import BASE, authenticate
        parsed = urlsplit(BASE)
        if parsed.scheme != 'https' or parsed.hostname != 'headlinearena.com' or parsed.path not in ('', '/'):
            raise CaptureError('unexpected Arena base URL; no requests sent')
        import requests
        requester = requests.get
        headers = authenticate()
        base = BASE.rstrip('/')
    else:
        base = 'https://headlinearena.com'
    attempted, http200, failed = 0, 0, 0
    for cid in missing[:max_requests]:
        url = base + ROUTE.format(challenge_id=cid)
        try:
            res = requester(url, headers=headers, timeout=(5, 20), allow_redirects=False)
        except Exception as exc:  # No token/URL/response text in exceptions.
            print('NETWORK_ERROR_TYPE:', type(exc).__name__)
            failed += 1
            break
        status = int(res.status_code)
        ctype = str(res.headers.get('Content-Type', ''))[:80]
        payload = bytes(res.content)
        store_response(output, cid, status, ctype, payload)
        attempted += 1
        if status != 200:
            failed += 1
            print('HTTP_ERROR_STATUS:', status, '(body preserved privately; stopping)')
            break
        http200 += 1
        shape, revisions = readback_shape(payload, cid)
        print('CAPTURED: HTTP_200 SHAPE:', shape, 'REVISION_ROWS:', revisions)
        if shape not in ('REVISIONS_RETURNED', 'NO_REVISIONS'):
            failed += 1
            print('UNEXPECTED_READBACK_SHAPE: stopping; raw body preserved')
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    covered = sum(validate_existing(output, cid) is not None for cid in ids)
    print('REQUESTS_ATTEMPTED_THIS_RUN:', attempted)
    print('HTTP_200_THIS_RUN:', http200)
    print('CAPTURES_PRESENT:', covered)
    print('CAPTURES_EXPECTED:', len(ids))
    print('CAPTURES_REMAINING:', len(ids) - covered)
    print('ERRORS_THIS_RUN:', failed)
    print('PREDICTION_POSTS: 0 | EXISTING_BODIES_OVERWRITTEN: 0')
    return 0 if failed == 0 else 2


def load_m10a(path: Path):
    if not path.is_file():
        raise CaptureError('M10A auditor module not found')
    spec = importlib.util.spec_from_file_location('m10a_offline_verifier', path)
    if spec is None or spec.loader is None:
        raise CaptureError('M10A auditor module cannot be imported')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ('build_report', 'AuditError'):
        if not hasattr(module, attr):
            raise CaptureError('unexpected M10A auditor interface')
    return module


def summarize(audit, audit_sha, ids, output: Path, m10a_module: Path, report_path=None):
    require_private(output)
    if report_path is not None:
        require_private(report_path)
        if report_path.exists():
            raise CaptureError('private summary report already exists; refusing overwrite')
    module = load_m10a(m10a_module)
    state = Counter()
    timings = Counter()
    details = []
    for cid in ids:
        meta = validate_existing(output, cid)
        if meta is None:
            state['MISSING_READBACK'] += 1
            details.append({'challenge_id': cid, 'capture_state': 'MISSING_READBACK'})
            continue
        body = (output / cid / 'body.json').read_bytes()
        if meta['http_status'] != 200:
            state['HTTP_NOT_200'] += 1
            details.append({'challenge_id': cid, 'capture_state': 'HTTP_NOT_200', 'body_sha256': meta['body_sha256']})
            continue
        shape, count = readback_shape(body, cid)
        if shape != 'REVISIONS_RETURNED':
            state[shape] += 1
            details.append({'challenge_id': cid, 'capture_state': shape, 'body_sha256': meta['body_sha256']})
            continue
        try:
            report = module.build_report(json.loads(body), audit, meta['body_sha256'], audit_sha, cid)
        except (module.AuditError, ValueError, KeyError, TypeError) as exc:
            state['VALIDATION_FAILED'] += 1
            details.append({'challenge_id': cid, 'capture_state': 'VALIDATION_FAILED',
                            'error_type': type(exc).__name__, 'body_sha256': meta['body_sha256']})
            continue
        state['VALIDATED_READBACK'] += 1
        timings.update(report['timing_counts'])
        details.append({'challenge_id': cid, 'capture_state': 'VALIDATED_READBACK',
                        'body_sha256': meta['body_sha256'], 'revision_count': report['revision_count'],
                        'timing_counts': report['timing_counts'],
                        'model_execution_certified': False, 'scored_revision_certified': False})
    result = {
        'schema_version': 'm10a_batch_readbacks_v0_1',
        'public_audit_sha256': audit_sha,
        'source_coverage_complete': audit['coverage'].get('complete') is True,
        'scope': 'remaining_retained_challenges_only; previously verified single challenge excluded',
        'target_challenges': len(ids),
        'capture_states': dict(sorted(state.items())),
        'revision_timing_counts': dict(sorted(timings.items())),
        'records': details,
        'historical_eligible_opportunity_census_certified': False,
        'model_execution_certified': False,
        'scored_revision_certified': False,
        'no_post_requests_by_capture_code': True,
    }
    print('M10A BATCH OFFLINE SUMMARY: OK')
    print('TARGET_CHALLENGES:', len(ids))
    print('CAPTURE_STATES:', json.dumps(result['capture_states'], sort_keys=True))
    print('REVISION_TIMING_COUNTS:', json.dumps(result['revision_timing_counts'], sort_keys=True))
    print('HISTORICAL_COVERAGE_COMPLETE:', result['source_coverage_complete'])
    print('MODEL_EXECUTION_CERTIFIED: NO | SCORED_REVISION_CERTIFIED: NO')
    print('NETWORK_REQUESTS_DURING_SUMMARY: 0 | SUBMISSIONS: 0')
    if report_path is not None:
        # Do not write a report that appears to be an exhaustive verification if incomplete.
        if state['MISSING_READBACK']:
            raise CaptureError('summary contains missing captures; no final private report written')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
        # Hash the exact bytes written on every platform (including Windows).
        report_bytes = payload.encode('utf-8')
        with report_path.open('xb') as f:
            f.write(report_bytes)
        print('PRIVATE_SUMMARY_WRITTEN: True')
        print('PRIVATE_SUMMARY_SHA256:', sha256(report_bytes))
    else:
        print('PRIVATE_SUMMARY_WRITTEN: False')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'fetch', 'summarize'), default='plan')
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--expected-audit-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--expected-count', type=int, default=19)
    p.add_argument('--max-requests', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--m10a-module', type=Path)
    p.add_argument('--private-report', type=Path)
    args = p.parse_args(argv)
    try:
        if args.max_requests < 1 or args.max_requests > 25:
            raise CaptureError('max-requests must be between 1 and 25')
        audit, identifiers = read_audit(args.audit, args.expected_audit_sha256, args.expected_count)
        ids = target_ids(identifiers, args.exclude)
        require_private(args.out)
        print('PINNED_PUBLIC_AUDIT_SHA256:', args.expected_audit_sha256.lower())
        print('RETAINED_CHALLENGES_IN_AUDIT:', len(identifiers))
        print('EXCLUDED_ALREADY_VERIFIED:', len(args.exclude))
        print('TARGET_CHALLENGES:', len(ids))
        print('HISTORICAL_COVERAGE_COMPLETE:', audit.get('coverage', {}).get('complete') is True)
        if args.mode == 'plan':
            print('PLAN_ONLY: True | NETWORK_REQUESTS: 0 | FILES_WRITTEN: 0')
            return 0
        if args.mode == 'fetch':
            if args.private_report is not None:
                raise CaptureError('--private-report is only available in summarize mode')
            return fetch_batch(audit, ids, args.out, args.repo, resume=args.resume,
                               max_requests=args.max_requests)
        if args.resume:
            raise CaptureError('--resume only applies to fetch mode')
        if args.m10a_module is None:
            raise CaptureError('summarize mode requires --m10a-module')
        summarize(audit, args.expected_audit_sha256.lower(), ids, args.out,
                  args.m10a_module, args.private_report)
        return 0
    except (CaptureError, OSError, KeyError, TypeError) as exc:
        print('M10A BATCH ERROR:', str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
