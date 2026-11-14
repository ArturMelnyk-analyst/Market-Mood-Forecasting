#!/usr/bin/env python3
"""M11 Stage 4: READ-ONLY, OFFLINE ES prospective-opportunity gate.

Checks pinned M11 Stage3 adoption/baseline, private candidate journal integrity,
then optional genuine preserved Arena discovery JSON. Does not enroll challenge IDs,
submit predictions, use the network, copy evidence, or change research files.

IMPORTANT: authenticated active feed and challenge created_at do NOT prove
independent account eligibility or official open_at_utc. A positive review
means only a candidate for a separate owner evidence review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import phase_a_m11_stage2_journal_v0_1 as journal_api
import phase_a_m11_stage3_immutable_baseline_v0_1 as baseline_api

ADOPT_COMMIT = '6f089df837e5dcd1aa5fc198490e287a98fa7fd8'
BRANCH = 'experiment/headline-arena-es-daily'
EFFECTIVE = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
DEFAULT_JOURNAL = 'm11_future/events/m11_prospective_candidate_events_v0_1.jsonl'
SHA = re.compile(r'^[a-f0-9]{64}$')
UUID = re.compile(r'^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$')
ES_CONTRACT = re.compile(r'^ES[FGHJKMNQUVXZ][0-9]$')

class GateError(ValueError):
    pass


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def utc(s: str, field: str):
    if not isinstance(s, str) or not s:
        raise GateError(field + '_MISSING')
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError as exc:
        raise GateError(field + '_INVALID') from exc
    if d.tzinfo is None or d.utcoffset() is None:
        raise GateError(field + '_TIMEZONE_MISSING')
    return d.astimezone(timezone.utc)


def inside(file: Path, roots: list[Path]) -> Path:
    p = file.resolve()
    if not any(p.is_relative_to(r.resolve()) for r in roots):
        raise GateError('PATH_OUTSIDE_ALLOWED_RESEARCH_ROOTS')
    return p


def verify_git(repo: Path):
    def git(*args):
        r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
        if r.returncode:
            raise GateError('GIT_CHECK_FAILED_' + '_'.join(args[:2]))
        return r.stdout.strip()
    if git('branch', '--show-current') != BRANCH:
        raise GateError('WRONG_BRANCH')
    git('merge-base', '--is-ancestor', ADOPT_COMMIT, 'HEAD')
    return git('rev-parse', 'HEAD')


def preflight(repo: Path, root: Path, journal: Path, now: datetime | None = None):
    repo, root = repo.resolve(), root.resolve()
    if not repo.is_dir() or not root.is_dir() or root.is_relative_to(repo):
        raise GateError('INVALID_REPO_OR_PRIVATE_ROOT')
    head = verify_git(repo)
    baseline = baseline_api.audit(repo, root, baseline_only=False)
    if baseline.get('status') != 'PASS' or baseline.get('research_design') != 'ORIGINAL_150_TOTAL_DESCRIPTIVE':
        raise GateError('BASELINE_OR_ADOPTION_NOT_VERIFIED')
    if baseline.get('owner_decision') != 'AFFIRMATIVELY_ADOPTED_BY_OWNER' or baseline.get('approved_protocol_effective_utc') != '2026-11-15T12:00:00Z':
        raise GateError('OWNER_ADOPTION_OR_EFFECTIVE_TIME_MISMATCH')
    j = inside(journal, [root])
    raw = j.read_bytes() if j.exists() else b''
    stage2 = journal_api.audit(root, j, digest(raw))
    if stage2['files_written'] != 0:
        raise GateError('JOURNAL_AUDIT_WROTE_FILES')
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise GateError('NAIVE_CURRENT_TIME')
    current = current.astimezone(timezone.utc)
    return {
        'status': 'PASS',
        'protocol_status': 'ADOPTED_EFFECTIVE' if current >= EFFECTIVE else 'ADOPTED_PENDING_EFFECTIVE_TIME',
        'current_utc': current.isoformat().replace('+00:00', 'Z'),
        'effective_utc': '2026-11-15T12:00:00Z',
        'branch_head': head,
        'adopted_protocol_commit': ADOPT_COMMIT,
        'historical_accepted_checkpoint': 19,
        'historical_opportunity_census_certified': False,
        'candidate_journal_path': str(j),
        'candidate_journal_sha256': digest(raw),
        'candidate_journal_events': stage2['events'],
        'candidate_journal_distinct_challenges': stage2['distinct_challenges'],
        'prospective_opportunities_enrolled_by_stage4': 0,
        'files_written': 0, 'network_calls': 0, 'forecast_submissions': 0,
    }


def get_entries(body):
    if isinstance(body, list):
        entries = body
    elif isinstance(body, dict):
        entries = body.get('challenges', body.get('items'))
    else:
        return []
    if not isinstance(entries, list):
        return []
    result = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        obj = item.get('challenge', item)
        if isinstance(obj, dict):
            result.append(obj)
    return result


def review_discovery(record: dict, *, expected_hash: str, actual_hash: str):
    if actual_hash != expected_hash:
        raise GateError('DISCOVERY_FILE_SHA256_MISMATCH')
    if record.get('schema_version') != 'arena_discovery_v0_1':
        raise GateError('UNSUPPORTED_DISCOVERY_SCHEMA')
    if record.get('status') != 'success' or record.get('http_status') != 200:
        return {'discovery_review': 'INCOMPLETE_OR_FAILED', 'challenge_reviews': [],
                'enrollment_authorized': False, 'notes': 'Failed/empty retrieval cannot prove no opportunity'}
    started = utc(record.get('request_started_at_utc'), 'REQUEST_STARTED')
    completed = utc(record.get('record_completed_at_utc'), 'RECORD_COMPLETED')
    if completed < started:
        raise GateError('DISCOVERY_TIME_ORDER_INVALID')
    candidates = record.get('es_daily_challenges')
    if not isinstance(candidates, list):
        raise GateError('DISCOVERY_MISSING_CHALLENGE_LIST')
    native = get_entries(record.get('response_json'))
    seen = set()
    result = []
    for item in candidates:
        if not isinstance(item, dict):
            raise GateError('INVALID_CHALLENGE_SUMMARY')
        cid = item.get('challenge_id')
        if not isinstance(cid, str) or not UUID.fullmatch(cid) or cid in seen:
            raise GateError('INVALID_OR_DUPLICATE_CHALLENGE_UUID')
        seen.add(cid)
        matches = [c for c in native if (c.get('id') or c.get('challenge_id')) == cid and
                   str(c.get('asset', c.get('scope_key', ''))).upper() == 'ES' and
                   c.get('challenge_type') == 'daily']
        if len(matches) != 1:
            result.append({'challenge_id': cid, 'review': 'OFFICIAL_RAW_BODY_MATCH_UNAVAILABLE',
                           'enrollment_authorized': False})
            continue
        official = matches[0]
        contract = official.get('contract_symbol')
        if not isinstance(contract, str) or not ES_CONTRACT.fullmatch(contract) or item.get('contract_symbol') != contract:
            result.append({'challenge_id': cid, 'review': 'CONTRACT_MISMATCH_OR_UNSUPPORTED',
                           'enrollment_authorized': False})
            continue
        deadline_raw = official.get('deadline_utc') or official.get('deadline_at') or official.get('deadline')
        try:
            deadline = utc(deadline_raw, 'DEADLINE')
        except GateError:
            result.append({'challenge_id': cid, 'review': 'DEADLINE_INVALID', 'enrollment_authorized': False})
            continue
        if item.get('deadline') is not None:
            try:
                if utc(item['deadline'], 'SUMMARY_DEADLINE') != deadline:
                    result.append({'challenge_id': cid, 'review': 'DEADLINE_SUMMARY_CONFLICT', 'enrollment_authorized': False})
                    continue
            except GateError:
                result.append({'challenge_id': cid, 'review': 'SUMMARY_DEADLINE_INVALID', 'enrollment_authorized': False})
                continue
        if completed >= deadline:
            reason = 'POST_DEADLINE_OBSERVATION_NOT_PREDEADLINE_EVIDENCE'
        else:
            official_open = official.get('open_at_utc') or official.get('opens_at') or official.get('opened_at')
            if not official_open:
                reason = 'OFFICIAL_OPEN_TIME_NOT_EVIDENCED_CREATED_AT_NOT_OPEN_AT'
            else:
                try:
                    opened = utc(official_open, 'OFFICIAL_OPEN')
                    if opened >= deadline or opened > completed:
                        reason = 'OPENING_TIME_INCONSISTENT_WITH_OBSERVATION'
                    elif opened < EFFECTIVE:
                        reason = 'OPENED_BEFORE_PROSPECTIVE_EFFECTIVE_TIME'
                    else:
                        reason = 'CANDIDATE_ACCOUNT_ELIGIBILITY_AND_TIMING_REVIEW_REQUIRED'
                except GateError:
                    reason = 'OFFICIAL_OPEN_TIME_INVALID'
        result.append({'challenge_id': cid, 'review': reason, 'contract': contract,
                       'deadline_utc': deadline.isoformat().replace('+00:00', 'Z'),
                       'enrollment_authorized': False})
    return {'discovery_review': 'REVIEW_REQUIRED' if result else 'SUCCESS_EMPTY_FEED_COVERAGE_UNKNOWN',
            'challenge_reviews': result, 'enrollment_authorized': False,
            'notes': 'Authenticated feed, local hash and HTTP Date are not independent account-specific eligibility proof'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight', 'review-discovery'))
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--private-root', required=True, type=Path)
    p.add_argument('--journal', type=Path)
    p.add_argument('--discovery-file', type=Path)
    p.add_argument('--expected-discovery-sha256')
    args = p.parse_args(argv)
    try:
        repo, root = args.repo.resolve(), args.private_root.resolve()
        j = args.journal or root / DEFAULT_JOURNAL
        report = preflight(repo, root, j)
        if args.mode == 'review-discovery':
            if not args.discovery_file or not args.expected_discovery_sha256 or not SHA.fullmatch(args.expected_discovery_sha256):
                raise GateError('DISCOVERY_FILE_AND_EXACT_SHA256_REQUIRED')
            file = inside(args.discovery_file, [repo, root])
            # Only local retained write-once JSON files, no remote fetches.
            raw = file.read_bytes()
            obj = json.loads(raw.decode('utf-8-sig'))
            if not isinstance(obj, dict):
                raise GateError('DISCOVERY_NOT_JSON_OBJECT')
            report['discovery_file_sha256'] = digest(raw)
            report.update(review_discovery(obj, expected_hash=args.expected_discovery_sha256, actual_hash=digest(raw)))
        print('M11_STAGE4_GATE:', report['status'])
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print('M11_STAGE4_STOP:', type(exc).__name__, str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
