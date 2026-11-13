"""Phase A ES opportunity reconciliation -- strictly read-only.

Joins the PINNED November 12 accepted-candidate snapshot with prospective,
archived ES discovery and original local forecast/attempt/receipt files.
Settlements are joined separately. Does not infer weekdays, misses, full census,
independent acceptance time, retrospective phase ordinals, or amendments.

Requires the existing phase_a_opportunity_audit_v0_1.py alongside this file.
Run: python phase_a_prospective_opportunity_reconcile_v0_1.py
     python phase_a_prospective_opportunity_reconcile_v0_1.py --json
No network, file writes, model loading, market-data purchases or submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from phase_a_opportunity_audit_v0_1 import (
    load_accepted, load_official, discovery_observations,
    utc_timestamp, header_date,
)

ROOT = Path(__file__).resolve().parent
MODEL_SHA = '5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d'
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
RECEIPT_PREFIX = 'receipt_'


def read_object(path: Path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object in {path.name}')
    return value


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_native_receipt(root: Path, challenge_id: str):
    """Classify original native evidence without changing files.

    A legitimate accepted receipt can exist with a missing local artifact, but
    this verifier deliberately labels it REVIEW_REQUIRED, not certified.
    """
    base = root / 'data/processed/es_arena_live'
    receipt_file = base / 'receipts' / f'receipt_{challenge_id}.json'
    attempt_file = base / 'receipts' / f'attempt_{challenge_id}.json'
    matches = list((base / 'forecasts').glob(f'forecast_*_{challenge_id}.json'))
    if not receipt_file.is_file() and not matches:
        return {'status': 'NO_NATIVE_EVIDENCE', 'target_date': None, 'problems': []}
    problems = []
    if len(matches) != 1:
        problems.append(f'Expected one original forecast file; found {len(matches)}')
    if not receipt_file.is_file():
        return {'status':'NO_ACCEPTED_RECEIPT_EVIDENCE', 'target_date':None, 'problems':problems}
    if len(matches) != 1:
        return {'status':'REVIEW_CONFLICT', 'target_date':None, 'problems':problems}
    forecast_file = matches[0]
    try:
        forecast = read_object(forecast_file)
        attempt = read_object(attempt_file) if attempt_file.is_file() else None
        receipt = read_object(receipt_file)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return {'status':'REVIEW_CONFLICT', 'target_date':None,
                'problems':[f'Unreadable original evidence: {type(exc).__name__}']}
    target = forecast.get('target_session_date')
    if not isinstance(target, str) or not DATE_RE.fullmatch(target):
        problems.append('Missing/invalid target_session_date')
        target = None
    elif forecast_file.name != f'forecast_{target.replace("-", "")}_{challenge_id}.json':
        problems.append('Forecast filename does not match forecast date and challenge ID')
    if forecast.get('challenge_id') != challenge_id:
        problems.append('Forecast challenge ID mismatch')
    if forecast.get('contract_symbol') != 'ESZ6':
        problems.append('Forecast contract is not ESZ6')
    if forecast.get('model_sha256') != MODEL_SHA:
        problems.append('Forecast is not pinned Phase A M1')
    deadline = utc_timestamp(forecast.get('deadline_utc'))
    if not deadline:
        problems.append('Forecast deadline has no timezone-aware value')
    prepared = utc_timestamp(forecast.get('prepared_at_utc'))
    if prepared and deadline and prepared >= deadline:
        problems.append('Locally recorded forecast preparation not before deadline')
    p = forecast.get('probabilities')
    if not (isinstance(p, dict) and set(p) == {'bearish','neutral','bullish'} and
            all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in p.values()) and
            abs(sum(p.values()) - 1) < 1e-8):
        problems.append('Invalid forecast probability vector')
    if attempt is None:
        problems.append('No original attempt evidence')
    if attempt is not None and attempt.get('challenge_id') != challenge_id:
        problems.append('Attempt challenge ID mismatch')
    forecast_hash = sha(forecast_file)
    if attempt is not None and attempt.get('forecast_sha256') != forecast_hash:
        problems.append('Attempt forecast SHA-256 mismatch')
    if receipt.get('forecast_sha256') != forecast_hash:
        problems.append('Receipt forecast SHA-256 mismatch')
    response = receipt.get('response')
    if not isinstance(response, dict):
        response = {}
    if not (receipt.get('http_status') == 201 and
            response.get('counts_for_score') is True and
            response.get('challenge_id') == challenge_id and
            isinstance(response.get('prediction_id'), str) and response['prediction_id'].strip()):
        problems.append('No valid stored scored acceptance response')
    if response.get('probabilities') is not None:
        fp = forecast.get('probabilities')
        rp = response['probabilities']
        if not (isinstance(fp, dict) and isinstance(rp, dict) and
                set(fp) == set(rp) == {'bearish','neutral','bullish'} and
                all(type(fp[k]) in (float,int) and type(rp[k]) in (float,int) and
                    abs(fp[k]-rp[k]) <= 1e-8 for k in fp)):
            problems.append('Original server and forecast probabilities disagree')
    return {
        'status': 'ACCEPTED_SCORED_LOCAL_CHAIN' if not problems else 'REVIEW_CONFLICT',
        'target_date': target,
        'problems': problems,
        'server_acceptance_time_zone_certified': False,  # never certify via saved local receipt
    }


def build_report(root: Path = ROOT):
    root = root.resolve()
    research = root / 'data/processed/es_arena_research_v0_1'
    live = root / 'data/processed/es_arena_live'
    pinned = load_accepted(research / 'arena_opportunity_candidate_20261112.csv')
    official = load_official(research / 'arena_official_settlements.csv')
    observations, checks = discovery_observations(
        discovery_dir=live / 'discovery',
        historical_path=live / 'active_challenges_inspection_v0_1.json',
        root=root,
    )
    # A successful discovery archive may still contain incomplete ES metadata.
    observations_by_id = defaultdict(list)
    for obs in observations:
        if obs.get('challenge_id'):
            observations_by_id[obs['challenge_id']].append(obs)
    candidate_ids = set(pinned)
    ids = candidate_ids | set(observations_by_id)
    # Receipts without a matching pinned candidate or discovery are included
    # only with an authentic-looking ESZ6 native forecast; never infer weekdays.
    receipt_dir = live / 'receipts'
    if receipt_dir.is_dir():
        for receipt_file in receipt_dir.glob('receipt_*.json'):
            cid = receipt_file.name[len(RECEIPT_PREFIX):-len('.json')]
            if not cid or cid in ids:
                continue
            forecasts_dir = live / 'forecasts'
            native_matches = list(forecasts_dir.glob(f'forecast_*_{cid}.json'))
            try:
                if len(native_matches) == 1 and read_object(native_matches[0]).get('contract_symbol') == 'ESZ6':
                    ids.add(cid)
            except (ValueError, OSError, json.JSONDecodeError):
                # Unknown market or damaged file: no invented ES opportunity.
                continue

    # Validate untrusted challenge identifiers BEFORE using them in filesystem paths.
    for cid in ids:
        if not UUID_RE.fullmatch(cid):
            raise ValueError('Unsafe/noncanonical challenge ID in retained evidence: '+repr(cid)[:130])
    records = []
    issues = []
    for cid in sorted(ids):
        ev = observations_by_id.get(cid, [])
        native = record_native_receipt(root, cid)
        legacy = cid in pinned
        statuses = []
        date_from_pinned = pinned[cid].get('settlement_date') if legacy else None
        date_from_native = native['target_date']
        if date_from_pinned and date_from_native and date_from_pinned != date_from_native:
            statuses.append('Pinned accepted-candidate date disagrees with native forecast')
        known_contracts = {x['contract'] for x in ev if x.get('contract')}
        known_deadlines = {x['deadline'] for x in ev if x.get('deadline')}
        if len(known_contracts) > 1 or (known_contracts and known_contracts != {'ESZ6'}):
            statuses.append('Discovery contract conflict')
        if len(known_deadlines) > 1:
            statuses.append('Discovery deadline conflict')
        if native['target_date']:
            forecast_path = live / 'forecasts' / f'forecast_{native["target_date"].replace("-", "")}_{cid}.json'
            if forecast_path.is_file():
                try:
                    forecast_deadline = utc_timestamp(read_object(forecast_path).get('deadline_utc'))
                    if known_deadlines and forecast_deadline and any(dl != forecast_deadline for dl in known_deadlines):
                        statuses.append('Discovery deadline differs from original forecast deadline')
                except (OSError, ValueError, json.JSONDecodeError):
                    statuses.append('Original forecast unreadable during deadline comparison')
        # independently successful and metadata-complete predeadline discovery;
        # both remote HTTP Date and locally logged completion must precede deadline.
        predeadline = False
        for x in ev:
            if x.get('evidence_kind') != 'authenticated_active_feed' or x.get('collection_status') != 'success':
                continue
            if x.get('contract') != 'ESZ6':
                continue
            deadline = x.get('deadline')
            server_date = x.get('server_date_at_capture')
            source_path = root / x['source_file']
            try:
                data = read_object(source_path)
                local_finished = utc_timestamp(data.get('record_completed_at_utc'))
                if (deadline and server_date and local_finished and server_date < deadline and local_finished < deadline
                        and data.get('http_status') == 200):
                    predeadline = True
            except (OSError, ValueError, json.JSONDecodeError):
                statuses.append('Discovery archive unreadable during strict timing check')
        if native['status'] == 'REVIEW_CONFLICT':
            statuses.extend(native['problems'])
        if native['status'] == 'NO_ACCEPTED_RECEIPT_EVIDENCE' and native['problems']:
            statuses.extend(native['problems'])
        if cid in official and not isinstance(official[cid], dict):
            statuses.append('Invalid official settlement record')
        if legacy:
            acceptance = 'PINNED_HISTORICAL_ACCEPTED'
        elif native['status'] == 'ACCEPTED_SCORED_LOCAL_CHAIN':
            acceptance = 'NATIVE_ACCEPTED_SCORED'
        else:
            acceptance = 'NO_CONFIRMED_ACCEPTANCE_IN_THIS_AUDIT'
        if statuses:
            issues.append({'challenge_id':cid,'problems':statuses})
        records.append({
            'challenge_id':cid,
            'target_session_date':date_from_native or date_from_pinned,
            'pinned_20261112_candidate':legacy,
            'acceptance_evidence':acceptance,
            'native_chain_status':native['status'],
            'settled_in_local_official_csv':cid in official,
            'discovery_observation_count':len(ev),
            'predeadline_discovery_http_date_evidence':predeadline,
            'discovery_archive_paths':[x['source_file'] for x in ev],
            'review_issues':statuses,
        })
    recorded_ids = {x['challenge_id'] for x in records}
    unmatched_official = sorted(set(official)-recorded_ids)
    if unmatched_official:
        issues.append({'challenge_id':None,'problems':[
            'Official settlements not matched to pinned/observed/native evidence: '+','.join(unmatched_official)]})
    archive_errors = [x['source'] for x in checks if x.get('status') == 'REVIEW_ERROR']
    if archive_errors:
        issues.append({'challenge_id':None,'problems':['Discovery archive read/parse error: '+','.join(archive_errors)]})
    acceptance_counts = Counter(r['acceptance_evidence'] for r in records)
    return {
        'schema_version':'phase_a_prospective_opportunity_reconcile_v0_1',
        'read_only':True,
        'pinned_20261112_accepted_count':len(pinned),
        'new_native_scored_count':acceptance_counts['NATIVE_ACCEPTED_SCORED'],
        'total_acceptance_evidenced_in_union':acceptance_counts['PINNED_HISTORICAL_ACCEPTED'] + acceptance_counts['NATIVE_ACCEPTED_SCORED'],
        'settled_records_in_union':sum(r['settled_in_local_official_csv'] for r in records),
        'eligible_opportunity_census_complete':False,
        'phase_a_ordinal_or_boundary_certified':False,
        'independent_utc_acceptance_time_certified':False,
        'original_notebook12_cohort':'first 30 accepted forecasts; UNCHANGED',
        'protocol_amendment_status':'NOT AUTOMATICALLY ADOPTED BY THIS TOOL',
        'discovery_archive_checks':checks,
        'records':records,
        'unmatched_official_challenge_ids':unmatched_official,
        'review_issues':issues,
        'limitations':[
            'This is an observed evidence union, not proof all eligible opportunities were discovered.',
            'Native receipts and local hashes do not independently certify UTC acceptance time.',
            'Missing receipt is not proof of a missed or failed submission.',
            'Historical accepted candidate is pinned to its original SHA; no source is rewritten.',
            'Historical and post-adoption boundaries/ordinals are NOT inferred from dates, weekdays or outcome.',
            'No independent authentication of saved discovery HTTP Date headers or original model execution.',
        ]
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(ROOT), help='Existing project root (defaults to this script directory)')
    p.add_argument('--json', action='store_true', help='Print full evidence index as JSON to stdout; no file writes')
    a = p.parse_args(argv)
    try:
        r = build_report(Path(a.root))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print('PHASE A PROSPECTIVE OPPORTUNITY RECONCILIATION (READ ONLY)')
        print('PINNED NOV 12 ACCEPTED:', r['pinned_20261112_accepted_count'])
        print('ADDITIONAL NATIVE SCORED:', r['new_native_scored_count'])
        print('TOTAL ACCEPTANCE EVIDENCED IN UNION:', r['total_acceptance_evidenced_in_union'])
        print('OFFICIAL SETTLED IN UNION:', r['settled_records_in_union'])
        for x in r['records']:
            if not x['pinned_20261112_candidate'] or x['review_issues']:
                print('REVIEW', x['challenge_id'], 'STATUS:', x['acceptance_evidence'],
                      'SETTLED:', x['settled_in_local_official_csv'],
                      'PREDEADLINE_DISCOVERY:', x['predeadline_discovery_http_date_evidence'],
                      'ISSUES:', x['review_issues'])
        print('ELIGIBLE OPPORTUNITY CENSUS: NOT ESTABLISHED')
        print('PROSPECTIVE AMENDMENT: NOT ADOPTED BY THIS TOOL')
        print('FILES WRITTEN: 0 | NETWORK REQUESTS: 0 | PREDICTION SUBMISSIONS: 0')
        if r['review_issues']:
            print('REVIEW ISSUES:', len(r['review_issues']))
    return 1 if r['review_issues'] else 0

if __name__ == '__main__':
    sys.exit(main())
