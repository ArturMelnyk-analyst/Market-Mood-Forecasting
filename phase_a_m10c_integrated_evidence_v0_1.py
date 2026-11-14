#!/usr/bin/env python3
"""M10C conservative, offline ES evidence join (observed historical 19 only).

Never sends network requests, makes forecasts, modifies source files, or changes
Notebook 12. Does not infer prior model execution, account-specific opportunity
visibility, prospective adoption, or the scored revision from server timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCHEMA_VERSION = 'm10c_integrated_evidence_v0_1'
AUDIT_SHA = '5a4bc95d600dbd1af49cb566cebd4f91644ff62ede678f818b98f4ea071ca2ad'
M10B_SHA = 'eba38ac15dd4b9d011f62dab58c51186d1b81505c97878395ef646bab2b06c8f'
SINGLE_SHA = '43d136992bef1f2d7bff812ad5400bba60f00d1aa4dfd6c9b329b6ae5db78532'
SINGLE_RAW_SHA = 'bb70769d7014d96c5563418b53168858e19ed8a8462768c17324f4f15891e78f'
BATCH_SHA = 'ca9bb6cd43d1be2d3015ad264287b3f978bd3388fc20f148c3c5b13fe1a43170'
MODEL_SHA = '5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d'
SINGLE_CID = '9e7a3255-e4dc-4a1d-a8f9-a30f18c4fa2c'
CLASSES = ('bearish', 'bullish', 'neutral')
UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

class EvidenceError(ValueError):
    pass


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path, sha: str | None = None):
    raw = path.read_bytes()
    actual = digest(raw)
    if sha is not None and actual != sha:
        raise EvidenceError('Pinned SHA256 does not match: ' + path.name)
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise EvidenceError('Unreadable JSON: ' + path.name) from exc
    if not isinstance(obj, dict):
        raise EvidenceError('Expected JSON object: ' + path.name)
    return obj, actual


def utc(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None or dt.utcoffset() is None:
            return None
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def canonical_vector(v):
    if not isinstance(v, dict) or set(v) != set(CLASSES):
        raise EvidenceError('Probability vector must include exactly 3 ES classes')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in v.values()):
        raise EvidenceError('Invalid probability values')
    total = sum(v.values())
    if all(x <= 1 for x in v.values()) and abs(total - 1) <= 0.002:
        return {k: float(v[k]) for k in CLASSES}, 'unit_interval'
    if all(x <= 100 for x in v.values()) and abs(total - 100) <= 0.2:
        return {k: float(v[k])/100 for k in CLASSES}, 'percentage'
    raise EvidenceError('Unexpected probability normalization')


def vector_comparison(local, retained, representation):
    if local is None:
        return 'NO_VERIFIED_LOCAL_VECTOR'
    if all(abs(local[k] - retained[k]) <= 1e-12 for k in CLASSES):
        return 'EXACT_NUMERIC_MATCH'
    if representation == 'percentage' and all(abs(round(100*local[k]) - 100*retained[k]) <= 1e-8 for k in CLASSES):
        return 'PERCENT_ROUNDING_CONSISTENT_NOT_EXACT'
    return 'VECTOR_MISMATCH_REVIEW'


def validated_local_source(repo: Path, cid: str):
    files = list((repo/'data/processed/es_arena_live/forecasts').glob('forecast_*_'+cid+'.json'))
    if len(files) != 1:
        return {'status': 'ORIGINAL_FORECAST_ABSENT' if not files else 'MULTIPLE_ORIGINAL_FORECASTS',
                'probabilities': None, 'forecast_sha256': None}
    path = files[0]
    raw = path.read_bytes()
    try:
        f = json.loads(raw)
        if not isinstance(f, dict):
            raise ValueError('not object')
    except (ValueError, UnicodeError):
        return {'status':'ORIGINAL_FORECAST_INVALID', 'probabilities':None, 'forecast_sha256':digest(raw)}
    if (f.get('challenge_id') != cid or f.get('contract_symbol') != 'ESZ6'
        or f.get('model_sha256') != MODEL_SHA):
        return {'status':'ORIGINAL_FORECAST_ID_OR_MODEL_CONFLICT', 'probabilities':None, 'forecast_sha256':digest(raw)}
    try:
        prob, repr_name = canonical_vector(f.get('probabilities'))
        if repr_name != 'unit_interval' or abs(sum(prob.values())-1)>1e-8:
            raise EvidenceError('non-unit or malformed local vector')
    except EvidenceError:
        return {'status':'ORIGINAL_FORECAST_VECTOR_INVALID', 'probabilities':None, 'forecast_sha256':digest(raw)}
    return {'status':'LOCAL_FILE_CLAIMS_FROZEN_M1', 'probabilities':prob,
            'forecast_sha256':digest(raw), 'filename':path.name}


def receipt_link(repo: Path, cid: str, forecast_sha: str | None, readback_rows):
    path = repo/'data/processed/es_arena_live/receipts'/('receipt_'+cid+'.json')
    if not path.is_file():
        return 'LOCAL_RECEIPT_UNAVAILABLE'
    try:
        receipt, _ = load_json(path)
    except (EvidenceError, OSError):
        return 'LOCAL_RECEIPT_UNREADABLE'
    response = receipt.get('response')
    if not isinstance(response, dict):
        return 'LOCAL_RECEIPT_RESPONSE_MISSING'
    if receipt.get('http_status') != 201 or response.get('challenge_id') != cid or response.get('counts_for_score') is not True:
        return 'LOCAL_RECEIPT_ACCEPTANCE_CONFLICT'
    if forecast_sha is None or receipt.get('forecast_sha256') != forecast_sha:
        return 'LOCAL_RECEIPT_FORECAST_HASH_UNVERIFIED'
    pid = response.get('prediction_id')
    if not isinstance(pid, str) or not pid:
        return 'LOCAL_RECEIPT_PREDICTION_ID_UNAVAILABLE'
    matches = [x for x in readback_rows if x.get('prediction_id') == pid]
    if len(matches) != 1:
        return 'LOCAL_RECEIPT_PREDICTION_ID_MISMATCH'
    rev = response.get('revision_number')
    if rev is not None and rev != matches[0].get('revision_number'):
        return 'LOCAL_RECEIPT_REVISION_MISMATCH'
    if response.get('probabilities') is not None:
        try:
            a, _ = canonical_vector(response['probabilities'])
            b, _ = canonical_vector(matches[0]['prediction']['probabilities'])
            if vector_comparison(a,b,'unit_interval') != 'EXACT_NUMERIC_MATCH':
                return 'LOCAL_RECEIPT_SERVER_VECTOR_CONFLICT'
        except (EvidenceError, KeyError, TypeError):
            return 'LOCAL_RECEIPT_VECTOR_INVALID'
    return 'LOCAL_ACCEPTANCE_RECEIPT_ID_LINKED_NOT_FINAL_SCORED_PROOF'


def validate_id_sets(audit, m10b, single, batch, reconciliation):
    if audit.get('api_contract_version') != 'financial-opportunity-audit-v1' or audit.get('asset') != 'ES':
        raise EvidenceError('Unexpected public audit contract')
    if audit.get('next_cursor') is not None:
        raise EvidenceError('Audit result is paginated; no complete captured retained set')
    if audit.get('coverage', {}).get('complete') is not False:
        raise EvidenceError('Unexpected historical coverage semantics; review required')
    entries = audit.get('entries')
    if not isinstance(entries, list) or len(entries) != 19:
        raise EvidenceError('Unexpected historical audit count')
    ids = [e.get('challenge_id') for e in entries]
    if len(set(ids)) != 19 or any(not isinstance(i,str) or not UUID.fullmatch(i) for i in ids):
        raise EvidenceError('Duplicate/unsafe audit IDs')
    m10b_records = m10b.get('challenge_records')
    if not isinstance(m10b_records, list) or {r.get('challenge_id') for r in m10b_records} != set(ids):
        raise EvidenceError('M10B UUID set mismatch')
    if m10b.get('source_sha256') != AUDIT_SHA:
        raise EvidenceError('M10B report not linked to pinned audit')
    if single.get('challenge_id') != SINGLE_CID or single.get('public_opportunity_audit_sha256') != AUDIT_SHA:
        raise EvidenceError('Single M10A report mismatch')
    if batch.get('public_audit_sha256') != AUDIT_SHA:
        raise EvidenceError('Batch M10A audit hash mismatch')
    records = batch.get('records')
    if not isinstance(records, list) or len(records) != 18 or set(r.get('challenge_id') for r in records) != set(ids)-{SINGLE_CID}:
        raise EvidenceError('Batch M10A UUID set mismatch')
    if any(r.get('capture_state') != 'VALIDATED_READBACK' for r in records):
        raise EvidenceError('Batch M10A validation not complete')
    recs = reconciliation.get('records')
    if not isinstance(recs,list) or {r.get('challenge_id') for r in recs} != set(ids):
        raise EvidenceError('Local reconciliation UUID set mismatch')
    if reconciliation.get('review_issues'):
        raise EvidenceError('Local reconciliation has review issues')
    return {e['challenge_id']:e for e in entries}, {r['challenge_id']:r for r in records}, {r['challenge_id']:r for r in recs}


def build_integrated(repo: Path, audit:dict, m10b:dict, single:dict, batch:dict,
                     single_readback:dict, batch_dir:Path, reconciliation:dict):
    amap, bmap, lmap = validate_id_sets(audit,m10b,single,batch,reconciliation)
    if single.get('server_readback_sha256') != SINGLE_RAW_SHA:
        raise EvidenceError('Single M10A body fingerprint mismatch')
    records=[]
    findings=[]
    summary=Counter()
    for cid in sorted(amap):
        entry=amap[cid]
        if cid == SINGLE_CID:
            blob=single_readback
        else:
            capdir=batch_dir/cid
            meta,_=load_json(capdir/'metadata.json')
            if meta.get('challenge_id') != cid or meta.get('http_status') != 200:
                raise EvidenceError('Invalid capture metadata for ID '+cid)
            blob,body_sha=load_json(capdir/'body.json',meta.get('body_sha256'))
            if bmap[cid].get('body_sha256') != body_sha or meta.get('body_bytes') != (capdir/'body.json').stat().st_size:
                raise EvidenceError('Batch capture evidence hash mismatch for ID '+cid)
        rawrows=blob.get('predictions')
        if not isinstance(rawrows,list) or not rawrows:
            raise EvidenceError('Empty or malformed readback for ID '+cid)
        numbers=set()
        local=validated_local_source(repo,cid)
        status=receipt_link(repo,cid,local.get('forecast_sha256'),rawrows)
        deadline=utc(entry.get('deadline'))
        if deadline is None:
            raise EvidenceError('Missing official deadline for ID '+cid)
        if len(rawrows)!=1:
            findings.append('MULTIPLE_REVISIONS_REVIEW_REQUIRED')
        for rev in rawrows:
            if not isinstance(rev,dict) or rev.get('challenge_id')!=cid or not isinstance(rev.get('prediction_id'),str) or not rev['prediction_id']:
                raise EvidenceError('Malformed retained revision for ID '+cid)
            n=rev.get('revision_number')
            if type(n)!=int or n<1 or n in numbers:
                raise EvidenceError('Invalid or duplicate revision number for ID '+cid)
            numbers.add(n)
            retained, representation=canonical_vector((rev.get('prediction') or {}).get('probabilities'))
            submission=utc(rev.get('submitted_at'))
            if submission is None:
                timing='UNKNOWN_SERVER_TIMESTAMP'; margin=None
            else:
                delta=deadline-submission
                margin=round(delta.total_seconds(),6)
                timing=('AT_OR_AFTER_DEADLINE' if delta<=timedelta(0) else
                        'PREDEADLINE_INSIDE_FIVE_MINUTE_BUFFER' if delta<=timedelta(minutes=5) else
                        'PREDEADLINE_BUFFER_MET')
            compare=vector_comparison(local['probabilities'],retained,representation)
            obs={
                'challenge_id':cid,'prediction_id':rev['prediction_id'],'revision_number':n,
                'acceptance_evidence':lmap[cid]['acceptance_evidence'],
                'official_settlement_record_present':bool(lmap[cid]['settled_in_local_official_csv']),
                'round_status_at_query_snapshot':entry.get('status'),
                'public_open_before_as_of':(entry.get('public_open_before_as_of') or {}).get('status'),
                'reader_delivery_before_as_of':(entry.get('reader_delivery_before_as_of') or {}).get('status'),
                'submission_eligibility_at_as_of':entry.get('submission_eligibility_at_as_of'),
                'readback_probability_source':rev.get('probability_source'),
                'readback_probability_representation':representation,
                'server_submitted_at':rev.get('submitted_at'),
                'retained_deadline_at_query_snapshot':entry.get('deadline'),
                'margin_seconds':margin,'timing_classification':timing,
                'local_forecast_status':local['status'],
                'local_forecast_sha256':local.get('forecast_sha256'),
                'original_vs_retained_vector':compare,
                'receipt_identity_linkage':status,
                'model_execution_certified':False,
                'final_scored_revision_certified':False,
                'historical_account_eligibility_certified':False,
            }
            records.append(obs)
            summary['timing_'+timing]+=1
            summary['vector_'+compare]+=1
            summary['receipt_'+status]+=1
            if compare=='VECTOR_MISMATCH_REVIEW' or status.endswith(('CONFLICT','MISMATCH')):
                findings.append('REVIEW_CONFLICT_FOR_ONE_OR_MORE_FORECASTS')
    records.sort(key=lambda x:(x['challenge_id'],x['revision_number']))
    if len(records)!=19:
        findings.append('UNEXPECTED_TOTAL_REVISION_COUNT')
    return {
        'schema_version':SCHEMA_VERSION,
        'scope':'observed_19_acceptance_evidenced_challenges_NOT_150_prospective_opportunities',
        'read_only':True,
        'source_audit_sha256':AUDIT_SHA,
        'source_m10b_report_sha256':M10B_SHA,
        'source_m10a_single_report_sha256':SINGLE_SHA,
        'source_m10a_batch_report_physical_sha256':BATCH_SHA,
        'source_m10a_single_readback_sha256':SINGLE_RAW_SHA,
        'retained_challenges':len(amap),'revision_records':len(records),
        'counts':dict(sorted(summary.items())),
        'records':records,
        'review_flags':sorted(set(findings)),
        'historical_eligible_opportunity_census_certified':False,
        'account_specific_prior_visibility_certified':False,
        'model_execution_certified':False,
        'final_scored_revision_certified':False,
        'phase_a_protocol_adopted':False,
        'deadline_immutability_since_challenge_open_certified':False,
        'limitations':[
            'Server submitted_at certifies readback-reported timing against retained deadline snapshot; not model execution.',
            'Public heartbeat coverage is partial; account-specific eligibility and historical opportunity census remain unknown.',
            'Current resolved status and an official settlement file do not prove which revision was scored.',
            'Exact local/readback probability agreement verifies consistency of retained values, not actual frozen model execution.',
            'A five-minute buffer is a separate safety policy; do not retrospectively exclude valid historical submissions.',
            'Original Notebook 12 first-30 accepted cohort and future prospective 150-opportunity study are unchanged.',
        ],
        'files_modified_by_audit':0,'network_requests':0,'prediction_submissions':0,
    }


def private_destination(path:Path):
    resolved=path.resolve()
    if 'MarketMood_Private' not in resolved.parts:
        raise EvidenceError('Private report must be under MarketMood_Private')
    if any((p/'.git').exists() for p in [resolved.parent,*resolved.parent.parents]):
        raise EvidenceError('Private report must not be inside a Git worktree')
    if path.exists():
        raise EvidenceError('Private output file exists; refusing overwrite')


def save_report(path:Path, obj):
    private_destination(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as f:
        f.write(raw)
    return digest(raw)


def load_repo_reconciliation(repo:Path):
    if not (repo/'phase_a_prospective_opportunity_reconcile_v0_1.py').is_file():
        raise EvidenceError('Expected local reconciliation module is missing')
    # Import user's pre-existing project code to construct the current local union.
    # No repo files are written. Use the canonical module path; no private data is printed.
    sys.path.insert(0,str(repo.resolve()))
    from phase_a_prospective_opportunity_reconcile_v0_1 import build_report
    return build_report(repo)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--private-report',type=Path)
    args=p.parse_args(argv)
    try:
        repo=args.repo.resolve()
        root=args.private_root.resolve()
        if root.name != 'phase_a_m10' or 'MarketMood_Private' not in root.parts:
            raise EvidenceError('Expected private phase_a_m10 evidence directory')
        audit,_=load_json(root/'public_probe_20261114_B'/'body.json',AUDIT_SHA)
        m10b,_=load_json(root/'m10b_private_report_20261114_A.json',M10B_SHA)
        single,_=load_json(root/'m10a_revision_timing_private_20261114_A.json',SINGLE_SHA)
        batch,_=load_json(root/'m10a_batch_private_20261114_A.json',BATCH_SHA)
        single_body,_=load_json(root/'own_prediction_probe_20261114_A'/'body.json',SINGLE_RAW_SHA)
        reconciliation=load_repo_reconciliation(repo)
        report=build_integrated(repo,audit,m10b,single,batch,single_body,
                                root/'own_prediction_batch_20261114_A',reconciliation)
        print('M10C INTEGRATED OFFLINE AUDIT: OK')
        print('RETAINED_CHALLENGES:',report['retained_challenges'])
        print('REVISION_RECORDS:',report['revision_records'])
        print('CLASSIFICATIONS:',json.dumps(report['counts'],sort_keys=True))
        print('REVIEW_FLAGS:',report['review_flags'])
        print('HISTORICAL_CENSUS_CERTIFIED: NO | MODEL_EXECUTION_CERTIFIED: NO | FINAL_SCORED_REVISION_CERTIFIED: NO')
        print('NETWORK_REQUESTS: 0 | SUBMISSIONS: 0 | INPUT_FILES_MODIFIED: 0')
        if args.private_report:
            print('PRIVATE_REPORT_SHA256:',save_report(args.private_report,report))
            print('PRIVATE_REPORT_WRITTEN: True')
        else:
            print('PRIVATE_REPORT_WRITTEN: False')
        return 0 if not report['review_flags'] else 1
    except (EvidenceError,OSError,ValueError,TypeError,KeyError,ImportError) as exc:
        print('M10C VALIDATION ERROR:',type(exc).__name__,str(exc),file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
