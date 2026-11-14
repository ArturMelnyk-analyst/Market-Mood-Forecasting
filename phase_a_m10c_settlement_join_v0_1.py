#!/usr/bin/env python3
"""Read-only, source-hash-pinned M10C settlement-presence/outcome reconciliation.

This is strictly an observed historical ES cohort audit; it DOES NOT certify
scored-revision identity, model execution, account visibility or opportunity
census completeness. It never computes or claims an official metric.
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
from datetime import datetime, timezone, date
from pathlib import Path

VERSION = 'm10c_settlement_join_v0_1'
INTEGRATED_VERSION = 'm10c_integrated_evidence_v0_1'
PUBLIC_AUDIT_SHA256 = '5a4bc95d600dbd1af49cb566cebd4f91644ff62ede678f818b98f4ea071ca2ad'
EXPECTED_COLS = {'challenge_id','settlement_date','asset','contract_symbol','official_label',
                 'settlement_return_pct','official_deadline_utc','official_settled_at_utc',
                 'arena_score','source_url','verified_by','verified_at_utc'}
CLASS_NAMES = {'bearish','bullish','neutral'}
UUID_RE = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$')
SHA_RE = re.compile(r'^[a-f0-9]{64}$')

class EvidenceError(ValueError):
    pass

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def require_hash(value:str) -> str:
    if not isinstance(value,str) or not SHA_RE.fullmatch(value):
        raise EvidenceError('Expected a 64-character lowercase SHA256 hex digest')
    return value

def load_pinned(path: Path, expected_hash: str) -> tuple[bytes,str]:
    require_hash(expected_hash)
    raw=path.read_bytes()
    digest=sha256(raw)
    if digest!=expected_hash:
        raise EvidenceError('SOURCE_HASH_MISMATCH: '+path.name)
    return raw,digest

def parse_utc(s):
    if not isinstance(s,str) or not s.strip(): return None
    try:
        value=datetime.fromisoformat(s.replace('Z','+00:00'))
        if value.utcoffset() is None: return None
        return value.astimezone(timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None

def load_integrated(raw: bytes):
    try: x=json.loads(raw)
    except (ValueError,UnicodeDecodeError) as exc: raise EvidenceError('Invalid M10C JSON') from exc
    if not isinstance(x,dict) or x.get('schema_version')!=INTEGRATED_VERSION:
        raise EvidenceError('Unexpected M10C integrated schema')
    if x.get('source_audit_sha256')!=PUBLIC_AUDIT_SHA256:
        raise EvidenceError('Integrated source audit SHA mismatch')
    if x.get('review_flags')!=[]:
        raise EvidenceError('Integrated reconciliation has review flags')
    if x.get('retained_challenges')!=19 or x.get('revision_records')!=19:
        raise EvidenceError('Expected frozen observed 19-challenge cohort')
    for field in ('historical_eligible_opportunity_census_certified','account_specific_prior_visibility_certified',
                  'model_execution_certified','final_scored_revision_certified','phase_a_protocol_adopted'):
        if x.get(field) is not False:
            raise EvidenceError('Unexpected certification flag: '+field)
    rows=x.get('records')
    if not isinstance(rows,list) or len(rows)!=19:
        raise EvidenceError('Integrated records must contain 19 revisions')
    ids=set()
    for r in rows:
        if not isinstance(r,dict):raise EvidenceError('Malformed integrated revision')
        cid=r.get('challenge_id')
        if not isinstance(cid,str) or not UUID_RE.fullmatch(cid):
            raise EvidenceError('Bad challenge ID')
        if cid in ids:raise EvidenceError('Multiple revisions require explicit scored-revision policy; not supported in this version')
        ids.add(cid)
        if r.get('model_execution_certified') is not False or r.get('final_scored_revision_certified') is not False:
            raise EvidenceError('Unexpected per-record certification')
        if r.get('timing_classification') not in {'PREDEADLINE_BUFFER_MET','PREDEADLINE_INSIDE_FIVE_MINUTE_BUFFER',
                                                   'AT_OR_AFTER_DEADLINE','UNKNOWN_SERVER_TIMESTAMP'}:
            raise EvidenceError('Unknown server timing classification')
    return x

def load_settlements(raw: bytes):
    try:
        content=raw.decode('utf-8-sig')
        reader=csv.DictReader(io.StringIO(content,newline=''))
        if not reader.fieldnames or not EXPECTED_COLS.issubset(set(reader.fieldnames)):
            raise EvidenceError('Settlement CSV column contract mismatch')
        rows=list(reader)
    except UnicodeDecodeError as exc:
        raise EvidenceError('Settlement CSV is not UTF-8') from exc
    seen=set(); by_id={}
    for index,row in enumerate(rows,2):
        if None in row:
            raise EvidenceError('Malformed CSV row '+str(index))
        cid=(row.get('challenge_id') or '').strip()
        if not UUID_RE.fullmatch(cid):
            raise EvidenceError('Invalid settlement challenge UUID on row '+str(index))
        if cid in seen:
            raise EvidenceError('Duplicate settlement challenge UUID')
        seen.add(cid)
        by_id[cid]=row
    return by_id

def build_report(integrated: dict, settlements: dict, integrated_sha: str, csv_sha: str):
    require_hash(integrated_sha); require_hash(csv_sha)
    by_id={r['challenge_id']:r for r in integrated['records']}
    counts=Counter(); flags=[]; records=[]
    for cid in sorted(by_id):
        forecast=by_id[cid]
        source=settlements.get(cid)
        present=source is not None
        stated=forecast.get('official_settlement_record_present')
        if type(stated) is not bool:
            raise EvidenceError('Missing original local settlement-presence indicator')
        if present != stated:
            flags.append('LOCAL_SETTLEMENT_PRESENCE_CONFLICT')
        result={'challenge_id':cid,
                'official_settlement_row_present':present,
                'prior_reconciliation_says_present':stated,
                'timing_classification':forecast['timing_classification'],
                'original_vs_retained_vector':forecast.get('original_vs_retained_vector'),
                'receipt_identity_linkage':forecast.get('receipt_identity_linkage'),
                'model_execution_certified':False,
                'final_scored_revision_certified':False,
                'historical_account_eligibility_certified':False}
        counts['settlement_PRESENT' if present else 'settlement_NOT_IN_LOCAL_CSV']+=1
        if present:
            label=(source.get('official_label') or '').strip().lower()
            asset=(source.get('asset') or '').strip().upper()
            if label not in CLASS_NAMES: flags.append('UNRECOGNIZED_OFFICIAL_LABEL')
            if asset!='ES': flags.append('SETTLEMENT_ASSET_CONFLICT')
            retained_dt=parse_utc(forecast.get('retained_deadline_at_query_snapshot'))
            official_dt=parse_utc(source.get('official_deadline_utc'))
            if retained_dt is None or official_dt is None:
                deadline_status='UNKNOWN_DEADLINE_COMPARISON'
                flags.append('DEADLINE_TIMESTAMP_UNAVAILABLE')
            elif retained_dt != official_dt:
                deadline_status='DEADLINE_DIFFERENCE_REVIEW'
                flags.append('OFFICIAL_SETTLEMENT_VS_RETAINED_DEADLINE_CONFLICT')
            else:
                deadline_status='DEADLINES_MATCH_AS_RECORDED'
            settled_dt=parse_utc(source.get('official_settled_at_utc'))
            if settled_dt is None:
                settlement_time_status='SETTLEMENT_TIMESTAMP_UNKNOWN'
            elif official_dt is not None and settled_dt < official_dt:
                settlement_time_status='SETTLED_BEFORE_RECORDED_DEADLINE_REVIEW'
                flags.append('SETTLEMENT_BEFORE_DEADLINE')
            else:
                settlement_time_status='SETTLEMENT_TIMESTAMP_PRESENT'
            settlement_date=(source.get('settlement_date') or '').strip()
            if settlement_date:
                try: date.fromisoformat(settlement_date)
                except ValueError: flags.append('SETTLEMENT_DATE_MALFORMED')
            result.update({'official_label':label if label in CLASS_NAMES else None,
                           'official_deadline_utc':source.get('official_deadline_utc') or None,
                           'official_settled_at_utc':source.get('official_settled_at_utc') or None,
                           'settlement_date':settlement_date or None,
                           'deadline_cross_source_comparison':deadline_status,
                           'settlement_timestamp_status':settlement_time_status,
                           'arena_score_field_present':bool((source.get('arena_score') or '').strip()),
                           'settlement_return_field_present':bool((source.get('settlement_return_pct') or '').strip())})
            counts['label_'+(label if label in CLASS_NAMES else 'UNRECOGNIZED')]+=1
            counts['deadline_'+deadline_status]+=1
            counts['settled_at_'+settlement_time_status]+=1
        records.append(result)
    extra_csv_ids=set(settlements)-set(by_id)
    report={
        'schema_version': VERSION,
        'scope': '19_observed_historical_ES_challenges_NOT_registered_150_opportunity_trial',
        'sources': {'m10c_integrated_physical_sha256':integrated_sha,
                    'local_official_settlements_csv_physical_sha256':csv_sha,
                    'public_opportunity_audit_sha256_in_integrated':PUBLIC_AUDIT_SHA256},
        'challenge_count':len(by_id), 'settlement_rows_in_observed_cohort':counts['settlement_PRESENT'],
        'settlement_rows_outside_observed_cohort':len(extra_csv_ids),
        'counts':dict(sorted(counts.items())), 'records':records,
        'review_flags':sorted(set(flags)),
        'official_scoring_calculated':False,
        'final_scored_revision_certified':False,
        'model_execution_certified':False,
        'historical_opportunity_census_certified':False,
        'historical_account_visibility_certified':False,
        'phase_a_150_opportunity_protocol_adopted':False,
        'source_files_modified':0,'network_requests':0,'forecast_submissions':0,
        'limitations':[
            'Settlement records supply published labels and timing, not proof of which retained revision was finally scored.',
            'Retained deadline at API snapshot is compared to local official CSV deadline; a match is not independent proof of deadline immutability.',
            'Arena score and settlement returns are not recomputed, validated as final score, or identified with multiclass Brier here.',
            'Public opportunity heartbeat coverage is historically partial; historical account-specific availability remains unknown.',
            'The 150-opportunity prospective protocol is separate and remains unadopted.',
            'Neither local model SHA claims nor probability agreement prove that the M1 artifact actually executed.',
        ],
    }
    return report

def private_destination(path: Path):
    resolved=path.resolve()
    if 'MarketMood_Private' not in resolved.parts:
        raise EvidenceError('Output must remain inside MarketMood_Private')
    if any((p/'.git').exists() for p in (resolved.parent,*resolved.parent.parents)):
        raise EvidenceError('Output path lies under a Git repository')
    if resolved.exists():
        raise EvidenceError('Output exists; cannot overwrite evidence')

def write_private(path: Path, report: dict):
    private_destination(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as f: f.write(raw)
    return sha256(raw)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrated-report',required=True,type=Path)
    parser.add_argument('--expected-integrated-sha256',required=True)
    parser.add_argument('--settlements-csv',required=True,type=Path)
    parser.add_argument('--expected-settlements-sha256',required=True)
    parser.add_argument('--private-report',type=Path)
    args=parser.parse_args(argv)
    try:
        raw_i, i_sha=load_pinned(args.integrated_report,args.expected_integrated_sha256)
        raw_s, s_sha=load_pinned(args.settlements_csv,args.expected_settlements_sha256)
        integrated=load_integrated(raw_i)
        settlement=load_settlements(raw_s)
        report=build_report(integrated,settlement,i_sha,s_sha)
        print('M10C SETTLEMENT RECONCILIATION: OK')
        print('OBSERVED_CHALLENGES:',report['challenge_count'])
        print('SETTLEMENT_ROWS_IN_OBSERVED_COHORT:',report['settlement_rows_in_observed_cohort'])
        print('SETTLEMENT_ROWS_OUTSIDE_OBSERVED_COHORT:',report['settlement_rows_outside_observed_cohort'])
        print('CLASSIFICATIONS:',json.dumps(report['counts'],sort_keys=True))
        print('REVIEW_FLAGS:',json.dumps(report['review_flags']))
        print('FINAL_SCORED_REVISION_CERTIFIED: NO | MODEL_EXECUTION_CERTIFIED: NO | PROSPECTIVE_PROTOCOL_ADOPTED: NO')
        print('NETWORK_REQUESTS: 0 | FORECAST_SUBMISSIONS: 0 | ORIGINAL_FILES_MODIFIED: 0')
        if args.private_report:
            print('PRIVATE_REPORT_PHYSICAL_SHA256:',write_private(args.private_report,report))
            print('PRIVATE_REPORT_WRITTEN: True')
        else:print('PRIVATE_REPORT_WRITTEN: False')
        return 1 if report['review_flags'] else 0
    except (EvidenceError,OSError,ValueError,TypeError,KeyError) as e:
        print('M10C SETTLEMENT VALIDATION ERROR:',type(e).__name__,str(e),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
