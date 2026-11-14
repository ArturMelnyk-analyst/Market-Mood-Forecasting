#!/usr/bin/env python3
"""Offline/read-only parser for the Arena financial-opportunity-audit-v1 response.

No network, no repository modifications, and no inference of account eligibility.
The optional private report uses exclusive creation and includes UUIDs for later join.
"""
from __future__ import annotations
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

CONTRACT = 'financial-opportunity-audit-v1'
_REQUIRED = ('api_contract_version','asset','window','as_of','snapshot_at','entries','coverage','next_cursor')

class AuditError(ValueError):
    pass


def utc_dt(value, name):
    if not isinstance(value, str):
        raise AuditError(f'{name}: expected timezone-aware string')
    try:
        x = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise AuditError(f'{name}: invalid timestamp') from exc
    if x.tzinfo is None or x.utcoffset() is None:
        raise AuditError(f'{name}: timezone required')
    return x.astimezone(timezone.utc)


def _obj(value, name):
    if not isinstance(value, dict):
        raise AuditError(f'{name}: expected object')
    return value


def _status(obj, nested, name):
    v = _obj(obj.get(nested), f'{name}.{nested}').get('status')
    if not isinstance(v, str) or not v.strip():
        raise AuditError(f'{name}.{nested}.status: nonempty string required')
    return v


def summarize(raw: bytes, expected_sha256: str | None = None) -> dict:
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and expected_sha256.lower() != sha:
        raise AuditError('SHA-256 mismatch: saved source differs from expected snapshot')
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuditError('invalid JSON') from exc
    d = _obj(payload, '$')
    for key in _REQUIRED:
        if key not in d:
            raise AuditError(f'missing required top-level field: {key}')
    if d['api_contract_version'] != CONTRACT:
        raise AuditError('unsupported API contract version')
    asset = d['asset']
    if asset != 'ES':
        raise AuditError('expected asset ES; do not mix markets')
    w = _obj(d['window'], 'window')
    start = utc_dt(w.get('start'), 'window.start')
    end = utc_dt(w.get('end'), 'window.end')
    asof = utc_dt(d['as_of'], 'as_of')
    snap = utc_dt(d['snapshot_at'], 'snapshot_at')
    if not start < end:
        raise AuditError('invalid window bounds')
    if w.get('bounds') != '[start,end)':
        raise AuditError('unexpected window-bound semantics')
    # A later snapshot is not equivalent to earlier as-of evidence.
    if not isinstance(d['entries'], list):
        raise AuditError('entries must be a list')
    coverage = _obj(d['coverage'], 'coverage')
    if not isinstance(coverage.get('complete'), bool):
        raise AuditError('coverage.complete must be a boolean')
    gaps = coverage.get('gaps')
    if not isinstance(gaps, list):
        raise AuditError('coverage.gaps must be a list')
    for idx, gap in enumerate(gaps):
        g = _obj(gap, f'coverage.gaps[{idx}]')
        a,b = utc_dt(g.get('start'), 'gap.start'),utc_dt(g.get('end'),'gap.end')
        if not (start <= a < b <= end):
            raise AuditError('coverage gap lies outside retained-deadline window')
    next_cursor = d['next_cursor']
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise AuditError('next_cursor must be null or a string')
    statuses = {k: collections.Counter() for k in ('round_status','state_at_as_of','reconstruction','public_open','reader_delivery','submission_eligibility','cancellation')}
    normalized=[]
    seen=set()
    evidence_items=0
    for i, entry in enumerate(d['entries']):
        p=f'entries[{i}]'
        e=_obj(entry,p)
        challenge_id=e.get('challenge_id')
        if not isinstance(challenge_id,str) or not challenge_id.strip():
            raise AuditError(f'{p}: challenge_id missing')
        if challenge_id in seen:
            raise AuditError(f'{p}: duplicate challenge_id')
        seen.add(challenge_id)
        if e.get('asset') != asset:
            raise AuditError(f'{p}: asset mismatch')
        deadline=utc_dt(e.get('deadline'),f'{p}.deadline')
        if not start <= deadline < end:
            raise AuditError(f'{p}: deadline outside requested retained window')
        for k in ('status','state_at_as_of','reconstruction','submission_eligibility_at_as_of'):
            if not isinstance(e.get(k),str):
                raise AuditError(f'{p}.{k}: expected string')
        pub=_status(e,'public_open_before_as_of',p)
        delivery=_status(e,'reader_delivery_before_as_of',p)
        cancel = _obj(e.get('cancellation'), f'{p}.cancellation').get('status')
        if cancel is not None and not isinstance(cancel,str):
            raise AuditError(f'{p}.cancellation.status: string/null required')
        ev=e.get('evidence')
        if not isinstance(ev,list):
            raise AuditError(f'{p}.evidence: expected array')
        evidence_items+=len(ev)
        statuses['round_status'][e['status']]+=1
        statuses['state_at_as_of'][e['state_at_as_of']]+=1
        statuses['reconstruction'][e['reconstruction']]+=1
        statuses['public_open'][pub]+=1
        statuses['reader_delivery'][delivery]+=1
        statuses['submission_eligibility'][e['submission_eligibility_at_as_of']]+=1
        statuses['cancellation'][str(cancel) if cancel is not None else 'null']+=1
        normalized.append({'challenge_id':challenge_id,'deadline_utc':deadline.isoformat().replace('+00:00','Z'),
         'round_status_at_snapshot':e['status'],'state_at_as_of':e['state_at_as_of'],
         'public_open_before_as_of_status':pub,'reader_delivery_before_as_of_status':delivery,
         'submission_eligibility_at_as_of':e['submission_eligibility_at_as_of'],
         'reconstruction':e['reconstruction'],'cancellation_status':cancel,'evidence_item_count':len(ev)})
    return {'schema_version':'m10b_offline_opportunity_review_v0_1','source_sha256':sha,
      'source_bytes':len(raw),'api_contract_version':CONTRACT,'asset':asset,
      'window':w,'as_of':d['as_of'],'snapshot_at':d['snapshot_at'],
      'snapshot_is_after_as_of':snap>asof,'record_count':len(normalized),
      'unique_challenge_count':len(seen),'next_cursor_present':bool(next_cursor),
      'pagination_complete':next_cursor is None,
      'coverage':{'complete':coverage['complete'],'status':coverage.get('status'),
       'semantics':coverage.get('semantics'),'gap_count':len(gaps),'gaps':gaps},
      'counts':{k:dict(v) for k,v in statuses.items()},'total_evidence_items':evidence_items,
      'challenge_records':normalized,
      'certification_notes':[
       'Retained challenge records do not certify account-level eligibility.',
       'Current status does not imply same status at as_of; use state_at_as_of.',
       'Unknown public/reader/eligibility status is neither true nor false.',
       'Public observation alone does not establish account-specific availability.',
       'Incomplete coverage and null next_cursor are independent concepts.',
       'This parser does not certify when a submitted revision was accepted.',
       'No research protocol or phase boundary is adopted by this audit.'
      ]}


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',required=True, help='private immutable response JSON')
    a.add_argument('--expected-sha256',required=True, help='previously recorded sha256 of input')
    a.add_argument('--private-report',help='write report outside Git, create only if nonexistent')
    ns=a.parse_args(argv)
    try:
        raw=Path(ns.input).read_bytes()
        report=summarize(raw,ns.expected_sha256)
        if ns.private_report:
            output=Path(ns.private_report)
            if output.exists():
                raise AuditError('private report already exists (will not overwrite)')
            if not output.parent.is_dir():
                raise AuditError('private report parent directory missing')
            with output.open('x',encoding='utf-8',newline='\n') as f:
                json.dump(report,f,indent=2,sort_keys=True,ensure_ascii=False)
                f.write('\n')
    except (OSError,AuditError) as exc:
        print(f'M10B ERROR: {exc}',file=sys.stderr)
        return 2
    print('M10B OFFLINE REVIEW: OK')
    print('SOURCE_SHA256:',report['source_sha256'])
    print('RETAINED_CHALLENGES:',report['record_count'])
    print('UNIQUE_IDS:',report['unique_challenge_count'])
    print('PAGINATION_COMPLETE:',report['pagination_complete'])
    print('HISTORICAL_COVERAGE_COMPLETE:',report['coverage']['complete'])
    print('COVERAGE_GAPS:',report['coverage']['gap_count'])
    for k,v in report['counts'].items():
        print(k.upper()+':',json.dumps(v,sort_keys=True))
    print('EVIDENCE_ITEMS:',report['total_evidence_items'])
    print('ACCOUNT_ELIGIBILITY_CERTIFIED_BY_PUBLIC_AUDIT: NO')
    print('PROTOCOL_ADOPTED: NO')
    print('NETWORK_REQUESTS: 0 | SUBMISSIONS: 0 | ORIGINAL_FILES_MODIFIED: 0')
    print('PRIVATE_REPORT_WRITTEN:',bool(ns.private_report))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
