#!/usr/bin/env python3
"""M11 Stage 9: read-only dedup/consistency audit of successive ES session CSV snapshots.

Counts sessions for engineering feasibility ONLY; never certifies labels, vintage,
licenses, point-in-time availability, market calendar, or usable training rows.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

import phase_b_m11_stage7_source_inventory_v0_1 as s7
import phase_b_m11_stage8_es_csv_coverage_v0_1 as s8

RECORD_TYPE = 'M11_STAGE9_ES_UNIQUE_SESSION_COVERAGE_V0_1'
SNAPSHOT_RE = re.compile(r'^es_sessions_(\d{8})_[a-f0-9]{16}\.csv$', re.I)
HASH_RE = re.compile(r'^[a-f0-9]{64}$')
REQUIRED = ('session_date', 'session_start_utc', 'session_end_utc')


def verify_report(repo: Path, private: Path, report_path: Path, expected_hash: str) -> dict:
    s7.validate_environment(repo, private)
    if not HASH_RE.fullmatch(expected_hash):
        raise ValueError('Expected Stage 8 SHA-256 invalid')
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError('Stage 8 report absent or symlink')
    if not report_path.resolve(strict=True).is_relative_to((private/'m11_future/source_inventories').resolve()):
        raise ValueError('Stage 8 report outside private inventory directory')
    if s7.digest(report_path) != expected_hash:
        raise ValueError('Stage 8 report hash differs')
    d = json.loads(report_path.read_text(encoding='utf-8'))
    if d.get('record_type') != s8.TYPE or d.get('status') != 'PASS_WITH_LIMITATIONS':
        raise ValueError('Stage 8 report must be successful audited report')
    for key, required in [('upstream_registry_sha256',s7.PINNED['phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json']),
                          ('upstream_contract_sha256',s7.PINNED['phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json'])]:
        if d.get(key)!=required:
            raise ValueError('Upstream registry or contract mismatch')
    items = d.get('csv_details_private_only')
    if not isinstance(items,list) or len(items)!=d.get('csv_candidates_inspected_or_skipped') or len(items)!=d.get('category_counts',{}).get('csv_candidates'):
        raise ValueError('Stage 8 CSV candidate list inconsistent')
    if len({i.get('relative_path') for i in items if isinstance(i,dict)}) != len(items):
        raise ValueError('Duplicate candidate path in report')
    return d


def parse_utc_timestamp(v: str) -> datetime:
    try:
        dt = datetime.fromisoformat(v.strip().replace('Z','+00:00'))
        if dt.tzinfo is None or dt.utcoffset()!=timedelta(0):
            raise ValueError('Timestamp must have explicit UTC offset')
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('Non-UTC or unparsable session time') from exc


def audit_snapshot(path: Path, meta: dict, *, max_bytes: int, max_rows: int) -> dict:
    if path.stat().st_size > max_bytes:
        raise ValueError('Snapshot exceeds file cap')
    if meta.get('inventory_sha256') is None or not HASH_RE.fullmatch(meta['inventory_sha256']):
        raise ValueError('Snapshot lacks Stage 7 pinned content hash')
    if s7.digest(path)!=meta['inventory_sha256'] or path.stat().st_size!=meta.get('inventory_size'):
        raise ValueError('Snapshot differs from recorded Stage 7 file')
    rows = {}
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, restkey='__extra__',restval=None)
        head = reader.fieldnames
        if not head or any(not x for x in head) or len(head)!=len(set(head)) or not all(x in head for x in REQUIRED):
            raise ValueError('Unsupported session CSV header')
        expected_header = meta.get('header_fields')
        if head!=expected_header or s8.sha_bytes(json.dumps(head,ensure_ascii=False,separators=(',',':')).encode())!=meta.get('header_sha256'):
            raise ValueError('CSV header changed from Stage 8')
        for idx,row in enumerate(reader):
            if idx>=max_rows or None in row or row.get('__extra__') is not None or any(v is None for v in row.values()):
                raise ValueError('CSV incomplete, row width problem or row cap exceeded')
            d=date.fromisoformat(row['session_date'].strip()).isoformat()
            start=parse_utc_timestamp(row['session_start_utc']); end=parse_utc_timestamp(row['session_end_utc'])
            if end<=start:
                raise ValueError('Session end not after start')
            if d != end.date().isoformat():
                raise ValueError('End UTC date differs from listed session date; review calendar semantics')
            if d in rows:
                raise ValueError('Duplicate session_date within a snapshot')
            rows[d]={'row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                     'start':start.isoformat(),'end':end.isoformat(),'header_sha256':meta['header_sha256']}
    if len(rows)!=meta.get('rows_inspected'):
        raise ValueError('Row count differs from Stage 8 complete scan')
    return rows


def audit(repo: Path, private: Path, stage8_file: Path, expected_hash: str, *, max_bytes: int=16*1024*1024, max_rows:int=200000)->dict:
    if max_bytes<=0 or max_rows<=0:
        raise ValueError('Invalid caps')
    d=verify_report(repo,private,stage8_file,expected_hash)
    picked=[]; excluded=[]; all_dates=defaultdict(list); schema_groups=Counter(); snapshots=[]
    for entry in d['csv_details_private_only']:
        if not isinstance(entry,dict) or not isinstance(entry.get('relative_path'),str):
            raise ValueError('Invalid candidate entry')
        rel=entry['relative_path']; name=Path(rel).name
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe relative inventory path')
        m=SNAPSHOT_RE.fullmatch(name)
        if not m:
            excluded.append(name)
            continue
        if entry.get('inspection_status')!='CSV_STRUCTURAL_SCAN_COMPLETE' or entry.get('complete_csv_scanned') is not True:
            raise ValueError('Selected sessions snapshot did not complete Stage 8 scan')
        expected_end=date(int(m.group(1)[:4]),int(m.group(1)[4:6]),int(m.group(1)[6:8])).isoformat()
        p,checked=s8.safe_path(repo, {'source_bucket_by_directory_only':'ES','path':str(repo/rel)})
        if checked!=rel or not p.is_file() or p.is_symlink():
            raise ValueError('Unsafe or missing session snapshot')
        rows=audit_snapshot(p,entry,max_bytes=max_bytes,max_rows=max_rows)
        if not rows or max(rows)!=expected_end:
            raise ValueError('Session snapshot maximum date does not match filename')
        schema_groups[entry['header_sha256']]+=1
        # Data values never printed; only comparisons of complete row fingerprints.
        for day, val in rows.items():
            all_dates[day].append((name,val))
        snapshots.append({'filename':name,'ending_session_date':expected_end,'rows':len(rows),'first_session_date':min(rows),'last_session_date':max(rows),'header_sha256':entry['header_sha256']})
    snapshots.sort(key=lambda x:(x['ending_session_date'],x['filename']))
    duplicates=sum(len(v)-1 for v in all_dates.values())
    differing=[]
    for day,recs in sorted(all_dates.items()):
        if len({(r['row_sha256'],r['header_sha256']) for _,r in recs})>1:
            differing.append({'session_date':day,'versions':len(recs),'distinct_row_versions':len({(r['row_sha256'],r['header_sha256']) for _,r in recs})})
    # Inconsistent schema or observations require review, not silent canonicalization.
    unique_days=sorted(all_dates)
    calendar_days = None if not unique_days else (date.fromisoformat(unique_days[-1])-date.fromisoformat(unique_days[0])).days+1
    latest = snapshots[-1] if snapshots else None
    status = 'REVIEW_REQUIRED' if differing or len(schema_groups)>1 else 'COMPLETE_WITH_LIMITATIONS'
    if not snapshots:
        status='NO_ELIGIBLE_SNAPSHOT_REVIEW_REQUIRED'
    return {
        'record_type':RECORD_TYPE,'status':status,'source_stage8_report_sha256':expected_hash,
        'stage8_candidate_csv_count':len(d['csv_details_private_only']),
        'selected_es_sessions_snapshots':len(snapshots),'excluded_other_csv_candidates':len(excluded),
        'raw_total_rows_in_selected_snapshots':sum(x['rows'] for x in snapshots),
        'distinct_session_dates_observed':len(unique_days),
        'repeated_session_date_occurrences':duplicates,
        'dates_with_nonidentical_versions_or_schemas':len(differing),
        'sample_conflict_dates_private_only':differing[:20],
        'schema_fingerprint_count':len(schema_groups),
        'min_session_date':unique_days[0] if unique_days else None,
        'max_session_date':unique_days[-1] if unique_days else None,
        'calendar_days_between_min_and_max_inclusive_not_exchange_sessions':calendar_days,
        'latest_snapshot_by_filename_private_only':latest,
        'snapshot_metadata_private_only':snapshots,
        'excluded_filenames_private_only':sorted(excluded),
        'independent_training_rows_certified':0,
        'training_labels_certified':False,'point_in_time_vintage_certified':False,
        'market_calendar_verified':False,'license_verified':False,'training_feasibility_certified':False,
        'm1_changed':False,'models_trained':0,'network_requests':0,'forecast_submissions':0,
        'explanation':'Exact session dates only; duplicates across accumulating snapshots are not independent examples. Conflicting rows/schema require review; no market calendar, labeling, vendor license, training eligibility or as-of proof is certified.'
    }


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['audit-sessions'])
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--private-root',required=True,type=Path)
    ap.add_argument('--stage8-report',required=True,type=Path)
    ap.add_argument('--expected-stage8-sha256',required=True)
    ap.add_argument('--max-csv-mib',type=int,default=16)
    ap.add_argument('--max-rows',type=int,default=200000)
    ap.add_argument('--snapshot-file',type=Path)
    args=ap.parse_args(argv)
    try:
        repo=args.repo.resolve(strict=True);private=args.private_root.resolve(strict=True)
        out=None
        if args.snapshot_file is not None:
            out=args.snapshot_file.absolute()
            container=(private/'m11_future/source_inventories').resolve(strict=True)
            if out.exists() or out.is_symlink() or not out.parent.is_dir() or not out.parent.resolve(strict=True).is_relative_to(container):
                raise ValueError('Snapshot must be new inside private source_inventories')
        report=audit(repo,private,args.stage8_report,args.expected_stage8_sha256,max_bytes=args.max_csv_mib*1024*1024,max_rows=args.max_rows)
        print('M11_STAGE9_SESSION_AUDIT:',report['status'])
        for key in ['selected_es_sessions_snapshots','excluded_other_csv_candidates','raw_total_rows_in_selected_snapshots','distinct_session_dates_observed','repeated_session_date_occurrences','dates_with_nonidentical_versions_or_schemas','schema_fingerprint_count','min_session_date','max_session_date']:
            print(key.upper()+':',report[key])
        print('TRAINING_ROWS_CERTIFIED: 0 | LABELS_CERTIFIED: NO | VINTAGE_CERTIFIED: NO')
        print('FORECAST_SUBMISSIONS: 0 | NETWORK_REQUESTS: 0 | MODELS_TRAINED: 0')
        if out is not None:
            data=(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode()
            with out.open('xb') as fh:fh.write(data)
            print('PRIVATE_REPORT_WRITTEN:',out)
            print('PRIVATE_REPORT_SHA256:',hashlib.sha256(data).hexdigest())
        else:
            print('FILES_WRITTEN: 0')
        return 0 if report['status']=='COMPLETE_WITH_LIMITATIONS' else 3
    except (ValueError,OSError,TypeError,KeyError,OverflowError,UnicodeError,json.JSONDecodeError,csv.Error) as exc:
        print('M11_STAGE9_STOP:',type(exc).__name__,str(exc),file=sys.stderr)
        return 2

if __name__=='__main__':sys.exit(main())
