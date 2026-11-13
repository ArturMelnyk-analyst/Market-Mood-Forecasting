"""Offline read-only Phase A ES evidence chain audit.

No network, no writes, no model inference, no submission, no retraining.
Usage:
 python phase_a_daily_chain_audit_v0_1.py --challenge-id UUID --date YYYY-MM-DD
Only evidence actually inspected is checked; a locally consistent receipt does not
independently certify Arena's acceptance time or the original model execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

MODEL_FILENAME = 'es_daily_lr_deployment_v0_1.joblib'
FROZEN_M1_SHA256 = '5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d'
CLASSES = {'bearish', 'neutral', 'bullish'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError(f'Expected JSON object: {path.name}')
    return obj


def utc(raw: object):
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.strip().replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo and dt.utcoffset() is not None else None
    except ValueError:
        return None


def http_date(raw: object):
    if not isinstance(raw, str):
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc) if dt.tzinfo and dt.utcoffset() is not None else None
    except (TypeError, ValueError, IndexError):
        return None


def unpack(body: object):
    if isinstance(body, list):
        arr = body
    elif isinstance(body, dict):
        arr = body.get('challenges', body.get('items', []))
    else:
        return []
    if not isinstance(arr, list):
        return []
    return [z.get('challenge', z) for z in arr if isinstance(z, dict) and isinstance(z.get('challenge', z), dict)]


def number_probabilities(p):
    return (isinstance(p, dict) and set(p) == CLASSES and
            all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in p.values()) and
            abs(sum(p.values())-1) < 1e-8)


def under_root(root: Path, raw: object):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Missing evidence path')
    p = Path(raw)
    resolved = (p if p.is_absolute() else root / p).resolve()
    # Never follow an arbitrary source path outside the repository.
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Evidence path escapes repository root')
    return resolved


def audit(root: Path, cid: str, target_date: str, *, expected_model_sha: str = FROZEN_M1_SHA256) -> dict:
    root = root.resolve()
    base = root / 'data/processed/es_arena_live'
    results = []
    def add(level, key, message):
        results.append({'level':level,'check':key,'detail':message})
    def verify_hash(key, path, expected):
        if not path.is_file():
            add('INCOMPLETE',key,f'Missing: {path.name}')
        elif not isinstance(expected,str) or len(expected)!=64:
            add('INCOMPLETE',key,'No usable stored SHA-256 to compare')
        else:
            actual=sha(path)
            add('PASS' if actual==expected else 'FAIL',key,'SHA-256 matches' if actual==expected else 'SHA-256 mismatch')

    forecast_path = base / 'forecasts' / f'forecast_{target_date.replace("-", "")}_{cid}.json'
    attempt_path = base / 'receipts' / f'attempt_{cid}.json'
    receipt_path = base / 'receipts' / f'receipt_{cid}.json'
    forecast = None
    if forecast_path.is_file():
        try:
            forecast=json_file(forecast_path)
            add('PASS','forecast_file',f'Found {forecast_path.name} (SHA-256 {sha(forecast_path)})')
        except (ValueError,OSError,json.JSONDecodeError) as exc:
            add('FAIL','forecast_file',type(exc).__name__)
    else:
        add('INCOMPLETE','forecast_file',f'Missing: {forecast_path.name}')

    if forecast is not None:
        for key, expected in [('challenge_id',cid),('target_session_date',target_date),('contract_symbol','ESZ6')]:
            add('PASS' if forecast.get(key)==expected else 'FAIL',key,
                'Matches expected value' if forecast.get(key)==expected else 'Unexpected or missing value')
        add('PASS' if number_probabilities(forecast.get('probabilities')) else 'FAIL',
            'probabilities','Three finite probabilities in [0,1] summing to one' if number_probabilities(forecast.get('probabilities')) else 'Invalid probability distribution')
        add('PASS' if forecast.get('model_sha256')==expected_model_sha else 'FAIL',
            'phase_a_frozen_m1_identity',
            'Matches independently pinned frozen M1 SHA-256' if forecast.get('model_sha256')==expected_model_sha else 'Model SHA-256 is not the independently pinned M1')
        if forecast.get('status')=='PREVIEW_NOT_SUBMITTED':
            add('PASS','forecast_status','Stored immutable forecast status unchanged; does not imply no submission')
        else:
            add('FAIL','forecast_status',f'Unexpected status: {forecast.get("status")!r}')
        if not isinstance(forecast.get('last_eligible_session'),str) or forecast['last_eligible_session'] >= target_date:
            add('FAIL','source_session_cutoff','Missing or nonprior eligible session')
        else:
            add('PASS','source_session_cutoff',f'Prior session: {forecast["last_eligible_session"]}')
        t1=utc(forecast.get('prepared_at_utc')); dl=utc(forecast.get('deadline_utc'))
        if t1 is None or dl is None:
            add('INCOMPLETE','forecast_timing','No timezone-aware local preparation/deadline evidence')
        else:
            add('PASS' if t1<dl else 'FAIL','forecast_timing',
                'Local recorded preparation time before deadline (not independent certification)' if t1<dl else 'Forecast recorded after deadline')
        for key, raw, expected in [
            ('source_csv_hash',forecast.get('source_csv'),forecast.get('source_sha256')),
            ('source_manifest_hash',None,forecast.get('source_manifest_sha256')),
            ('frozen_model_hash',None,forecast.get('model_sha256')),
        ]:
            try:
                if key.startswith('source'):
                    src=under_root(root,forecast.get('source_csv'))
                    fpath=src if key=='source_csv_hash' else src.with_suffix('.manifest.json')
                else:
                    fpath=under_root(root,forecast.get('model_dir'))/MODEL_FILENAME
                verify_hash(key,fpath,expected)
                if key=='source_manifest_hash' and fpath.is_file() and src.is_file():
                    manifest=json_file(fpath)
                    stored=manifest.get('snapshot_sha256')
                    if stored:
                        verify_hash('manifest_snapshot_content',src,stored)
            except (ValueError,OSError,json.JSONDecodeError) as exc:
                add('FAIL',key,type(exc).__name__+': '+str(exc)[:120])

    attempt=None
    if attempt_path.is_file():
        try:
            attempt=json_file(attempt_path)
            good=attempt.get('challenge_id')==cid
            add('PASS' if good else 'FAIL','attempt_challenge','Challenge ID matches' if good else 'Challenge ID mismatch')
            if forecast is not None:
                verify_hash('attempt_forecast_hash',forecast_path,attempt.get('forecast_sha256'))
            else:
                add('INCOMPLETE','attempt_forecast_hash','No forecast to compare')
        except (OSError,ValueError,json.JSONDecodeError) as exc:
            add('FAIL','attempt_file',type(exc).__name__)
    else:
        add('INCOMPLETE','attempt_file','No original attempt record')

    if receipt_path.is_file():
        try:
            receipt=json_file(receipt_path)
            response=receipt.get('response')
            response=response if isinstance(response,dict) else {}
            accepted=(receipt.get('http_status')==201 and response.get('counts_for_score') is True and
                      str(response.get('challenge_id'))==cid and bool(response.get('prediction_id')))
            add('PASS' if accepted else 'FAIL','receipt_acceptance',
                'Stored HTTP 201 / scored / matching challenge / server prediction ID' if accepted else 'Missing or conflicting server acceptance evidence')
            if forecast is not None:
                verify_hash('receipt_forecast_hash',forecast_path,receipt.get('forecast_sha256'))
                server_p=response.get('probabilities')
                if server_p is not None:
                    local_p=forecast.get('probabilities')
                    same=number_probabilities(server_p) and number_probabilities(local_p) and all(abs(server_p[k]-local_p[k])<1e-8 for k in CLASSES)
                    add('PASS' if same else 'FAIL','server_probabilities','Match saved forecast' if same else 'Server and forecast probabilities differ')
                else:
                    add('WARN','server_probabilities','Response does not include probabilities to compare')
            created=utc(response.get('created_at'))
            add('PASS' if created else 'WARN','server_acceptance_timestamp',
                'Timezone-aware server timestamp in receipt (not cryptographically signed)' if created else 'Server created_at missing timezone or unavailable; independent UTC acceptance time not certified')
        except (OSError,ValueError,json.JSONDecodeError) as exc:
            add('FAIL','receipt_file',type(exc).__name__)
    else:
        add('INCOMPLETE','receipt_file','No original scored receipt; do NOT automatically retry a POST')

    observations=0
    predeadline=0
    discovery_conflicts=0
    all_discovery=list(sorted((base/'discovery').glob('arena_discovery_*.json')))
    for path in all_discovery:
        try:
            record=json_file(path)
            if record.get('status')!='success' or record.get('http_status')!=200:
                continue
            for c in unpack(record.get('response_json')):
                if (c.get('id') or c.get('challenge_id'))!=cid:
                    continue
                observations+=1
                if c.get('contract_symbol') and c.get('contract_symbol')!='ESZ6':
                    discovery_conflicts+=1
                deadline=utc(c.get('deadline_utc') or c.get('deadline_at') or c.get('deadline'))
                server=http_date(record.get('server_date_header'))
                completed=utc(record.get('record_completed_at_utc'))
                forecast_deadline=utc(forecast.get('deadline_utc')) if forecast is not None else None
                if deadline and forecast_deadline and deadline!=forecast_deadline:
                    discovery_conflicts+=1
                if deadline and server and completed and server<deadline:
                    predeadline+=1
        except (OSError,ValueError,json.JSONDecodeError):
            # Other archives cannot be presumed to be valid; separate audit handles them.
            continue
    if discovery_conflicts:
        add('FAIL','discovery_metadata_match',f'{discovery_conflicts} conflicting contract/deadline fields across matching discovery records')
    elif observations:
        add('PASS','discovery_metadata_match','No matching discovery record conflicts on contract/deadline')
    if observations:
        add('PASS','challenge_discovery',f'{observations} authenticated successful archived observation(s) contain the ID')
        add('PASS' if predeadline else 'WARN','discovery_predeadline',
            f'{predeadline} archive(s) with HTTP Date preceding observed deadline; this does not certify subsequent submission time' if predeadline else 'No independently server-dated predeadline discovery established')
    else:
        add('INCOMPLETE','challenge_discovery','No matching successful discovery record; do not infer no opportunity existed')
    add('WARN','historical_opportunity_census','One-day chain audit is not a complete historical eligible-opportunity census')
    failures=sum(r['level']=='FAIL' for r in results)
    incomplete=sum(r['level']=='INCOMPLETE' for r in results)
    return {'schema_version':'phase_a_daily_chain_audit_v0_1','challenge_id':cid,
            'target_session_date':target_date,'read_only':True,
            'summary':'FAIL' if failures else ('INCOMPLETE' if incomplete else 'LOCALLY_CONSISTENT_WITH_LIMITATIONS'),
            'checks':results,'failure_count':failures,'incomplete_count':incomplete}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--challenge-id',required=True)
    p.add_argument('--date',required=True)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--json',action='store_true')
    a=p.parse_args(argv)
    report=audit(Path(a.root),a.challenge_id,a.date)
    if a.json:
        print(json.dumps(report,indent=2))
    else:
        print('PHASE A DAILY CHAIN AUDIT (READ-ONLY)')
        for item in report['checks']:
            print(f"{item['level']:>10}  {item['check']:<29} {item['detail']}")
        print('SUMMARY:',report['summary'])
        print('Independent acceptance-time certification: NOT ESTABLISHED by local receipt alone')
    return 1 if report['failure_count'] else (2 if report['incomplete_count'] else 0)

if __name__=='__main__':
    sys.exit(main())
