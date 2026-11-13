"""Prospective ES forecast commitment prototype, offline and write-once.

create: BEFORE deadline only (mandatory UTC deadline cutoff). Never commit forecasts retroactively.
verify: read-only. Public hash alone cannot prove independent publication time.

No networking, no submission, no model training. Private payload contains a secret nonce
and full forecast; do not publish or commit it to a public repository.
The private output is deliberately refused inside Git checkouts to prevent accidental staging.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA='es_arena_commitment_v0_1'
CLASSES={'bearish','neutral','bullish'}

def canonical(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')

def digest(v):
    return hashlib.sha256((SCHEMA+'\n').encode('ascii')+canonical(v)).hexdigest()

def json_obj(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj,dict): raise ValueError('Expected JSON object')
    return obj

def parse_utc(v):
    if not isinstance(v,str): raise ValueError('Missing UTC deadline')
    d=datetime.fromisoformat(v.replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset() is None: raise ValueError('Timezone-aware deadline required')
    return d.astimezone(timezone.utc)

def valid_forecast(f):
    p=f.get('probabilities')
    if not (isinstance(p,dict) and set(p)==CLASSES and
            all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in p.values()) and
            abs(sum(p.values())-1)<1e-8):
        raise ValueError('Invalid original three-class probability vector')
    if not isinstance(f.get('challenge_id'),str) or not f['challenge_id'].strip():
        raise ValueError('Missing challenge id')
    parse_utc(f.get('deadline_utc'))


def in_git_checkout(output_path: Path) -> bool:
    """Check both directory- and file-based .git markers (including worktrees).

    This is a local safety guard, not a substitute for access controls or a
    review of the public file and staging list.
    """
    parent = Path(output_path).resolve().parent
    return any((ancestor / '.git').exists() for ancestor in (parent, *parent.parents))


def validate_output_locations(forecast_path: Path, private: Path, public: Path) -> None:
    f, secret, share = (Path(x).resolve() for x in (forecast_path, private, public))
    if len({f, secret, share}) != 3:
        raise ValueError('Forecast, private and public files must have distinct paths')
    if in_git_checkout(secret):
        raise ValueError('Private nonce/forecast file must be OUTSIDE every Git checkout')


def create(forecast_path:Path, private:Path, public:Path, *, now=None):
    forecast_path=Path(forecast_path)
    f=json_obj(forecast_path)
    valid_forecast(f)
    validate_output_locations(forecast_path, private, public)
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None: raise ValueError('Naive current time refused')
    deadline=parse_utc(f['deadline_utc'])
    if now.astimezone(timezone.utc)>=deadline-timedelta(minutes=5):
        raise ValueError('Pre-deadline commitment cutoff reached. No retroactive commits.')
    if private.exists() or public.exists():
        raise FileExistsError('Existing commitment artifact. Refusing to overwrite.')
    payload={'schema_version':SCHEMA,'challenge_id':f['challenge_id'],
             'target_session_date':f.get('target_session_date'),
             'model_sha256':f.get('model_sha256'),
             'forecast_sha256':hashlib.sha256(forecast_path.read_bytes()).hexdigest(),
             'forecast':f,'nonce_hex':secrets.token_hex(32)}
    public_record={'schema_version':SCHEMA,'challenge_id':f['challenge_id'],
                   'target_session_date':f.get('target_session_date'),
                   'model_sha256':f.get('model_sha256'),
                   'commitment_sha256':digest(payload),
                   'warning':'A locally generated commitment is NOT independent proof of pre-deadline publication.'}
    # Require existing destination dirs, discourage accidental creation inside private evidence folders.
    if not private.parent.is_dir() or not public.parent.is_dir():
        raise FileNotFoundError('Create separate private/public parent folders explicitly before running')
    try:
        with private.open('x',encoding='utf-8') as fh:
            fh.write(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        with public.open('x',encoding='utf-8') as fh:
            fh.write(json.dumps(public_record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    except BaseException:
        # Fail closed: never remove a newly created private commitment after a partial failure.
        raise
    return public_record


def verify(forecast_path:Path, private:Path, public:Path):
    f=json_obj(forecast_path); valid_forecast(f)
    p=json_obj(private); q=json_obj(public)
    if p.get('schema_version')!=SCHEMA or q.get('schema_version')!=SCHEMA:
        raise ValueError('Schema mismatch')
    if any(key in q for key in ('nonce_hex', 'forecast', 'probabilities')):
        raise ValueError('Public record contains private forecast or nonce fields')
    if p.get('forecast_sha256') != hashlib.sha256(Path(forecast_path).read_bytes()).hexdigest():
        raise ValueError('Original forecast file hash does not match private evidence')
    if p.get('forecast') != f:
        raise ValueError('Stored forecast differs')
    if p.get('challenge_id')!=f['challenge_id'] or q.get('challenge_id')!=f['challenge_id']:
        raise ValueError('Challenge ID mismatch')
    if not isinstance(p.get('nonce_hex'),str) or len(bytes.fromhex(p['nonce_hex']))<32:
        raise ValueError('Nonce missing or too short')
    if q.get('commitment_sha256')!=digest(p):
        raise ValueError('Private payload does not match public commitment')
    if p.get('model_sha256')!=f.get('model_sha256') or q.get('model_sha256')!=f.get('model_sha256'):
        raise ValueError('Model hash mismatch')
    if p.get('target_session_date')!=f.get('target_session_date') or q.get('target_session_date')!=f.get('target_session_date'):
        raise ValueError('Target date mismatch')
    return q['commitment_sha256']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('create'); b=sub.add_parser('verify')
    for s in (a,b):
        s.add_argument('--forecast',required=True)
        s.add_argument('--private',required=True)
        s.add_argument('--public',required=True)
    ns=parser.parse_args(argv)
    if ns.cmd=='create':
        q=create(Path(ns.forecast),Path(ns.private),Path(ns.public))
        print('COMMITMENT CREATED OFFLINE (NOT PUBLICATION EVIDENCE):',q['commitment_sha256'])
        print('PRIVATE: Do not publish the private payload or nonce')
    else:
        print('COMMITMENT VERIFIED AGAINST FORECAST:',verify(Path(ns.forecast),Path(ns.private),Path(ns.public)))
        print('Independent publication time NOT established by this verification alone')
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,FileNotFoundError,FileExistsError,KeyError) as e:
        print('ERROR:',str(e),file=sys.stderr)
        sys.exit(1)
