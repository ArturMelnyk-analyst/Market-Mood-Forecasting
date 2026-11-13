"""Phase A prospective ES challenge-level commitment + post-settlement reveal.

Offline, write-once, no network and NO Arena submissions. Uses the unchanged
phase_a_forecast_commitment_v0_1 canonical JSON and UTC/Git safety helpers.
This creates cryptographic evidence of *content*, NOT external publication time.
All live use requires a genuinely future deadline and independently reviewable
pre-deadline publication evidence. Never retroactively create commitments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import secrets
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from phase_a_forecast_commitment_v0_1 import canonical, in_git_checkout, parse_utc

PLAN_SCHEMA = 'es_arena_phase_a_commitment_plan_v0_1'
SCHEMA = 'es_arena_phase_a_challenge_commitment_v0_1'
REVEAL_SCHEMA = 'es_arena_phase_a_challenge_reveal_v0_1'
SETTLEMENT_SCHEMA = 'es_arena_reviewed_settlement_v0_1'
CLASSES = ('bearish', 'neutral', 'bullish')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
MODEL_ID = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,31}$')
FAIL_CODES = {'DATA_UNAVAILABLE', 'MODEL_ERROR', 'PREPROCESSING_ERROR',
              'RUNTIME_ERROR', 'FORECAST_NOT_GENERATED', 'DEADLINE_MISSED',
              'CANCELLED', 'OTHER_DOCUMENTED_FAILURE'}
DOMAIN = b'ES_ARENA_PHASE_A_CHALLENGE_COMMITMENT_V1\n'


def load_json(path: Path) -> dict:
    o = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(o, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return o


def _require_sha(val: str, name: str) -> None:
    if type(val) is not str or not HEX64.fullmatch(val):
        raise ValueError(f'{name} must be a lowercase SHA-256 hex string')


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_payload(p: dict) -> str:
    return hashlib.sha256(DOMAIN + canonical(p)).hexdigest()


def _uuid(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('Challenge ID must be a UUID string')
    u = uuid.UUID(value)
    if str(u) != value:
        raise ValueError('Challenge ID must be the canonical lowercase UUID')
    return value


def _target_date(s: str) -> str:
    if type(s) is not str or date.fromisoformat(s).isoformat() != s:
        raise ValueError('target_session_date must be YYYY-MM-DD')
    return s


def _probabilities(p: dict) -> dict:
    if not isinstance(p, dict) or set(p) != set(CLASSES):
        raise ValueError('Exact bearish/neutral/bullish probability keys required')
    if any(type(p[k]) not in (float, int) or not math.isfinite(p[k]) or not (0 <= p[k] <= 1) for k in CLASSES):
        raise ValueError('Probabilities must be finite numeric values in [0, 1]')
    if abs(sum(p[k] for k in CLASSES) - 1.0) > 1e-8:
        raise ValueError('Probabilities do not sum to one')
    return {k: p[k] for k in CLASSES}


def _observed_now(now=None):
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Naive local time refused')
    return now.astimezone(timezone.utc)


def _plain_paths(*paths):
    abs_paths = [str(Path(p).resolve()) for p in paths]
    if len(abs_paths) != len(set(abs_paths)):
        raise ValueError('Input/private/public/reveal paths must be distinct')


def _safe_private(path):
    if in_git_checkout(Path(path)):
        raise ValueError('Private record/reveal must be OUTSIDE a Git checkout')


def _forecast_projection(f: dict, challenge_id: str, target: str, *, raw_hash: str):
    if f.get('challenge_id') != challenge_id or f.get('target_session_date') != target:
        raise ValueError('Forecast challenge/date mismatch')
    m = f.get('model_sha256')
    _require_sha(m, 'model_sha256')
    p = _probabilities(f.get('probabilities'))
    projection = {'challenge_id': challenge_id, 'target_session_date': target,
                  'model_sha256': m, 'probabilities': p,
                  'original_forecast_file_sha256': raw_hash}
    for k in ('source_snapshot_sha256', 'source_manifest_sha256'):
        if k in f:
            _require_sha(f[k], k)
            projection[k] = f[k]
    return projection


def _plan(plan: dict, *, now=None) -> tuple[dict, dict]:
    if plan.get('schema_version') != PLAN_SCHEMA:
        raise ValueError('Unsupported plan schema')
    cid = _uuid(plan.get('challenge_id'))
    target = _target_date(plan.get('target_session_date'))
    contract = plan.get('contract')
    if type(contract) is not str or not re.fullmatch(r'ES[A-Z0-9]{1,12}', contract):
        raise ValueError('ES futures contract required')
    deadline = parse_utc(plan.get('deadline_utc'))
    now = _observed_now(now)
    if now >= deadline - timedelta(minutes=5):
        raise ValueError('Freeze cutoff reached (deadline minus five minutes); no retrospective commitments')
    if date.fromisoformat(target) < now.date() - timedelta(days=1):
        raise ValueError('Past target session date refused')
    expected = plan.get('expected_models')
    entries = plan.get('entries')
    if not isinstance(expected, list) or not expected or not isinstance(entries, list):
        raise ValueError('Nonempty expected_models and entries lists required')
    if any(type(m) is not str or not MODEL_ID.fullmatch(m) for m in expected) or len(expected) != len(set(expected)):
        raise ValueError('Invalid/duplicate expected model IDs')
    if len(entries) != len(expected) or any(not isinstance(e, dict) for e in entries):
        raise ValueError('Exactly one explicit entry for every expected model required')
    by_id = {e.get('model_id'): e for e in entries}
    if len(by_id) != len(expected) or set(by_id) != set(expected):
        raise ValueError('Missing/extra/duplicate model entries')
    proof = []
    for model_id in expected:
        entry = by_id[model_id]
        status = entry.get('status')
        if status == 'forecast':
            if 'forecast_path' not in entry or 'failure_code' in entry:
                raise ValueError('Forecast entry must reference a file, not a failure code')
            path = Path(entry['forecast_path'])
            if not path.is_file():
                raise FileNotFoundError('Forecast file missing: ' + str(path))
            sha = sha_bytes(path)
            fore = load_json(path)
            # Check file again after parsing to catch simple accidental in-process edits.
            if sha != sha_bytes(path):
                raise ValueError('Forecast source changed while reading')
            proj = _forecast_projection(fore, cid, target, raw_hash=sha)
            proof.append({'model_id': model_id, 'status': status, 'forecast': proj})
        elif status == 'failure':
            if 'forecast_path' in entry or entry.get('failure_code') not in FAIL_CODES:
                raise ValueError('Failure must have a recognized code and no forecast file')
            observed = parse_utc(entry.get('observed_at_utc'))
            if observed > now or observed > deadline:
                raise ValueError('Failure observation timestamp is future or postdeadline')
            description = entry.get('failure_description', '')
            if type(description) is not str or len(description) > 512:
                raise ValueError('Failure description must be <=512 characters')
            proof.append({'model_id': model_id, 'status': status,
                          'failure_code': entry['failure_code'],
                          'observed_at_utc': observed.isoformat(),
                          'failure_description': description})
        else:
            raise ValueError('Unknown/unrecorded model outcome; every model must be explicit')
    spec = {'challenge_id': cid, 'target_session_date': target,
            'contract': contract, 'deadline_utc': deadline.isoformat(),
            'expected_models': expected}
    return spec, {'entries': proof}


def _safe_public(public: dict):
    required = {'schema_version', 'challenge_id', 'target_session_date', 'contract',
                'deadline_utc', 'model_count', 'commitment_sha256', 'notice'}
    if set(public) != required:
        raise ValueError('Public record has unexpected/missing fields (possible private leak)')
    if public['schema_version'] != SCHEMA:
        raise ValueError('Public schema mismatch')
    _uuid(public['challenge_id'])
    _require_sha(public['commitment_sha256'], 'commitment_sha256')
    if type(public['model_count']) is not int or public['model_count'] < 1:
        raise ValueError('Invalid model count')


def _private_payload(p: dict):
    if p.get('schema_version') != SCHEMA:
        raise ValueError('Private payload schema mismatch')
    nonce = p.get('nonce_hex')
    if not isinstance(nonce, str) or len(nonce) != 64 or not re.fullmatch(r'[0-9a-f]{64}', nonce):
        raise ValueError('256-bit secret nonce required')
    if not isinstance(p.get('spec'), dict) or not isinstance(p.get('manifest'), dict):
        raise ValueError('Missing private manifest/spec')
    spec = p['spec']
    _uuid(spec.get('challenge_id'))
    _target_date(spec.get('target_session_date'))
    parse_utc(spec.get('deadline_utc'))
    expected = spec.get('expected_models')
    records = p['manifest'].get('entries')
    if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
        raise ValueError('Invalid expected model roster')
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('Incomplete manifest roster')
    if [x.get('model_id') for x in records] != expected:
        raise ValueError('Manifest model roster mismatch')
    for e in records:
        if e.get('status') == 'forecast':
            f = e.get('forecast')
            if not isinstance(f, dict) or f.get('challenge_id') != spec['challenge_id'] or f.get('target_session_date') != spec['target_session_date']:
                raise ValueError('Forecast projection mismatch')
            _require_sha(f.get('model_sha256'), 'model_sha256')
            _require_sha(f.get('original_forecast_file_sha256'), 'original forecast sha')
            _probabilities(f.get('probabilities'))
            for k in ('source_snapshot_sha256', 'source_manifest_sha256'):
                if k in f: _require_sha(f[k], k)
            if set(f) - {'challenge_id','target_session_date','model_sha256','original_forecast_file_sha256','probabilities','source_snapshot_sha256','source_manifest_sha256'}:
                raise ValueError('Unknown forecast fields')
        elif e.get('status') == 'failure':
            if e.get('failure_code') not in FAIL_CODES or 'forecast' in e:
                raise ValueError('Invalid failure entry')
            parse_utc(e.get('observed_at_utc'))
        else:
            raise ValueError('Unrecognized manifest status')


def _public_for(p):
    spec = p['spec']
    return {'schema_version': SCHEMA, 'challenge_id': spec['challenge_id'],
            'target_session_date': spec['target_session_date'],
            'contract': spec['contract'], 'deadline_utc': spec['deadline_utc'],
            'model_count': len(spec['expected_models']),
            'commitment_sha256': hash_payload(p),
            'notice': 'Local commitment only; no independent publication-time evidence.'}


def freeze(plan_path, private_path, public_path, *, now=None):
    plan_path, private_path, public_path = map(Path, (plan_path, private_path, public_path))
    _plain_paths(plan_path, private_path, public_path)
    _safe_private(plan_path)
    _safe_private(private_path)
    if not private_path.parent.is_dir() or not public_path.parent.is_dir():
        raise FileNotFoundError('Create destination folders explicitly before freezing')
    if private_path.exists() or public_path.exists():
        raise FileExistsError('Existing commitment files: no overwrite')
    now = _observed_now(now)
    spec, manifest = _plan(load_json(plan_path), now=now)
    p = {'schema_version': SCHEMA, 'spec': spec, 'manifest': manifest,
         'nonce_hex': secrets.token_hex(32), 'local_freeze_utc': now.isoformat()}
    _private_payload(p)
    public = _public_for(p)
    _safe_public(public)
    # Exclusive creation prevents accidental overwrites; if public write fails,
    # keep private record for forensic recovery and do not claim publication.
    with private_path.open('x', encoding='utf-8') as h:
        h.write(json.dumps(p, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    try:
        with public_path.open('x', encoding='utf-8') as h:
            h.write(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    except BaseException:
        raise RuntimeError('Partial freeze: PRIVATE CREATED but PUBLIC NOT CREATED. Preserve evidence; no retroactive repair')
    return public


def verify(private_path, public_path):
    _plain_paths(private_path, public_path)
    private, public = load_json(Path(private_path)), load_json(Path(public_path))
    _private_payload(private); _safe_public(public)
    expected = _public_for(private)
    if public != expected:
        raise ValueError('Public/private commitment mismatch or tampering')
    return expected


def _settlement(settlement_path, source_path, challenge_id, *, now=None):
    o = load_json(Path(settlement_path))
    if o.get('schema_version') != SETTLEMENT_SCHEMA or o.get('challenge_id') != challenge_id:
        raise ValueError('Reviewed settlement schema/challenge mismatch')
    if o.get('outcome') not in CLASSES:
        raise ValueError('Official result must be bearish/neutral/bullish')
    resolved = parse_utc(o.get('resolved_at_utc'))
    if resolved > _observed_now(now):
        raise ValueError('Cannot reveal before settlement time')
    if not isinstance(o.get('official_reference'), str) or not o['official_reference'].strip():
        raise ValueError('Manual-reviewed official source reference required')
    _require_sha(o.get('official_source_sha256'), 'official_source_sha256')
    if sha_bytes(Path(source_path)) != o['official_source_sha256']:
        raise ValueError('Official source file hash mismatch')
    return o


def reveal(private_path, public_path, settlement_path, source_path, reveal_path, *, now=None):
    _plain_paths(private_path, public_path, settlement_path, source_path, reveal_path)
    _safe_private(Path(reveal_path))
    reveal_path = Path(reveal_path)
    if reveal_path.exists():
        raise FileExistsError('Reveal already exists; no overwrite')
    if not reveal_path.parent.is_dir():
        raise FileNotFoundError('Reveal output directory must already exist')
    public = verify(private_path, public_path)
    payload = load_json(Path(private_path))
    o = _settlement(settlement_path, source_path, public['challenge_id'], now=now)
    if parse_utc(o['resolved_at_utc']) < parse_utc(public['deadline_utc']):
        raise ValueError('Official result time precedes deadline')
    if _observed_now(now) <= parse_utc(public['deadline_utc']):
        raise ValueError('Cannot reveal before deadline')
    rev = {'schema_version': REVEAL_SCHEMA, 'public_record': public,
           'private_payload': payload, 'reviewed_settlement': o,
           'notice': 'Manual-reviewed settlement, not independently certified by this offline tool. Public release requires privacy/licensing review.'}
    verify_reveal_object(rev)
    with reveal_path.open('x', encoding='utf-8') as h:
        h.write(json.dumps(rev, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return rev


def verify_reveal_object(r):
    if r.get('schema_version') != REVEAL_SCHEMA:
        raise ValueError('Reveal schema mismatch')
    p = r.get('private_payload')
    q = r.get('public_record')
    if not isinstance(p, dict) or not isinstance(q, dict):
        raise ValueError('Reveal missing records')
    _private_payload(p); _safe_public(q)
    if _public_for(p) != q:
        raise ValueError('Reveal payload/nonce does not match public commitment')
    s = r.get('reviewed_settlement')
    if not isinstance(s, dict) or s.get('schema_version') != SETTLEMENT_SCHEMA or s.get('challenge_id') != q['challenge_id'] or s.get('outcome') not in CLASSES:
        raise ValueError('Settlement evidence in reveal invalid')
    _require_sha(s.get('official_source_sha256'), 'official_source_sha256')
    if parse_utc(s.get('resolved_at_utc')) < parse_utc(q['deadline_utc']):
        raise ValueError('Settlement time earlier than deadline')
    if not isinstance(s.get('official_reference'), str) or not s['official_reference'].strip():
        raise ValueError('Official reference missing')
    return q['commitment_sha256']


def verify_reveal(public_path, reveal_path, *, source_path=None):
    public = load_json(Path(public_path))
    r = load_json(Path(reveal_path))
    h = verify_reveal_object(r)
    if public != r['public_record']:
        raise ValueError('Reveal does not match original public artifact')
    if source_path is not None and sha_bytes(Path(source_path)) != r['reviewed_settlement']['official_source_sha256']:
        raise ValueError('Official source changed or incorrect')
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    fr = sub.add_parser('freeze', help='Before deadline minus five minutes, create write-once private and public files')
    fr.add_argument('--plan', required=True); fr.add_argument('--private', required=True); fr.add_argument('--public', required=True)
    ve = sub.add_parser('verify', help='Offline check public hash matches private content')
    ve.add_argument('--private', required=True); ve.add_argument('--public', required=True)
    rv = sub.add_parser('reveal', help='After reviewed official settlement, create PRIVATE reveal for review')
    rv.add_argument('--private', required=True); rv.add_argument('--public', required=True)
    rv.add_argument('--settlement', required=True); rv.add_argument('--official-source', required=True)
    rv.add_argument('--out', required=True)
    vr = sub.add_parser('verify-reveal', help='Public verifiability of revealed nonce/projections (not timestamp)')
    vr.add_argument('--public', required=True); vr.add_argument('--reveal', required=True)
    vr.add_argument('--official-source', required=False)
    ns = ap.parse_args(argv)
    if ns.cmd == 'freeze':
        q = freeze(ns.plan, ns.private, ns.public)
        print('LOCAL CONTENT COMMITMENT CREATED:', q['commitment_sha256'])
        print('NOT PUBLICATION EVIDENCE. Publish publicly BEFORE deadline, retain provider-timed record.')
    elif ns.cmd == 'verify':
        q = verify(ns.private, ns.public)
        print('CONTENT VERIFIED:', q['commitment_sha256'])
    elif ns.cmd == 'reveal':
        r = reveal(ns.private, ns.public, ns.settlement, ns.official_source, ns.out)
        print('PRIVATE REVEAL CREATED:', r['public_record']['commitment_sha256'])
        print('Review for sensitive/license-restricted content BEFORE sharing; no public upload performed.')
    else:
        print('REVEAL CONTENT VERIFIED:', verify_reveal(ns.public, ns.reveal, source_path=ns.official_source))
    print('INDEPENDENT PREDEADLINE PUBLICATION CERTIFICATION: NOT PERFORMED')
    print('NETWORK REQUESTS: 0 | ARENA SUBMISSIONS: 0 | M1 CHANGES: 0')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as e:
        print('ERROR:', str(e), file=sys.stderr)
        sys.exit(1)
