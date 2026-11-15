#!/usr/bin/env python3
"""M11 Stage 5: read-only Phase B registry and point-in-time evidence checks.

DRAFT validation does not freeze a protocol, train models or certify live data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SOURCES = {
    'M1':['ES'],
    'M2':['ES','GDELT'],
    'M3':['ES','VIX'],
    'M4':['ES','GDELT','VIX'],
    'M5':['ES','GDELT','VIX','GOOGLE_TRENDS'],
    'M6':['ES','GDELT','VIX','UNEMPLOYMENT'],
    'M7':['ES','GDELT','VIX','GOOGLE_TRENDS','UNEMPLOYMENT']
}
MODEL_SHA = '5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d'
PROTOCOL_COMMIT = '6f089df837e5dcd1aa5fc198490e287a98fa7fd8'
HEX_SHA = re.compile(r'^[a-f0-9]{64}$')

class AuditError(ValueError):
    pass

def must(test, message):
    if not test:
        raise AuditError(message)

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_json(p: Path):
    must(p.is_file(), f'Missing file: {p}')
    return json.loads(p.read_text(encoding='utf-8-sig'))

def parse_utc(s: str) -> datetime:
    must(isinstance(s,str) and bool(s), 'Required aware UTC timestamp missing')
    must(s.endswith('Z') or ('+' in s[10:]), 'Timestamp has no UTC timezone')
    try:
        dt = datetime.fromisoformat(s.replace('Z','+00:00'))
    except (ValueError,TypeError) as e:
        raise AuditError(f'Invalid timestamp: {s}') from e
    must(dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0, 'Timestamp must be UTC')
    return dt

def validate_registry(r):
    must(isinstance(r,dict), 'Registry must be a JSON object')
    must(r.get('record_type')=='M11_PHASE_B_SEVEN_CONFIGURATION_REGISTRY_DRAFT_V0_1', 'Wrong record type')
    must(r.get('status')=='DRAFT_NOT_FROZEN', 'Only DRAFT is supported by this tool; do not claim freeze')
    must(r.get('study_design')=='ORIGINAL_150_TOTAL_DESCRIPTIVE', 'Wrong accepted-count design')
    must(r.get('adopted_protocol_commit')==PROTOCOL_COMMIT,'Wrong adopted protocol commit')
    must(r.get('official_submitting_baseline')=='M1', 'Frozen official baseline changed')
    for key, positions in [('phase_a_accepted_positions',[1,30]),('phase_b_accepted_positions',[31,90]),('phase_c_accepted_positions',[91,150])]:
        must(r.get(key)==positions, f'Phase accepted-count boundary changed: {key}')
    must(r.get('phase_b_transition_rule')=='FREEZE_BEFORE_FIRST_VERIFIED_ELIGIBLE_OPENING_AFTER_ACCEPTED_30','Phase B freeze transition changed')
    must(r.get('phase_c_transition_rule')=='FREEZE_FINALIST_BEFORE_FIRST_VERIFIED_ELIGIBLE_OPENING_AFTER_ACCEPTED_90','Phase C freeze transition changed')
    models=r.get('models')
    must(isinstance(models,list) and len(models)==7, 'Exactly seven model configurations required')
    must([x.get('id') for x in models if isinstance(x,dict)] == list(EXPECTED_SOURCES), 'Model IDs/order changed')
    for m in models:
        mid=m['id']
        must(m.get('sources')==EXPECTED_SOURCES[mid], f'Forbidden or missing features: {mid}')
        if mid=='M1':
            must(m.get('role')=='FROZEN_OFFICIAL_BASELINE','M1 role cannot change')
            must(m.get('artifact_status')=='PRESERVE_EXISTING_FROZEN_M1','M1 status changed')
            must(m.get('artifact_sha256')==MODEL_SHA,'M1 frozen model SHA mismatch')
            must(bool(m.get('algorithm')) and bool(m.get('training_cutoff_date')), 'Missing original M1 provenance')
            must(re.fullmatch(r'\d{4}-\d{2}-\d{2}', m['training_cutoff_date']) is not None, 'M1 training cutoff date invalid')
        else:
            must(m.get('role')=='SHADOW_CANDIDATE','Candidate must be shadow')
            must(m.get('artifact_status')=='NOT_TRAINED','Draft candidate claims trained artifact')
            must(m.get('artifact_sha256') is None and m.get('algorithm') is None and m.get('training_cutoff_utc') is None,'Unfrozen artifact or algorithm asserted')
    ctrl=r.get('matched_training_control') or {}
    must(ctrl.get('name')=='MATCHED_TRAINING_ES_ONLY_UNWEIGHTED','Required matched ES-only control missing')
    must(ctrl.get('status')=='NOT_TRAINED' and ctrl.get('artifact_sha256') is None,'Premature control training claim')
    must(ctrl.get('must_use_same_training_window_as_challenger') is True,'Matched-training control mismatch')
    cmp=r.get('comparison_plan') or {}
    must(cmp.get('primary_metric')=='UNSCALED_MULTICLASS_BRIER','Comparison metric must be unscaled multiclass Brier')
    must(cmp.get('paired_challenge_ids_only') is True,'Require paired challenges')
    must(cmp.get('max_primary_finalists_including_M1')==4,'Four-finalist ceiling changed')
    must(cmp.get('primary_finalists')==[] and cmp.get('finalist_selection_status')=='NOT_FROZEN','Primary finalists cannot be claimed frozen in draft')
    for k in ('minimum_improvement_threshold','missing_pair_rule','tie_break_rule','fallback_rule','block_bootstrap_rule'):
        must(cmp.get(k) is None, f'Final selection rule prematurely claimed in draft: {k}')
    must(r.get('external_source_readiness')=={k:'NOT_VALIDATED' for k in ('GDELT','VIX','GOOGLE_TRENDS','UNEMPLOYMENT')},'External sources not yet validated')
    must(r.get('input_time_rule')=='NO_FEATURE_USABLE_AFTER_DECISION_UTC_OR_WITH_UNKNOWN_POINT_IN_TIME_AVAILABILITY','Leakage rule changed')
    must(r.get('do_not_add_sources')==['SP500_INDEX'],'Do-not-add S&P500 index constraint missing')
    must(r.get('private_data_and_model_artifacts_publication')=='FORBIDDEN','Private artifact disclosure restriction missing')
    must(r.get('official_phase_b_forecasts')=='M1_ONLY_UNLESS_SEPARATELY_AUTHORIZED_BY_ADOPTED_PROTOCOL_AND_ARENA','Official submitter restriction changed')
    must(r.get('phase_b_frozen_decision_record') is None and r.get('phase_b_freeze_certified') is False,'A draft cannot be a Phase B freeze')
    must(r.get('prospective_shadow_predictions_created')==0,'Draft cannot claim shadow outputs')
    return {'status':'DRAFT_REGISTRY_VALIDATED_NOT_FROZEN', 'design':r['study_design'],
            'configuration_ids':list(EXPECTED_SOURCES), 'trained_new_models':0,
            'primary_finalists_frozen':0, 'phase_b_freeze_certified':False,
            'official_submitter':'M1', 'files_written':0, 'network_calls':0,
            'forecasts_submitted':0}

def validate_evidence(registry, manifest, model_id, decision_utc):
    must(model_id in EXPECTED_SOURCES, 'Unknown model ID')
    cutoff=parse_utc(decision_utc)
    must(isinstance(manifest,dict) and manifest.get('record_type')=='M11_STAGE5_CANDIDATE_INPUT_EVIDENCE_V0_1','Evidence record type mismatch')
    must(manifest.get('model_id')==model_id,'Evidence model ID mismatch')
    must(parse_utc(manifest.get('decision_cutoff_utc'))==cutoff,'Evidence cutoff mismatch')
    rows=manifest.get('inputs')
    must(isinstance(rows,list),'Input evidence must be a list')
    names=[r.get('source') if isinstance(r,dict) else None for r in rows]
    must(len(names)==len(EXPECTED_SOURCES[model_id]) and len(set(names))==len(names) and set(names)==set(EXPECTED_SOURCES[model_id]),'Missing, extra or duplicate required sources')
    for row in rows:
        src=row['source']
        must(row.get('asof_evidence_status')=='VERIFIED_PREDECISION',f'Unknown point-in-time evidence: {src}')
        for key in ('observation_end_utc','source_publication_utc','source_available_utc','retrieved_at_utc'):
            t=parse_utc(row.get(key))
            must(t<=cutoff,f'LOOKAHEAD {src}.{key} occurs after forecast cutoff')
        must(parse_utc(row['source_publication_utc'])<=parse_utc(row['source_available_utc']),f'Publication after available: {src}')
        must(parse_utc(row['source_available_utc'])<=parse_utc(row['retrieved_at_utc']),f'Unverified retrieval availability sequence: {src}')
        must(isinstance(row.get('provenance_sha256'),str) and HEX_SHA.fullmatch(row['provenance_sha256']) is not None,f'Missing source provenance hash: {src}')
        must(isinstance(row.get('licensing_restriction'),str) and row['licensing_restriction'] in {'PRIVATE_LICENSED','PUBLIC_REUSE_REVIEW_REQUIRED','PUBLIC_REUSE_CONFIRMED'},f'License state missing: {src}')
        must(isinstance(row.get('source_revision_or_vintage'),str) and row['source_revision_or_vintage'].strip(),f'Missing source vintage: {src}')
        must(row.get('includes_target_date_settlement') is False,f'Target leakage: {src}')
    return {'status':'INPUT_TIMING_CONSISTENT_WITH_SUPPLIED_EVIDENCE_NOT_EXTERNALLY_CERTIFIED',
            'model_id':model_id,'decision_cutoff_utc':decision_utc,
            'sources_checked':len(rows),'files_written':0,'network_calls':0,'forecasts_submitted':0,
            'phase_b_freeze_certified':False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('audit-registry')
    r.add_argument('--registry',type=Path,required=True)
    r.add_argument('--expected-registry-sha256')
    e=sub.add_parser('audit-input-evidence')
    e.add_argument('--registry',type=Path,required=True)
    e.add_argument('--manifest',type=Path,required=True)
    e.add_argument('--expected-manifest-sha256',required=True)
    e.add_argument('--model-id',required=True)
    e.add_argument('--decision-utc',required=True)
    x=p.parse_args(argv)
    try:
        reg=load_json(x.registry)
        a=validate_registry(reg)
        digest=sha256_file(x.registry)
        if x.command=='audit-registry':
            if x.expected_registry_sha256:
                must(digest==x.expected_registry_sha256.lower(), 'Registry file SHA-256 mismatch')
            a['registry_sha256']=digest
        else:
            must(bool(HEX_SHA.fullmatch(x.expected_manifest_sha256.lower())),'Invalid expected manifest SHA')
            must(sha256_file(x.manifest)==x.expected_manifest_sha256.lower(),'Input evidence manifest SHA mismatch')
            a=validate_evidence(reg,load_json(x.manifest),x.model_id,x.decision_utc)
            a['input_manifest_sha256']=x.expected_manifest_sha256.lower()
        print('M11_STAGE5_READ_ONLY_AUDIT: PASS')
        print(json.dumps(a,indent=2,sort_keys=True))
        return 0
    except (AuditError, OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        print('M11_STAGE5_AUDIT: FAIL - '+str(exc), file=sys.stderr)
        return 2

if __name__=='__main__':
    sys.exit(main())
