"""Read-only Phase A Milestone 9 protocol READINESS audit (not an adoption action).

Dependencies: Python >=3.10 standard library and existing
phase_a_prospective_opportunity_reconcile_v0_1.py in the project root.
No network, Git mutations, model loading, private nonce handling, or outputs written.

Example:
 python phase_a_m9_protocol_readiness_v0_1.py --root . --amendment PATH --adoption-template PATH
 python phase_a_m9_protocol_readiness_v0_1.py --root . --amendment PATH --adoption-template PATH --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PINNED_CANDIDATE_SHA256 = '620816c8161312508e7220c85a064776f60e6af4ae4acbeb1fd5a846b851a575'
REFERENCE_AMENDMENT_SHA256 = 'db572143c52dccbce10c0089a2866667d0aaf17490096c0f57a9e88fc511afa2'
EXPECTED_AMENDMENT_FILENAME = 'PHASE_A_OPPORTUNITY_ACCOUNTING_AMENDMENT_v0_1_2026-11-12.md'
NEEDED_IMPLEMENTATIONS = (
    'phase_a_opportunity_audit_v0_1.py',
    'phase_a_prospective_opportunity_reconcile_v0_1.py',
    'phase_a_commitment_lifecycle_v0_1.py',
    'phase_a_forecast_commitment_v0_1.py',
    'run_phase_a_milestone7_v0_1.py',
)

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Adoption template must be a JSON object')
    return value


def summarize(reconciliation: dict) -> dict:
    """Reject reconciliation conflicts, never infer unobserved challenges or phase order."""
    for k in ('pinned_20261112_accepted_count','new_native_scored_count',
              'total_acceptance_evidenced_in_union','settled_records_in_union'):
        if type(reconciliation.get(k)) is not int or reconciliation[k] < 0:
            raise ValueError('Missing/invalid reconciliation count: '+k)
    pinned = reconciliation['pinned_20261112_accepted_count']
    native = reconciliation['new_native_scored_count']
    total = reconciliation['total_acceptance_evidenced_in_union']
    settled = reconciliation['settled_records_in_union']
    if pinned != 18 or pinned + native != total or settled > total:
        raise ValueError('Reconciliation cardinality/pinned-cohort conflict')
    if reconciliation.get('review_issues'):
        raise ValueError('Reconciliation contains review issues; investigate before proceeding')
    if reconciliation.get('eligible_opportunity_census_complete') is not False:
        raise ValueError('Audit cannot certify the historical eligible-opportunity census')
    if reconciliation.get('phase_a_ordinal_or_boundary_certified') is not False:
        raise ValueError('Audit cannot certify the Phase A/B boundary')
    if reconciliation.get('independent_utc_acceptance_time_certified') is not False:
        raise ValueError('Audit cannot certify independent acceptance timing')
    return {
        'pinned_accepted_20261112':pinned,
        'additional_native_scored':native,
        'acceptance_evidenced_union':total,
        'official_settled_in_union':settled,
        'historical_eligible_census_complete':False,
        'phase_a_boundary_certified':False,
        'independent_utc_acceptance_certified':False,
    }


def audit(root:Path, amendment:Path, adoption_template:Path, *, reconcile_fn=None) -> dict:
    root = root.resolve()
    amendment = amendment.resolve()
    adoption_template = adoption_template.resolve()
    problems=[]
    for name in NEEDED_IMPLEMENTATIONS:
        if not (root/name).is_file():
            problems.append('Missing existing implementation: '+name)
    candidate=root/'data/processed/es_arena_research_v0_1/arena_opportunity_candidate_20261112.csv'
    candidate_digest=None
    try:
        candidate_digest=sha256_file(candidate)
        if candidate_digest != PINNED_CANDIDATE_SHA256:
            problems.append('Pinned historical candidate SHA-256 mismatch: do not replace or rewrite')
    except OSError as e:
        problems.append('Pinned historical candidate missing or unreadable: '+type(e).__name__)
    amendment_digest=None
    try:
        amendment_digest=sha256_file(amendment)
        if amendment.name != EXPECTED_AMENDMENT_FILENAME or amendment_digest != REFERENCE_AMENDMENT_SHA256:
            problems.append('Reference amendment differs from reviewed v0.1: review/version separately')
    except OSError as e:
        problems.append('Reference amendment missing or unreadable: '+type(e).__name__)
    record=None
    try:
        record=load_json_object(adoption_template)
        if record.get('status') != 'NOT_ADOPTED':
            problems.append('Adoption template must remain NOT_ADOPTED; this tool never approves adoption')
        if record.get('decision') != 'PENDING_REVIEW':
            problems.append('Adoption decision must remain PENDING_REVIEW in readiness template')
        if any(record.get(k) not in (None,'PENDING') for k in (
            'actual_decision_utc','full_public_commit_sha','independent_publication_evidence',
            'first_future_challenge_id','phase_boundary_decision')):
            problems.append('Template contains an adoption/evidence claim: verify separately, do not infer certification')
    except (OSError, ValueError, json.JSONDecodeError) as e:
        problems.append('Adoption template invalid/unreadable: '+type(e).__name__)
    counts=None
    if not problems:
        try:
            if reconcile_fn is None:
                from phase_a_prospective_opportunity_reconcile_v0_1 import build_report
                reconcile_fn = build_report
            counts=summarize(reconcile_fn(root))
        except (ImportError, ValueError, OSError, json.JSONDecodeError) as e:
            problems.append('Real evidence reconciliation not clean: '+type(e).__name__+': '+str(e)[:180])
    return {
        'schema_version':'phase_a_m9_protocol_readiness_v0_1',
        'read_only':True,
        'status':'READY_FOR_HUMAN_POLICY_REVIEW' if not problems else 'REVIEW_REQUIRED',
        'verified_pinned_candidate_sha256':candidate_digest,
        'reference_amendment_sha256':amendment_digest,
        'observed_reconciliation':counts,
        'protocol_adopted':False,
        'independent_publication_certified':False,
        'historical_eligible_opportunities_certified':False,
        'phase_a_or_b_boundary_certified':False,
        'milestone9_local_software_readiness':not problems,
        'human_decisions_still_required':[
            'Explicit dated adoption and prospective scope after review',
            'First future eligible ES challenge after adoption, from real independent evidence',
            'A/B boundary treatment given uncertified historical opportunity coverage',
            'Choice of 30/60/60 versus 30/30/45/45 and pre-Phase B freeze',
            'Real external predeadline publication and post-settlement reveal policy',
        ],
        'issues':problems,
        'files_written':0,
        'network_requests':0,
        'forecast_submissions':0,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.',help='Current Market-Mood-Forecasting repository root')
    p.add_argument('--amendment',required=True,help='Reviewed unchanged v0.1 amendment reference file')
    p.add_argument('--adoption-template',required=True,help='NOT_ADOPTED JSON template, outside repository')
    p.add_argument('--json',action='store_true',help='Print metadata-only report as JSON')
    args=p.parse_args(argv)
    result=audit(Path(args.root),Path(args.amendment),Path(args.adoption_template))
    if args.json:
        print(json.dumps(result,indent=2,sort_keys=True))
    else:
        print('PHASE A MILESTONE 9 PROTOCOL READINESS (READ ONLY)')
        print('STATUS:',result['status'])
        if result['observed_reconciliation']:
            for k,v in result['observed_reconciliation'].items():
                print(k.upper()+':',v)
        print('PROTOCOL ADOPTED: NO')
        print('INDEPENDENT PUBLICATION CERTIFIED: NO')
        print('HISTORICAL ELIGIBLE CENSUS / PHASE B BOUNDARY CERTIFIED: NO')
        for issue in result['issues']:
            print('ISSUE:',issue)
        print('FILES WRITTEN: 0 | NETWORK REQUESTS: 0 | PREDICTION SUBMISSIONS: 0')
    return 0 if not result['issues'] else 1

if __name__=='__main__':
    sys.exit(main())
