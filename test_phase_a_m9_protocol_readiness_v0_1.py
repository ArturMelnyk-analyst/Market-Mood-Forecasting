"""Isolated synthetic tests for the read-only M9 readiness gate."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase_a_m9_protocol_readiness_v0_1 as m

class M9ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        for name in m.NEEDED_IMPLEMENTATIONS:
            (self.root/name).write_text('synthetic')
        self.candidate = self.root/'data/processed/es_arena_research_v0_1/arena_opportunity_candidate_20261112.csv'
        self.candidate.parent.mkdir(parents=True)
        self.candidate.write_text('synthetic pinned candidate')
        self.amendment=Path(self.temp.name)/m.EXPECTED_AMENDMENT_FILENAME
        self.amendment.write_text('synthetic amendment')
        self.adoption=Path(self.temp.name)/'adoption.json'
        self.adoption.write_text(json.dumps({'status':'NOT_ADOPTED','decision':'PENDING_REVIEW',
            'actual_decision_utc':None,'full_public_commit_sha':None,
            'independent_publication_evidence':None,'first_future_challenge_id':None,
            'phase_boundary_decision':None}))
        self.good_recon={
          'pinned_20261112_accepted_count':18,
          'new_native_scored_count':1,
          'total_acceptance_evidenced_in_union':19,
          'settled_records_in_union':18,
          'eligible_opportunity_census_complete':False,
          'phase_a_ordinal_or_boundary_certified':False,
          'independent_utc_acceptance_time_certified':False,
          'review_issues':[],
        }
        self.patches=[
            patch.object(m, 'PINNED_CANDIDATE_SHA256',m.sha256_file(self.candidate)),
            patch.object(m, 'REFERENCE_AMENDMENT_SHA256',m.sha256_file(self.amendment)),
        ]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
    def run_audit(self,recon=None):
        if recon is None: recon=self.good_recon
        return m.audit(self.root,self.amendment,self.adoption,reconcile_fn=lambda _:recon)
    def test_clean_pending_review(self):
        x=self.run_audit()
        self.assertEqual(x['status'],'READY_FOR_HUMAN_POLICY_REVIEW')
        self.assertFalse(x['protocol_adopted'])
        self.assertFalse(x['historical_eligible_opportunities_certified'])
        self.assertEqual(x['files_written'],0)
    def test_no_new_files_written(self):
        before={str(x.relative_to(self.root)) for x in self.root.rglob('*')}
        self.run_audit()
        self.assertEqual(before,{str(x.relative_to(self.root)) for x in self.root.rglob('*')})
    def test_wrong_candidate_hash_blocks(self):
        self.candidate.write_text('changed')
        self.assertEqual(self.run_audit()['status'],'REVIEW_REQUIRED')
    def test_wrong_amendment_hash_blocks(self):
        self.amendment.write_text('changed')
        self.assertEqual(self.run_audit()['status'],'REVIEW_REQUIRED')
    def test_missing_dependency_blocks(self):
        (self.root/m.NEEDED_IMPLEMENTATIONS[0]).unlink()
        self.assertEqual(self.run_audit()['status'],'REVIEW_REQUIRED')
    def test_adoption_claim_blocks(self):
        record=json.loads(self.adoption.read_text()); record['status']='ADOPTED'
        self.adoption.write_text(json.dumps(record))
        self.assertEqual(self.run_audit()['status'],'REVIEW_REQUIRED')
    def test_fake_decision_time_blocks(self):
        record=json.loads(self.adoption.read_text());record['actual_decision_utc']='2026-11-13T14:00:00Z'
        self.adoption.write_text(json.dumps(record))
        self.assertEqual(self.run_audit()['status'],'REVIEW_REQUIRED')
    def test_review_conflict_blocks(self):
        report=dict(self.good_recon,review_issues=[{'problems':['bad source']}])
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')
    def test_pinned_count_changed_blocks(self):
        report=dict(self.good_recon,pinned_20261112_accepted_count=19)
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')
    def test_count_inconsistency_blocks(self):
        report=dict(self.good_recon,total_acceptance_evidenced_in_union=20)
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')
    def test_unproven_census_must_remain_unproven(self):
        report=dict(self.good_recon,eligible_opportunity_census_complete=True)
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')
    def test_no_automatic_phase_boundary_certification(self):
        report=dict(self.good_recon,phase_a_ordinal_or_boundary_certified=True)
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')
    def test_no_automatic_time_certification(self):
        report=dict(self.good_recon,independent_utc_acceptance_time_certified=True)
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')
    def test_invalid_template_rejected(self):
        self.adoption.write_text('[]')
        self.assertEqual(self.run_audit()['status'],'REVIEW_REQUIRED')
    def test_newer_prospective_count_not_hardcoded(self):
        report=dict(self.good_recon,new_native_scored_count=2,total_acceptance_evidenced_in_union=20,settled_records_in_union=19)
        self.assertEqual(self.run_audit(report)['status'],'READY_FOR_HUMAN_POLICY_REVIEW')
    def test_noninteger_counts_rejected(self):
        report=dict(self.good_recon,new_native_scored_count='1')
        self.assertEqual(self.run_audit(report)['status'],'REVIEW_REQUIRED')

if __name__=='__main__':
    unittest.main()
