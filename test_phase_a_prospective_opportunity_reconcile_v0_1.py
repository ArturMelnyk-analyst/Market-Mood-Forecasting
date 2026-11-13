"""Synthetic read-only tests for Phase A opportunity reconciliation."""
import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import phase_a_opportunity_audit_v0_1 as opportunity
import phase_a_prospective_opportunity_reconcile_v0_1 as reconcile

FIRST = '118cefc4-96a8-4e47-b617-73af53c67d03'
NEW = '9e7a3255-e4dc-4a1d-a8f9-a30f18c4fa2c'
DATE = '2026-11-13'


class ProspectiveOpportunityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='phase_a_ledger_synthetic_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'data/processed/es_arena_live'
        self.research = self.root / 'data/processed/es_arena_research_v0_1'
        self.research.mkdir(parents=True)
        self.candidate = self.research / 'arena_opportunity_candidate_20261112.csv'
        self.new_forecast = self.base / 'forecasts' / f'forecast_20261113_{NEW}.json'
        self.attempt = self.base / 'receipts' / f'attempt_{NEW}.json'
        self.receipt = self.base / 'receipts' / f'receipt_{NEW}.json'
        self.discovery = self.base / 'discovery' / 'arena_discovery_synthetic.json'
        for path in [self.new_forecast, self.attempt, self.receipt, self.discovery]:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.ids = [FIRST] + [f'00000000-0000-4000-8000-{i:012d}' for i in range(1, 18)]
        with self.candidate.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['challenge_id', 'settlement_date', 'opportunity_status'])
            writer.writeheader()
            for i, cid in enumerate(self.ids):
                writer.writerow({'challenge_id': cid, 'settlement_date':f'2026-10-{20+i:02d}' if i < 12 else '2026-11-11',
                                 'opportunity_status':'confirmed_accepted'})
        self.candidate_sha = self.hash(self.candidate)
        with (self.research / 'arena_official_settlements.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['challenge_id','settlement_date','settled_outcome'])
            writer.writeheader()
            for cid in self.ids[:17]:
                writer.writerow({'challenge_id':cid, 'settlement_date':'2026-11-11', 'settled_outcome':'bullish'})
            writer.writerow({'challenge_id':NEW, 'settlement_date':DATE, 'settled_outcome':'neutral'})
        self.forecast = {'challenge_id':NEW, 'target_session_date':DATE,'contract_symbol':'ESZ6',
                         'model_sha256':reconcile.MODEL_SHA,'deadline_utc':'2026-11-13T15:00:00Z',
                         'probabilities':{'bearish':0.37,'neutral':0.18,'bullish':0.45}}
        self.save(self.new_forecast, self.forecast)
        self.save(self.attempt, {'challenge_id':NEW,'forecast_sha256':self.hash(self.new_forecast)})
        self.save(self.receipt,{'http_status':201,'forecast_sha256':self.hash(self.new_forecast),
                                'response':{'challenge_id':NEW,'prediction_id':'synthetic-accepted',
                                            'counts_for_score':True,'probabilities':self.forecast['probabilities'],
                                            'created_at':'2026-11-13 11:22:00'}})
        self.discovery_obj = {'status':'success','http_status':200,
                 'record_completed_at_utc':'2026-11-13T09:12:12Z',
                 'server_date_header':'Fri, 13 Nov 2026 09:12:11 GMT',
                 'response_json':{'challenges':[{'id':NEW,'asset':'ES', 'challenge_type':'daily',
                            'contract_symbol':'ESZ6','deadline':'2026-11-13T15:00:00Z'}]}}
        self.save(self.discovery, self.discovery_obj)

    @staticmethod
    def save(p, obj):
        p.write_text(json.dumps(obj), encoding='utf-8')

    @staticmethod
    def hash(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def report(self):
        with mock.patch.object(opportunity,'CANDIDATE_SHA256',self.candidate_sha):
            return reconcile.build_report(self.root)

    def new_row(self):
        return next(x for x in self.report()['records'] if x['challenge_id'] == NEW)

    def test_new_native_accepted_and_official_settlement_join_once(self):
        r = self.report()
        self.assertEqual(r['pinned_20261112_accepted_count'],18)
        self.assertEqual(r['new_native_scored_count'],1)
        self.assertEqual(r['total_acceptance_evidenced_in_union'],19)
        self.assertEqual(r['settled_records_in_union'],18)
        self.assertEqual(len(r['records']),19)
        self.assertEqual(r['review_issues'],[])
        self.assertTrue(self.new_row()['predeadline_discovery_http_date_evidence'])

    def test_original_cohort_and_phase_boundary_not_recategorized(self):
        r=self.report()
        self.assertFalse(r['eligible_opportunity_census_complete'])
        self.assertFalse(r['phase_a_ordinal_or_boundary_certified'])
        self.assertEqual(r['original_notebook12_cohort'],'first 30 accepted forecasts; UNCHANGED')

    def test_no_receipt_not_automatically_missed(self):
        self.receipt.unlink()
        r=self.report()
        self.assertEqual(r['total_acceptance_evidenced_in_union'],18)
        self.assertEqual(self.new_row()['acceptance_evidence'],'NO_CONFIRMED_ACCEPTANCE_IN_THIS_AUDIT')
        self.assertFalse(r['eligible_opportunity_census_complete'])

    def test_afterdeadline_server_date_not_predeadline(self):
        self.discovery_obj['server_date_header']='Fri, 13 Nov 2026 15:12:11 GMT'
        self.save(self.discovery,self.discovery_obj)
        self.assertFalse(self.new_row()['predeadline_discovery_http_date_evidence'])

    def test_afterdeadline_local_completion_not_predeadline(self):
        self.discovery_obj['record_completed_at_utc']='2026-11-13T15:12:12Z'
        self.save(self.discovery,self.discovery_obj)
        self.assertFalse(self.new_row()['predeadline_discovery_http_date_evidence'])

    def test_receipt_hash_mismatch_detected_not_counted(self):
        obj=json.loads(self.receipt.read_text()); obj['forecast_sha256']='0'*64
        self.save(self.receipt,obj)
        self.assertEqual(self.report()['new_native_scored_count'],0)
        self.assertTrue(self.new_row()['review_issues'])

    def test_attempt_hash_mismatch_detected(self):
        obj=json.loads(self.attempt.read_text()); obj['forecast_sha256']='0'*64
        self.save(self.attempt,obj)
        self.assertEqual(self.report()['new_native_scored_count'],0)
        self.assertTrue(self.report()['review_issues'])

    def test_model_identity_mismatch_detected(self):
        self.forecast['model_sha256']='f'*64
        self.save(self.new_forecast,self.forecast)
        self.assertEqual(self.report()['new_native_scored_count'],0)

    def test_discovery_contract_disagreement_flagged(self):
        self.discovery_obj['response_json']['challenges'][0]['contract_symbol']='ESU6'
        self.save(self.discovery,self.discovery_obj)
        self.assertTrue(self.report()['review_issues'])
        self.assertFalse(self.new_row()['predeadline_discovery_http_date_evidence'])

    def test_discovery_deadline_disagrees_with_forecast(self):
        self.discovery_obj['response_json']['challenges'][0]['deadline']='2026-11-13T16:00:00Z'
        self.save(self.discovery, self.discovery_obj)
        self.assertTrue(any('deadline differs' in x.lower() for x in self.new_row()['review_issues']))

    def test_unsafe_challenge_id_rejected(self):
        self.discovery_obj['response_json']['challenges'][0]['id']='../../../../etc/passwd'
        self.save(self.discovery, self.discovery_obj)
        with self.assertRaisesRegex(ValueError, 'Unsafe/noncanonical challenge ID'):
            self.report()

    def test_does_not_write_any_files(self):
        original={str(p):self.hash(p) for p in self.root.rglob('*') if p.is_file()}
        self.report()
        actual={str(p):self.hash(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(original,actual)

    def test_unmatched_official_raises_review_not_fabricated_challenge(self):
        off=self.research/'arena_official_settlements.csv'
        with off.open('a',encoding='utf-8',newline='') as f:
            csv.writer(f).writerow(['11111111-1111-4111-8111-111111111111',DATE,'neutral'])
        r=self.report()
        self.assertEqual(len(r['records']),19)
        self.assertEqual(len(r['unmatched_official_challenge_ids']),1)
        self.assertTrue(r['review_issues'])

    def test_duplicate_discovery_archives_do_not_create_duplicate_opportunity(self):
        other=self.discovery.with_name('arena_discovery_second.json')
        other.write_bytes(self.discovery.read_bytes())
        r=self.report()
        self.assertEqual(len(r['records']),19)
        self.assertEqual(self.new_row()['discovery_observation_count'],2)


if __name__=='__main__': unittest.main(verbosity=2)
