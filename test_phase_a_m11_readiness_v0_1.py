import csv
import hashlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import phase_a_m11_readiness_v0_1 as m

CID='9e7a3255-e4dc-4a1d-a8f9-a30f18c4fa2c'

def fixture_row(sha, path='source.json'):
    return {
        'challenge_id':CID,'asset':'ES','challenge_type':'daily','contract_symbol':'ESZ6',
        'open_at_utc':'2026-11-16T00:00:00Z',
        'deadline_utc':'2026-11-16T15:00:00Z',
        'first_observed_at_utc':'2026-11-16T10:00:00Z',
        'opportunity_first_evidence_path':path,'opportunity_first_evidence_sha256':sha,
        'availability_status':'CONFIRMED_OPEN','submission_status':'NO_ACCEPTED_RECORD_YET',
        'outcome_status':'PENDING','provenance_level':'SOURCE_DISCOVERY_PREDEADLINE'}

def reports():
    ids=[str(__import__('uuid').UUID(int=i+1)) for i in range(19)]
    integrated={
        'retained_challenges':19,'revision_records':19,'review_flags':[],
        'records':[{'challenge_id':cid} for cid in ids],
        'model_execution_certified':False,'final_scored_revision_certified':False,
        'historical_eligible_opportunity_census_certified':False,'phase_a_protocol_adopted':False}
    settlement={
        'challenge_count':19,'settlement_rows_in_observed_cohort':19,
        'settlement_rows_outside_observed_cohort':0,
        'counts':{'deadline_DEADLINES_MATCH_AS_RECORDED':19},'review_flags':[],
        'sources':{'m10c_integrated_physical_sha256':m.PIN['integrated'],
                   'local_official_settlements_csv_physical_sha256':m.PIN['csv']},
        'records':[{'challenge_id':cid,'official_settlement_row_present':True} for cid in ids],
        'final_scored_revision_certified':False,'model_execution_certified':False,
        'historical_opportunity_census_certified':False,
        'phase_a_150_opportunity_protocol_adopted':False}
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=['challenge_id','official_label']);w.writeheader()
    for i,cid in enumerate(ids):
        w.writerow({'challenge_id':cid,'official_label':'bearish' if i<8 else 'bullish' if i<16 else 'neutral'})
    return integrated,settlement,out.getvalue()

class TestM11(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)/'MarketMood_Private'/'phase_a_m11';self.root.mkdir(parents=True)
        self.source=self.root/'source.json';self.source.write_text('{"source":"synthetic"}',encoding='utf-8')
        self.row=fixture_row(m.digest(self.source.read_bytes()))
    def validate(self,rows=None):
        rows=[self.row] if rows is None else rows
        raw=('\n'.join(json.dumps(x) for x in rows)+'\n').encode()
        return m.validate_ledger(raw,self.root)
    def test_clean_future_candidate(self):
        result=self.validate();self.assertEqual(result['review_issues'],[])
        self.assertEqual(result['candidate_confirmed_open'],1)
        self.assertEqual(result['new_150_opportunity_trial_enrolled_count'],0)
    def test_empty_candidate_does_not_adopt(self):
        r=self.validate([])
        self.assertEqual(r['candidate_rows'],0);self.assertFalse(r['automatically_adopted'])
    def test_duplicate_ids_review(self):
        self.assertIn('invalid_or_duplicate_challenge_id',self.validate([self.row,self.row])['review_issues'][0]['issues'])
    def test_bad_uuid_review(self):
        row=dict(self.row,challenge_id='not-a-uuid')
        self.assertIn('invalid_or_duplicate_challenge_id',self.validate([row])['review_issues'][0]['issues'])
    def test_wrong_market_scope(self):
        row=dict(self.row,asset='SPX')
        self.assertIn('invalid_market_scope',self.validate([row])['review_issues'][0]['issues'])
    def test_future_contract_roll_is_allowed(self):
        row=dict(self.row,contract_symbol='ESH7')
        self.assertEqual(self.validate([row])['review_issues'],[])
    def test_naive_time_rejected(self):
        row=dict(self.row,deadline_utc='2026-11-16T15:00:00')
        self.assertIn('invalid_deadline_utc',self.validate([row])['review_issues'][0]['issues'])
    def test_open_deadline_order(self):
        row=dict(self.row,open_at_utc='2026-11-16T15:30:00Z')
        self.assertIn('invalid_open_deadline_order',self.validate([row])['review_issues'][0]['issues'])
    def test_postdeadline_observation_not_predeadline(self):
        row=dict(self.row,first_observed_at_utc='2026-11-16T15:00:00Z')
        self.assertIn('postdeadline_evidence_claimed_predate',self.validate([row])['review_issues'][0]['issues'])
    def test_source_tamper(self):
        self.source.write_bytes(self.source.read_bytes()+b'!')
        self.assertIn('source_sha_mismatch',self.validate()['review_issues'][0]['issues'])
    def test_source_outside_private_rejected(self):
        outside=Path(self.t.name)/'outside.json';outside.write_text('ok')
        row=dict(self.row,opportunity_first_evidence_path=str(outside),opportunity_first_evidence_sha256=m.digest(b'ok'))
        self.assertIn('source_not_under_private_evidence_root',self.validate([row])['review_issues'][0]['issues'])
    def test_missing_source_rejected(self):
        row=dict(self.row,opportunity_first_evidence_sha256='')
        self.assertIn('missing_source_and_sha',self.validate([row])['review_issues'][0]['issues'])
    def test_accepted_without_receipt_rejected(self):
        row=dict(self.row,submission_status='ACCEPTED_SCORED')
        self.assertIn('missing_scored_acceptance_reference',self.validate([row])['review_issues'][0]['issues'])
    def test_accepted_with_receipt_reference_not_certified(self):
        row=dict(self.row,submission_status='ACCEPTED_SCORED',accepted_receipt_sha256='a'*64,prediction_id='server-prediction')
        result=self.validate([row]);self.assertEqual(result['review_issues'],[])
        self.assertEqual(result['candidate_accepted_scored'],1)
        self.assertEqual(result['new_150_opportunity_trial_enrolled_count'],0)
    def test_missed_without_evidence_rejected(self):
        row=dict(self.row,submission_status='MISSED_VERIFIED')
        self.assertIn('missing_independent_miss_evidence',self.validate([row])['review_issues'][0]['issues'])
    def test_miss_before_deadline_rejected(self):
        row=dict(self.row,submission_status='MISSED_VERIFIED',miss_evidence_sha256='a'*64,miss_reviewed_at_utc='2026-11-16T12:00:00Z')
        self.assertIn('miss_declared_before_deadline',self.validate([row])['review_issues'][0]['issues'])
    def test_settled_without_ground_truth_rejected(self):
        row=dict(self.row,outcome_status='OFFICIALLY_SETTLED')
        self.assertIn('missing_official_settlement_reference',self.validate([row])['review_issues'][0]['issues'])
    def test_cancellation_not_evaluable(self):
        row=dict(self.row,availability_status='CANCELLED_AFTER_OPEN',outcome_status='CANCELLED_NO_OUTCOME')
        result=self.validate([row]);self.assertEqual(result['candidate_cancelled'],1)
        self.assertEqual(result['candidate_officially_settled'],0)
    def test_historical_19_settled(self):
        i,s,c=reports();r=m.evaluate_historical(i,s,c)
        self.assertEqual(r['review_flags'],[]);self.assertFalse(r['historical_eligible_census_certified'])
    def test_historical_missing_settlement_review(self):
        i,s,c=reports();s['settlement_rows_in_observed_cohort']=18
        self.assertIn('Unexpected 19-settlement count',m.evaluate_historical(i,s,c)['review_flags'])
    def test_historical_wrong_sha_link_review(self):
        i,s,c=reports();s['sources']['m10c_integrated_physical_sha256']='f'*64
        self.assertIn('Integrated source SHA mismatch',m.evaluate_historical(i,s,c)['review_flags'])
    def test_historical_false_scored_certification_review(self):
        i,s,c=reports();s['final_scored_revision_certified']=True
        self.assertTrue(any('incorrectly certifies' in x for x in m.evaluate_historical(i,s,c)['review_flags']))
    def test_historical_id_conflict(self):
        i,s,c=reports();s['records'][0]['challenge_id']='00000000-0000-0000-0000-000000000066'
        self.assertIn('M10C record identity mismatch',m.evaluate_historical(i,s,c)['review_flags'])
    def test_pinned_source_tamper(self):
        expected=m.digest(self.source.read_bytes());self.assertEqual(m.pinned_bytes(self.source,expected),self.source.read_bytes())
        self.source.write_bytes(self.source.read_bytes()+b'x')
        with self.assertRaises(m.EvidenceError): m.pinned_bytes(self.source,expected)
    def test_full_run_offline_preserves_files(self):
        i,s,c=reports()
        repo=Path(self.t.name)/'repo';repo.mkdir()
        root=Path(self.t.name)/'MarketMood_Private'/'phase_a_m10';root.mkdir()
        path1=root/'m10c_integrated_post_nov13_20261114_A.json'
        path2=root/'m10c_settlement_post_nov13_20261114_A.json'
        path3=repo/'data/processed/es_arena_research_v0_1/arena_official_settlements.csv';path3.parent.mkdir(parents=True)
        data1=json.dumps(i).encode();data2=json.dumps(s).encode();data3=c.encode()
        path1.write_bytes(data1);path2.write_bytes(data2);path3.write_bytes(data3)
        pins=dict(m.PIN)
        pins['integrated']=m.digest(data1);pins['csv']=m.digest(data3);pins['settlement']=m.digest(data2)
        s['sources']['m10c_integrated_physical_sha256']=pins['integrated']
        s['sources']['local_official_settlements_csv_physical_sha256']=pins['csv']
        data2=json.dumps(s).encode();path2.write_bytes(data2);pins['settlement']=m.digest(data2)
        before=[p.read_bytes() for p in (path1,path2,path3)]
        with patch.object(m,'PIN',pins):
            r=m.run(repo,root)
        self.assertEqual(r['historical']['review_flags'],[])
        self.assertEqual([p.read_bytes() for p in (path1,path2,path3)],before)
        self.assertEqual(r['candidate_protocol_status'],'DRAFT_NOT_ADOPTED')
        self.assertFalse(r['no_files_written'] is False)

if __name__=='__main__':unittest.main()
