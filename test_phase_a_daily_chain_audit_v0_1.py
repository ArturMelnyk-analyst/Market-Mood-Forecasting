"""Offline synthetic fixtures for the read-only daily chain verifier."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from phase_a_daily_chain_audit_v0_1 import audit

CID='synthetic-challenge-id'
DATE='2026-11-13'

class DailyChainAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='es_daily_chain_')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base=self.root/'data/processed/es_arena_live'
        self.forecast_path=self.base/'forecasts'/f'forecast_20261113_{CID}.json'
        self.attempt=self.base/'receipts'/f'attempt_{CID}.json'
        self.receipt=self.base/'receipts'/f'receipt_{CID}.json'
        self.discovery=self.base/'discovery'/'arena_discovery_synthetic.json'
        for x in (self.forecast_path,self.attempt,self.receipt,self.discovery):x.parent.mkdir(parents=True,exist_ok=True)
        self.source=self.root/'data/processed/es_arena_nb11_v0_1/snapshots/es_sessions_fake.csv'
        self.source.parent.mkdir(parents=True,exist_ok=True)
        self.source.write_text('session_date,close\n2026-11-12,7600\n')
        self.manifest=self.source.with_suffix('.manifest.json')
        self.manifest.write_text(json.dumps({'snapshot_sha256':self.sha(self.source)}))
        self.model=self.root/'data/processed/es_arena_nb10_v0_1/deployment_fake/es_daily_lr_deployment_v0_1.joblib'
        self.model.parent.mkdir(parents=True,exist_ok=True)
        self.model.write_bytes(b'synthetic model, do not load')
        self.forecast={
            'challenge_id':CID,'target_session_date':DATE,'last_eligible_session':'2026-11-12',
            'contract_symbol':'ESZ6','deadline_utc':'2026-11-13T15:00:00+00:00',
            'prepared_at_utc':'2026-11-13T10:00:00+00:00',
            'probabilities':{'bearish':.3,'neutral':.3,'bullish':.4},
            'status':'PREVIEW_NOT_SUBMITTED',
            'source_csv':str(self.source.relative_to(self.root)),
            'source_sha256':self.sha(self.source),
            'source_manifest_sha256':self.sha(self.manifest),
            'model_dir':str(self.model.parent.relative_to(self.root)),
            'model_sha256':self.sha(self.model),
        }
        self.save(self.forecast_path,self.forecast)
        h=self.sha(self.forecast_path)
        self.save(self.attempt,{'challenge_id':CID,'forecast_sha256':h,'attempt_started_at_utc':'2026-11-13T10:02:00+00:00'})
        self.save(self.receipt,{'http_status':201,'forecast_sha256':h,
            'response':{'challenge_id':CID,'prediction_id':'synthetic-only','counts_for_score':True,
                        'probabilities':self.forecast['probabilities'], 'created_at':'2026-11-13 10:03:00'}})
        self.save(self.discovery,{'status':'success','http_status':200,
            'record_completed_at_utc':'2026-11-13T09:12:12+00:00',
            'server_date_header':'Fri, 13 Nov 2026 09:12:11 GMT',
            'response_json':{'challenges':[{'id':CID,'asset':'ES','challenge_type':'daily',
                  'contract_symbol':'ESZ6','deadline':'2026-11-13T15:00:00Z'}]}})
    @staticmethod
    def save(p,v):p.write_text(json.dumps(v),encoding='utf-8')
    @staticmethod
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def check(self):return audit(self.root,CID,DATE,expected_model_sha=self.sha(self.model))
    def entries(self):return {a['check']:a['level'] for a in self.check()['checks']}
    def test_valid_synthetic_chain_has_no_failures(self):
        r=self.check();self.assertEqual(r['failure_count'],0);self.assertEqual(r['incomplete_count'],0)
        self.assertEqual(r['summary'],'LOCALLY_CONSISTENT_WITH_LIMITATIONS')
    def test_source_tamper_detected(self):
        self.source.write_bytes(self.source.read_bytes()+b'tamper')
        self.assertEqual(self.entries()['source_csv_hash'],'FAIL')
    def test_forecast_tamper_detected(self):
        self.forecast['probabilities']={'bearish':.2,'neutral':.2,'bullish':.6}
        self.save(self.forecast_path,self.forecast)
        self.assertEqual(self.entries()['receipt_forecast_hash'],'FAIL')
    def test_missing_receipt_never_inferred_accepted(self):
        self.receipt.unlink()
        self.assertEqual(self.entries()['receipt_file'],'INCOMPLETE')
    def test_missing_discovery_is_not_no_opportunity(self):
        self.discovery.unlink()
        self.assertEqual(self.entries()['challenge_discovery'],'INCOMPLETE')
    def test_afterdeadline_discovery_is_not_predeadline(self):
        v=json.loads(self.discovery.read_text());v['server_date_header']='Fri, 13 Nov 2026 16:12:11 GMT';self.save(self.discovery,v)
        self.assertEqual(self.entries()['discovery_predeadline'],'WARN')
    def test_missing_timezone_server_created_at_is_warning(self):
        self.assertEqual(self.entries()['server_acceptance_timestamp'],'WARN')
    def test_discovery_deadline_mismatch_detected(self):
        v=json.loads(self.discovery.read_text());v['response_json']['challenges'][0]['deadline']='2026-11-13T16:00:00Z';self.save(self.discovery,v)
        self.assertEqual(self.entries()['discovery_metadata_match'],'FAIL')
    def test_wrong_server_probabilities_detected(self):
        r=json.loads(self.receipt.read_text());r['response']['probabilities']={'bearish':.4,'neutral':.3,'bullish':.3};self.save(self.receipt,r)
        self.assertEqual(self.entries()['server_probabilities'],'FAIL')

if __name__=='__main__':unittest.main(verbosity=2)
