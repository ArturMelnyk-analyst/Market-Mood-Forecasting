import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import phase_a_m10_revision_timing_audit_v0_1 as m

CID = '9e7a3255-e4dc-4a1d-a8f9-a30f18c4fa2c'
READ = {'predictions': [
    {'challenge_id': CID, 'prediction_id': 'pred-1', 'revision_number': 1,
     'created_at': '2026-11-13T10:10:00Z',
     'submitted_at': '2026-11-13T10:50:12.574030Z',
     'probability_source': 'stored_native_ternary',
     'prediction': {'probabilities': {'bearish': 0.3, 'neutral': 0.2, 'bullish': 0.5}}}
]}
AUDIT = {'api_contract_version': 'financial-opportunity-audit-v1',
         'asset': 'ES', 'as_of': '2026-11-14T00:00:00Z', 'snapshot_at': '2026-11-14T11:34:49Z',
         'window': {'start': '2026-10-20T00:00:00Z', 'end': '2026-11-14T00:00:00Z'},
         'coverage': {'complete': False, 'status': 'partial', 'gaps': []},
         'next_cursor': None,
         'entries': [{'challenge_id': CID, 'asset': 'ES', 'deadline': '2026-11-13T15:00:00Z',
                      'public_open_before_as_of': {'status': 'unknown'},
                      'reader_delivery_before_as_of': {'status': 'unknown'},
                      'submission_eligibility_at_as_of': 'unknown',
                      'state_at_as_of': 'unknown'}]}


def clone(o):
    return json.loads(json.dumps(o))


class TestM10A(unittest.TestCase):
    def build(self, read=None, audit=None):
        return m.build_report(clone(READ if read is None else read), clone(AUDIT if audit is None else audit), 'r'*64, 'a'*64, CID)

    def test_valid_one_revision(self):
        r=self.build(); self.assertEqual(r['records'][0]['timing_status'],'SERVER_TIME_PREDEADLINE')
        self.assertEqual(r['records'][0]['margin_seconds'],14987.42597)

    def test_safety_margin(self):
        self.assertEqual(self.build()['records'][0]['five_minute_safety_status'],'BEFORE_FIVE_MINUTE_CUTOFF')

    def test_at_deadline(self):
        d=clone(READ); d['predictions'][0]['submitted_at']='2026-11-13T15:00:00Z'
        r=self.build(d)['records'][0]
        self.assertEqual(r['timing_status'],'SERVER_TIME_AT_OR_AFTER_DEADLINE')

    def test_at_safety_cutoff(self):
        d=clone(READ); d['predictions'][0]['submitted_at']='2026-11-13T14:55:00Z'
        r=self.build(d)['records'][0]
        self.assertEqual(r['five_minute_safety_status'],'AT_OR_AFTER_FIVE_MINUTE_CUTOFF')
        self.assertEqual(r['timing_status'],'SERVER_TIME_PREDEADLINE')

    def test_null_time_unknown(self):
        d=clone(READ); d['predictions'][0]['submitted_at']=None
        self.assertEqual(self.build(d)['records'][0]['timing_status'],'SUBMISSION_TIME_UNKNOWN')

    def test_no_created_at_fallback(self):
        d=clone(READ); d['predictions'][0].pop('submitted_at')
        self.assertEqual(self.build(d)['records'][0]['timing_status'],'SUBMISSION_TIME_UNKNOWN')

    def test_naive_time_not_certified(self):
        d=clone(READ); d['predictions'][0]['submitted_at']='2026-11-13T10:50:00'
        self.assertEqual(self.build(d)['records'][0]['timing_status'],'SUBMISSION_TIME_INVALID')

    def test_offset_time_normalized(self):
        d=clone(READ); d['predictions'][0]['submitted_at']='2026-11-13T11:50:12.574030+01:00'
        self.assertEqual(self.build(d)['records'][0]['margin_seconds'],14987.42597)

    def test_bad_time_not_certified(self):
        d=clone(READ); d['predictions'][0]['submitted_at']='yesterday'
        self.assertEqual(self.build(d)['records'][0]['timing_status'],'SUBMISSION_TIME_INVALID')

    def test_wrong_challenge(self):
        d=clone(READ); d['predictions'][0]['challenge_id']='OTHER'
        with self.assertRaises(m.AuditError): self.build(d)

    def test_duplicate_revision(self):
        d=clone(READ); d['predictions'].append(clone(d['predictions'][0]))
        with self.assertRaises(m.AuditError): self.build(d)

    def test_multiple_revisions_sorted(self):
        d=clone(READ); row=clone(d['predictions'][0]); row['revision_number']=2; row['submitted_at']=None; d['predictions'].insert(0,row)
        r=self.build(d); self.assertEqual([x['revision_number'] for x in r['records']],[1,2])

    def test_missing_challenge_audit(self):
        a=clone(AUDIT); a['entries']=[]
        with self.assertRaises(m.AuditError): self.build(audit=a)

    def test_incomplete_pagination(self):
        a=clone(AUDIT); a['next_cursor']='NEXT'
        with self.assertRaises(m.AuditError): self.build(audit=a)

    def test_audit_unknown_not_promoted(self):
        r=self.build(); self.assertFalse(r['historical_eligible_opportunity_census_certified'])
        self.assertEqual(r['account_eligibility_status_at_as_of'],'unknown')

    def test_source_unsupported(self):
        a=clone(AUDIT); a['api_contract_version']='not-allowed'
        with self.assertRaises(m.AuditError): self.build(audit=a)

    def test_audit_contract_window_exclusive(self):
        a=clone(AUDIT); a['entries'][0]['deadline']='2026-11-14T00:00:00Z'
        with self.assertRaises(m.AuditError): self.build(audit=a)

    def test_prohibit_percentage_overrun(self):
        d=clone(READ); d['predictions'][0]['prediction']['probabilities']['bearish']=-1
        with self.assertRaises(m.AuditError): self.build(d)

    def test_percentage_vector(self):
        d=clone(READ); d['predictions'][0]['prediction']['probabilities']={'bearish':30,'neutral':20,'bullish':50}
        self.assertEqual(self.build(d)['records'][0]['probability_representation'],'percentage')

    def test_no_m1_execution_claim(self):
        self.assertFalse(self.build()['model_execution_certified'])
        self.assertFalse(self.build()['scored_revision_certified'])

    def test_read_json_hash_and_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json'; raw=json.dumps(READ).encode(); p.write_bytes(raw)
            sha=hashlib.sha256(raw).hexdigest(); self.assertEqual(m.read_json(p,sha)[1],sha)
            with self.assertRaises(m.AuditError): m.read_json(p,'0'*64)

    def test_exclusive_report(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'report.json'; m.write_private_report(p,self.build()); self.assertTrue(p.exists())
            with self.assertRaises(FileExistsError): m.write_private_report(p,self.build())

    def test_private_report_outside_git(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'project'; p.mkdir(); (p/'.git').mkdir()
            with self.assertRaises(m.AuditError): m.write_private_report(p/'report.json',self.build())

    def test_cli_default_no_write(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)/'r.json'; a=Path(t)/'a.json'
            rr=json.dumps(READ).encode(); aa=json.dumps(AUDIT).encode()
            r.write_bytes(rr); a.write_bytes(aa)
            args=['--readback',str(r),'--audit',str(a),'--expected-readback-sha256',hashlib.sha256(rr).hexdigest(), '--expected-audit-sha256', hashlib.sha256(aa).hexdigest(), '--challenge-id',CID]
            self.assertEqual(m.main(args),0)
            self.assertEqual(len(list(Path(t).iterdir())),2)

if __name__ == '__main__': unittest.main()
