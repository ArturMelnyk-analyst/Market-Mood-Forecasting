import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from phase_a_m10_public_opportunity_audit_v0_1 import AuditError, summarize

HERE=Path(__file__).resolve().parent

def fixture():
    return {'api_contract_version':'financial-opportunity-audit-v1','site':'arena','asset':'ES',
      'time_basis':'retained_deadline_at_query_snapshot',
      'window':{'start':'2026-10-20T00:00:00Z','end':'2026-11-14T00:00:00Z','bounds':'[start,end)'},
      'as_of':'2026-11-14T00:00:00Z','snapshot_at':'2026-11-14T11:34:49Z',
      'entries':[{'asset':'ES','challenge_id':'mock-uuid-1','deadline':'2026-11-13T15:00:00Z',
       'status':'resolved','state_at_as_of':'unknown','reconstruction':'retained_challenge_today',
       'submission_eligibility_at_as_of':'unknown','public_open_before_as_of':{'status':'unknown'},
       'reader_delivery_before_as_of':{'status':'unknown'},
       'cancellation':{'status':None},'evidence':[]},],
      'coverage':{'complete':False,'status':'partial','semantics':'point_observations_not_exhaustive_lifecycle_coverage',
       'gaps':[{'start':'2026-10-20T00:00:00Z','end':'2026-11-13T22:58:14Z'}]},
      'next_cursor':None}

def run_report(x):
    return summarize(json.dumps(x).encode())

class TestM10B(unittest.TestCase):
    def test_base(self):
        r=run_report(fixture())
        self.assertEqual(r['record_count'],1)
        self.assertEqual(r['counts']['submission_eligibility'],{'unknown':1})
        self.assertEqual(r['counts']['state_at_as_of'],{'unknown':1})
        self.assertFalse(r['coverage']['complete'])
    def test_source_hash(self):
        b=json.dumps(fixture()).encode()
        self.assertEqual(summarize(b,hashlib.sha256(b).hexdigest())['source_bytes'],len(b))
    def test_wrong_hash(self):
        with self.assertRaisesRegex(AuditError,'SHA-256 mismatch'): summarize(b'{}','0'*64)
    def test_invalid_json(self):
        with self.assertRaises(AuditError): summarize(b'{')
    def test_unsupported_contract(self):
        d=fixture();d['api_contract_version']='v2'
        with self.assertRaisesRegex(AuditError,'unsupported'):run_report(d)
    def test_duplicate(self):
        d=fixture();d['entries'].append(copy.deepcopy(d['entries'][0]))
        with self.assertRaisesRegex(AuditError,'duplicate'):run_report(d)
    def test_utc_required(self):
        d=fixture();d['entries'][0]['deadline']='2026-11-13T15:00:00'
        with self.assertRaisesRegex(AuditError,'timezone'):run_report(d)
    def test_offset_datetime_accepted(self):
        d=fixture();d['entries'][0]['deadline']='2026-11-13T16:00:00+01:00'
        self.assertIn('15:00:00Z',run_report(d)['challenge_records'][0]['deadline_utc'])
    def test_window_exclusive(self):
        d=fixture();d['entries'][0]['deadline']=d['window']['end']
        with self.assertRaisesRegex(AuditError,'outside'):run_report(d)
    def test_asset_mismatch(self):
        d=fixture();d['entries'][0]['asset']='NQ'
        with self.assertRaisesRegex(AuditError,'asset mismatch'):run_report(d)
    def test_missing_eligibility(self):
        d=fixture();del d['entries'][0]['submission_eligibility_at_as_of']
        with self.assertRaisesRegex(AuditError,'expected string'):run_report(d)
    def test_cursor_preserved(self):
        d=fixture();d['next_cursor']='dummy'
        self.assertFalse(run_report(d)['pagination_complete'])
    def test_bad_gap(self):
        d=fixture();d['coverage']['gaps'][0]['end']='2026-11-16T00:00:00Z'
        with self.assertRaisesRegex(AuditError,'outside'):run_report(d)
    def test_current_resolved_not_asof_certified(self):
        r=run_report(fixture())
        self.assertEqual(r['challenge_records'][0]['round_status_at_snapshot'],'resolved')
        self.assertEqual(r['challenge_records'][0]['state_at_as_of'],'unknown')
    def test_snapshot_later_not_backdated(self):
        self.assertTrue(run_report(fixture())['snapshot_is_after_as_of'])
    def test_evidence_count_not_eligibility(self):
        d=fixture();d['entries'][0]['evidence']=[{'some':'unknown item'}]
        r=run_report(d)
        self.assertEqual(r['total_evidence_items'],1)
        self.assertEqual(r['counts']['submission_eligibility'],{'unknown':1})
    def test_no_write_default(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source.json';p.write_text(json.dumps(fixture()))
            b=p.read_bytes(); h=hashlib.sha256(b).hexdigest()
            c=subprocess.run([sys.executable,str(HERE/'phase_a_m10_public_opportunity_audit_v0_1.py'),'--input',str(p),'--expected-sha256',h],cwd=td,capture_output=True,text=True)
            self.assertEqual(c.returncode,0,c.stderr)
            self.assertEqual(sorted(x.name for x in Path(td).iterdir()),['source.json'])
            self.assertEqual(p.read_bytes(),b)
    def test_exclusive_write(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source.json';p.write_text(json.dumps(fixture()))
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            dest=Path(td)/'report.json'
            cmd=[sys.executable,str(HERE/'phase_a_m10_public_opportunity_audit_v0_1.py'),'--input',str(p),'--expected-sha256',h,'--private-report',str(dest)]
            a=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr)
            previous=dest.read_bytes()
            b=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(b.returncode,2)
            self.assertEqual(dest.read_bytes(),previous)

if __name__=='__main__':unittest.main()
