import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import m10a_batch_readbacks_v0_1 as m

ROOT = Path(__file__).resolve().parent
M10A = ROOT / 'phase_a_m10_revision_timing_audit_v0_1.py'


def cid(n):
    return str(uuid.UUID(int=n))


def audit_three():
    return {'api_contract_version': m.CONTRACT, 'asset': 'ES', 'next_cursor': None,
            'coverage': {'complete': False, 'status': 'partial'},
            'window': {'start': '2026-11-13T00:00:00Z', 'end': '2026-11-14T00:00:00Z'},
            'as_of': '2026-11-14T00:00:00Z', 'snapshot_at': '2026-11-14T01:00:00Z',
            'entries': [{'challenge_id': cid(i), 'asset': 'ES', 'deadline': '2026-11-13T15:00:00Z',
                         'public_open_before_as_of': {'status': 'unknown'},
                         'reader_delivery_before_as_of': {'status': 'unknown'},
                         'submission_eligibility_at_as_of': 'unknown'} for i in (1, 2, 3)]}


def body_for(n, submitted='2026-11-13T10:50:12.574030Z'):
    return json.dumps({'predictions': [{'challenge_id': cid(n), 'prediction_id': 'pred1',
                                       'revision_number': 1, 'submitted_at': submitted,
                                       'prediction': {'probabilities': {'bearish': 0.3, 'neutral': 0.3, 'bullish': 0.4}}}]}).encode('utf-8')


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.private = Path(self.tmp.name) / 'MarketMood_Private'
        self.private.mkdir()
        self.output = self.private / 'batch'
        self.source = self.private / 'audit.json'
        self.audit = audit_three()
        self.save_audit()

    def save_audit(self):
        raw = json.dumps(self.audit).encode('utf-8')
        self.source.write_bytes(raw)
        self.digest = m.sha256(raw)

    def request(self, status=200):
        calls = []
        def get(url, **kwargs):
            self.assertEqual(kwargs['allow_redirects'], False)
            self.assertEqual(kwargs['headers'], {'Authorization': 'private-test-credential'})
            self.assertEqual(kwargs['timeout'], (5, 20))
            self.assertTrue(url.startswith('https://headlinearena.com/api/v1/eval/challenges/'))
            which = int(uuid.UUID(url.split('/')[-2]))
            calls.append(which)
            return SimpleNamespace(status_code=status,
                                   headers={'Content-Type': 'application/json'},
                                   content=body_for(which) if status==200 else b'{"detail":"not available"}')
        return get, calls

    def test_pin_audit(self):
        d, ids = m.read_audit(self.source, self.digest, 3)
        self.assertEqual(len(ids), 3)
        self.assertFalse(d['coverage']['complete'])

    def test_hash_mismatch(self):
        with self.assertRaisesRegex(m.CaptureError, 'SHA-256 mismatch'):
            m.read_audit(self.source, 'f'*64, 3)

    def test_asset_mismatch(self):
        self.audit['asset'] = 'SPX'
        self.save_audit()
        with self.assertRaises(m.CaptureError):
            m.read_audit(self.source, self.digest, 3)

    def test_pagination_guard(self):
        self.audit['next_cursor'] = 'page2'
        self.save_audit()
        with self.assertRaisesRegex(m.CaptureError, 'pagination'):
            m.read_audit(self.source, self.digest, 3)

    def test_duplicate_ids(self):
        self.audit['entries'][1]['challenge_id'] = cid(1)
        self.save_audit()
        with self.assertRaisesRegex(m.CaptureError, 'duplicate'):
            m.read_audit(self.source, self.digest, 3)

    def test_wrong_count(self):
        with self.assertRaisesRegex(m.CaptureError, 'count'):
            m.read_audit(self.source, self.digest, 19)

    def test_exclusion_unknown_rejected(self):
        with self.assertRaisesRegex(m.CaptureError, 'excluded'):
            m.target_ids([cid(1), cid(2)], [cid(99)])

    def test_private_path_guard(self):
        with self.assertRaisesRegex(m.CaptureError, 'MarketMood_Private'):
            m.require_private(Path(self.tmp.name) / 'outside.json')

    def test_plan_no_network_or_file_writes(self):
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc=m.main(['--mode','plan','--audit',str(self.source),'--expected-audit-sha256',self.digest,
                       '--out',str(self.output),'--exclude',cid(3),'--expected-count','3'])
        self.assertEqual(rc,0)
        self.assertFalse(self.output.exists())
        self.assertIn('TARGET_CHALLENGES: 2',buf.getvalue())
        self.assertIn('NETWORK_REQUESTS: 0',buf.getvalue())

    def test_fetch_pilot_then_resume_no_rewrites(self):
        get, calls=self.request()
        ids=[cid(1),cid(2),cid(3)]
        with contextlib.redirect_stdout(io.StringIO()):
            rc=m.fetch_batch(self.audit,ids,self.output,ROOT,requester=get,
                             headers={'Authorization':'private-test-credential'},max_requests=2,sleep_seconds=0)
        self.assertEqual(rc,0)
        self.assertEqual(len(calls),2)
        before=(self.output / cid(1) / 'body.json').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            rc=m.fetch_batch(self.audit,ids,self.output,ROOT,requester=get,
                             headers={'Authorization':'private-test-credential'},resume=True,max_requests=18,sleep_seconds=0)
        self.assertEqual(rc,0)
        self.assertEqual(len(calls),3)
        self.assertEqual((self.output / cid(1) / 'body.json').read_bytes(),before)

    def test_existing_directory_requires_resume(self):
        self.output.mkdir()
        get,_=self.request()
        with self.assertRaisesRegex(m.CaptureError, 'already exists'):
            m.fetch_batch(self.audit,[cid(1)],self.output,ROOT,requester=get,
                          headers={'Authorization': 'private-test-credential'},sleep_seconds=0)

    def test_tamper_prevents_resume(self):
        get,_=self.request()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fetch_batch(self.audit,[cid(1)],self.output,ROOT,requester=get,headers={'Authorization': 'private-test-credential'},sleep_seconds=0)
        (self.output/cid(1)/'body.json').write_text('tampered',encoding='utf-8')
        with self.assertRaisesRegex(m.CaptureError, 'hash/metadata mismatch'):
            m.fetch_batch(self.audit,[cid(1)],self.output,ROOT,resume=True,requester=get,headers={'Authorization': 'private-test-credential'},sleep_seconds=0)

    def test_incomplete_capture_guard(self):
        folder=self.output/cid(1)
        folder.mkdir(parents=True)
        (folder/'body.json').write_text('a',encoding='utf-8')
        with self.assertRaisesRegex(m.CaptureError, 'partial capture'):
            m.validate_existing(self.output,cid(1))

    def test_non200_preserved_and_stops(self):
        get,calls=self.request(401)
        with contextlib.redirect_stdout(io.StringIO()):
            rc=m.fetch_batch(self.audit,[cid(1),cid(2)],self.output,ROOT,
                             requester=get,headers={'Authorization': 'private-test-credential'},max_requests=2,sleep_seconds=0)
        self.assertEqual(rc,2)
        self.assertEqual(len(calls),1)
        meta=m.validate_existing(self.output,cid(1))
        self.assertEqual(meta['http_status'],401)
        self.assertEqual((self.output/cid(1)/'body.json').read_bytes(),b'{"detail":"not available"}')

    def test_network_exception_cannot_erase_old_capture(self):
        get,_=self.request()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fetch_batch(self.audit,[cid(1)],self.output,ROOT,requester=get,headers={'Authorization': 'private-test-credential'},sleep_seconds=0)
        def fail(*args,**kwargs):
            raise RuntimeError('secret should not print')
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc=m.fetch_batch(self.audit,[cid(1),cid(2)],self.output,ROOT,resume=True,
                             requester=fail,headers={'Authorization': 'private-test-credential'},sleep_seconds=0)
        self.assertEqual(rc,2)
        self.assertNotIn('secret',buf.getvalue())
        self.assertEqual(len(list(self.output.iterdir())),1)

    def test_empty_readback_not_claimed(self):
        shape,n=m.readback_shape(b'{"predictions":[]}',cid(1))
        self.assertEqual((shape,n),('NO_REVISIONS',0))

    def test_challenge_mismatch_readback(self):
        shape,n=m.readback_shape(body_for(2),cid(1))
        self.assertEqual(shape,'CHALLENGE_MISMATCH')

    def test_offline_summary_with_real_m10a_module(self):
        self.assertTrue(M10A.exists())
        get,_=self.request()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fetch_batch(self.audit,[cid(1),cid(2)],self.output,ROOT,requester=get,headers={'Authorization': 'private-test-credential'},
                          max_requests=2,sleep_seconds=0)
            result=m.summarize(self.audit,self.digest,[cid(1),cid(2)],self.output,M10A)
        self.assertEqual(result['capture_states']['VALIDATED_READBACK'],2)
        self.assertEqual(result['revision_timing_counts']['SERVER_TIME_PREDEADLINE'],2)
        self.assertFalse(result['model_execution_certified'])
        self.assertFalse(result['scored_revision_certified'])

    def test_summary_missing_not_written(self):
        if not M10A.exists(): self.skipTest('M10A module absent')
        with self.assertRaisesRegex(m.CaptureError, 'missing captures'):
            with contextlib.redirect_stdout(io.StringIO()):
                m.summarize(self.audit,self.digest,[cid(1)],self.output,M10A,self.private/'report.json')
        self.assertFalse((self.private/'report.json').exists())

    def test_summary_hash_matches_actual_saved_bytes(self):
        if not M10A.exists(): self.skipTest('M10A module absent')
        get, _ = self.request()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fetch_batch(self.audit,[cid(1)],self.output,ROOT,requester=get,
                          headers={'Authorization':'private-test-credential'},sleep_seconds=0)
        report = self.private/'report_bytes.json'
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            m.summarize(self.audit,self.digest,[cid(1)],self.output,M10A,report)
        actual = hashlib.sha256(report.read_bytes()).hexdigest()
        self.assertIn('PRIVATE_SUMMARY_SHA256: '+actual, output.getvalue())
        self.assertIn(b'\n', report.read_bytes())
        self.assertNotIn(b'\r\n', report.read_bytes())
        self.assertEqual(json.loads(report.read_bytes())['target_challenges'],1)

    def test_summary_report_exclusive(self):
        if not M10A.exists(): self.skipTest('M10A module absent')
        get,_=self.request()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fetch_batch(self.audit,[cid(1)],self.output,ROOT,requester=get,headers={'Authorization': 'private-test-credential'},sleep_seconds=0)
            m.summarize(self.audit,self.digest,[cid(1)],self.output,M10A,self.private/'report.json')
        self.assertTrue((self.private/'report.json').is_file())
        with self.assertRaisesRegex(m.CaptureError, 'already exists'):
            m.summarize(self.audit,self.digest,[cid(1)],self.output,M10A,self.private/'report.json')


if __name__=='__main__':
    unittest.main()
