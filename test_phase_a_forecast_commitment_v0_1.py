"""Offline synthetic tests for a write-once prospective forecast commitment."""
import json
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from phase_a_forecast_commitment_v0_1 import create,verify

class CommitmentTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='commitment_synth_')
        self.addCleanup(t.cleanup)
        r=Path(t.name)
        self.f=r/'forecast.json';self.private=r/'private.json';self.public=r/'public.json'
        self.f.write_text(json.dumps({'challenge_id':'fake-id','target_session_date':'2026-11-13',
            'deadline_utc':'2026-11-13T15:00:00+00:00','model_sha256':'deadbeef',
            'probabilities':{'bearish':.21,'neutral':.32,'bullish':.47}}))
        self.now=datetime(2026,11,13,14,0,tzinfo=timezone.utc)
    def build(self):return create(self.f,self.private,self.public,now=self.now)
    def test_commitment_verifies(self):
        p=self.build();self.assertEqual(verify(self.f,self.private,self.public),p['commitment_sha256'])
    def test_refuses_duplicate_write(self):
        self.build()
        with self.assertRaises(FileExistsError):self.build()
    def test_refuses_cutoff(self):
        self.now=datetime(2026,11,13,14,55,tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError,'cutoff'):self.build()
        self.assertFalse(self.private.exists());self.assertFalse(self.public.exists())
    def test_tampered_forecast_cannot_verify(self):
        self.build();self.f.write_bytes(self.f.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'hash'):verify(self.f,self.private,self.public)
    def test_tampered_public_commitment_cannot_verify(self):
        self.build();p=json.loads(self.public.read_text());p['commitment_sha256']='0'*64;self.public.write_text(json.dumps(p))
        with self.assertRaisesRegex(ValueError,'commitment'):verify(self.f,self.private,self.public)
    def test_private_nonce_not_in_public(self):
        self.build()
        nonce=json.loads(self.private.read_text())['nonce_hex']
        self.assertNotIn(nonce,self.public.read_text())
    def test_refuses_private_public_same_path_without_writing(self):
        with self.assertRaisesRegex(ValueError,'distinct paths'):
            create(self.f,self.public,self.public,now=self.now)
        self.assertFalse(self.public.exists())

    def test_refuses_private_in_git_checkout(self):
        (self.f.parent / '.git').mkdir()
        with self.assertRaisesRegex(ValueError,'OUTSIDE every Git checkout'):
            self.build()
        self.assertFalse(self.private.exists())
        self.assertFalse(self.public.exists())

    def test_rejects_public_target_date_tampering(self):
        self.build()
        q=json.loads(self.public.read_text())
        q['target_session_date']='2026-11-14'
        self.public.write_text(json.dumps(q))
        with self.assertRaisesRegex(ValueError,'Target date mismatch'):
            verify(self.f,self.private,self.public)

    def test_refuses_public_nonce_leak_on_verify(self):
        self.build()
        q=json.loads(self.public.read_text())
        q['nonce_hex']=json.loads(self.private.read_text())['nonce_hex']
        self.public.write_text(json.dumps(q))
        with self.assertRaisesRegex(ValueError,'private forecast or nonce'):
            verify(self.f,self.private,self.public)

    def test_refuses_invalid_probabilities(self):
        f=json.loads(self.f.read_text());f['probabilities']['bullish']=.9;self.f.write_text(json.dumps(f))
        with self.assertRaisesRegex(ValueError,'probability'):self.build()

if __name__=='__main__':unittest.main(verbosity=2)
