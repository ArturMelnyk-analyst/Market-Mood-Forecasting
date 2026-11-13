"""Offline synthetic tests; no Arena authentication, network, forecasts or submissions."""
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phase_a_commitment_lifecycle_v0_1 import (
    freeze, verify, reveal, verify_reveal, verify_reveal_object,
    hash_payload, load_json, _plan, PLAN_SCHEMA, SETTLEMENT_SCHEMA,
)

BASE = datetime(2032, 6, 1, 10, 0, tzinfo=timezone.utc)
CID = 'b6baec39-37b4-4e49-bb08-aae31ccbfb64'


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private_dir = self.root / 'private'
        self.public_dir = self.root / 'public'
        self.private_dir.mkdir()
        self.public_dir.mkdir()
        self.forecast = self.private_dir / 'm1_forecast.json'
        self.plan = self.private_dir / 'plan.json'
        self.priv = self.private_dir / 'private.json'
        self.pub = self.public_dir / 'public.json'
        self.rev = self.private_dir / 'reveal.json'
        self.official = self.private_dir / 'official_snapshot.html'
        self.official.write_text('<html>synthetic, NOT real Arena</html>', encoding='utf-8')
        self.settle = self.private_dir / 'reviewed_settlement.json'
        self.deadline = BASE + timedelta(hours=5)
        self.resolution = BASE + timedelta(hours=7)
        self.f = {'challenge_id':CID, 'target_session_date':'2032-06-01',
                  'model_sha256':'a'*64, 'probabilities':{'bearish':0.2345678912345,
                  'neutral':0.3456789123456,'bullish':0.4197531964199},
                  'source_snapshot_sha256':'b'*64}
        self.forecast.write_text(json.dumps(self.f),encoding='utf-8')
        self.data = {'schema_version':PLAN_SCHEMA,'challenge_id':CID,
                     'target_session_date':'2032-06-01','contract':'ESM32',
                     'deadline_utc': self.deadline.isoformat(),
                     'expected_models':['M1','M2'],
                     'entries':[{'model_id':'M1','status':'forecast','forecast_path':str(self.forecast)},
                                {'model_id':'M2','status':'failure','failure_code':'DATA_UNAVAILABLE',
                                 'observed_at_utc':BASE.isoformat(),
                                 'failure_description':'synthetic provider outage'}]}
        self._write_plan()
        self.s = {'schema_version':SETTLEMENT_SCHEMA,'challenge_id':CID,
                  'outcome':'neutral','resolved_at_utc':self.resolution.isoformat(),
                  'official_reference':'synthetic://not-official',
                  'official_source_sha256':hashlib.sha256(self.official.read_bytes()).hexdigest()}
        self._write_settlement()

    def _write_plan(self): self.plan.write_text(json.dumps(self.data),encoding='utf-8')
    def _write_settlement(self): self.settle.write_text(json.dumps(self.s),encoding='utf-8')
    def _freeze(self): return freeze(self.plan,self.priv,self.pub,now=BASE)
    def _reveal(self): return reveal(self.priv,self.pub,self.settle,self.official,self.rev,now=BASE+timedelta(hours=8))

    def test_end_to_end_forecast_and_failure(self):
        pub = self._freeze()
        self.assertEqual(verify(self.priv,self.pub),pub)
        r = self._reveal()
        self.assertEqual(verify_reveal(self.pub,self.rev,source_path=self.official),pub['commitment_sha256'])
        entries = r['private_payload']['manifest']['entries']
        self.assertEqual([x['status'] for x in entries],['forecast','failure'])
        self.assertEqual(entries[0]['forecast']['probabilities'],self.f['probabilities'])
        self.assertNotIn('probabilities',pub)
        self.assertNotIn('nonce_hex',pub)
        self.assertNotIn('forecast_path',str(r))

    def test_freeze_rejects_after_cutoff(self):
        with self.assertRaisesRegex(ValueError,'Freeze cutoff'):
            freeze(self.plan,self.priv,self.pub,now=self.deadline-timedelta(minutes=4))
        self.assertFalse(self.priv.exists()); self.assertFalse(self.pub.exists())

    def test_freeze_exact_cutoff_rejected(self):
        with self.assertRaises(ValueError):
            freeze(self.plan,self.priv,self.pub,now=self.deadline-timedelta(minutes=5))

    def test_no_duplicate_freeze(self):
        self._freeze()
        with self.assertRaises(FileExistsError): self._freeze()

    def test_missing_model_declared(self):
        self.data['entries'].pop(); self._write_plan()
        with self.assertRaisesRegex(ValueError,'Exactly one'): self._freeze()

    def test_extra_model_declared(self):
        self.data['entries'].append(dict(self.data['entries'][1],model_id='M3'))
        self._write_plan()
        with self.assertRaises(ValueError): self._freeze()

    def test_duplicate_model_declared(self):
        self.data['expected_models']=['M1','M1']; self._write_plan()
        with self.assertRaises(ValueError): self._freeze()

    def test_unknown_status_rejected(self):
        self.data['entries'][1]['status']='not sure'; self._write_plan()
        with self.assertRaises(ValueError): self._freeze()

    def test_invalid_probability_rejected(self):
        self.f['probabilities']['bullish']=0.7
        self.forecast.write_text(json.dumps(self.f),encoding='utf-8')
        with self.assertRaises(ValueError): self._freeze()

    def test_wrong_challenge_rejected(self):
        self.f['challenge_id']='ffffffff-ffff-4fff-8fff-ffffffffffff'
        self.forecast.write_text(json.dumps(self.f),encoding='utf-8')
        with self.assertRaises(ValueError): self._freeze()

    def test_future_failure_timestamp_rejected(self):
        self.data['entries'][1]['observed_at_utc']=(BASE+timedelta(seconds=1)).isoformat()
        self._write_plan()
        with self.assertRaises(ValueError): self._freeze()

    def test_private_inside_git_rejected(self):
        (self.root/'.git').mkdir()
        with self.assertRaisesRegex(ValueError,'OUTSIDE'): self._freeze()

    def test_plan_in_git_rejected(self):
        (self.private_dir/'.git').mkdir()
        with self.assertRaises(ValueError): self._freeze()

    def test_public_tampering_detected(self):
        self._freeze(); p=load_json(self.pub); p['model_count']=7
        self.pub.write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaises(ValueError): verify(self.priv,self.pub)

    def test_private_nonce_tampering_detected(self):
        self._freeze(); p=load_json(self.priv);p['nonce_hex']='0'*64
        self.priv.write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaises(ValueError): verify(self.priv,self.pub)

    def test_public_field_leak_rejected(self):
        self._freeze(); p=load_json(self.pub);p['nonce_hex']='bad'
        self.pub.write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaises(ValueError): verify(self.priv,self.pub)

    def test_bad_settlement_source_hash_rejected(self):
        self._freeze(); self.s['official_source_sha256']='f'*64; self._write_settlement()
        with self.assertRaisesRegex(ValueError,'source file hash mismatch'): self._reveal()
        self.assertFalse(self.rev.exists())

    def test_wrong_settlement_id_rejected(self):
        self._freeze();self.s['challenge_id']='ffffffff-ffff-4fff-8fff-ffffffffffff';self._write_settlement()
        with self.assertRaisesRegex(ValueError,'schema/challenge'): self._reveal()

    def test_early_reveal_rejected(self):
        self._freeze()
        with self.assertRaises(ValueError):
            reveal(self.priv,self.pub,self.settle,self.official,self.rev,now=BASE+timedelta(hours=6))
        self.assertFalse(self.rev.exists())

    def test_no_overwrite_reveal(self):
        self._freeze();self._reveal()
        with self.assertRaises(FileExistsError):self._reveal()

    def test_reveal_modified_nonce_detected(self):
        self._freeze();self._reveal();r=load_json(self.rev)
        r['private_payload']['nonce_hex']='c'*64
        self.rev.write_text(json.dumps(r),encoding='utf-8')
        with self.assertRaises(ValueError): verify_reveal(self.pub,self.rev)

    def test_reveal_changed_probabilities_detected(self):
        self._freeze();self._reveal();r=load_json(self.rev)
        r['private_payload']['manifest']['entries'][0]['forecast']['probabilities']={'bearish':0.1,'neutral':0.2,'bullish':0.7}
        self.rev.write_text(json.dumps(r),encoding='utf-8')
        with self.assertRaises(ValueError): verify_reveal(self.pub,self.rev)

    def test_reveal_wrong_public_detected(self):
        self._freeze();self._reveal(); pub=load_json(self.pub);pub['commitment_sha256']='0'*64
        self.pub.write_text(json.dumps(pub),encoding='utf-8')
        with self.assertRaises(ValueError): verify_reveal(self.pub,self.rev)

    def test_only_intended_files_written(self):
        initial=set(self.root.rglob('*'))
        self._freeze(); self._reveal()
        additions={x.relative_to(self.root).as_posix() for x in set(self.root.rglob('*'))-initial}
        self.assertEqual(additions, {'private/private.json','public/public.json','private/reveal.json'})

    def test_no_legacy_schema_rewrite(self):
        self._freeze();q=load_json(self.pub)
        self.assertNotEqual(q['schema_version'],'es_arena_commitment_v0_1')
        self.assertEqual(len(q['commitment_sha256']),64)


if __name__ == '__main__': unittest.main()
