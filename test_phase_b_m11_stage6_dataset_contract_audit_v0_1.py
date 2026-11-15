"""Synthetic M11 Stage 6 chronological-split and point-in-time contract tests."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import phase_b_m11_stage6_dataset_contract_audit_v0_1 as mod

ROOT = Path(__file__).resolve().parent
CONTRACT_FILE = ROOT / 'phase_b' / 'phase_b_dataset_split_contract_DRAFT_v0_1.json'
REGISTRY_FILE = ROOT / 'phase_b' / 'phase_b_seven_configuration_registry_DRAFT_v0_1.json'


def raw(path):
    return json.loads(path.read_text(encoding='utf-8'))


def z(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def make_manifest(model='M2', mode='PROSPECTIVE_CAPTURED'):
    t0 = datetime(2024, 1, 2, 13, tzinfo=timezone.utc)
    source_names = ['ES'] if model == mod.CONTROL else mod.EXPECTED_SOURCES[model]
    recs=[]
    for split, offsets in [('train',[0,3,6]),('validation',[9,12]),('test',[15,18])]:
        for delta in offsets:
            ts=t0+timedelta(days=delta)
            sources=[]
            for src in source_names:
                sources.append({
                    'source':src,
                    'observation_end_utc':z(ts-timedelta(hours=4)),
                    'source_publication_utc':z(ts-timedelta(hours=3)),
                    'source_available_utc':z(ts-timedelta(hours=2)),
                    'retrieved_at_utc':z(ts-timedelta(hours=1)) if mode=='PROSPECTIVE_CAPTURED' else z(ts+timedelta(days=60)),
                    'source_revision_or_vintage':'test-vintage-'+src,
                    'provenance_sha256':'a'*64,
                    'licensing_restriction':'PRIVATE_LICENSED',
                    'includes_target_date_settlement':False,
                    'asof_evidence_status': 'VERIFIED_PREDECISION' if mode=='PROSPECTIVE_CAPTURED'
                                            else 'LOCAL_HISTORICAL_VINTAGE_REVIEW_REQUIRED',
                })
            recs.append({
                'challenge_id':str(uuid.uuid5(uuid.NAMESPACE_OID, f'{model}-{delta}')),
                'decision_cutoff_utc':z(ts),
                'target_session':(ts.date()).isoformat(),
                'split':split,
                'label_available_at_utc':z(ts+timedelta(days=1)) if split!='test' else None,
                'input_evidence':sources,
            })
    return {
        'record_type':'M11_STAGE6_CHRONOLOGICAL_FOLD_MANIFEST_V0_1',
        'status':'CANDIDATE_NOT_FROZEN',
        'model_id':model,
        'contract_sha256':hashlib.sha256(CONTRACT_FILE.read_bytes()).hexdigest(),
        'evidence_mode':mode,
        'fit_at_utc':z(t0+timedelta(days=8)),
        'selection_at_utc':z(t0+timedelta(days=14)),
        'rows':recs,
    }


class Stage6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry=raw(REGISTRY_FILE)
        cls.contract=raw(CONTRACT_FILE)
        cls.contract_sha=mod.digest(CONTRACT_FILE)

    def validate(self, manifest):
        return mod.audit_candidate_folds(manifest, self.contract, self.registry, self.contract_sha)

    def reject(self, manifest, phrase=None):
        with self.assertRaises(ValueError) as exc:
            self.validate(manifest)
        if phrase:
            self.assertIn(phrase.lower(),str(exc.exception).lower())

    def test_contract_passes(self):
        self.assertEqual(mod.audit_contract(self.contract,self.registry)['status'],
                         'DRAFT_CONTRACT_VALIDATED_NOT_FROZEN')

    def test_registry_pin(self):
        self.assertEqual(mod.digest(REGISTRY_FILE),mod.REGISTRY_SHA)

    def test_contract_is_not_frozen(self):
        self.assertFalse(self.contract['phase_b_model_freeze_certified'])
        self.assertFalse(self.contract['fold_dates_or_sample_ids_frozen'])

    def test_all_six_candidate_models(self):
        for mid in ('M2','M3','M4','M5','M6','M7'):
            with self.subTest(mid=mid):
                r=self.validate(make_manifest(mid))
                self.assertEqual(r['fold_row_counts'],{'train':3,'validation':2,'test':2})
                self.assertFalse(r['phase_b_model_freeze_certified'])

    def test_matched_es_only_control(self):
        r=self.validate(make_manifest(mod.CONTROL))
        self.assertEqual(r['model_id'],mod.CONTROL)

    def test_no_frozen_m1_retraining(self):
        x=make_manifest();x['model_id']='M1';self.reject(x,'M1')

    def test_reconstruction_mode_allowed_but_not_certified(self):
        x=make_manifest(mode='HISTORICAL_RECONSTRUCTED')
        r=self.validate(x)
        self.assertFalse(r['historical_vintages_externally_certified'])

    def test_reconstruction_does_not_claim_independent_proof(self):
        x=make_manifest(mode='HISTORICAL_RECONSTRUCTED')
        x['rows'][0]['input_evidence'][0]['asof_evidence_status']='VERIFIED_PREDECISION'
        self.reject(x,'unverified')

    def test_bad_release_after_decision(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][0]['source_publication_utc']='2025-01-01T00:00:00Z'
        self.reject(x,'LOOKAHEAD')

    def test_bad_source_availability(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][0]['source_available_utc']='2025-01-01T00:00:00Z'
        self.reject(x,'LOOKAHEAD')

    def test_bad_observation_end(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][0]['observation_end_utc']='2025-01-01T00:00:00Z'
        self.reject(x,'LOOKAHEAD')

    def test_bad_retrieval_prospective(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][0]['retrieved_at_utc']='2025-01-01T00:00:00Z'
        self.reject(x,'LOOKAHEAD')

    def test_backtest_later_retrieval_is_not_proof_but_allowed(self):
        x=make_manifest(mode='HISTORICAL_RECONSTRUCTED')
        self.assertEqual(self.validate(x)['status'],'CANDIDATE_FOLD_METADATA_CONSISTENT_NOT_CERTIFIED')

    def test_unverified_historical_asof_after_cutoff_rejected(self):
        x=make_manifest(mode='HISTORICAL_RECONSTRUCTED')
        x['rows'][0]['input_evidence'][0]['source_available_utc']='2025-01-01T00:00:00Z'
        self.reject(x,'lookahead')

    def test_target_settlement_in_feature_rejected(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][0]['includes_target_date_settlement']=True
        self.reject(x,'Target leakage')

    def test_duplicate_id_rejected(self):
        x=make_manifest()
        x['rows'][3]['challenge_id']=x['rows'][0]['challenge_id']
        self.reject(x,'Repeated')

    def test_overlapping_target_session_rejected(self):
        x=make_manifest()
        x['rows'][3]['target_session']=x['rows'][0]['target_session']
        self.reject(x,'same target-session')

    def test_chronological_overlap_train_validation_rejected(self):
        x=make_manifest()
        x['rows'][3]['decision_cutoff_utc']='2024-01-04T13:00:00Z'
        for rec in x['rows'][3]['input_evidence']:
            rec['observation_end_utc']='2024-01-04T09:00:00Z'
            rec['source_publication_utc']='2024-01-04T10:00:00Z'
            rec['source_available_utc']='2024-01-04T11:00:00Z'
            rec['retrieved_at_utc']='2024-01-04T12:00:00Z'
        self.reject(x,'chronology')

    def test_chronological_overlap_validation_test_rejected(self):
        x=make_manifest()
        x['rows'][5]['decision_cutoff_utc']='2024-01-10T13:00:00Z'
        for rec in x['rows'][5]['input_evidence']:
            rec['observation_end_utc']='2024-01-10T09:00:00Z'
            rec['source_publication_utc']='2024-01-10T10:00:00Z'
            rec['source_available_utc']='2024-01-10T11:00:00Z'
            rec['retrieved_at_utc']='2024-01-10T12:00:00Z'
        self.reject(x,'chronology')

    def test_train_label_after_fit_rejected(self):
        x=make_manifest()
        x['rows'][2]['label_available_at_utc']='2024-02-01T13:00:00Z'
        self.reject(x,'Training label')

    def test_validation_label_after_selection_rejected(self):
        x=make_manifest()
        x['rows'][4]['label_available_at_utc']='2024-02-01T13:00:00Z'
        self.reject(x,'Validation label')

    def test_missing_train_label_time_rejected(self):
        x=make_manifest()
        x['rows'][0]['label_available_at_utc']=None
        self.reject(x,'availability')

    def test_wrong_source_combination_rejected(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][1]['source']='SP500_INDEX'
        self.reject(x,'model source')

    def test_source_missing_vintage_rejected(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][1]['source_revision_or_vintage']=''
        self.reject(x,'source vintage')

    def test_missing_license_rejected(self):
        x=make_manifest()
        x['rows'][0]['input_evidence'][1]['licensing_restriction']='UNKNOWN'
        self.reject(x,'License')

    def test_non_utc_rejected(self):
        x=make_manifest()
        x['rows'][0]['decision_cutoff_utc']='2024-01-02T13:00:00+02:00'
        self.reject(x,'Timestamp must be UTC')

    def test_future_test_label_time_permitted_for_offline_audit(self):
        x=make_manifest()
        x['rows'][5]['label_available_at_utc']='2024-03-15T13:00:00Z'
        self.assertEqual(self.validate(x)['status'],'CANDIDATE_FOLD_METADATA_CONSISTENT_NOT_CERTIFIED')

    def test_fold_manifest_status_must_not_claim_frozen(self):
        x=make_manifest()
        x['status']='FROZEN'
        self.reject(x,'cannot be called frozen')

    def test_frozen_contract_false_claim_rejected(self):
        c=copy.deepcopy(self.contract)
        c['phase_b_model_freeze_certified']=True
        with self.assertRaisesRegex(ValueError,'phase_b_model_freeze_certified'):
            mod.audit_contract(c,self.registry)

    def test_altered_registry_rejected(self):
        r=copy.deepcopy(self.registry)
        r['official_submitting_baseline']='M2'
        with self.assertRaises(ValueError):
            mod.audit_contract(self.contract,r)

    def test_contract_unknown_field_rejected(self):
        c=copy.deepcopy(self.contract)
        c['source_already_licensed']=True
        with self.assertRaises(ValueError):
            mod.audit_contract(c,self.registry)

    def test_cli_contract_read_only(self):
        files=[CONTRACT_FILE,REGISTRY_FILE]
        before=[mod.digest(p) for p in files]
        result=subprocess.run([sys.executable,str(Path(mod.__file__)), 'audit-contract',
                              '--registry',str(REGISTRY_FILE),'--contract',str(CONTRACT_FILE)],
                              text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('M11_STAGE6_READ_ONLY_AUDIT: PASS',result.stdout)
        self.assertEqual(before,[mod.digest(p) for p in files])

    def test_cli_wrong_registry_hash_rejected(self):
        result=subprocess.run([sys.executable,str(Path(mod.__file__)), 'audit-contract',
                              '--registry',str(REGISTRY_FILE),'--contract',str(CONTRACT_FILE),
                              '--expected-registry-sha256','0'*64],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('SHA-256',result.stderr)

    def test_cli_fold_manifest_with_pinned_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'candidate.json'
            p.write_text(json.dumps(make_manifest()),encoding='utf-8')
            result=subprocess.run([sys.executable,str(Path(mod.__file__)), 'audit-folds',
                '--registry',str(REGISTRY_FILE),'--contract',str(CONTRACT_FILE),
                '--expected-contract-sha256',mod.digest(CONTRACT_FILE),
                '--fold-manifest',str(p),'--expected-fold-manifest-sha256',mod.digest(p)],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('CANDIDATE_FOLD_METADATA_CONSISTENT_NOT_CERTIFIED',result.stdout)

    def test_cli_wrong_fold_manifest_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'candidate.json';p.write_text(json.dumps(make_manifest()),encoding='utf-8')
            result=subprocess.run([sys.executable,str(Path(mod.__file__)), 'audit-folds',
                '--registry',str(REGISTRY_FILE),'--contract',str(CONTRACT_FILE),
                '--expected-contract-sha256',mod.digest(CONTRACT_FILE),
                '--fold-manifest',str(p),'--expected-fold-manifest-sha256','b'*64],
                capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('SHA-256',result.stderr)

    def test_empty_fold_rejected(self):
        x=make_manifest();x['rows']=[];self.reject(x,'Too few')

    def test_wrong_contract_digest_embedded_rejected(self):
        x=make_manifest();x['contract_sha256']='a'*64;self.reject(x,'pin mismatch')

if __name__ == '__main__':
    unittest.main()
