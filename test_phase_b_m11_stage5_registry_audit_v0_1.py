"""Synthetic-only tests for M11 Stage 5 read-only registry and as-of controls."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('stage5_registry_audit',SRC/'phase_b_m11_stage5_registry_audit_v0_1.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
REGISTRY=SRC/'phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json'
T='2026-11-16T13:00:00Z'
H='a'*64

def evidence_for(model_id='M2'):
    return {'record_type':'M11_STAGE5_CANDIDATE_INPUT_EVIDENCE_V0_1','model_id':model_id,
            'decision_cutoff_utc':T,'inputs':[{
                'source':s,'asof_evidence_status':'VERIFIED_PREDECISION',
                'observation_end_utc':'2026-11-13T20:00:00Z',
                'source_publication_utc':'2026-11-13T20:30:00Z',
                'source_available_utc':'2026-11-13T20:35:00Z',
                'retrieved_at_utc':'2026-11-16T12:00:00Z',
                'provenance_sha256':H,'licensing_restriction':'PRIVATE_LICENSED',
                'source_revision_or_vintage':'SYNTHETIC_FIXTURE_NOT_REAL',
                'includes_target_date_settlement':False
            } for s in mod.EXPECTED_SOURCES[model_id]]}

class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry=json.loads(REGISTRY.read_text(encoding='utf-8'))

    def rejects(self, mutate):
        r=copy.deepcopy(self.registry);mutate(r)
        with self.assertRaises(mod.AuditError):mod.validate_registry(r)

    def test_valid_draft(self):
        self.assertEqual(mod.validate_registry(self.registry)['status'],'DRAFT_REGISTRY_VALIDATED_NOT_FROZEN')
    def test_exact_seven_configs(self):
        self.assertEqual(list(mod.EXPECTED_SOURCES),[m['id'] for m in self.registry['models']])
    def test_reject_m1_artifact_swap(self):self.rejects(lambda x:x['models'][0].update(artifact_sha256=H))
    def test_reject_m1_retraining(self):self.rejects(lambda x:x['models'][0].update(artifact_status='RETRAINED'))
    def test_reject_removed_m1(self):self.rejects(lambda x:x['models'].pop(0))
    def test_reject_duplicate_config(self):self.rejects(lambda x:x['models'].__setitem__(1,x['models'][0]))
    def test_reject_sp500_index(self):self.rejects(lambda x:x['models'][2]['sources'].append('SP500_INDEX'))
    def test_reject_m5_missing_gdelt(self):self.rejects(lambda x:x['models'][4]['sources'].remove('GDELT'))
    def test_reject_modified_phase_boundary(self):self.rejects(lambda x:x.update(phase_b_accepted_positions=[31,91]))
    def test_reject_pretend_freeze(self):self.rejects(lambda x:x.update(status='FROZEN'))
    def test_reject_pretend_model_trained(self):self.rejects(lambda x:x['models'][1].update(artifact_status='TRAINED'))
    def test_reject_pretend_primary_finalists(self):self.rejects(lambda x:x['comparison_plan'].update(primary_finalists=['M1','M2']))
    def test_reject_missing_matched_control(self):self.rejects(lambda x:x['matched_training_control'].update(name='NONE'))
    def test_reject_metric_change(self):self.rejects(lambda x:x['comparison_plan'].update(primary_metric='ACCURACY'))
    def test_reject_unfrozen_threshold_value(self):self.rejects(lambda x:x['comparison_plan'].update(minimum_improvement_threshold=0.02))
    def test_reject_wrong_study_design(self):self.rejects(lambda x:x.update(study_design='FRESH_150'))
    def test_reject_official_challenger(self):self.rejects(lambda x:x.update(official_submitting_baseline='M2'))
    def test_reject_disclosed_artifacts(self):self.rejects(lambda x:x.update(private_data_and_model_artifacts_publication='ALLOWED'))
    def test_reject_false_external_source_readiness(self):self.rejects(lambda x:x['external_source_readiness'].update(GDELT='VALIDATED'))
    def test_reject_false_shadow_pred_count(self):self.rejects(lambda x:x.update(prospective_shadow_predictions_created=1))

class InputTests(unittest.TestCase):
    def setUp(self):self.r=json.loads(REGISTRY.read_text(encoding='utf-8'))
    def rejects(self, mutate, model_id='M2'):
        e=evidence_for(model_id);mutate(e)
        with self.assertRaises(mod.AuditError):mod.validate_evidence(self.r,e,model_id,T)
    def test_correct_7_model_source_sets(self):
        for mid in mod.EXPECTED_SOURCES:
            with self.subTest(mid=mid):
                self.assertEqual(mod.validate_evidence(self.r,evidence_for(mid),mid,T)['sources_checked'],len(mod.EXPECTED_SOURCES[mid]))
    def test_reject_future_observation(self):self.rejects(lambda e:e['inputs'][0].update(observation_end_utc='2026-11-16T14:00:00Z'))
    def test_reject_future_publication(self):self.rejects(lambda e:e['inputs'][0].update(source_publication_utc='2026-11-16T14:00:00Z'))
    def test_reject_future_availability(self):self.rejects(lambda e:e['inputs'][0].update(source_available_utc='2026-11-16T14:00:00Z'))
    def test_reject_future_retrieval(self):self.rejects(lambda e:e['inputs'][0].update(retrieved_at_utc='2026-11-16T14:00:00Z'))
    def test_reject_naive_timestamp(self):self.rejects(lambda e:e['inputs'][0].update(retrieved_at_utc='2026-11-16T12:00:00'))
    def test_reject_timezone_offset(self):self.rejects(lambda e:e['inputs'][0].update(retrieved_at_utc='2026-11-16T12:00:00+01:00'))
    def test_reject_unknown_point_in_time_availability(self):self.rejects(lambda e:e['inputs'][0].update(asof_evidence_status='UNKNOWN'))
    def test_reject_missing_source_vintage(self):self.rejects(lambda e:e['inputs'][1].update(source_revision_or_vintage=''))
    def test_reject_missing_license(self):self.rejects(lambda e:e['inputs'][0].update(licensing_restriction=''))
    def test_reject_missing_hash(self):self.rejects(lambda e:e['inputs'][0].update(provenance_sha256=''))
    def test_reject_included_target_settlement(self):self.rejects(lambda e:e['inputs'][0].update(includes_target_date_settlement=True))
    def test_reject_missing_required_source(self):self.rejects(lambda e:e['inputs'].pop(1))
    def test_reject_extra_source(self):self.rejects(lambda e:e['inputs'].append(e['inputs'][0]))
    def test_reject_wrong_model(self):self.rejects(lambda e:e.update(model_id='M3'))
    def test_reject_wrong_cutoff(self):self.rejects(lambda e:e.update(decision_cutoff_utc='2026-11-16T13:01:00Z'))
    def test_reject_publication_after_available(self):self.rejects(lambda e:e['inputs'][0].update(source_publication_utc='2026-11-16T12:30:00Z'))
    def test_reject_availability_after_retrieval(self):self.rejects(lambda e:e['inputs'][0].update(source_available_utc='2026-11-16T12:30:00Z'))
    def test_no_file_write_in_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.json';p.write_text(json.dumps(evidence_for()),encoding='utf-8')
            before=(hashlib.sha256(p.read_bytes()).hexdigest(),hashlib.sha256(REGISTRY.read_bytes()).hexdigest())
            self.assertEqual(mod.validate_evidence(self.r,json.loads(p.read_text()),'M2',T)['files_written'],0)
            self.assertEqual(before,(hashlib.sha256(p.read_bytes()).hexdigest(),hashlib.sha256(REGISTRY.read_bytes()).hexdigest()))
    def test_cli_registry_no_writes(self):
        r=subprocess.run([sys.executable,str(SRC/'phase_b_m11_stage5_registry_audit_v0_1.py'),'audit-registry','--registry',str(REGISTRY)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('DRAFT_REGISTRY_VALIDATED_NOT_FROZEN',r.stdout)
    def test_cli_input_with_pinned_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json';p.write_text(json.dumps(evidence_for()))
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            r=subprocess.run([sys.executable,str(SRC/'phase_b_m11_stage5_registry_audit_v0_1.py'),'audit-input-evidence','--registry',str(REGISTRY),'--manifest',str(p),'--expected-manifest-sha256',sha,'--model-id','M2','--decision-utc',T],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('NOT_EXTERNALLY_CERTIFIED',r.stdout)
    def test_cli_wrong_manifest_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json';p.write_text(json.dumps(evidence_for()))
            r=subprocess.run([sys.executable,str(SRC/'phase_b_m11_stage5_registry_audit_v0_1.py'),'audit-input-evidence','--registry',str(REGISTRY),'--manifest',str(p),'--expected-manifest-sha256',H,'--model-id','M2','--decision-utc',T],capture_output=True,text=True)
            self.assertEqual(r.returncode,2)

if __name__=='__main__':unittest.main()
