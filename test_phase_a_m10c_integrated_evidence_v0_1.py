import json
import tempfile
import unittest
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import phase_a_m10c_integrated_evidence_v0_1 as m


def fixture():
    t=tempfile.TemporaryDirectory()
    root=Path(t.name)
    repo=root/'repo'
    batchdir=root/'batch'
    repo.mkdir()
    batchdir.mkdir()
    ids=[str(uuid.UUID(int=i+1)) for i in range(18)]+[m.SINGLE_CID]
    entries=[]
    br=[]
    local=[]
    b_records=[]
    single_body=None
    for i,cid in enumerate(ids):
        dl=datetime(2026,10,20,tzinfo=timezone.utc)+timedelta(days=i,hours=14)
        margin=0.069164 if i==0 else 247.305982 if i==1 else 7200
        sub=dl-timedelta(seconds=margin)
        deadline=dl.isoformat().replace('+00:00','Z')
        submitted=sub.isoformat().replace('+00:00','Z')
        entries.append({'challenge_id':cid,'deadline':deadline,'status':'resolved','asset':'ES',
            'public_open_before_as_of':{'status':'unknown'},'reader_delivery_before_as_of':{'status':'unknown'},
            'submission_eligibility_at_as_of':'unknown'})
        br.append({'challenge_id':cid})
        local.append({'challenge_id':cid,'acceptance_evidence':'PINNED_HISTORICAL_ACCEPTED' if i<18 else 'NATIVE_ACCEPTED_SCORED',
                      'settled_in_local_official_csv':i<18})
        read={'predictions':[{'challenge_id':cid,'prediction_id':'prediction_'+str(i),'revision_number':1,
                              'submitted_at':submitted,'probability_source':'stored_native_ternary',
                              'prediction':{'probabilities':{'bearish':0.2,'bullish':0.7,'neutral':0.1}}}]}
        if cid == m.SINGLE_CID:
            single_body=read
        else:
            place=batchdir/cid
            place.mkdir()
            raw=json.dumps(read).encode()
            (place/'body.json').write_bytes(raw)
            metadata={'challenge_id':cid,'http_status':200,'body_sha256':m.digest(raw),'body_bytes':len(raw)}
            (place/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
            b_records.append({'challenge_id':cid,'capture_state':'VALIDATED_READBACK','body_sha256':m.digest(raw)})
    audit={'api_contract_version':'financial-opportunity-audit-v1','asset':'ES','next_cursor':None,
           'coverage':{'complete':False},'entries':entries}
    m10b={'source_sha256':m.AUDIT_SHA,'challenge_records':br}
    single={'challenge_id':m.SINGLE_CID,'public_opportunity_audit_sha256':m.AUDIT_SHA,'server_readback_sha256':m.SINGLE_RAW_SHA}
    batch={'public_audit_sha256':m.AUDIT_SHA,'records':b_records}
    reconciliation={'records':local,'review_issues':[]}
    return t,repo,batchdir,audit,m10b,single,batch,single_body,reconciliation,ids


class TestM10C(unittest.TestCase):
    def setUp(self):
        (self.t,self.repo,self.batch_dir,self.audit,self.m10b,self.single,self.batch,
         self.single_body,self.reconciliation,self.ids)=fixture()
        self.addCleanup(self.t.cleanup)

    def run_build(self):
        return m.build_integrated(self.repo,self.audit,self.m10b,self.single,self.batch,
                                  self.single_body,self.batch_dir,self.reconciliation)

    def test_basic_count_and_timing(self):
        rep=self.run_build()
        self.assertEqual(rep['retained_challenges'],19)
        self.assertEqual(rep['revision_records'],19)
        self.assertEqual(rep['counts']['timing_PREDEADLINE_BUFFER_MET'],17)
        self.assertEqual(rep['counts']['timing_PREDEADLINE_INSIDE_FIVE_MINUTE_BUFFER'],2)
        self.assertEqual(rep['review_flags'],[])

    def test_preserves_scientific_limits(self):
        rep=self.run_build()
        for key in ('historical_eligible_opportunity_census_certified','account_specific_prior_visibility_certified',
                    'model_execution_certified','final_scored_revision_certified','phase_a_protocol_adopted'):
            self.assertIs(rep[key],False)
        self.assertTrue(all(x['model_execution_certified'] is False for x in rep['records']))

    def test_own_audit_uuid_differing(self):
        self.m10b['challenge_records'].pop()
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_bad_source_hash_in_batch(self):
        cid=self.ids[0]
        path=self.batch_dir/cid/'body.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_mismatched_batch_report_body(self):
        self.batch['records'][0]['body_sha256']='a'*64
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_bad_local_reconciliation_id_set(self):
        self.reconciliation['records'].pop()
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_unresolved_local_review_issue_rejected(self):
        self.reconciliation['review_issues']=[{'problems':['synthetic']}]
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_audit_incomplete_cursor(self):
        self.audit['next_cursor']='abc'
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_single_challenge_mismatch(self):
        self.single['challenge_id']='wrong'
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_server_readback_id_conflict(self):
        self.single_body['predictions'][0]['challenge_id']='wrong'
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_duplicate_revision_number(self):
        x=deepcopy(self.single_body['predictions'][0])
        self.single_body['predictions'].append(x)
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_no_created_at_fallback(self):
        rev=self.single_body['predictions'][0]
        rev['created_at']=rev.pop('submitted_at')
        rep=self.run_build()
        row=next(x for x in rep['records'] if x['challenge_id']==m.SINGLE_CID)
        self.assertEqual(row['timing_classification'],'UNKNOWN_SERVER_TIMESTAMP')

    def test_time_after_deadline(self):
        self.single_body['predictions'][0]['submitted_at']='2026-12-01T00:00:00Z'
        rep=self.run_build()
        self.assertEqual(rep['counts']['timing_AT_OR_AFTER_DEADLINE'],1)

    def test_valid_local_forecast_matching(self):
        cid=self.ids[0]
        dir=self.repo/'data/processed/es_arena_live/forecasts'
        dir.mkdir(parents=True)
        content={'challenge_id':cid,'contract_symbol':'ESZ6','model_sha256':m.MODEL_SHA,
                 'probabilities':{'bearish':0.2,'bullish':0.7,'neutral':0.1}}
        (dir/('forecast_20261020_'+cid+'.json')).write_text(json.dumps(content),encoding='utf-8')
        rep=self.run_build()
        row=next(x for x in rep['records'] if x['challenge_id']==cid)
        self.assertEqual(row['original_vs_retained_vector'],'EXACT_NUMERIC_MATCH')
        self.assertFalse(row['model_execution_certified'])

    def test_wrong_model_forecast_not_matched(self):
        cid=self.ids[0]
        dir=self.repo/'data/processed/es_arena_live/forecasts';dir.mkdir(parents=True)
        (dir/('forecast_a_'+cid+'.json')).write_text(json.dumps({'challenge_id':cid,'contract_symbol':'ESZ6',
                            'model_sha256':'bad','probabilities':{'bearish':0.2,'bullish':0.7,'neutral':0.1}}),encoding='utf-8')
        rep=self.run_build()
        row=next(x for x in rep['records'] if x['challenge_id']==cid)
        self.assertEqual(row['original_vs_retained_vector'],'NO_VERIFIED_LOCAL_VECTOR')

    def test_percentage_rounded_not_exact(self):
        local={'bearish':.202,'bullish':.697,'neutral':.101}
        retained={'bearish':.2,'bullish':.7,'neutral':.1}
        self.assertEqual(m.vector_comparison(local,retained,'percentage'),'PERCENT_ROUNDING_CONSISTENT_NOT_EXACT')

    def test_percentage_class_check(self):
        v,r=m.canonical_vector({'bearish':20,'bullish':70,'neutral':10})
        self.assertEqual(r,'percentage')
        self.assertAlmostEqual(sum(v.values()),1)

    def test_invalid_boolean_probability(self):
        with self.assertRaises(m.EvidenceError):m.canonical_vector({'bearish':True,'bullish':.8,'neutral':.2})

    def test_bad_timestamp_naive_unknown(self):
        self.single_body['predictions'][0]['submitted_at']='2026-11-13T10:00:00'
        rep=self.run_build()
        self.assertEqual(rep['counts']['timing_UNKNOWN_SERVER_TIMESTAMP'],1)

    def test_single_report_pinned_hash(self):
        self.single['public_opportunity_audit_sha256']='bad'
        with self.assertRaises(m.EvidenceError):self.run_build()

    def test_private_report_actual_bytes_hash(self):
        dest=Path(self.t.name)/'MarketMood_Private'/'report.json'
        sha=m.save_report(dest,{'hello':['world']})
        self.assertEqual(sha,m.digest(dest.read_bytes()))
        with self.assertRaises(m.EvidenceError):m.save_report(dest,{'hello':['new']})

    def test_private_output_reject_repo(self):
        dest=self.repo/'report.json'
        with self.assertRaises(m.EvidenceError):m.save_report(dest,{'x':1})

    def test_receipt_link_matching_and_no_scored_claim(self):
        cid=self.ids[0]
        fdir=self.repo/'data/processed/es_arena_live/forecasts'; fdir.mkdir(parents=True)
        rp=self.repo/'data/processed/es_arena_live/receipts';rp.mkdir(parents=True)
        forecast={'challenge_id':cid,'contract_symbol':'ESZ6','model_sha256':m.MODEL_SHA,
                 'probabilities':{'bearish':0.2,'bullish':0.7,'neutral':0.1}}
        fp=fdir/('forecast_20261020_'+cid+'.json'); fp.write_text(json.dumps(forecast),encoding='utf-8')
        receipt={'http_status':201,'forecast_sha256':m.digest(fp.read_bytes()),
                 'response':{'challenge_id':cid,'counts_for_score':True,'prediction_id':'prediction_0',
                             'revision_number':1,'probabilities':forecast['probabilities']}}
        (rp/('receipt_'+cid+'.json')).write_text(json.dumps(receipt),encoding='utf-8')
        rep=self.run_build()
        row=next(x for x in rep['records'] if x['challenge_id']==cid)
        self.assertEqual(row['receipt_identity_linkage'],'LOCAL_ACCEPTANCE_RECEIPT_ID_LINKED_NOT_FINAL_SCORED_PROOF')
        self.assertFalse(row['final_scored_revision_certified'])

    def test_received_receipt_id_mismatch_review(self):
        cid=self.ids[0]
        rp=self.repo/'data/processed/es_arena_live/receipts';rp.mkdir(parents=True)
        (rp/('receipt_'+cid+'.json')).write_text(json.dumps({'http_status':201,'response':{'challenge_id':cid,
                                              'counts_for_score':True,'prediction_id':'wrong'}}),encoding='utf-8')
        rep=self.run_build()
        row=next(x for x in rep['records'] if x['challenge_id']==cid)
        self.assertEqual(row['receipt_identity_linkage'],'LOCAL_RECEIPT_FORECAST_HASH_UNVERIFIED')

if __name__ == '__main__':unittest.main()
