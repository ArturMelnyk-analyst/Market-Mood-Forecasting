import csv
import hashlib
import io
import json
import tempfile
import unittest
import uuid
from copy import deepcopy
from pathlib import Path

import phase_a_m10c_settlement_join_v0_1 as m


def fixture(nsettled=18):
    ids=[str(uuid.UUID(int=i+1)) for i in range(19)]
    rows=[]
    for i,cid in enumerate(ids):
        rows.append({'challenge_id':cid,
                     'revision_number':1,
                     'timing_classification':'PREDEADLINE_INSIDE_FIVE_MINUTE_BUFFER' if i<2 else 'PREDEADLINE_BUFFER_MET',
                     'retained_deadline_at_query_snapshot':'2026-11-13T15:00:00Z',
                     'official_settlement_record_present':i<nsettled,
                     'model_execution_certified':False,
                     'final_scored_revision_certified':False,
                     'original_vs_retained_vector':'EXACT_NUMERIC_MATCH' if i<18 else 'NO_VERIFIED_LOCAL_VECTOR',
                     'receipt_identity_linkage':'LOCAL_ACCEPTANCE_RECEIPT_ID_LINKED_NOT_FINAL_SCORED_PROOF' if i<18 else 'LOCAL_RECEIPT_UNAVAILABLE'})
    report={'schema_version':m.INTEGRATED_VERSION,'source_audit_sha256':m.PUBLIC_AUDIT_SHA256,
            'review_flags':[],'retained_challenges':19,'revision_records':19,'records':rows,
            'historical_eligible_opportunity_census_certified':False,
            'account_specific_prior_visibility_certified':False,
            'model_execution_certified':False,
            'final_scored_revision_certified':False,
            'phase_a_protocol_adopted':False}
    cols=['challenge_id','settlement_date','asset','contract_symbol','official_label',
          'settlement_return_pct','official_deadline_utc','official_settled_at_utc',
          'arena_score','source_url','verified_by','verified_at_utc']
    data={}
    for i,cid in enumerate(ids[:nsettled]):
        data[cid]={'challenge_id':cid,'settlement_date':'2026-11-13','asset':'ES','contract_symbol':'ESZ6',
                   'official_label':['bullish','neutral','bearish'][i%3],
                   'settlement_return_pct':'-0.35','official_deadline_utc':'2026-11-13T15:00:00Z',
                   'official_settled_at_utc':'2026-11-13T22:00:00Z','arena_score':'41',
                   'source_url':'https://example.test','verified_by':'tester','verified_at_utc':'2026-11-14T00:00:00Z'}
    return report,data,cols


def csv_bytes(rows,cols):
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    return f.getvalue().encode('utf-8')


class TestSettlement(unittest.TestCase):
    def setUp(self):
        self.report,self.data,self.cols=fixture()
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
    def build(self):
        return m.build_report(self.report,self.data,'a'*64,'b'*64)
    def test_valid_settlements(self):
        x=self.build()
        self.assertEqual(x['challenge_count'],19)
        self.assertEqual(x['settlement_rows_in_observed_cohort'],18)
        self.assertEqual(x['counts']['settlement_NOT_IN_LOCAL_CSV'],1)
        self.assertEqual(x['counts']['deadline_DEADLINES_MATCH_AS_RECORDED'],18)
        self.assertEqual(x['review_flags'],[])
    def test_no_scoring_or_execution_claims(self):
        x=self.build()
        self.assertFalse(x['official_scoring_calculated'])
        self.assertFalse(x['final_scored_revision_certified'])
        self.assertFalse(x['model_execution_certified'])
        self.assertFalse(x['phase_a_150_opportunity_protocol_adopted'])
        self.assertTrue(all(r['final_scored_revision_certified'] is False for r in x['records']))
    def test_missing_one_more_settlement_reviews(self):
        self.data.pop(next(iter(self.data)))
        self.assertIn('LOCAL_SETTLEMENT_PRESENCE_CONFLICT',self.build()['review_flags'])
    def test_extra_settlement_reviews(self):
        self.data[self.report['records'][-1]['challenge_id']]=dict(next(iter(self.data.values())))
        self.assertIn('LOCAL_SETTLEMENT_PRESENCE_CONFLICT',self.build()['review_flags'])
    def test_deadline_mismatch_reviews(self):
        next(iter(self.data.values()))['official_deadline_utc']='2026-11-13T14:00:00Z'
        self.assertIn('OFFICIAL_SETTLEMENT_VS_RETAINED_DEADLINE_CONFLICT',self.build()['review_flags'])
    def test_equivalent_timezones_match(self):
        next(iter(self.data.values()))['official_deadline_utc']='2026-11-13T16:00:00+01:00'
        self.assertNotIn('OFFICIAL_SETTLEMENT_VS_RETAINED_DEADLINE_CONFLICT',self.build()['review_flags'])
    def test_missing_deadline_reviews(self):
        next(iter(self.data.values()))['official_deadline_utc']=''
        self.assertIn('DEADLINE_TIMESTAMP_UNAVAILABLE',self.build()['review_flags'])
    def test_naive_deadline_reviews(self):
        next(iter(self.data.values()))['official_deadline_utc']='2026-11-13T15:00:00'
        self.assertIn('DEADLINE_TIMESTAMP_UNAVAILABLE',self.build()['review_flags'])
    def test_early_settlement_reviews(self):
        next(iter(self.data.values()))['official_settled_at_utc']='2026-11-13T12:00:00Z'
        self.assertIn('SETTLEMENT_BEFORE_DEADLINE',self.build()['review_flags'])
    def test_missing_settled_timestamp_is_unknown(self):
        next(iter(self.data.values()))['official_settled_at_utc']=''
        x=self.build(); self.assertEqual(x['counts']['settled_at_SETTLEMENT_TIMESTAMP_UNKNOWN'],1)
    def test_bad_label_reviews(self):
        next(iter(self.data.values()))['official_label']='up'
        self.assertIn('UNRECOGNIZED_OFFICIAL_LABEL',self.build()['review_flags'])
    def test_bad_asset_reviews(self):
        next(iter(self.data.values()))['asset']='NQ'
        self.assertIn('SETTLEMENT_ASSET_CONFLICT',self.build()['review_flags'])
    def test_bad_settlement_date_reviews(self):
        next(iter(self.data.values()))['settlement_date']='2026-11-99'
        self.assertIn('SETTLEMENT_DATE_MALFORMED',self.build()['review_flags'])
    def test_header_check(self):
        with self.assertRaises(m.EvidenceError):m.load_settlements(b'challenge_id,official_label\nabc,bullish\n')
    def test_duplicate_csv_ids(self):
        row=next(iter(self.data.values()))
        with self.assertRaises(m.EvidenceError):m.load_settlements(csv_bytes([row,row],self.cols))
    def test_csv_utf8_bom(self):
        raw=b'\xef\xbb\xbf'+csv_bytes(list(self.data.values()),self.cols)
        self.assertEqual(len(m.load_settlements(raw)),18)
    def test_csv_invalid_id(self):
        row=next(iter(self.data.values())).copy(); row['challenge_id']='bad'
        with self.assertRaises(m.EvidenceError):m.load_settlements(csv_bytes([row],self.cols))
    def test_raw_file_hash_tamper(self):
        path=self.root/'source.txt';path.write_bytes(b'hello')
        with self.assertRaises(m.EvidenceError):m.load_pinned(path,'0'*64)
    def test_integrated_source_hash_rejected(self):
        self.report['source_audit_sha256']='wrong'
        with self.assertRaises(m.EvidenceError):m.load_integrated(json.dumps(self.report).encode())
    def test_integrated_review_flags_rejected(self):
        self.report['review_flags']=['problem']
        with self.assertRaises(m.EvidenceError):m.load_integrated(json.dumps(self.report).encode())
    def test_integrated_scoring_flag_rejected(self):
        self.report['final_scored_revision_certified']=True
        with self.assertRaises(m.EvidenceError):m.load_integrated(json.dumps(self.report).encode())
    def test_duplicate_integrated_id_rejected(self):
        self.report['records'][1]['challenge_id']=self.report['records'][0]['challenge_id']
        with self.assertRaises(m.EvidenceError):m.load_integrated(json.dumps(self.report).encode())
    def test_extra_rows_outside_cohort(self):
        self.data[str(uuid.uuid4())]=dict(next(iter(self.data.values())))
        x=self.build();self.assertEqual(x['settlement_rows_outside_observed_cohort'],1)
    def test_binary_report_write_and_no_overwrite(self):
        dest=self.root/'MarketMood_Private'/'report.json'
        result=m.write_private(dest,self.build())
        self.assertEqual(result,m.sha256(dest.read_bytes()))
        with self.assertRaises(m.EvidenceError):m.write_private(dest,self.build())
    def test_report_not_in_private_directory(self):
        with self.assertRaises(m.EvidenceError):m.write_private(self.root/'report.json',self.build())
    def test_malformed_csv_row(self):
        bad=b','.join(x.encode() for x in self.cols)+b'\n'+b'a,b,c,d,e,f,g,h,i,j,k,l,m\n'
        with self.assertRaises(m.EvidenceError):m.load_settlements(bad)
    def test_cli_dry_no_file_write(self):
        integrated_path=self.root/'integrated.json'
        csv_path=self.root/'settlements.csv'
        integrated_raw=json.dumps(self.report).encode()
        csv_raw=csv_bytes(list(self.data.values()),self.cols)
        integrated_path.write_bytes(integrated_raw);csv_path.write_bytes(csv_raw)
        from contextlib import redirect_stdout
        f=io.StringIO()
        args=['--integrated-report',str(integrated_path),'--expected-integrated-sha256',m.sha256(integrated_raw),
              '--settlements-csv',str(csv_path),'--expected-settlements-sha256',m.sha256(csv_raw)]
        with redirect_stdout(f):rc=m.main(args)
        self.assertEqual(rc,0)
        self.assertIn('PRIVATE_REPORT_WRITTEN: False',f.getvalue())
        self.assertEqual(list(self.root.glob('*.json')),[integrated_path])
    def test_cli_private_write_hashed(self):
        integrated_path=self.root/'integrated.json';csv_path=self.root/'settlements.csv'
        a=json.dumps(self.report).encode();b=csv_bytes(list(self.data.values()),self.cols)
        integrated_path.write_bytes(a);csv_path.write_bytes(b)
        dest=self.root/'MarketMood_Private'/'settlement.json'
        from contextlib import redirect_stdout
        f=io.StringIO()
        with redirect_stdout(f):
            rc=m.main(['--integrated-report',str(integrated_path),'--expected-integrated-sha256',m.sha256(a),
              '--settlements-csv',str(csv_path),'--expected-settlements-sha256',m.sha256(b),
              '--private-report',str(dest)])
        self.assertEqual(rc,0)
        self.assertIn(m.sha256(dest.read_bytes()),f.getvalue())

if __name__=='__main__':unittest.main()
