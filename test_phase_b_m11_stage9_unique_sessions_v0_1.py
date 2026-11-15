"""Synthetic-only Stage 9 tests; no real licensed data or network."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import phase_b_m11_stage9_unique_sessions_v0_1 as s9
import phase_b_m11_stage8_es_csv_coverage_v0_1 as s8
import phase_b_m11_stage7_source_inventory_v0_1 as s7

HEAD=['session_date','session_start_utc','session_end_utc','open','close']

def make_csv(repo, ymd, dates, *, override=None, extra=None):
    dir=repo/'data/processed/es_arena_nb11_v0_1/snapshots';dir.mkdir(parents=True,exist_ok=True)
    p=dir/f'es_sessions_{ymd}_aaaaaaaaaaaaaaaa.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=HEAD);w.writeheader()
        for day in dates:
            v='101' if day==override else '100'
            w.writerow({'session_date':day,'session_start_utc':day+'T00:00:00Z','session_end_utc':day+'T22:00:00Z','open':'99','close':v})
    meta={'relative_path':p.relative_to(repo).as_posix(),'inventory_sha256':s7.digest(p),'inventory_size':p.stat().st_size,'inspection_status':'CSV_STRUCTURAL_SCAN_COMPLETE','complete_csv_scanned':True,'rows_inspected':len(dates),'header_fields':HEAD,'header_sha256':s8.sha_bytes(json.dumps(HEAD,ensure_ascii=False,separators=(',',':')).encode())}
    return p,meta

class Stage9Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name);self.repo=self.root/'repo';self.repo.mkdir();self.private=self.root/'private';self.private.mkdir()
    def entries(self,*metas):
        return {'record_type':s8.TYPE,'status':'PASS_WITH_LIMITATIONS','csv_details_private_only':list(metas), 'csv_candidates_inspected_or_skipped':len(metas),'category_counts':{'csv_candidates':len(metas)}}
    def run_audit(self,*metas):
        with patch.object(s9,'verify_report',return_value=self.entries(*metas)):
            return s9.audit(self.repo,self.private,self.root/'fake.json','a'*64)
    def test_exact_two_snapshots_dedup(self):
        _,a=make_csv(self.repo,'20261102',['2026-10-30','2026-11-02'])
        _,b=make_csv(self.repo,'20261103',['2026-10-30','2026-11-02','2026-11-03'])
        r=self.run_audit(a,b)
        self.assertEqual(r['distinct_session_dates_observed'],3)
        self.assertEqual(r['raw_total_rows_in_selected_snapshots'],5)
        self.assertEqual(r['repeated_session_date_occurrences'],2)
        self.assertEqual(r['dates_with_nonidentical_versions_or_schemas'],0)
        self.assertEqual(r['status'],'COMPLETE_WITH_LIMITATIONS')
        self.assertEqual(r['independent_training_rows_certified'],0)
        self.assertFalse(r['training_feasibility_certified'])
    def test_conflict_detected(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        _,b=make_csv(self.repo,'20261103',['2026-11-02','2026-11-03'],override='2026-11-02')
        r=self.run_audit(a,b)
        self.assertEqual(r['dates_with_nonidentical_versions_or_schemas'],1)
        self.assertEqual(r['status'],'REVIEW_REQUIRED')
        self.assertEqual(r['sample_conflict_dates_private_only'][0]['session_date'],'2026-11-02')
    def test_no_snapshots_explicitly_review(self):
        r=self.run_audit()
        self.assertEqual(r['status'],'NO_ELIGIBLE_SNAPSHOT_REVIEW_REQUIRED')
    def test_other_csv_excluded(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        extra={'relative_path':'data/processed/es_arena_nb11_v0_1/snapshots/review_draft.csv','inspection_status':'CSV_STRUCTURAL_SCAN_COMPLETE'}
        r=self.run_audit(a,extra)
        self.assertEqual(r['excluded_other_csv_candidates'],1)
        self.assertEqual(r['selected_es_sessions_snapshots'],1)
    def test_changed_content_hash_stops(self):
        p,a=make_csv(self.repo,'20261102',['2026-11-02']);p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'differs'):
            self.run_audit(a)
    def test_file_size_cap(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        with patch.object(s9,'verify_report',return_value=self.entries(a)):
            with self.assertRaisesRegex(ValueError,'cap'):
                s9.audit(self.repo,self.private,self.root/'fake','a'*64,max_bytes=20)
    def test_row_cap(self):
        _,a=make_csv(self.repo,'20261102',['2026-10-30','2026-11-02'])
        with patch.object(s9,'verify_report',return_value=self.entries(a)):
            with self.assertRaisesRegex(ValueError,'cap'):
                s9.audit(self.repo,self.private,self.root/'fake','a'*64,max_rows=1)
    def test_later_file_date_mismatch(self):
        _,a=make_csv(self.repo,'20261103',['2026-11-02'])
        with self.assertRaisesRegex(ValueError,'filename'):
            self.run_audit(a)
    def test_same_date_twice_rejected(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02','2026-11-02'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.run_audit(a)
    def test_bad_utc(self):
        self.assertRaises(ValueError,s9.parse_utc_timestamp,'2026-11-02T12:00:00')
        self.assertRaises(ValueError,s9.parse_utc_timestamp,'2026-11-02T12:00:00+01:00')
        self.assertEqual(s9.parse_utc_timestamp('2026-11-02T12:00:00Z').tzinfo,timezone.utc)
    def test_end_before_start(self):
        p,a=make_csv(self.repo,'20261102',['2026-11-02'])
        contents=p.read_text().replace('22:00:00','00:00:00');p.write_text(contents)
        a['inventory_sha256']=s7.digest(p);a['inventory_size']=p.stat().st_size
        with self.assertRaisesRegex(ValueError,'end not after'):
            self.run_audit(a)
    def test_report_hash_guard(self):
        file=self.root/'private'/'report.json';file.write_text('{}')
        with patch.object(s7,'validate_environment',return_value=None):
            with self.assertRaisesRegex(ValueError,'hash differs|outside'):
                s9.verify_report(self.repo,self.private,file,'a'*64)
    def test_pins_in_report_guard(self):
        d=self.entries()
        d['record_type']=s8.TYPE;d['upstream_registry_sha256']='a'*64
        pr=self.private/'m11_future/source_inventories';pr.mkdir(parents=True)
        p=pr/'r.json';p.write_text(json.dumps(d))
        with patch.object(s7,'validate_environment',return_value=None):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                s9.verify_report(self.repo,self.private,p,s7.digest(p))
    def test_report_duplicate_paths_guard(self):
        d=self.entries({'relative_path':'x'},{'relative_path':'x'})
        d['upstream_registry_sha256']=s7.PINNED['phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json']
        d['upstream_contract_sha256']=s7.PINNED['phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json']
        pr=self.private/'m11_future/source_inventories';pr.mkdir(parents=True)
        p=pr/'r.json';p.write_text(json.dumps(d))
        with patch.object(s7,'validate_environment',return_value=None):
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                s9.verify_report(self.repo,self.private,p,s7.digest(p))
    def test_missing_metadata_hash(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        a['inventory_sha256']=None
        with self.assertRaisesRegex(ValueError,'hash'):
            self.run_audit(a)
    def test_unsafe_escape_stops(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        a['relative_path']='../outside.csv'
        with self.assertRaises(ValueError):
            self.run_audit(a)
    def test_main_no_snapshot_writes_none(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        with patch.object(s9,'verify_report',return_value=self.entries(a)):
            x=s9.main(['audit-sessions','--repo',str(self.repo),'--private-root',str(self.private),'--stage8-report',str(self.root/'notreal'),'--expected-stage8-sha256','a'*64])
        self.assertEqual(x,0)
        self.assertFalse(list(self.private.rglob('*.json')))
    def test_main_no_snapshots_returns_3(self):
        with patch.object(s9,'verify_report',return_value=self.entries()):
            self.assertEqual(s9.main(['audit-sessions','--repo',str(self.repo),'--private-root',str(self.private),'--stage8-report',str(self.root/'notreal'),'--expected-stage8-sha256','a'*64]),3)
    def test_no_model_changes_and_no_training(self):
        _,a=make_csv(self.repo,'20261102',['2026-11-02'])
        r=self.run_audit(a)
        self.assertFalse(r['m1_changed']);self.assertEqual(r['models_trained'],0)
        self.assertEqual(r['network_requests'],0);self.assertEqual(r['forecast_submissions'],0)

if __name__=='__main__':unittest.main()
