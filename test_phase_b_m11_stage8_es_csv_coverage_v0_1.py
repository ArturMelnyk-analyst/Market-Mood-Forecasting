"""Synthetic offline tests, no licensed data, API calls, or model predictions."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phase_b_m11_stage8_es_csv_coverage_v0_1 as s8


class Stage8Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        self.repo=root/'repo'; self.repo.mkdir()
        self.private=root/'private'; self.private.mkdir()
        (self.private/'m11_future/source_inventories').mkdir(parents=True)
        self.inventory=self.private/'m11_future/source_inventories/stage7.json'
        self.entries=[]
        self.environment=patch.object(s8.s7,'validate_environment')
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.make_inventory()
    def make_inventory(self):
        obj={'record_type':'M11_STAGE7_LOCAL_SOURCE_INVENTORY_V0_1',
             'status':'LOCAL_INVENTORY_ONLY_NOT_DATASET_VALIDATION',
             'upstream_registry_sha256':s8.s7.PINNED['phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json'],
             'upstream_contract_sha256':s8.s7.PINNED['phase_b/phase_b_dataset_split_contract_DRAFT_v0_1.json'],
             'discovered_file_count':len(self.entries), 'file_metadata':self.entries}
        self.inventory.write_text(json.dumps(obj),encoding='utf-8')
        self.h=s8.s7.digest(self.inventory)
    def add(self,rel,content,source='ES',hashed=True):
        p=self.repo/rel; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(content)
        entry={'path':str(p.resolve()),'size_bytes':len(content),'sha256':s8.s7.digest(p) if hashed else None,
               'source_bucket_by_directory_only':source}
        self.entries.append(entry);self.make_inventory();return p
    def csvadd(self,header=('session_date','close'),rows=(('2026-10-01','2'),('2026-10-02','3')),rel='data/processed/es_arena_nb11_v0_1/snapshots/test.csv'):
        import io
        buf=io.StringIO(); w=csv.writer(buf,lineterminator='\n');w.writerow(header);w.writerows(rows)
        return self.add(rel,buf.getvalue().encode())
    def audit(self,**kw):
        return s8.audit_inventory(self.repo,self.private,self.inventory,self.h,**kw)
    def test_empty_inventory(self):
        r=self.audit();self.assertEqual(r['es_bucket_count_by_directory_only'],0)
        self.assertEqual(r['training_rows_certified'],0)
    def test_csv_valid_schema_range(self):
        self.csvadd();r=self.audit()
        self.assertEqual(r['csv_status_counts']['CSV_STRUCTURAL_SCAN_COMPLETE'],1)
        c=r['csv_details_private_only'][0]
        self.assertEqual(c['date_column_observations']['session_date']['max_date'],'2026-10-02')
        self.assertEqual(r['rows_structurally_inspected_not_validated'],2)
        self.assertFalse(c['source_market_calendar_verified'])
    def test_sha_mismatch_stops(self):
        self.csvadd()
        with self.assertRaisesRegex(ValueError,'SHA-256 changed'):
            s8.audit_inventory(self.repo,self.private,self.inventory,'a'*64)
    def test_bad_sha_format(self):
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            s8.audit_inventory(self.repo,self.private,self.inventory,'NOT_SHA')
    def test_wrong_type(self):
        obj=json.loads(self.inventory.read_text());obj['record_type']='fake';self.inventory.write_text(json.dumps(obj));self.h=s8.s7.digest(self.inventory)
        with self.assertRaisesRegex(ValueError,'schema'):
            self.audit()
    def test_upstream_pin_changed(self):
        obj=json.loads(self.inventory.read_text());obj['upstream_contract_sha256']='x';self.inventory.write_text(json.dumps(obj))
        self.h=s8.s7.digest(self.inventory)
        with self.assertRaisesRegex(ValueError,'upstream pins'):
            self.audit()
    def test_inventory_count_wrong(self):
        obj=json.loads(self.inventory.read_text());obj['discovered_file_count']=100;self.inventory.write_text(json.dumps(obj));self.h=s8.s7.digest(self.inventory)
        with self.assertRaisesRegex(ValueError,'file count'):
            self.audit()
    def test_not_private_source_inventory(self):
        outside=self.private/'else.json';outside.write_bytes(self.inventory.read_bytes())
        with self.assertRaisesRegex(ValueError,'designated private'):
            s8.audit_inventory(self.repo,self.private,outside,s8.s7.digest(outside))
    def test_csv_mutated_after_snapshot(self):
        p=self.csvadd();p.write_text('session_date,close\n2026-10-01,44\n')
        self.assertIn('FILE_SIZE_CHANGED_AFTER_STAGE7',self.audit()['csv_status_counts'])
    def test_csv_same_size_different_hash(self):
        p=self.csvadd();data=p.read_bytes();p.write_bytes(data.replace(b',2\n',b',9\n'))
        self.assertEqual(self.audit()['status'],'REVIEW_REQUIRED')
        self.assertIn('FILE_HASH_CHANGED_AFTER_STAGE7',self.audit()['csv_status_counts'])
    def test_missing_file(self):
        p=self.csvadd();p.unlink();self.assertIn('MISSING_OR_SYMLINK_AFTER_STAGE7',self.audit()['csv_status_counts'])
    def test_binary_not_parsed(self):
        self.add('data/raw/databento/incremental_v0_1/a.dbn.zst',b'not real DBN')
        r=self.audit();self.assertEqual(r['category_counts']['raw_dbn_files_not_parsed'],1)
        self.assertEqual(r['csv_candidates_inspected_or_skipped'],0)
    def test_purchase_json_not_csv(self):
        self.add('data/raw/databento/incremental_v0_1/PURCHASE_ATTEMPT_foo.json',b'{}')
        self.assertEqual(self.audit()['category_counts']['purchase_attempt_records_not_market_rows'],1)
    def test_other_source_skipped(self):
        self.add('data/raw/vix/test.csv',b'a\nb\n',source='VIX')
        self.assertEqual(self.audit()['es_bucket_count_by_directory_only'],0)
    def test_path_escape(self):
        outside=self.private/'secrets.csv';outside.write_text('x')
        self.entries.append({'source_bucket_by_directory_only':'ES','path':str(outside),'size_bytes':1,'sha256':s8.s7.digest(outside)})
        self.make_inventory()
        with self.assertRaisesRegex(ValueError,'escapes'):
            self.audit()
    def test_unapproved_repo_scope(self):
        self.add('other/unapproved.csv',b'a\nb\n')
        with self.assertRaisesRegex(ValueError,'outside Stage 7'):
            self.audit()
    def test_duplicate_candidate(self):
        self.csvadd();self.entries.append(dict(self.entries[0]));self.make_inventory()
        with self.assertRaisesRegex(ValueError,'Duplicate ES'):
            self.audit()
    def test_duplicate_header(self):
        self.csvadd(header=('close','CLOSE'))
        self.assertIn('INVALID_CSV_HEADER',self.audit()['csv_status_counts'])
    def test_missing_header(self):
        self.add('data/raw/databento/incremental_v0_1/empty.csv',b'')
        self.assertIn('EMPTY_CSV',self.audit()['csv_status_counts'])
    def test_row_width_mismatch(self):
        self.csvadd(rows=(('2026-10-01','2','extra'),))
        self.assertIn('ROW_WIDTH_MISMATCH',self.audit()['csv_status_counts'])
    def test_wrong_encoding(self):
        self.add('data/raw/databento/incremental_v0_1/bad.csv',b'head\n\xff\n')
        self.assertIn('CSV_PARSE_ERROR',self.audit()['csv_status_counts'])
    def test_size_cap(self):
        self.csvadd();r=self.audit(max_csv_bytes=3)
        self.assertIn('SKIPPED_FILE_SIZE_CAP',r['csv_status_counts'])
    def test_row_cap(self):
        self.csvadd();r=self.audit(max_rows=1)
        self.assertIn('TRUNCATED_ROW_CAP',r['csv_status_counts'])
        self.assertEqual(r['rows_structurally_inspected_not_validated'],1)
    def test_negative_limits(self):
        self.csvadd()
        with self.assertRaises(ValueError):self.audit(max_rows=0)
    def test_label_columns_flagged_not_validated(self):
        self.csvadd(header=('session_date','target','price'))
        c=self.audit()['csv_details_private_only'][0]
        self.assertEqual(c['target_like_columns_review_required'],['target'])
        self.assertFalse(c['source_target_label_semantics_verified'])
    def test_naive_timestamp_unverified_timezone(self):
        self.csvadd(header=('timestamp','value'),rows=(('2026-10-01T12:45:00','5'),))
        c=self.audit()['csv_details_private_only'][0]
        self.assertIn('NAIVE_TIMESTAMP_UNVERIFIED_ZONE',c['date_column_observations']['timestamp']['timestamp_kinds'])
    def test_utc_timestamp(self):
        self.csvadd(header=('ts_event','value'),rows=(('2026-10-01T12:45:00Z','5'),))
        c=self.audit()['csv_details_private_only'][0]
        self.assertEqual(c['date_column_observations']['ts_event']['timestamp_kinds']['UTC_OFFSET_TIMESTAMP'],1)
    def test_unparseable_date_marked(self):
        self.csvadd(rows=(('OCTOBER 1','2'),))
        self.assertEqual(self.audit()['csv_details_private_only'][0]['date_column_observations']['session_date']['unparseable_count'],1)
    def test_multiple_csv_schema_groups(self):
        self.csvadd(); self.csvadd(rel='data/processed/es_arena_nb11_v0_1/audits/x.csv')
        self.assertEqual(len(self.audit()['full_scanned_csv_schema_fingerprints']),1)
    def test_missing_stage7_hash_allowed_but_not_certified(self):
        self.csvadd();self.entries[0]['sha256']=None;self.make_inventory()
        self.assertTrue(self.audit()['csv_details_private_only'][0]['stage7_file_sha256_not_recorded'])
    def test_cli_readonly(self):
        self.csvadd()
        with patch.object(s8.s7,'validate_environment'), redirect_stdout(StringIO()) as out:
            code=s8.run(['audit-inventory','--repo',str(self.repo),'--private-root',str(self.private),'--inventory-file',str(self.inventory),'--expected-inventory-sha256',self.h])
        self.assertEqual(code,0);self.assertIn('FILES_WRITTEN: 0',out.getvalue())
    def test_cli_private_report_and_no_overwrite(self):
        self.csvadd();report=self.inventory.parent/'stage8.json'
        params=['audit-inventory','--repo',str(self.repo),'--private-root',str(self.private),'--inventory-file',str(self.inventory),'--expected-inventory-sha256',self.h,'--snapshot-file',str(report)]
        with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            self.assertEqual(s8.run(params),0)
            self.assertEqual(s8.run(params),2)
        self.assertTrue(report.exists())
    def test_licensing_and_training_not_certified(self):
        self.csvadd();r=self.audit();self.assertFalse(r['point_in_time_vintages_certified']);self.assertFalse(r['phase_b_freeze_certified']);self.assertEqual(r['training_rows_certified'],0)

if __name__ == '__main__':unittest.main()
