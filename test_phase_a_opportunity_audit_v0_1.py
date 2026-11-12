"""Fully offline tests for Phase A opportunity evidence audit; no Arena/API calls."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase_a_opportunity_audit_v0_1 as m


class OpportunityAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.research = self.root / 'data/processed/es_arena_research_v0_1'
        self.live = self.root / 'data/processed/es_arena_live'
        self.discovery = self.live / 'discovery'
        self.research.mkdir(parents=True)
        self.discovery.mkdir(parents=True)
        self.cid = m.ORIGINAL_FIRST_ID
        with (self.research/'arena_opportunity_candidate_20261112.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['challenge_id','settlement_date','opportunity_status']); w.writeheader()
            w.writerow(dict(challenge_id=self.cid,settlement_date='2026-10-20',opportunity_status='confirmed_accepted'))
            for i in range(17):
                w.writerow(dict(challenge_id=f'fake-accepted-{i}',settlement_date='2026-11-01',opportunity_status='confirmed_accepted'))
        self.pinned_candidate=hashlib.sha256((self.research/'arena_opportunity_candidate_20261112.csv').read_bytes()).hexdigest()
        self.hist={'challenges':[{'challenge':{'id':self.cid,'asset':'ES','challenge_type':'daily','deadline':'2026-10-20T14:00:00Z','created_at':'2026-10-19T21:00:00Z','status':'open','contract_symbol':'ESZ6'}}]}
        p=self.live/'active_challenges_inspection_v0_1.json';p.write_text(json.dumps(self.hist),encoding='utf-8')
        self.pinned_hist=hashlib.sha256(p.read_bytes()).hexdigest()
        self.patch1=patch.object(m,'CANDIDATE_SHA256',self.pinned_candidate)
        self.patch2=patch.object(m,'HISTORICAL_SHA256',self.pinned_hist)
        self.patch1.start();self.patch2.start();self.addCleanup(self.patch1.stop);self.addCleanup(self.patch2.stop)

    def save_capture(self,name,cid=None,server_date='Thu, 12 Nov 2026 14:50:00 GMT',deadline='2026-11-12T15:00:00Z',status='success',body=None):
        if body is None:
            body={'challenges':[{'challenge':{'id':cid or 'new-id','asset':'ES','challenge_type':'daily','deadline':deadline,'contract_symbol':'ESZ6'}}]}
        d={'status':status,'http_status':200,'response_json':body,'server_date_header':server_date,'record_completed_at_utc':'2026-11-12T14:50:02+00:00'}
        (self.discovery/f'arena_discovery_{name}.json').write_text(json.dumps(d),encoding='utf-8')

    def test_historical_matches_one_accepted_but_no_server_time(self):
        r=m.build_report(self.root)
        self.assertEqual(r['accepted_candidate_count'],18)
        self.assertEqual(r['historical_inspection_matched_accepted_count'],1)
        self.assertEqual(r['positive_discovery_with_server_predeadline_count'],0)

    def test_post_deadline_empty_feed_is_not_missing_opportunity(self):
        self.save_capture('after',body={'challenges':[{'id':'palm-id','asset':'PALM','challenge_type':'daily'}]},server_date='Thu, 12 Nov 2026 15:41:30 GMT')
        r=m.build_report(self.root)
        self.assertEqual(len(r['records']),18)
        self.assertEqual(r['record_status_counts']['ACCEPTED_DOCUMENTED'],18)
        self.assertNotIn('DISCOVERED_NOT_IN_ACCEPTED_CANDIDATE_REVIEW_REQUIRED',r['record_status_counts'])

    def test_new_predeadline_es_is_unresolved_not_automatically_missed(self):
        self.save_capture('pre')
        r=m.build_report(self.root)
        self.assertEqual(len(r['records']),19)
        new=next(x for x in r['records'] if x['challenge_id']=='new-id')
        self.assertEqual(new['record_status'],'DISCOVERED_NOT_IN_ACCEPTED_CANDIDATE_REVIEW_REQUIRED')
        self.assertTrue(new['predeadline_server_date_discovery_evidenced'])
        self.assertIsNone(new['settlement_date'])

    def test_postdeadline_es_does_not_count_as_predeadline_evidence(self):
        self.save_capture('post',server_date='Thu, 12 Nov 2026 15:41:30 GMT')
        r=m.build_report(self.root)
        new=next(x for x in r['records'] if x['challenge_id']=='new-id')
        self.assertFalse(new['predeadline_server_date_discovery_evidenced'])

    def test_incomplete_es_has_observation_but_no_certified_predeadline(self):
        self.save_capture('incomplete',status='incomplete',deadline=None)
        r=m.build_report(self.root)
        new=next(x for x in r['records'] if x['challenge_id']=='new-id')
        self.assertFalse(new['predeadline_server_date_discovery_evidenced'])

    def test_cannot_silently_replace_pinned_candidate(self):
        with (self.research/'arena_opportunity_candidate_20261112.csv').open('ab') as f:f.write(b'\n')
        with self.assertRaisesRegex(ValueError,'hash differs'):
            m.build_report(self.root)

    def test_malformed_archive_flagged_without_fabricating_opportunity(self):
        self.save_capture('malformed',body={'invalid':[]})
        r=m.build_report(self.root)
        self.assertEqual(len(r['records']),18)
        self.assertIn('REVIEW_ERROR',[x['status'] for x in r['archive_checks']])


if __name__=='__main__':
    unittest.main(verbosity=2)
