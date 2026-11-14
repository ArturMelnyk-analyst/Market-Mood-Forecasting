"""Offline synthetic tests for M11 Stage4; no live API, forecast or private record writes."""
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import phase_a_m11_stage4_prospective_gate_v0_1 as mod

CID='123e4567-e89b-42d3-a456-426614174000'


def record(*, open_time='2026-11-16T14:00:00Z', completed='2026-11-16T14:01:00Z'):
    raw={'id':CID,'asset':'ES','challenge_type':'daily','contract_symbol':'ESZ6',
         'deadline_utc':'2026-11-16T19:00:00Z','created_at':'2026-11-16T13:59:00Z'}
    if open_time is not None: raw['open_at_utc']=open_time
    return {'schema_version':'arena_discovery_v0_1','status':'success','http_status':200,
            'request_started_at_utc':'2026-11-16T14:00:10Z','record_completed_at_utc':completed,
            'response_json':{'challenges':[raw]},
            'es_daily_challenges':[{'challenge_id':CID,'asset':'ES','contract_symbol':'ESZ6','deadline':None}]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'repo').mkdir(); (self.root/'private').mkdir()
        self.repo=self.root/'repo'; self.priv=self.root/'private'

    def test_preflight_pending_before_effective(self):
        with patch.object(mod,'verify_git',return_value='f'*40), \
             patch.object(mod.baseline_api,'audit',return_value={
                 'status':'PASS','research_design':'ORIGINAL_150_TOTAL_DESCRIPTIVE',
                 'owner_decision':'AFFIRMATIVELY_ADOPTED_BY_OWNER','approved_protocol_effective_utc':'2026-11-15T12:00:00Z'}), \
             patch.object(mod.journal_api,'audit',return_value={
                 'files_written':0,'events':0,'distinct_challenges':0}):
            r=mod.preflight(self.repo,self.priv,self.priv/'journal.jsonl',now=datetime(2026,11,14,17,tzinfo=timezone.utc))
            self.assertEqual(r['protocol_status'],'ADOPTED_PENDING_EFFECTIVE_TIME')
            self.assertEqual(r['prospective_opportunities_enrolled_by_stage4'],0)
            self.assertEqual(r['candidate_journal_sha256'],mod.digest(b''))
            self.assertEqual(r['files_written'],0)

    def test_preflight_after_effective(self):
        with patch.object(mod,'verify_git',return_value='f'*40), \
             patch.object(mod.baseline_api,'audit',return_value={
                 'status':'PASS','research_design':'ORIGINAL_150_TOTAL_DESCRIPTIVE',
                 'owner_decision':'AFFIRMATIVELY_ADOPTED_BY_OWNER','approved_protocol_effective_utc':'2026-11-15T12:00:00Z'}), \
             patch.object(mod.journal_api,'audit',return_value={
                 'files_written':0,'events':1,'distinct_challenges':1}):
            r=mod.preflight(self.repo,self.priv,self.priv/'journal.jsonl',now=datetime(2026,11,16,17,tzinfo=timezone.utc))
            self.assertEqual(r['protocol_status'],'ADOPTED_EFFECTIVE')
            self.assertEqual(r['prospective_opportunities_enrolled_by_stage4'],0)

    def test_preflight_rejects_unapproved(self):
        with patch.object(mod,'verify_git',return_value='f'*40), \
             patch.object(mod.baseline_api,'audit',return_value={'status':'PASS','research_design':'FRESH_150_OPPORTUNITIES'}):
            with self.assertRaises(mod.GateError):
                mod.preflight(self.repo,self.priv,self.priv/'x')

    def test_preflight_rejects_journal_outside_private(self):
        with patch.object(mod,'verify_git',return_value='f'*40), \
             patch.object(mod.baseline_api,'audit',return_value={
                 'status':'PASS','research_design':'ORIGINAL_150_TOTAL_DESCRIPTIVE',
                 'owner_decision':'AFFIRMATIVELY_ADOPTED_BY_OWNER','approved_protocol_effective_utc':'2026-11-15T12:00:00Z'}):
            with self.assertRaises(mod.GateError):
                mod.preflight(self.repo,self.priv,self.repo/'journal.jsonl')

    def review(self, r):
        return mod.review_discovery(r,expected_hash='a'*64,actual_hash='a'*64)

    def test_candidate_with_official_opening_still_not_enrolled(self):
        r=self.review(record())
        self.assertEqual(r['challenge_reviews'][0]['review'],'CANDIDATE_ACCOUNT_ELIGIBILITY_AND_TIMING_REVIEW_REQUIRED')
        self.assertFalse(r['enrollment_authorized'])

    def test_created_at_does_not_substitute_for_open_time(self):
        r=self.review(record(open_time=None))
        self.assertEqual(r['challenge_reviews'][0]['review'],'OFFICIAL_OPEN_TIME_NOT_EVIDENCED_CREATED_AT_NOT_OPEN_AT')

    def test_opened_before_effective_is_excluded(self):
        r=self.review(record(open_time='2026-11-13T14:00:00Z'))
        self.assertEqual(r['challenge_reviews'][0]['review'],'OPENED_BEFORE_PROSPECTIVE_EFFECTIVE_TIME')

    def test_postdeadline_capture_not_predeadline(self):
        r=self.review(record(completed='2026-11-16T19:05:00Z'))
        self.assertEqual(r['challenge_reviews'][0]['review'],'POST_DEADLINE_OBSERVATION_NOT_PREDEADLINE_EVIDENCE')

    def test_wrong_raw_body_challenge(self):
        x=record(); x['response_json']['challenges'][0]['id']='123e4567-e89b-42d3-a456-426614174001'
        r=self.review(x)
        self.assertEqual(r['challenge_reviews'][0]['review'],'OFFICIAL_RAW_BODY_MATCH_UNAVAILABLE')

    def test_invalid_contract(self):
        x=record(); x['response_json']['challenges'][0]['contract_symbol']='SPY'
        r=self.review(x)
        self.assertEqual(r['challenge_reviews'][0]['review'],'CONTRACT_MISMATCH_OR_UNSUPPORTED')

    def test_deadline_after_capture_was_verified(self):
        x=record(); x['response_json']['challenges'][0]['deadline_utc']='2026-11-16T13:59:59Z'
        r=self.review(x)
        self.assertEqual(r['challenge_reviews'][0]['review'],'POST_DEADLINE_OBSERVATION_NOT_PREDEADLINE_EVIDENCE')

    def test_duplicate_id_rejected(self):
        x=record(); x['es_daily_challenges'].append(copy.deepcopy(x['es_daily_challenges'][0]))
        with self.assertRaises(mod.GateError): self.review(x)

    def test_bad_hash_rejected(self):
        with self.assertRaises(mod.GateError):
            mod.review_discovery(record(),expected_hash='b'*64,actual_hash='a'*64)

    def test_empty_feed_does_not_prove_absence(self):
        x=record(); x['es_daily_challenges']=[]; x['response_json']['challenges']=[]
        r=self.review(x)
        self.assertEqual(r['discovery_review'],'SUCCESS_EMPTY_FEED_COVERAGE_UNKNOWN')

    def test_failed_feed_does_not_prove_absence(self):
        x=record(); x['status']='failed'; x['es_daily_challenges']=[]
        r=self.review(x)
        self.assertEqual(r['discovery_review'],'INCOMPLETE_OR_FAILED')

    def test_private_root_escape_rejected(self):
        with self.assertRaises(mod.GateError): mod.inside(self.repo/'a.json',[self.priv])

    def test_naive_time_rejected(self):
        with self.assertRaises(mod.GateError): mod.utc('2026-11-16T15:00:00','OPEN')


if __name__=='__main__': unittest.main()
