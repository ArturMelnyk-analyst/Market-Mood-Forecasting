import hashlib
import json
import tempfile
import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import phase_a_m11_stage2_journal_v0_1 as j
import phase_a_m11_stage2_adoption_review_v0_1 as a

CID = str(uuid.UUID(int=421))
OLDER = str(uuid.UUID(int=22))

class M11Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'private'
        self.root.mkdir()
        self.journal = self.root/'m11_future'/'journal.jsonl'
        self.source = self.root/'source.json'
        self.source.write_text('{"official_source":"synthetic_fixture_only"}',encoding='utf-8')
        self.row = {
            'challenge_id':CID, 'asset':'ES', 'challenge_type':'daily', 'contract_symbol':'ESZ6',
            'open_at_utc':'2026-11-16T00:00:00Z', 'deadline_utc':'2026-11-16T15:00:00Z',
            'first_observed_at_utc':'2026-11-16T10:00:00Z',
            'opportunity_first_evidence_path':'source.json',
            'opportunity_first_evidence_sha256':j.sha(self.source.read_bytes()),
            'availability_status':'CONFIRMED_OPEN', 'submission_status':'NO_ACCEPTED_RECORD_YET',
            'outcome_status':'PENDING', 'provenance_level':'SOURCE_DISCOVERY_PREDEADLINE',
        }
        self.event = self.root/'event.json'
        self.historical_patch=patch.object(j,'historical_ids',return_value={OLDER})
        self.historical_patch.start()
        self.addCleanup(self.historical_patch.stop)

    def put(self, op='CREATE', row=None, reason=None):
        x={'operation':op,'candidate_row':row or deepcopy(self.row)}
        if reason is not None:x['reason']=reason
        self.event.write_text(json.dumps(x,sort_keys=True),encoding='utf-8')
        return j.sha(self.event.read_bytes())

    def create(self):
        return j.append_event(self.root,self.journal,self.event,self.put(), 'NONE',True)

    def test_empty_audit(self):
        r=j.audit(self.root,self.journal,'NONE');self.assertEqual(r['events'],0)
    def test_dry_run_does_not_write(self):
        r=j.append_event(self.root,self.journal,self.event,self.put(),'NONE',False)
        self.assertEqual(r['files_written'],0);self.assertFalse(self.journal.exists())
    def test_apply_appends_event(self):
        r=self.create();self.assertEqual(r['files_written'],1);self.assertEqual(j.audit(self.root,self.journal,r['new_journal_sha256'])['events'],1)
    def test_journal_sha_stale_fails(self):
        self.create()
        with self.assertRaises(j.JournalError):j.audit(self.root,self.journal,'NONE')
    def test_event_sha_wrong_fails(self):
        self.put()
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,'f'*64,'NONE',True)
    def test_tamper_event_fails(self):
        r=self.create();self.journal.write_bytes(self.journal.read_bytes().replace(b'PENDING',b'UNKNOWN'))
        with self.assertRaises(j.JournalError):j.audit(self.root,self.journal,j.sha(self.journal.read_bytes()))
    def test_update_requires_reason(self):
        self.create();row=deepcopy(self.row);row['submission_status']='UNKNOWN'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put('UPDATE',row),'%s'%j.sha(self.journal.read_bytes()),False)
    def test_update_locked_identity(self):
        self.create();row=deepcopy(self.row);row['contract_symbol']='ESH7'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put('UPDATE',row,'correcting source after review'),j.sha(self.journal.read_bytes()),False)
    def test_update_valid(self):
        self.create();row=deepcopy(self.row);row['submission_status']='ATTEMPT_AMBIGUOUS'
        pin=j.sha(self.journal.read_bytes())
        result=j.append_event(self.root,self.journal,self.event,self.put('UPDATE',row,'POST returned inconclusive and needs review'),pin,True)
        audit=j.audit(self.root,self.journal,result['new_journal_sha256'])
        self.assertEqual(audit['distinct_challenges'],1);self.assertEqual(audit['events'],2)
        self.assertEqual(audit['rows'][0]['submission_status'],'ATTEMPT_AMBIGUOUS')
    def test_update_duplicate_noop_rejected(self):
        self.create()
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put('UPDATE',reason='no differences, user error'),j.sha(self.journal.read_bytes()))
    def test_duplicate_create_fails(self):
        self.create()
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(),j.sha(self.journal.read_bytes()))
    def test_historical_id_rejected(self):
        row=deepcopy(self.row);row['challenge_id']=OLDER
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_missing_source_fails(self):
        row=deepcopy(self.row);row['opportunity_first_evidence_path']='missing.json'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_wrong_source_hash_fails(self):
        row=deepcopy(self.row);row['opportunity_first_evidence_sha256']='f'*64
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_pre_open_observation_fails(self):
        row=deepcopy(self.row);row['first_observed_at_utc']='2026-11-15T20:00:00Z'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_post_deadline_falsely_predate_fails(self):
        row=deepcopy(self.row);row['first_observed_at_utc']='2026-11-16T16:00:00Z'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_offline_no_auto_enrollment(self):
        self.create()
        r=j.audit(self.root,self.journal,j.sha(self.journal.read_bytes()))
        self.assertEqual(r['enrolled_to_150'],0);self.assertEqual(r['phase_ordinals_assigned'],0)
    def test_snapshot_dry_run(self):
        self.create();destination=self.root/'snapshot.jsonl'
        r=j.snapshot(self.root,self.journal,j.sha(self.journal.read_bytes()),destination)
        self.assertFalse(destination.exists());self.assertEqual(r['rows'],1)
    def test_snapshot_apply(self):
        self.create();destination=self.root/'snapshot.jsonl'
        r=j.snapshot(self.root,self.journal,j.sha(self.journal.read_bytes()),destination,True)
        self.assertEqual(r['snapshot_sha256'],j.sha(destination.read_bytes()))
    def test_snapshot_no_overwrite(self):
        self.create();destination=self.root/'snapshot.jsonl'
        j.snapshot(self.root,self.journal,j.sha(self.journal.read_bytes()),destination,True)
        with self.assertRaises(j.JournalError):j.snapshot(self.root,self.journal,j.sha(self.journal.read_bytes()),destination,True)
    def test_journal_outside_private_fails(self):
        with self.assertRaises(j.JournalError):j.audit(self.root,Path(self.tmp.name)/'external.jsonl','NONE')
    def test_non_utc_time_rejected(self):
        row=deepcopy(self.row);row['deadline_utc']='2026-11-16T15:00:00'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_missing_receipt_accepted_rejected(self):
        row=deepcopy(self.row);row['submission_status']='ACCEPTED_SCORED'
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_missed_review_before_deadline_rejected(self):
        row=deepcopy(self.row);row.update(submission_status='MISSED_VERIFIED',miss_reviewed_at_utc='2026-11-16T14:00:00Z', miss_evidence_sha256='a'*64)
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')
    def test_source_not_under_root(self):
        row=deepcopy(self.row);row['opportunity_first_evidence_path']=str(Path(self.tmp.name)/'outside')
        with self.assertRaises(j.JournalError):j.append_event(self.root,self.journal,self.event,self.put(row=row),'NONE')

class AdoptionReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.protocol=self.root/'PROTOCOL_v1.md';self.protocol.write_text('Candidate reviewed prospective research policy',encoding='utf-8')
        self.hash=j.sha(self.protocol.read_bytes())
        self.record={
            'record_type':'M11_CANDIDATE_ADOPTION_REVIEW_V0_1', 'status':'PROPOSED_NOT_ADOPTED',
            'chosen_study_design':'FRESH_150_OPPORTUNITIES', 'planned_phase_lengths':[30,60,60],
            'historical_19_enrolled':False, 'original_notebook12_preserved':True,
            'protocol_sha256':self.hash, 'owner_reviewed_at_utc':'2026-11-14T13:00:00Z',
            'proposed_effective_at_utc':'2026-11-15T00:00:00Z',
            'first_eligible_challenge_id':None,'independent_publication_time_certified':False,
        }
    def problems(self, r=None):return a.evaluate(r or self.record,self.protocol,self.hash,self.root)
    def test_valid_candidate_passes(self):self.assertEqual(self.problems(),[])
    def test_no_auto_adopt(self):
        x=deepcopy(self.record);x['status']='ADOPTED'
        self.assertIn('automatic_or_false_adoption_claim',self.problems(x))
    def test_missing_decision_rejected(self):
        x=deepcopy(self.record);x['chosen_study_design']='UNDECIDED'
        self.assertIn('study_design_not_explicitly_selected',self.problems(x))
    def test_wrong_phase_lengths(self):
        x=deepcopy(self.record);x['planned_phase_lengths']=[30,30,45,45]
        self.assertIn('unapproved_phase_lengths',self.problems(x))
    def test_retrospective_enrollment_rejected(self):
        x=deepcopy(self.record);x['historical_19_enrolled']=True
        self.assertIn('historical_cohort_must_not_be_enrolled',self.problems(x))
    def test_bad_protocol_pin(self):
        x=deepcopy(self.record);x['protocol_sha256']='f'*64
        self.assertIn('protocol_hash_pin_mismatch',self.problems(x))
    def test_backdated_effective_rejected(self):
        x=deepcopy(self.record);x['proposed_effective_at_utc']='2026-11-14T12:00:00Z'
        self.assertIn('effective_time_not_after_owner_review',self.problems(x))
    def test_ambiguous_utc_rejected(self):
        x=deepcopy(self.record);x['owner_reviewed_at_utc']='2026-11-14T13:00:00'
        self.assertIn('missing_or_invalid_utc_times',self.problems(x))
    def test_fake_external_certification_rejected(self):
        x=deepcopy(self.record);x['independent_publication_time_certified']=True
        self.assertIn('independent_timestamp_claim_must_be_manual_separate_review',self.problems(x))
    def test_claim_first_eligible_rejected(self):
        x=deepcopy(self.record);x['first_eligible_challenge_id']=CID
        self.assertIn('first_future_challenge_cannot_be_claimed_in_candidate_record',self.problems(x))
    def test_publication_evidence_hash_missing(self):
        x=deepcopy(self.record);x['publication_evidence_path']='nonexistent.json';x['publication_evidence_sha256']='a'*64
        self.assertIn('publication_evidence_missing_or_hash_mismatch',self.problems(x))
    def test_publication_evidence_valid_bytes_not_attestation(self):
        proof=self.root/'gitproof.txt';proof.write_text('not independent timestamp',encoding='utf-8')
        x=deepcopy(self.record);x['publication_evidence_path']='gitproof.txt';x['publication_evidence_sha256']=j.sha(proof.read_bytes())
        self.assertEqual(self.problems(x),[])

if __name__ == '__main__':unittest.main()
