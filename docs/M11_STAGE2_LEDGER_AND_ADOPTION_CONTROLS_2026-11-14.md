# Milestone 11 — Stage 2: private ES opportunity journal and adoption review

**Version:** v0.1, prepared 2026-11-14. **STATUS: CODE READY; RESEARCH PROTOCOL NOT ADOPTED.**

This stage *extends*, but never edits, `phase_a_m11_readiness_v0_1.py`, its test, or its methods document. The historical 19 observed ES challenges and their 19 settled results remain a separate pinned evidence population; original Notebook 12 still means **first 30 accepted forecasts beginning October 20**, not 30 eligible opportunities. No prior observations are enrolled in a new 150-opportunity trial.

## Deliverables

- `phase_a_m11_stage2_journal_v0_1.py`: validates and appends **private candidate** `CREATE` and `UPDATE` events to a SHA-256-linked JSONL journal, with explicitly pinned prior journal/source file hashes, original full row retained, and no implicit writes. Refuses historical cohort IDs, duplicate CREATE, mutation of challenge identity/source, invalid UTC, missing source bytes, absent receipt references, or premature MISSED_VERIFIED claims. On audit it rechecks **every** journal event and referenced first-evidence file. Any change is recorded as a new event; no prior event is edited. The external event JSON must be preserved itself.
- Journal `snapshot` mode generates a separate new immutable candidate JSONL snapshot (one row per unique challenge), reuses the established M11 candidate schema, and refuses overwrite. Snapshots **do not** certify evidence truth, independent timing, challenge visibility or eligibility, assigned phase ordinals, protocol adoption, or enrollment.
- `phase_a_m11_stage2_adoption_review_v0_1.py`: checks a PRIVATE candidate adoption record against byte-pinned protocol text and explicitly future-effective UTC timestamps. It **only** reports a candidate review, never adopts a protocol. Publication screenshot/Git hash alone does not certify independent timestamp or platform/account visibility.
- Synthetic tests cover hash chain corruption, idempotency, updates and source integrity, prospective/historical partition, dry-run, no overwrite, adoption timing and false certification claims.

## Deliberate non-capabilities

- **No** live network, opportunity discovery, timing attestation, S&P 500 dataset ingestion, Databento calls, model inference, submissions, Git operations, or automated adoption.
- **No** determination that all opportunities have been captured, or that a claimed no-submission is a verified miss.
- **No** approval of the 30/60/60 design by installing code. Alternative 30/30/45/45 and historical-plus-prospective descriptive targets must remain separate explicitly chosen protocols, not silently interchanged.
- **No** automatic transition from Phase A to Phase B.

## Recommended physical storage

All live candidate journals, copies of source evidence, snapshots and adoption-review files go under
`%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\` or a subfolder of that private root. This lets the established M11 validator check original source hashes; nothing under this private folder should be added to Git.

Suggested paths:

```
%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\journal.jsonl
%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\evidence\<challenge-id>_discovery.json
%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\events\<challenge-id>_create.json
%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\snapshots\candidate_<date>_A.jsonl
%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\adoption_review_candidate.json
```

The journal does **not** have to exist before the first append; use `--expected-journal-sha256 NONE` for a new journal. Each subsequent append requires the SHA-256 of the existing complete journal. Do not move, edit, truncate or reorder previous journal lines.

## Create candidate event JSON (only after a real official ES opportunity is captured)

The event file uses this structure. Replace every `<...>` placeholder from actual, byte-preserved official evidence; do **not** fill in fabricated IDs, default dates, or arbitrary market parameters:

```json
{
  "operation": "CREATE",
  "candidate_row": {
    "challenge_id": "<EXACT_OFFICIAL_ARENA_UUID>",
    "asset": "ES",
    "challenge_type": "daily",
    "contract_symbol": "<ACTUAL_ES_CONTRACT_EG_ESZ6>",
    "open_at_utc": "<OFFICIAL_OPEN_TIME_IN_EXPLICIT_UTC>",
    "deadline_utc": "<OFFICIAL_DEADLINE_IN_EXPLICIT_UTC>",
    "first_observed_at_utc": "<PRESERVED_DISCOVERY_TIME_IN_EXPLICIT_UTC>",
    "opportunity_first_evidence_path": "m11_future/evidence/<ACTUAL_SOURCE_FILENAME>.json",
    "opportunity_first_evidence_sha256": "<ACTUAL_SOURCE_SHA256>",
    "availability_status": "CONFIRMED_OPEN",
    "submission_status": "NO_ACCEPTED_RECORD_YET",
    "outcome_status": "PENDING",
    "provenance_level": "SOURCE_DISCOVERY_PREDEADLINE"
  }
}
```

A source evidence hash proves only that saved bytes haven't changed. The recorded observation time must come from preserved supporting information; inserting a self-claimed UTC timestamp is not independent evidence.

Later updates use `"operation":"UPDATE"`, a **complete revised candidate_row**, and a substantive `"reason"` string. Original challenge ID, market, contract, opening, deadline and first-evidence source reference cannot be changed. Adding `prediction_id` + `accepted_receipt_sha256` is necessary for `ACCEPTED_SCORED`; **it is not independent scored-revision certification**. `MISSED_VERIFIED` also needs separate preserved evidence, postdeadline review time and human substantiation. On settlement, use `OFFICIALLY_SETTLED` with `official_label` and `official_settlement_evidence_sha256` from separate official outcome import. Do not rewrite earlier receipts or probability vectors.

## Adoption candidate (not yet adopted)

A reviewer can create a PRIVATE JSON record with:

```json
{
  "record_type": "M11_CANDIDATE_ADOPTION_REVIEW_V0_1",
  "status": "PROPOSED_NOT_ADOPTED",
  "chosen_study_design": "UNDECIDED",
  "planned_phase_lengths": [30, 60, 60],
  "historical_19_enrolled": false,
  "original_notebook12_preserved": true,
  "protocol_sha256": "<ACTUAL_HASH_OF_REVIEWED_POLICY_FILE>",
  "owner_reviewed_at_utc": "<GENUINE_UTC_REVIEW_TIME>",
  "proposed_effective_at_utc": "<FUTURE_UTC_TIME_AFTER_REVIEW>",
  "first_eligible_challenge_id": null,
  "independent_publication_time_certified": false
}
```

`UNDECIDED` is intentional until the owner selects a design. For the **separate future 150-opportunity proposal**, choose `FRESH_150_OPPORTUNITIES` after explicit review, with `[30,60,60]`. For the earlier 150-total descriptive target, `ORIGINAL_150_TOTAL_DESCRIPTIVE` is a separate design, never relabelled a fresh 150-opportunity preregistration. The validator will reject `UNDECIDED` because a research policy must be selected deliberately, but even a passing candidate review does **not** establish adoption.

A final adoption requires a genuine decision, the exact protocol text hash and committed version, external publication/timestamp evidence reviewed on its own merits, an effective time preceding the **first** eligible official open, and a confirmed operational collection plan. Only future eligible challenge IDs can enter a new prospective population. Do not backdate to October 20 or November 13.

## Windows CMD usage

From repo root, assuming an activated `.venv`:

```bat
set "M11ROOT=%USERPROFILE%\MarketMood_Private\phase_a_m10"
set "M11FUT=%M11ROOT%\m11_future"
set "M11JOURNAL=%M11FUT%\journal.jsonl"
set "M11EV=%M11FUT%\events\<ACTUAL_CHALLENGE_ID>_create.json"
```

Preserve actual evidence and source file hashes. Compute event JSON physical SHA:

```bat
certutil -hashfile "%M11EV%" SHA256
```

Before first append (dry run; makes no files):

```bat
python phase_a_m11_stage2_journal_v0_1.py append --private-root "%M11ROOT%" --journal "%M11JOURNAL%" --expected-journal-sha256 NONE --event-file "%M11EV%" --expected-event-sha256 <EVENT_SHA256>
```

After inspection only, repeat with `--apply`. For subsequent updates or audits use the *new full journal physical SHA* (not the per-event hash) in `--expected-journal-sha256`. Never call `--apply` after a nonzero dry run. A true first journal append can only happen after a **real** future challenge is observed and its supporting source is saved; do not fabricate a weekend entry.

Read-only journal audit:

```bat
python phase_a_m11_stage2_journal_v0_1.py audit --private-root "%M11ROOT%" --journal "%M11JOURNAL%" --expected-journal-sha256 <CURRENT_JOURNAL_SHA256>
```

Dry-run new candidate snapshot then add `--apply` after review (new path only):

```bat
python phase_a_m11_stage2_journal_v0_1.py snapshot --private-root "%M11ROOT%" --journal "%M11JOURNAL%" --expected-journal-sha256 <CURRENT_JOURNAL_SHA256> --snapshot-path "%M11FUT%\snapshots\candidate_20261116_A.jsonl"
```

Then check the snapshot with the existing readiness validator (the M10 pinned baseline still must match):

```bat
certutil -hashfile "%M11FUT%\snapshots\candidate_20261116_A.jsonl" SHA256
python phase_a_m11_readiness_v0_1.py --repo "%USERPROFILE%\Market-Mood-Forecasting" --private-root "%M11ROOT%" --ledger "%M11FUT%\snapshots\candidate_20261116_A.jsonl" --expected-ledger-sha256 <SNAPSHOT_SHA256>
```

**Known pinned-baseline limitation:** the existing M11 readiness module intentionally pins the *current live* 19-row `arena_official_settlements.csv` hash. Once future official settlements are appended to that same live CSV, its historical pin will fail. That does **not** mean the new journal is corrupt. Preserve the November 14 19-row CSV as a sealed historical snapshot and introduce a separately versioned historical-baseline reader in a future amendment; never silently update a hash or modify the committed M11 methods to make a test green.

For a local adoption candidate review, hash the candidate JSON and reviewed protocol file separately and invoke `phase_a_m11_stage2_adoption_review_v0_1.py --candidate ... --private-root ... --expected-candidate-sha256 ... --protocol ... --expected-protocol-sha256 ...`. It reports `RESEARCH_STATUS: NOT_ADOPTED` regardless of whether candidate validation passes.

## Source/research protections

- Do not stage any private evidence or source HTML/JSON, candidate adoption record, journal or snapshot.
- Do not change Hugging Face, model/joblib, S&P 500 market series, Notebook 11/12, earlier submission receipts, M10 JSON reports, or historic M10C hashes.
- No paper claim of preregistered experiments until protocol adoption and independent timing are evidenced.
- Verify source provenance, opportunity availability, official challenge opening/deadline, and retained final scored revision **separately**. The journal alone does not prove them.
