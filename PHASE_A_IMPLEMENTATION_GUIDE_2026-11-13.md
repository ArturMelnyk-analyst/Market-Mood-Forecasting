# Phase A implementation pack — 13 November 2026

**Project:** Market-Mood-Forecasting / Headline Arena ES daily  
**Branch:** `experiment/headline-arena-es-daily`  
**Scope:** Extend evidence quality around the frozen M1 model, never change today's or previous predictions.

## What is included

- `phase_a_daily_chain_audit_v0_1.py`: read-only, offline verification of one challenge's discovery, forecast, source snapshot and manifest, frozen M1 joblib, attempt and server receipt.
- `test_phase_a_daily_chain_audit_v0_1.py`: 9 isolated synthetic tests.
- `phase_a_forecast_commitment_v0_1.py`: write-once private/public **prospective proof-of-concept** for commitments, with time-cutoff enforcement and hash verification. No network or automatic posting.
- `test_phase_a_forecast_commitment_v0_1.py`: 11 isolated synthetic tests (including four privacy/integrity hardening tests).
- `PHASE_A_READINESS_CHECKLIST_2026-11-13.md`: limitations, next steps, model freeze conditions.

**20/20 isolated synthetic tests passed in the revised-package environment (9 daily-audit + 11 commitment tests).** The first-pack daily verifier and initial 16 tests were successfully executed by the user against the November 13 private-repository evidence. The revised commitment hardening and updated 20-test suite still require a fresh run in the user’s Windows environment before any commit.

## A. Safety constraints before installing

The M1 deployed model SHA-256 must remain:

```
5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d
```

Never overwrite existing forecast, discovery, attempt, receipt, reviewed audit, model or paid DBN artifacts. Do NOT run `prepare` or `submit` just to test these tools. A frozen M1 forecast remains untouched. The current successful local submission receipt does NOT prove an independently timezone-certified server acceptance timestamp.

**No public commitment should be created for November 13 after the official prediction was already submitted.** The prototype is for future pre-deadline forecasts, and requires approval of a publication protocol and confirmation of Arena's rules.

## B. Install in Windows CMD

1. Download `Phase_A_Implementation_Pack_2026-11-13.zip` to `C:\Users\melny\Downloads`.
2. Open CMD and activate the project's existing virtual environment.
3. Navigate to the root directory.

```bat
cd /d C:\Users\melny\Market-Mood-Forecasting
.venv\Scripts\activate
git branch --show-current
git status --short
```

Confirm the branch is `experiment/headline-arena-es-daily`. Your working tree may have other unrelated changes. Do not reset, restore or discard them.

Check these filenames do not already exist:

```bat
if exist phase_a_daily_chain_audit_v0_1.py echo STOP - FILE EXISTS
if exist phase_a_forecast_commitment_v0_1.py echo STOP - FILE EXISTS
if exist test_phase_a_daily_chain_audit_v0_1.py echo STOP - FILE EXISTS
if exist test_phase_a_forecast_commitment_v0_1.py echo STOP - FILE EXISTS
```

If any exists, inspect it instead of overwriting it. Otherwise extract:

```bat
powershell -NoProfile -Command "Expand-Archive -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\Phase_A_Implementation_Pack_2026-11-13.zip') -DestinationPath (Get-Location).Path"
```

This installs **only source code and documentation**, not private market data.

## C. Run the new offline synthetic tests

```bat
python -m unittest test_phase_a_daily_chain_audit_v0_1 test_phase_a_forecast_commitment_v0_1 -v
```

Expected:

```
Ran 20 tests in ...
OK
```

If you see `FAILED` or `ERROR`, do not treat the code as accepted. Inspect the traceback. A successful mocked scenario is not an Arena submission.

Also rerun the earlier 23 safeguards:

```bat
python -m unittest test_phase_a_opportunity_audit_v0_1 test_arena_deadline_guard_v0_1 test_arena_submission_no_post_integration_v0_1 -v
```

Expected from your earlier run: 23 tests OK. The mocked `HTTP STATUS: 201` is synthetic, not a second real POST.

## D. Audit November 13's existing evidence — no API calls

```bat
set "ES_ID=9e7a3255-e4dc-4a1d-a8f9-a30f18c4fa2c"
set "TARGET_DATE=2026-11-13"
python phase_a_daily_chain_audit_v0_1.py --challenge-id "%ES_ID%" --date "%TARGET_DATE%"
```

The verifier attempts to read existing paths derived from the challenge/date:

```
data\processed\es_arena_live\forecasts\forecast_20261113_<UUID>.json
data\processed\es_arena_live\receipts\attempt_<UUID>.json
data\processed\es_arena_live\receipts\receipt_<UUID>.json
data\processed\es_arena_live\discovery\arena_discovery_*.json
```

It also follows the original saved forecast's `source_csv`, `model_dir` and SHA-256 fields. It checks the snapshot manifest's `snapshot_sha256` when present. It does **not** run joblib, Databento, Arena GET/POST, model inference or retraining.

Results:
- `PASS`: a specified local condition was verified.
- `WARN`: a relevant limitation exists without a known contradiction (e.g., server `created_at` lacks a timezone).
- `INCOMPLETE`: evidence is missing. **A missing receipt is not permission to retry.**
- `FAIL`: an explicit inconsistency was detected. Investigate the original evidence, do not overwrite or repair it in place.
- `LOCALLY_CONSISTENT_WITH_LIMITATIONS`: no failed/missing checks; **not** a complete independent provenance certification.

Exit codes: 0 no FAIL/INCOMPLETE (warnings possible); 1 a contradictory check; 2 incomplete evidence without contradictions. You can get JSON output to inspect without writing a report file:

```bat
python phase_a_daily_chain_audit_v0_1.py --challenge-id "%ES_ID%" --date "%TARGET_DATE%" --json
```

Do not interpret an unsigned HTTP Date header as independent proof of the server accepting your scored POST at a particular time. The locally timestamped discovery event and separate receipt are distinct evidence classes.

## E. Run the existing historical audit separately

```bat
python phase_a_opportunity_audit_v0_1.py audit
```

The historical audit is **pinned to a November 12 candidate inventory of 18 accepted forecasts**; it does NOT append a November 13 row to that immutable CSV. Do not change that candidate, its SHA-256, or the 30-accepted versus 30-eligible protocol debate. Extend the research protocol only through a dated amendment and separately validated new opportunity records.

## F. Prospective cryptographic commitment prototype — TEST TODAY, DO NOT CREATE FOR NOV 13

The eleven synthetic tests now exercise:
- write-once creation and verification;
- 32-byte random private nonce (64 hex characters);
- canonical UTF-8 JSON and SHA-256;
- cutoff refusal at or after deadline minus five minutes;
- detection of modified forecast and modified commitment;
- keeping the nonce out of public commitment records;
- refusing identical forecast/private/public output paths **before writing**;
- refusing to create the private nonce/forecast file inside a Git checkout (including a worktree);
- detecting altered public target dates and unexpected sensitive fields during verification.

This is a **prototype**, not a production-activated commitment/reveal system. It creates a private object containing the *full forecast*, hashes it with a nonce and produces a public commitment object containing no probabilities. Publication of the public hash is a **separate manual action** that must occur before the deadline and must have independently inspectable timing evidence. Neither Git author-date nor local filesystem timestamp suffices. The full reveal workflow (post-settlement publication, model failures, nonces and policy) is NOT implemented here.

**Store any future private nonce/forecast file OUTSIDE the repository entirely**, e.g., in a restricted-access directory under your Windows profile that is not a Git checkout. This prototype now rejects private output inside Git checkouts and matching private/public destinations. The public output can be inside Git, but must never include full forecast values or nonce material. These guards do not substitute for reviewing staged files or independently establishing a publication timestamp.

**Do not run `create` against November 13 or earlier already-submitted forecasts.** A separate pre-deadline opportunity is required. Do not publish private data or nonces now. The prototype's `verify` command is read-only and may be used for synthetic private/public test fixtures.

For future operation, after agreeing on the protocol and verifying .gitignore and official rules, the CLI will support:

```bat
python phase_a_forecast_commitment_v0_1.py create --forecast "EXACT_NEW_UNSUBMITTED_FORECAST_PATH" --private "EXACT_PRIVATE_OUTPUT_PATH" --public "EXACT_PUBLIC_OUTPUT_PATH"
python phase_a_forecast_commitment_v0_1.py verify --forecast "EXACT_NEW_FORECAST_PATH" --private "EXACT_PRIVATE_OUTPUT_PATH" --public "EXACT_PUBLIC_OUTPUT_PATH"
```

The exact paths must be set by you; this is intentionally **not** a ready-to-run November 13 command. `create` produces write-once local artifacts and does not automatically upload anything. The public object alone cannot establish an independently verified publication time.

## G. Source control without accidentally uploading private evidence

```bat
git status --short
git diff --check
```

After reviewing the files, stage **only** these known-safe source/documentation artifacts:

```bat
git add phase_a_daily_chain_audit_v0_1.py test_phase_a_daily_chain_audit_v0_1.py phase_a_forecast_commitment_v0_1.py test_phase_a_forecast_commitment_v0_1.py PHASE_A_IMPLEMENTATION_GUIDE_2026-11-13.md PHASE_A_READINESS_CHECKLIST_2026-11-13.md
git diff --cached --stat
git diff --cached --check
```

Examine the staged list carefully; none should include `data/`, `.env`, joblib, Databento DBNs, originals, receipt JSON, secrets or private forecasts. No push is required to complete local testing. If it matches the intended files and tests pass, consider:

```bat
git commit -m "Add offline Phase A daily provenance audit and commitment prototype"
```

Publishing a **code implementation** commit is distinct from publishing actual pre-deadline forecast commitments. Confirm the hosting and public-evidence protocol before using the latter.

## H. What must remain unfinished or marked provisional

- Complete independently certified historical eligible-opportunity census: **NOT ESTABLISHED**.
- Independent UTC certification of server acceptance using only local receipt: **NOT ESTABLISHED**.
- Publicly timed pre-deadline forecast commitment: **NOT IMPLEMENTED / NOT YET ACTIVATED**.
- Public post-settlement reveal with failure/missingness policy: **NOT IMPLEMENTED**.
- M1-Balanced model and Phase B selection rules: **NOT FROZEN**; await Kopei's methodology feedback.
- External data availability/release/retrieval timestamp certification: separate Phase B preparation work.

## I. Report back for review

Share the output of the revised **20 tests** and confirm the earlier **23 tests** still pass. The previous November 13 read-only chain audit was already consistent with disclosed limitations. We can then inspect any warnings or mismatches and decide whether the prototype requires adjustments before being merged.

## J. November 13 local integration checkpoint and privacy hardening

The user ran the first pack in the project and reported 16/16 new tests, 23/23 earlier tests, and a November 13 daily audit summary of `LOCALLY_CONSISTENT_WITH_LIMITATIONS`. The stored scored receipt and saved probabilities were locally consistent. Independent UTC acceptance-time certification and complete historical opportunity census remain `NOT ESTABLISHED`.

Before the first public Git commit, a focused safety review added 4 synthetic checks and two preventative rules: secret private commitment files must be created outside every Git checkout, and private/public/forecast paths must be distinct. Verification additionally rejects a changed public target date and public records containing sensitive forecast/nonce fields. The adjusted pack has 20 new synthetic tests; rerun them in your own virtual environment and still rerun the earlier 23. A passing suite is necessary but does not authorize publication of actual forecast commitments.
