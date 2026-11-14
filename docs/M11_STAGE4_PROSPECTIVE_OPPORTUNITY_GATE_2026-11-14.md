# M11 Stage 4 — Read-only prospective opportunity gate v0.1

**Prepared:** 2026-11-14
**Effective study protocol:** ES Daily Protocol v1.0, `ORIGINAL_150_TOTAL_DESCRIPTIVE`, recorded adoption `2026-11-14T16:55:34Z`, effective `2026-11-15T12:00:00Z`.
**Adopted protocol Git commit:** `6f089df837e5dcd1aa5fc198490e287a98fa7fd8`
**Prior Stage 3 code commit:** `51432cd8e5cc75dd8bdb081cba43fdda531a3ce4`

## 1. Purpose

Protect the transition from the 19 historical accepted-evidenced ES challenges to *future* account-eligible opportunities. This stage is read-only and offline. It verifies the sealed historical baseline and local owner-adoption record through the existing Stage 3 source, verifies the Stage 2 private candidate journal through the existing Stage 2 source, and optionally reviews a **real, preserved** authenticated Arena discovery JSON from the existing capture tool. It does not make API calls, invent challenge IDs, write journal entries, submit forecasts, label misses, or certify independently timestamped adoption.

The overall study target remains **150 accepted forecasts including the historical 19**, with accepted-count phases 30/60/60. The prospective **opportunity** denominator is separate and not capped at 150. The first 19 are not newly prospectively enrolled.

## 2. Read-only preflight now

From Windows CMD inside the repository, with `.venv` activated:

```bat
cd /d "%USERPROFILE%\Market-Mood-Forecasting"
set "M11ROOT=%USERPROFILE%\MarketMood_Private\phase_a_m10"
set "M11JOURNAL=%M11ROOT%\m11_future\events\m11_prospective_candidate_events_v0_1.jsonl"
python phase_a_m11_stage4_prospective_gate_v0_1.py preflight --repo "%CD%" --private-root "%M11ROOT%" --journal "%M11JOURNAL%"
echo STAGE4_PREFLIGHT_EXIT=%ERRORLEVEL%
```

Expected before effective time: `M11_STAGE4_GATE: PASS`, `protocol_status: ADOPTED_PENDING_EFFECTIVE_TIME`, `candidate_journal_events: 0` (unless a genuine future candidate journal was already created), and `files_written: 0`, `network_calls: 0`, `forecast_submissions: 0`. **Absence of the journal is equivalent to empty for this preflight only; it is not evidence of an empty Arena feed.** After the effective time, the status becomes `ADOPTED_EFFECTIVE` without automatically enrolling any challenge.

A nonzero exit code or an unexpected journal count must be investigated before proceeding. The Stage 2 journal still prints the legacy `CANDIDATE_ONLY_NOT_ADOPTED` status if called directly: that refers to its old *automatic enrollment* behavior, not to the real separately recorded owner-adoption decision.

## 3. On the next real opportunity after the effective time

Use the project's already field-tested SOP. Make **one authenticated challenge discovery GET**, with no scored submission:

```bat
python arena_opportunity_capture_v0_1.py capture
echo DISCOVERY_EXIT=%ERRORLEVEL%
python es_arena_daily.py list
```

The capture tool prints the **actual discovery JSON path** and `RECORD SHA256`. Do not construct a filename by guesswork or reuse a previous day's ID. Ensure `STATUS: success`, `HTTP STATUS: 200`, actual ES daily challenge UUID and contract, terms and deadline. Failed or empty discovery proves neither absence of an earlier opportunity nor account ineligibility. An official `created_at` timestamp must **never** be silently substituted for the official opening timestamp `open_at_utc`.

For the **actual** path printed by the capture command (example variable intentionally not populated):

```bat
set "M11DISCOVERY=PASTE_ACTUAL_DISCOVERY_JSON_PATH_HERE"
if not exist "%M11DISCOVERY%" echo STOP - DISCOVERY FILE MISSING
certutil -hashfile "%M11DISCOVERY%" SHA256
for /f "delims=" %H in ('powershell -NoProfile -Command "(Get-FileHash -LiteralPath $env:M11DISCOVERY -Algorithm SHA256).Hash.ToLower()"') do @set "M11DISCOVERY_SHA=%H"
python phase_a_m11_stage4_prospective_gate_v0_1.py review-discovery --repo "%CD%" --private-root "%M11ROOT%" --journal "%M11JOURNAL%" --discovery-file "%M11DISCOVERY%" --expected-discovery-sha256 "%M11DISCOVERY_SHA%"
echo STAGE4_DISCOVERY_REVIEW_EXIT=%ERRORLEVEL%
```

**Do not run the commands with the placeholder path**. The source file must be inside the project repository or the private evidence root; do not upload secrets. A Stage 4 `PASS` means the *tool functioned*, not that any particular opportunity is certified. Inspect each `challenge_reviews[].review` value.

- `OFFICIAL_OPEN_TIME_NOT_EVIDENCED_CREATED_AT_NOT_OPEN_AT`: source did not establish actual opening. Retain a review gap and obtain official opening evidence; no retrospective opening inference.
- `OPENED_BEFORE_PROSPECTIVE_EFFECTIVE_TIME`: excluded from prospective enrollment regardless of when observed. If accepted, it can still count as a clearly described transition accepted forecast.
- `CANDIDATE_ACCOUNT_ELIGIBILITY_AND_TIMING_REVIEW_REQUIRED`: a real, post-effective official opening appears in the source, but that does not by itself certify account-specific eligibility, independently authenticated timing, or an accepted forecast. Review separately before any Stage 2 candidate journal CREATE.
- `POST_DEADLINE_OBSERVATION_NOT_PREDEADLINE_EVIDENCE`: retained source is not valid pre-deadline discovery evidence.
- `SUCCESS_EMPTY_FEED_COVERAGE_UNKNOWN`: no challenges returned; **not** evidence that zero eligible challenges opened earlier.

The actual launch may be after a weekend or market holiday. Do not infer a challenge merely from a weekday.

## 4. After genuine official validation — separate work, not automated here

Preserve original discovery JSON bytes and hash in the private evidence tree using a write-once, independently reviewed procedure, not by editing the source. Confirm the actual challenge's official UUID, actual `open_at_utc`, deadline, contract, neutral-band rules, account eligibility, the authentic pre-deadline observation, and any gaps. Only then prepare a **new event proposal** for the existing Stage 2 append-only journal, with the exact Stage 2 required fields and genuinely sourced status axes. Stage 2 `append` runs read-only by default and writes only on an explicit `--apply`; it never automatically certifies opportunity enrollment, accepted forecast status, or phase position. Do not use the historical 19 IDs in the prospective journal.

The existing separate, explicit one-submission procedure remains the only way to approve a real Arena scored POST. No submission authorization is conferred by Stage 4.

## 5. Tests and reproducibility

Run:

```bat
python -m unittest -v test_phase_a_m11_stage4_prospective_gate_v0_1.py
python -m unittest -q test_phase_a_m11_readiness_v0_1.py test_phase_a_m11_stage2_v0_1.py test_phase_a_m11_stage3_immutable_baseline_v0_1.py test_phase_a_m11_stage4_prospective_gate_v0_1.py
```

Stage 4 has 17 offline synthetic cases. The previous Stage 1/2/3 tested total is 81; local Phase A test total should be 98. The established full repository regression previously had 325 tests; adding the new module should give 342 if the same previous modules are selected. All synthetic discovery responses are fake fixtures, not real Arena observations.

## 6. Constraints

- Do not change, reset or stage Notebook 11 or Notebook 12, M1 joblib, licensed Databento files, Hugging Face or previous pinned evidence.
- Do not edit the adopted protocol, private adoption JSON, or sealed 19-row settlement snapshot.
- This gate does not independently certify the 2026-11-14 adoption timestamp, historical final scored revisions or a complete historical opportunity census.
- The existing `arena_opportunity_capture_v0_1.py` may use the authenticated API; this Stage 4 verifier itself makes **no** network request. Consult the existing v0.4 SOP before any live submission.
