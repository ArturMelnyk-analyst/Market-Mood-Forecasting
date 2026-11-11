# Market Mood Forecasting — Headline Arena ES Research Runbook

**Operational procedure and prospective research plan · Version 2.2 · Updated 2026-11-11**

> **Scope:** Reproducible, human-supervised predictions for Headline Arena's **ES (E-mini S&P 500 futures)** challenge using the existing frozen Logistic Regression deployment. The original Market Mood Forecasting Gradio/Hugging Face application remains unchanged. This is an updated operating guide, not a retrospective amendment to any frozen forecast or already recorded protocol.
>
> **Important:** Verify the current challenge, contract, submission deadline, market calendar, data coverage, quote, and original hashes **on every run**. Historical dates, challenge IDs, quotes, and hashes are **never** reusable authorization values. All commands assume **Windows CMD**, run from the repository root, unless stated otherwise.
>
> **Source-of-truth hierarchy:** Native immutable forecast and receipt files support *what was submitted*; official Arena resolution supports *the outcome*; independent source and server-timing evidence governs *provenance certification*. A local receipt time is **not** a verified server timestamp.

## 1. Research objective, phases and non-negotiable rules

**Current Phase A:** Evaluate the original **frozen ES-only Logistic Regression M1** over an initial target of **30 official ES forecast submissions**, while independently recording *every eligible challenge opportunity* (including failures, missed submissions, cancellations and unavailable outcomes). Do not extend or selectively choose opportunities merely to collect 30 successes. Analyze accepted-and-settled outcomes separately from opportunity coverage and strict provenance.

**Subsequent research design (planned; not yet prospectively frozen):** After the baseline, **60 consecutive eligible ES challenge opportunities** for a prespecified paired model comparison (Phase B) and **60 subsequent eligible opportunities** for confirmation (Phase C). These are **120 additional challenge opportunities**, yielding up to 150 opportunities across the programme, not 150 guaranteed accepted/settled predictions. Preserve opportunity-level denominators and do not substitute favorable dates for failures.

**Rules that protect the experiment:**

1. Make **at most one official scored submission per ES challenge** through the controlled M1 pipeline; do not perform outcome-driven revisions or blind retries.
2. **Freeze M1** throughout Phase A and B: no refitting, coefficient/scaler changes, feature additions, threshold tuning, probability adjustments, or retrospective edits to its original inputs and forecasts. The frozen deployment remains the reference in Phase C even if a separately selected challenger becomes official, subject to Arena rules.
3. Read **previously completed and independently reviewed source sessions**; avoid target-day or future-market information in original M1 features.
4. Preserve original Databento DBNs, original/new audits, reviewed snapshots, manifests, forecast JSONs, pre-submission attempt records, server receipts and challenge evidence **immutably**.
5. Quote before purchasing Databento data. Use the same authorized collector for subsequent research models as permitted by provider limits; never assume seven independent robots are available.
6. Keep correct and incorrect predictions and missing/failed opportunities; do not cherry-pick, backfill, re-score or extend the sample without a separately documented protocol amendment.
7. Differentiate **forecast preparation**, **server-accepted official prediction**, **officially settled outcome**, **locally hash-consistent evidence**, **independently time-certified evidence**, and **hypothetical trading performance**.
8. Leave the **original Hugging Face/Gradio application unchanged**; it is not the daily ES submission mechanism.
9. Keep licensed market data, private receipts, keys, and potentially restricted Arena HTML out of public Git history. An output-cleared notebook **still requires a source-cell security review**.
10. Set an operational submission target at least **five minutes before the verified Arena deadline**. If an unresolved attempt already exists, never send a second POST merely to meet that buffer.

### Provider and code separation

| Component | Responsibility |
|---|---|
| Databento / Notebook 11 | Quote/acquire authorized `GLBX.MDP3 / ohlcv-1m` ES data; reconstruct, audit, and export completed-session snapshots |
| `carry_forward_es_reviews.py` | Verify prior human-reviewed fields and document the new session review |
| Frozen Notebook 10 deployment | Load the unchanged scaler/model and compute three-class probabilities from eligible completed sessions |
| `es_arena_daily.py` | Read challenge, prepare immutable M1 forecast, attempt one official submission only with explicit approval, preserve receipt |
| `verify_es_forecast_provenance.py` | Read-only, per-challenge artifact, snapshot, forecast, attempt and receipt consistency checks; strict timing disclosure |
| Notebook 12 v0.3.2 | Descriptive accepted-and-settled M1 evaluation **separate from** strict provenance certification |
| `import_arena_peer_day_v1_1.py` | Offline, checksum-pinned, revision-aware official-settlement and peer-evidence importer (local private sources) |
| Notebooks 13/14 | Separate **draft** economic-utility and regime studies; cannot modify previous forecasts |
| Headline Arena | Official submission acceptance, scoring and settlement; only authoritative platform evidence can resolve server-time conventions |

A successful `es_arena_daily.py prepare` does **not** submit a forecast. A matching SHA-256 proves consistency of bytes, **not independently that a model was executed**.

## 2. Current historical checkpoint and file conventions

**Research checkpoint, recorded 2026-11-11 (not a live status API):**

| Item | Historical observation |
|---|---|
| Git research branch | `experiment/headline-arena-es-daily` |
| Frozen ES-only deployment | `data/processed/es_arena_nb10_v0_1/deployment_ab4d5c14423352c7` |
| Original M1 model file | `es_daily_lr_deployment_v0_1.joblib` in that deployment |
| Original model SHA-256 (recorded) | `5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d` |
| Training cutoff | 2026-09-30; 3,835 feature-complete training rows; 12 lagged ES features |
| Accepted predictions | **17** (16 original native receipt-backed, one October 20 recovered legacy) |
| Locally imported official settlements | **16** through November 10 |
| Local eligible peer probability rows | **52**, across imported challenges; not 52 independent market outcomes |
| Preliminary main descriptive sample | **n=16**, accuracy **8/16 = 50.0%**; unscaled three-class Brier **0.621310**; log loss **1.026332** |
| Matched uniform-probability Brier | **0.666667**; historical training-frequency Brier approximately **0.675155** (same 16 outcomes) |
| Local native artifact and scored-receipt integrity | 16/16 accepted native records checked without reported contradictions |
| Strict independently timezone-certified native receipts | **0/16**: `response.created_at` has no explicit UTC offset; local `recorded_at_utc` does not certify the server clock |
| October 20 | **Legacy provenance limitation**, preserved separately |
| Evaluation opportunity-calendar rows | **0** as of this checkpoint; fill from evidence, never invent gaps or mark all days successfully covered |
| November 11 | Native forecast with local scored receipt present; **official outcome not included** in this 16-settlement checkpoint |

**Do not treat an updated Arena dashboard, a new receipt, or a newer local CSV as covered by this dated table.** Re-run the local audit and Notebook 12 for each new dated checkpoint, preserving the old outputs and source hashes. Preliminary Brier advantage over a uniform comparator does **not** demonstrate a robust signal or profitable strategy.

### Active files and protected evidence

```text
Market-Mood-Forecasting/
  es_arena_daily.py
  carry_forward_es_reviews.py
  verify_es_forecast_provenance.py
  test_verify_es_forecast_provenance.py
  import_arena_peer_day.py                  # legacy v1.0; keep for reproducibility, not new revised peers
  arena_nov_2026_backfill/
    import_arena_peer_day_v1_1.py          # revision-aware current importer
    batch_import_nov_2026.py              # dated backfill; preserve, not blanket-publication material
    test_november_backfill.py
    evidence/                              # PRIVATE source HTML
    peer_revision_history_audit.json      # PRIVATE raw revision history
  notebooks/
    08_es_daily_data_and_events.ipynb
    09_es_daily_walkforward_modeling.ipynb
    10_es_arena_prospective_inference_v0_1.ipynb
    11_es_arena_incremental_data_v0_1.ipynb
    12_es_arena_prospective_evaluation_v0_3_2_separated_evaluation_and_provenance.ipynb  # PRIVATE EXECUTED ACTIVE
    12_es_arena_prospective_evaluation_v0_3_2_public.ipynb                      # PUBLIC COPY, if created/reviewed
    13_es_arena_paper_trading_roi_v0_1.ipynb                                    # LOCAL / UNFINISHED
    ES_ARENA_DAILY_RUNBOOK.md
    ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_1.md     # preserve original 2026-10-28 draft
    ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_2.md     # new dated revised DRAFT (this update)
    archive/                                # private historical notebooks and checkpoints
  docs/ES_DAILY_RESEARCH_STATUS_2026-11-11.md
  data/raw/databento/                      # PRIVATE licensed source bytes
  data/raw/arena_peer_evidence/           # PRIVATE original downloaded Arena HTML
  data/processed/es_arena_nb11_v0_1/
    audits/unreviewed_*.csv
    audits/reviewed_*.csv
    snapshots/es_sessions_YYYYMMDD_<hash>.csv
    snapshots/es_sessions_YYYYMMDD_<hash>.manifest.json
  data/processed/es_arena_live/
    forecasts/forecast_YYYYMMDD_<challenge_id>.json
    receipts/receipt_<challenge_id>.json
    ...                                     # private original submission-attempt evidence
  data/processed/es_arena_research_v0_1/
    arena_official_settlements.csv          # private official outcome ledger
    arena_peer_forecasts.csv                # private peers ledger
    evaluation_calendar.csv                 # may exist with zero records; not proof of full coverage
  backups/                                  # PRIVATE dated import and notebook backups
```

Filenames are descriptive conventions, **not proof that every optional/planned file exists** in a particular checkout. The public Notebook 12 copy is created intentionally and inspected before Git staging; it is not the same as the executed local notebook. Earlier Notebook 12 v0.1 and v0.3 remain accessible through Git history/private archive; they are **not** the current evaluator.

### Historical immutable snapshot example

For the 2026-11-11 target, the independently reviewed previous session was 2026-11-10 (ESZ6), with snapshot basename `es_sessions_20261110_30bda8bf399f4aa7.csv`; Notebook 11 reported snapshot verification. **This example is not a substitute for checking the original snapshot/manifest on future dates**.

## 3. When to run and which session to use

- For the standard ES schedule, a trading session runs from **18:00 New York time on the previous calendar day** to **17:00 New York time on its settlement date**, with the normal last minute bar beginning at **16:59**. Always verify CME special hours, holidays, interruptions and the current contract.
- Normally submit **Monday through Friday**, only when the matching Arena challenge is offered. The Sunday-evening futures reopening starts **Monday's session**; it is not a separate Sunday prediction. After a weekend, Monday usually uses **Friday's completed session**.
- Headline Arena's usual US daily-futures submission deadline has been **10:00 New York time**. **Never assume the corresponding German clock time**: the US and Germany switch daylight-saving time on different dates. The live challenge deadline and `prepare` output are authoritative. For example, the 2026-10-26 API output showed `2026-10-26 14:00:00+00:00`, corresponding to **15:00 Berlin time** that day.
- The target ES futures session has already started by a typical German-midday submission. Our Version 0.1 model intentionally uses **previous completed sessions only**, not target-day intraday bars.
- If the platform does not offer a valid ES challenge, if the market is closed, or if required source data is incomplete, **record the reason and do not force a submission**.

### Deadline safety rule

Use the **live Arena deadline** (UTC-aware if supplied) and compute its local representation from a time-zone library, not from an assumed German clock offset. The internal **five-minute buffer** is operational policy, not an Arena acceptance-time guarantee. A late/missed opportunity must be recorded instead of performing unsafe retries.

## 4. Daily preflight — Windows Command Prompt

Open CMD (not PowerShell) and run:

```bat
cd /d %USERPROFILE%\Market-Mood-Forecasting
.venv\Scripts\activate
python -c "import pandas, numpy, databento, joblib, requests, dotenv; print('Environment OK')"
python es_arena_daily.py list
```

From `list`, identify the **new** item where `asset` is `ES`. Confirm in the official Arena UI:

- target settlement date;
- `contract_symbol` (historically `ESZ6`, but a roll can change this);
- `dead_zone_pct` (historically `0.3`, i.e., ±0.30%);
- official deadline and open status;
- correct agent/account and no prior scored submission for that challenge.

Set its **actual**, newly returned ID. This is a **template**; replace the placeholder before execution:

```bat
set "ES_ID=PASTE_THIS_DAYS_ACTUAL_ES_CHALLENGE_ID"
echo %ES_ID%
```

**Do not infer the ID from the preceding session or reuse a previous challenge's ID.**

Before downloading, check for an existing complete DBN, reviewed audit, verified snapshot, or forecast/receipt covering the required session:

```bat
dir /b "data\raw\databento\incremental_v0_1\*.dbn.zst"
dir /b "data\processed\es_arena_nb11_v0_1\audits\reviewed_*.csv"
dir /b "data\processed\es_arena_nb11_v0_1\snapshots\es_sessions_*.csv"
```

A filename alone does **not** prove coverage. Verify the underlying requested UTC window and notebook audit output. Confirm there is no prior submission or unresolved POST attempt for the **exact** challenge ID before beginning. If today's snapshot is already valid, skip to **Section 10**; do not purchase or rebuild it unnecessarily.

## 5. Notebook 11 — quote only (safe, no purchase)

Open `notebooks/11_es_arena_incremental_data_v0_1.ipynb`. Edit its **one existing user-configuration cell**, never a duplicate.

For a normal session with settlement date **D**, construct request timestamps from New York local time rather than incrementing an old UTC date blindly:

- Request **start**: **17:00 ET on the calendar day before D**, providing a one-hour pre-session buffer.
- Request **end**: **17:00 ET on D**.
- Convert both to UTC using timezone-aware timestamps and check DST / exchange schedule.

Example for **Friday, 2026-10-23** (historical illustration only):

```python
REQUEST_START_UTC = "2026-10-22T21:00:00Z"
REQUEST_END_UTC = "2026-10-23T21:00:00Z"

ENABLE_METADATA_QUOTE = True
ALLOW_PAID_DOWNLOAD = False
PAID_DOWNLOAD_CONFIRMATION = ""
APPROVED_QUOTE_SHA256 = ""

WRITE_UNREVIEWED_AUDIT_TEMPLATE = False
REVIEWED_AUDIT_CSV = ""
EXPORT_REVIEWED_SESSION_SNAPSHOT = False
EXPORT_CONFIRMATION = ""
```

Keep the historical-context DBN and all already verified incremental DBN paths unchanged. Save, restart the kernel, run all cells. Record:

```text
Estimated USD: <ACTUAL_QUOTE>
Estimated records: <ACTUAL_COUNT>
APPROVED_QUOTE_SHA256 to use AFTER YOUR REVIEW: <ACTUAL_64_CHAR_HEX_HASH>
THIS DID NOT PURCHASE DATA
```

Check the intended date, contract, number of records, cost, quote expiration, and whether the market data is already local. A metadata quote is **not** permission to spend money.

## 6. Purchase only if needed, explicitly authorized, exactly once

Only when the full session is missing, no earlier purchase for that request remains unresolved, the quote is valid and the quoted amount is acceptable:

```python
ENABLE_METADATA_QUOTE = False
ALLOW_PAID_DOWNLOAD = True
PAID_DOWNLOAD_CONFIRMATION = "I_AUTHORIZE_THIS_DATABENTO_PURCHASE_V0_1"
APPROVED_QUOTE_SHA256 = "PASTE_THE_NEW_EXACT_64_CHARACTER_QUOTE_HASH"
MAX_AUTHORIZED_COST_USD = 0.10  # EXAMPLE cap; set only to an amount actually approved

WRITE_UNREVIEWED_AUDIT_TEMPLATE = False
REVIEWED_AUDIT_CSV = ""
EXPORT_REVIEWED_SESSION_SNAPSHOT = False
EXPORT_CONFIRMATION = ""
```

Replace the quote-hash placeholder with today's actual hash. **Do not execute while the placeholder remains.** Leave the existing DBN input lists unchanged during the purchase. Save and run all cells **once**.

Require evidence such as:

```text
PAID DOWNLOAD COMPLETED; do not request again:
.../es_incremental_<actual_request_id>.dbn.zst
DBN SHA-256: ...
```

Record the exact new filename and hash. **Immediately** restore `ALLOW_PAID_DOWNLOAD = False`, empty the approval string and quote hash. If the call errors or times out after an attempted purchase, do not retry automatically: reconcile the notebook's purchase-attempt marker, vendor billing, and local file first.

## 7. Reconstruct, write the candidate audit, and inspect raw minutes

With purchases disabled, append only the **verified** new `.dbn.zst` path to `INCREMENTAL_DBN_FILES`. Preserve all earlier DBNs. Use the notebook's normal symbol-resolution procedure. Set:

```python
ENABLE_METADATA_QUOTE = False
ALLOW_PAID_DOWNLOAD = False
PAID_DOWNLOAD_CONFIRMATION = ""
APPROVED_QUOTE_SHA256 = ""
WRITE_UNREVIEWED_AUDIT_TEMPLATE = True
REVIEWED_AUDIT_CSV = ""
EXPORT_REVIEWED_SESSION_SNAPSHOT = False
EXPORT_CONFIRMATION = ""
```

Save, restart the kernel, run all cells. Check frozen historical overlap reconciliation, candidate row count and last date. A standard complete additional eligible session normally adds **one** row; never force the expected total if the calendar or data says otherwise. Record the newly printed `Candidate audit path`:

```text
data/processed/es_arena_nb11_v0_1/audits/unreviewed_<actual_hash>.csv
```

### Independent read-only minute checks

Use a **diagnostic cell in Notebook 11 only**, after `resolved_bars` and `session_candidates` exist. Update the two ET timestamps for the session under review:

```python
import pandas as pd
import numpy as np

ET = "America/New_York"
start = pd.Timestamp("2026-10-22 18:00:00", tz=ET)  # example: Oct 23 session
end   = pd.Timestamp("2026-10-23 17:00:00", tz=ET)

ny = resolved_bars.tz_convert(ET)
g = ny.loc[(ny.index >= start) & (ny.index < end)].copy()
expected = pd.date_range(start, end, freq="min", inclusive="left")
missing = expected.difference(g.index)
duplicates = g.index.duplicated().sum()
cols = ["open", "high", "low", "close", "volume"]

print("Observed / expected:", len(g), "/", len(expected))
print("First/last:", g.index.min(), g.index.max())
print("Missing/duplicate:", len(missing), duplicates)
print("Symbols:", g["raw_symbol"].value_counts().to_dict())
print("Finite:", np.isfinite(g[cols].to_numpy(dtype=float)).all())
print("High valid:", (g["high"] >= g[["open", "low", "close"]].max(axis=1)).all())
print("Low valid:", (g["low"] <= g[["open", "high", "close"]].min(axis=1)).all())
print("Prices positive:", (g[["open","high","low","close"]] > 0).all().all())
print("Volume nonnegative:", (g["volume"] >= 0).all())
print("OHLCV:", g[cols].head(2), g[cols].tail(2))
```

For an **ordinary uninterrupted 23-hour session**, expect 1,380 minute bars, no missing/duplicate timestamps, a single correct underlying ES contract, and valid OHLCV. **Do not assume all sessions are ordinary**. Inspect the candidate row's `review_flags`, coverage, gaps, symbol mixing, target eligibility and the relevant CME schedule. Do not assert eligibility or sign an attestation unless the actual evidence supports it.

## 8. Carry forward prior reviews — protect history

The source audit and the human-reviewed audit are **separate immutable files**. Seeing `unreviewed_*.csv` after a successful review is normal; Git staging does not turn `unreviewed` into `reviewed`.

Identify the newest candidate and the most recent **actual prior reviewed** file. Inspect row counts/dates to avoid selecting the wrong hash:

```bat
dir /b /o-d "data\processed\es_arena_nb11_v0_1\audits\unreviewed_*.csv"
dir /b "data\processed\es_arena_nb11_v0_1\audits\reviewed_*.csv"
python -c "from pathlib import Path; import pandas as pd; p=Path('data/processed/es_arena_nb11_v0_1/audits'); [(lambda d: print(f.name,'rows=',len(d),'last=',d['session_date'].iloc[-1]))(pd.read_csv(f)) for f in sorted(p.glob('unreviewed_*.csv'))]"
```

Set the two **real** paths, with no remaining placeholders:

```bat
set "OLD_AUDIT=data\processed\es_arena_nb11_v0_1\audits\reviewed_<PRIOR_HASH>.csv"
set "NEW_AUDIT=data\processed\es_arena_nb11_v0_1\audits\unreviewed_<NEW_HASH>.csv"
dir "%OLD_AUDIT%"
dir "%NEW_AUDIT%"
python carry_forward_es_reviews.py --old "%OLD_AUDIT%" --new "%NEW_AUDIT%"
```

The helper must confirm prior OHLCV, contract, diagnostics, coverage and eligibility fields are preserved; only the documented source-observation timestamps may advance. **Stop on unexpected historical differences.** Its prompts require a human to:

1. Enter `I_CONFIRMED_PRIOR_AUDITS` only after verifying prior evidence remains valid.
2. Inspect the new session's raw minute evidence and calendar, then enter `I_REVIEWED_THIS_SESSION` only truthfully.
3. Supply a reviewer identifier and a **specific, evidence-based** review note of at least 25 characters.

The output should be the corresponding `reviewed_<NEW_HASH>.csv`. Verify it exists and ends on the target source date. Do not edit the audit CSV by hand, rename a candidate, or fake a review.

## 9. Export and verify the immutable daily snapshot

Return to Notebook 11. Keep paid modes off and all validated DBN paths in place. Only after the newly reviewed CSV exists, set:

```python
ENABLE_METADATA_QUOTE = False
ALLOW_PAID_DOWNLOAD = False
PAID_DOWNLOAD_CONFIRMATION = ""
APPROVED_QUOTE_SHA256 = ""
WRITE_UNREVIEWED_AUDIT_TEMPLATE = False
REVIEWED_AUDIT_CSV = "data/processed/es_arena_nb11_v0_1/audits/reviewed_<NEW_HASH>.csv"
EXPORT_REVIEWED_SESSION_SNAPSHOT = True
EXPORT_CONFIRMATION = "I_REVIEWED_THE_ES_SESSIONS_V0_1"
```

Replace `<NEW_HASH>` with the **actual** candidate hash. Save, restart the kernel, run all cells. Require successful checks such as:

```text
FROZEN NB08 HISTORICAL OVERLAP: PASS
IMMUTABLE SESSION SNAPSHOT: CREATED  # or verified existing identical snapshot
MANIFEST: CREATED                    # or verified existing manifest
SNAPSHOT VERIFIED: True
NOTEBOOK 10 INPUT: .../es_sessions_YYYYMMDD_<actual_hash>.csv
```

Use the printed snapshot filename, not an old canonical file or a made-up hash. Keep the intentionally write-once `data/processed/es_arena_completed_sessions_v0_1.csv` untouched.

If exactly **one** verified CSV matches the source date, the following works in **interactive CMD** (in `.bat`, use `%%F`):

```bat
set "SNAPSHOT="
for %F in ("data\processed\es_arena_nb11_v0_1\snapshots\es_sessions_YYYYMMDD_*.csv") do @if exist "%~fF" set "SNAPSHOT=%~fF"
echo %SNAPSHOT%
dir "%SNAPSHOT%"
```

Replace `YYYYMMDD` with the **source session** date, not the forecast date. If more than one file matches, choose the exact filename verified in Notebook 11. Validate its row count, latest date, contract, model schema, and matching `.manifest.json`; the `prepare` script must still independently verify its hash and provenance.

## 10. Prepare the frozen-model forecast — no submission yet

Use the **actual** current challenge ID and the **verified** snapshot path. The known frozen deployment is:

```bat
set "MODEL_DIR=data\processed\es_arena_nb10_v0_1\deployment_ab4d5c14423352c7"
dir "%MODEL_DIR%\es_daily_lr_deployment_v0_1.joblib"
echo %ES_ID%
echo %SNAPSHOT%
echo %MODEL_DIR%
```

Below is a **historical command example for 2026-10-26**, not a command to reuse for another day:

```bat
python es_arena_daily.py prepare --challenge-id "%ES_ID%" --date 2026-10-26 --last-session 2026-10-23 --contract ESZ6 --sessions "%SNAPSHOT%" --model-dir "%MODEL_DIR%"
```

On future days, replace **both dates**, verify the **actual challenge contract**, and set the challenge ID from `list`. Successful output should indicate:

```text
FROZEN MODEL VERIFIED; NO SUBMISSION
Challenge: <CURRENT_CHALLENGE_ID> deadline: <OFFICIAL_SERVER_DEADLINE>
Source date: <LAST_COMPLETED_SESSION> contract: <ACTUAL_CONTRACT>
Model probabilities: {'bearish': ..., 'neutral': ..., 'bullish': ...}
Saved immutable forecast: .../forecast_YYYYMMDD_<ACTUAL_CHALLENGE_ID>.json
```

The probabilities must be produced by the saved model. The script's validation should enforce the immutable snapshot, original model SHA-256, feature order, date availability and live challenge. If a verification fails or the deadline has passed, **stop**. This step is **not** an Arena submission.

## 11. Inspect and submit exactly once

Check the actual forecast path printed by `prepare` and confirm the live Arena UI matches the target date, contract, neutral band, open status, and one chosen agent. The following expected filename convention is valid only if `prepare` printed it:

```bat
set "FORECAST=data\processed\es_arena_live\forecasts\forecast_YYYYMMDD_%ES_ID%.json"
dir "%FORECAST%"
type "%FORECAST%"
```

Replace `YYYYMMDD` with the **forecast date**. The full JSON must contain the intended challenge ID, source session, immutable-model identity and correct three-probability vector. If the generated filename differs, use the exact printed filename.

Execute a real submission **only after checking** the forecast and deadline:

```bat
python es_arena_daily.py submit --forecast "%FORECAST%" --confirm-id "%ES_ID%" --approve I_APPROVE_ONE_SCORED_SUBMISSION
```

**Do not retry blindly on a timeout.** The client records an immutable submission attempt before the POST; a successful response produces a receipt. Confirm a genuine acceptance such as:

```text
HTTP STATUS: 201
COUNTS FOR SCORE: True
SUCCESS: ONE SCORED FORECAST ACCEPTED; do not submit again.
```

Verify the receipt:

```bat
dir "data\processed\es_arena_live\receipts\receipt_%ES_ID%.json"
```

### Read-only native receipt / provenance check

After a real receipt exists, execute from the project root:

```bat
python verify_es_forecast_provenance.py --root . --challenge-id "%ES_ID%"
```

Record the per-challenge hash/source/receipt checks **and** the timing-provenance limitations. `INCOMPLETE` because Arena's `response.created_at` lacks a timezone is **not** equivalent to a failed hash; conversely, a local `recorded_at_utc` does not establish independently certified UTC server acceptance. Keep the original unzoned response untouched. `VERIFIED_LOCAL_CHAIN` (when reached) still does not independently prove executable model bytes ran or the provider's first-publication timestamp.

Do **not** rerun an old submission to fix an audit status. On an ambiguous attempted POST, reconcile server/account state first. If the live deadline is within five minutes, mark the opportunity as operationally late rather than initiating a fresh POST under the new policy.

A successful submission is **not yet a correct prediction**. The final outcome and scoring appear only after the official settlement. Preserve model probabilities, source snapshot hash, challenge ID, submission time, response, receipt, and official outcome in the local evidence archive. If a screenshot or website label disagrees with local data, investigate rather than overwriting an existing record.

## 12. Audit the official target definition

For ES with a **±0.30%** band, a session's directional class is derived from its return relative to the published opening baseline:

- `bullish` if the return is above `+0.30%`;
- `bearish` if it is below `−0.30%`;
- `neutral` within the band, subject to the challenge's published boundary conventions.

Our session labels are constructed from Databento minute bars for the designated ET trading session. Arena's price sampling/resolution can differ slightly, especially near ±0.30%. **Never assume exact daily equality from a single matching example.**

In Notebook 12, reconcile the model's local open/close-based label with the **official settled Arena label** for each challenge, and record both return series, differences, threshold proximity, and any label disagreements. Use **official settled Arena outcomes** as ground truth for evaluating Arena submission accuracy. Never alter the frozen features or historical labels retroactively to conceal a mismatch.

## 13. Evaluation, prospective comparison and separate follow-on studies

### Notebook 12 — current v0.3.2 evaluation

**Current local executed notebook:** `12_es_arena_prospective_evaluation_v0_3_2_separated_evaluation_and_provenance.ipynb`. A separate output-cleared `12_es_arena_prospective_evaluation_v0_3_2_public.ipynb` can be committed only after **source-cell and metadata inspection**. The former v0.1 evaluator is historical, **not** the active analysis.

Run `python verify_es_forecast_provenance.py --root .` before interpreting new checkpoint results. Notebook 12 must maintain at least three distinct views:

1. **Opportunity coverage:** all eligible ES challenge opportunities, including no-data, no-model, missed deadline, failed/ambiguous attempt, canceled, pending settlement and accepted cases. `evaluation_calendar.csv` is presently unfilled; reconstruct historical opportunities from actual challenge/attempt logs and mark unknowns, never fabricated success.
2. **Primary descriptive forecasting cohort:** original fixed M1 probabilities with **original hash-linked native scored receipts or explicitly flagged October 20 recovered evidence**, plus independently imported official Arena settlement outcomes. Report complete denominator, accuracy, **unscaled three-class Brier**, log loss, calibration, class prevalence, uniform/training-only priors, and matched peer comparisons. Do not suppress an Arena-accepted forecast merely because strict server-time certification is missing; disclose that limitation and run sensitivity analyses.
3. **Strict provenance subset:** independently supported model and source artifact hashes, forecast-to-receipt link, attempt evidence, chronological checks and **independently corroborated server UTC acceptance**. Arena server `created_at` without a timezone is not automatically UTC. The current strict timing-certified native subset is zero; this is **separate from** the 16 locally settlement-qualified descriptive observations.

Reconcile local ES minute-bar directional labels with the **official settled Arena label**; the official label controls platform forecasting evaluation. An observed 16/16 label match is a dated audit result, not a guarantee. Keep original full-precision forecast probabilities; never reconstruct them from rounded Arena cards. Preserve source-level peer revisions and use only each current scored card's eligible probability vector.

**Current checkpoint:** 17 accepted M1 records, 16 official outcomes, 52 locally eligible peer vectors; descriptive M1 accuracy 8/16, Brier 0.621310; legacy October 20 separately flagged; 16 native artifact/receipt-consistent, strict independent UTC timing not certified. These are **preliminary and time-bounded**.

### Phase A/B/C — baseline, model comparison and confirmation

| Phase | Opportunity window | Official scored submitter | Shadow/research models | Primary purpose |
|---|---|---|---|---|
| A | Initial 30 M1 target challenge opportunities | **Original frozen M1** | Optional research development strictly separate | Baseline prospective record; preserve acceptance and failure coverage |
| B | Next **60 eligible challenge opportunities** | **M1 remains official** | Seven separately versioned research configurations, with **no more than four predeclared primary finalists including M1** | Same-challenge, pre-deadline **paired** Brier selection under protocol frozen before Phase B |
| C | Next **60 eligible challenge opportunities** | A qualifying **frozen selected winner**, only if permitted by Arena rules; otherwise M1 remains official | Original unchanged M1 comparator and any prespecified secondary shadow models | Untouched prospective confirmation of selected strategy/system; if no qualified winner, report M1 stability / inconclusive comparison |

**Proposed shadow family, subject to point-in-time data rights and feature/protocol freeze:** M1 frozen ES-only; M2 ES+GDELT; M3 ES+VIX; M4 ES+GDELT+VIX; M5 M4+Google Trends; M6 M4+published unemployment vintage; M7 M4+Trends+unemployment. Names are design labels, not deployed implementations. One shared authorized Databento collector should serve compatible model inputs; no assumption of seven paid/vendor robots.

**Mandatory Phase B freeze items before its first deadline:** challenge eligibility and fixed 60-opportunity window (not 60 successful pairs), target definition, common cutoff and observable data vintages, training windows, preprocessing and model hashes, missing-data/failure fallback for **all** candidates, uniform and training-frequency baselines, at most four eligible finalists, primary **paired unscaled multiclass Brier loss difference**, significance/uncertainty procedure with time dependence (block bootstrap sensitivity), minimum meaningful benefit/tie rule, exact model selection and no-improvement/M1-retention rule, official submission attribution, public nonce-based pre-deadline commitments **including failures**, after-settlement reveals and required evidence. Do not claim Git timestamps alone certify historical pre-deadline publication.

**Training/freshness controls:** maintain the frozen M1 as the whole-system baseline and, where possible, a separately versioned matched-training **ES-only control** for isolating incremental value of new sources. A newer retrained ES model versus older frozen M1 confounds retraining benefit with extra-source benefit. New external source values require observation, first-available (where documented), local retrieval and pre-deadline cutoff timestamps. GDELT collection timing, cash VIX vs VIX futures, Google Trends normalization and unemployment release/vintage controls must be explicit.

**Statistical limits:** 60 shared challenge outcomes per phase, not 7×60 independent outcomes. Keep all failures and no-forecasts. Select a winner *once* after Phase B; freeze it *before* Phase C; do not mix selection and untouched confirmation as if all 120 formed an unbiased confirmatory sample. Avoid equating Arena platform score, market BSS and monetary ROI with independently computed three-class Brier skill.

The detailed future 60+60 selection protocol is **not yet claimed to be prospectively finalized or committed** by this runbook; create and freeze that separate document before Phase B. No model or source changes are authorized in current M1 inference by this plan.

### Notebook 13 — ES paper-trading economic utility (separate DRAFT)

`13_es_arena_paper_trading_roi_v0_1.ipynb` exists locally as **unfinished paper-trading research**, **not a completed profitability study and not part of the current public research release**. Its proposed methodology is documented in the dated **draft** [NB13/NB14 research protocol v0.2](ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_2.md), preserving the original v0.1 draft.

The existing proposal stays on the **exact contract-matched ES futures** instrument (not a misleading hypothetical $100 stock/ETF account), mapping original frozen `bullish=long`, `bearish=short`, `neutral=flat`. The predeclared *candidate* normal-hours simulation window is **10:05–16:55 America/New_York**, with an observed, accepted forecast before entry; use contract-specific price proxy/timestamp checks, explicit fees/slippage, and normalized ES notional returns. Its original 2026-10-28 draft already knew earlier outcomes, so those days cannot be retrospectively called prospective for this *strategy*. The revised draft does **not** retroactively create registration; only a genuine commit/freeze before later eligible windows can do that.

Do not substitute Arena's open-to-resolve market return for a post-acceptance executable trade, or claim a historical filled order, account ROI or model profitability without execution, financing and risk evidence.

### Notebook 14 — volatility / geopolitical regime study (separate DRAFT)

Notebook 14 remains **planned**, not assumed present or completed. Its proposed design is documented in the same [draft protocol v0.2](ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_2.md). Prior-published cash VIX Index levels, lagged ES volatility and independently timestamped geopolitical/macro events must be evaluated **as of the forecast time**. Classify uncertainty honestly and report descriptive subgroup results; do not infer geopolitical causation from small selected windows. Label historical annotations made after the outcome as retrospective.

### Final integrated reporting

After the original Phase A target cohort is complete, publish the dated M1 evaluation with the **exact opportunity/acceptance/settlement/provenance denominators**, baseline/peer comparisons, target alignment, missingness, uncertainty and negative findings. ROI and regime studies have their **own** protocol-freeze dates and eligible sample sizes, possibly shorter or zero; do not claim that they automatically cover all 30 M1 dates. The future 60+60 experiment is a separately prospective continuation, not an adjustment to the already observed M1 results.

## 14. GitHub, documentation and privacy — controlled publication

**Branch:** `experiment/headline-arena-es-daily`. The research branch can receive reviewed commits before the initial 30 opportunities finish; a branch push is **not** a merged PR or a prospective timestamp proof for already submitted predictions. Do **not** merge the comprehensive research work into the stable public application without separately reviewing scope and tests. Hugging Face remains unchanged.

**Current publication objective:** Make the new Notebook 12 **sanitized code**, provenance verifier, portable tests, revision-aware peer importer, revised runbook, dated research status and separately labeled *draft* NB13/NB14 protocol reviewable by Kopei. Preserve the historical v0.1 evaluator in Git history and/or private archive. Superseding a historical notebook on the current branch does **not** erase historical commits; a deliberate tracked deletion is acceptable only when its source history/backup has been checked.

### Public-release candidates — review individually

- `notebooks/12_es_arena_prospective_evaluation_v0_3_2_public.ipynb` — **output-cleared copy** with all code/Markdown/metadata independently inspected, **not** the private executed notebook.
- `verify_es_forecast_provenance.py`, `test_verify_es_forecast_provenance.py` — locally tested; explain private artifacts needed for full historical verification.
- `arena_nov_2026_backfill/import_arena_peer_day_v1_1.py` — revision-aware importer; public test fixtures require a separate rights review. Do not stage the entire backfill directory.
- `notebooks/ES_ARENA_DAILY_RUNBOOK.md` (this document), `notebooks/ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_2.md` (**DRAFT**, not preregistration), `docs/ES_DAILY_RESEARCH_STATUS_2026-11-11.md`, `.gitignore` and reviewed installation/runbook docs.
- Portable synthetic-only tests after manual fixture review; optional audited sanitized statistics/manifests when data rights and privacy permit.

### Keep local / private or defer

- Executed Notebook 12 and its historical checkpoints, private original receipt/attempt and Databento inputs, reviewed audits/snapshots, complete Arena authenticated HTML and raw peer revision history, `backups/` and archives.
- Modified local Notebook 11 until its code changes, output/secret cleanup and inert purchase flags are reviewed **separately**. Do not discard or stage it merely because Git reports `M`.
- Unfinished Notebook 13 ROI and planned Notebook 14; never portray incomplete strategy analysis as published profitability.
- Legacy root `import_arena_peer_day.py` and historical October backfill helpers; retain for compatibility/history until dependencies are reviewed.
- Any existing v1.2.0 Git-tracked public legacy datasets remain in their existing Git history; `.gitignore` does **not** untrack them. Do not rewrite their history as part of this ES release.

### Windows CMD — verify, test, selectively stage and push

From project root, **before staging**:

```bat
cd /d %USERPROFILE%\Market-Mood-Forecasting
git branch --show-current
git status --short --untracked-files=all
git ls-files -- data/raw data/processed backups
python -m py_compile verify_es_forecast_provenance.py
python -m unittest -v test_verify_es_forecast_provenance.py
python -m py_compile arena_nov_2026_backfill\import_arena_peer_day_v1_1.py
python arena_nov_2026_backfill\test_november_backfill.py
```

**Only after the published copies, source cells, private evidence and staged diff have been reviewed**, stage explicit paths (use the exact paths actually present; an absent public notebook must be created and sanitized first):

```bat
git add -- .gitignore
git add -- docs/ES_DAILY_RESEARCH_STATUS_2026-11-11.md
git add -- notebooks/ES_ARENA_DAILY_RUNBOOK.md
git add -- notebooks/ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_2.md
git add -- verify_es_forecast_provenance.py
git add -- test_verify_es_forecast_provenance.py
git add -- arena_nov_2026_backfill/import_arena_peer_day_v1_1.py
git add -- notebooks/12_es_arena_prospective_evaluation_v0_3_2_public.ipynb
```

An **intentional** deletion of the former tracked Notebook 12 v0.1 from the current branch can be staged separately only after confirming retained history/archive; do not indiscriminately stage all deletions. A prior Git commit, private local timestamp or unzoned Arena server response does **not** prove pre-deadline publication/acceptance.

```bat
git diff --cached --name-status
git diff --cached --check
git diff --cached --stat
git diff --cached
```

**STOP** if the staged set includes private HTML, `data/processed`, original Databento files, source forecasts/receipts, Notebook 11, Notebook 13, old private Notebook 12 outputs, archives, backups, hard-coded credentials, or unreviewed additions. Use `git restore --staged -- "EXACT_PATH"` to unstage unintended files without discarding them. Never use `git add .`, `git clean -fd`, `git reset --hard`, `git push --force`, or automatic overwrite to solve mixed directory status.

Commit/push **only after all checks pass**:

```bat
git commit -m "Document ES daily research audit and prospective protocol drafts"
git show --stat --oneline HEAD
git push origin experiment/headline-arena-es-daily
git rev-parse HEAD
git status -sb
```

Record the actual commit SHA and verify the branch on GitHub **before sending Kopei** the commit URL. A proposed protocol only becomes prospective for **later** eligible observations if its version/time were committed and frozen before those observations; this document is not itself such evidence.

## 15. End-of-day checklist

- [ ] Today's Arena ES challenge, agent, contract, dead zone and submission deadline were independently verified.
- [ ] Required prior session is complete, correct for the CME calendar, and not already purchased or processed.
- [ ] Paid-download permissions were **explicitly authorized once if needed**, then reset to `False`.
- [ ] Candidate audit exists; new raw minute evidence, coverage, symbols and anomalies were personally reviewed.
- [ ] Previous reviewed history reconciled without unexplained changes; new `reviewed_*.csv` exists.
- [ ] Notebook 11 reported `SNAPSHOT VERIFIED: True`; snapshot date, schema and manifest match.
- [ ] `prepare` verified the unchanged frozen model and saved the correct probabilities **before settlement and before deadline**.
- [ ] Saved forecast JSON was inspected; Arena `submit` was called **at most once**, with explicit approval.
- [ ] `HTTP STATUS: 201` and `COUNTS FOR SCORE: True` and official receipt were verified; otherwise unresolved submission is documented, not blindly retried.
- [ ] All immutable evidence retained locally; no secrets or licensed data accidentally added to Git.
- [ ] New result added to the prospectively collected record **only when the official settlement becomes available**; no cherry-picking.
- [ ] `verify_es_forecast_provenance.py` run read-only; failed and incomplete checks preserved without editing historical files.
- [ ] Internal five-minute submission safety buffer respected, or shortfall explicitly recorded as an operational limitation.
- [ ] Forecast existence, Arena acceptance, official settlement, descriptive eligibility and strict provenance reported **separately**.
- [ ] Eligible challenge opportunity calendar updated from actual evidence (including failed and missed opportunities), or omissions labeled unknown.
- [ ] Public Git staging remains restricted to deliberately reviewed source/docs; no private original evidence is staged.

---

**Research principle:** A good experiment can show that a model is poorly calibrated, loses money, or performs worse than simpler baselines. Credibility comes from leakage controls, transparent prospective testing, matched comparisons, honest uncertainty, and immutable evidence—not from claiming every forecast is right.

## 16. Document revision history and audit boundaries

| Document date | Version | Summary | Prospective effect |
|---|---|---|---|
| 2026-10-28 | 2.1 source version | Original frozen daily workflow, 30-M1 protocol and draft NB13/14 design | Preserve original historical interpretation |
| 2026-11-11 | **2.2 (this revision)** | Current v0.3.2 evaluator, original-model hash/receipt audit, revision-aware peer imports, 16-outcome checkpoint, five-minute buffer, planned 60+60 research and controlled Git publication | **Documentation update**, not a retroactive forecast or a freeze of the future Phase B/C selection protocol |

**The full-precision original files remain the evidence of historical submissions; the local report and public Git commit are later descriptions, not substitutes for historical pre-deadline commitments.**
