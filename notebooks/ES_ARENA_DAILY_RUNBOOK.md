# Market Mood Forecasting — Headline Arena ES Research Runbook

**Operational procedure and prospective research plan · Version 2.1 · Updated 2026-10-28**

> **Scope:** Reproducible, human-supervised predictions for Headline Arena's **ES (E-mini S&P 500 futures)** challenge using the existing frozen Logistic Regression deployment. The original Market Mood Forecasting Gradio application remains unchanged. This document replaces the old runbook written specifically for the 2026-10-21 prediction.
>
> **Important:** The exact current challenge, contract, deadline, session calendar, data coverage, cost, and generated file hashes must be verified *on every run*. Dates and hashes in the historical example are **not** reusable authorization values.

## 1. Research objective and non-negotiable rules

The core prospective experiment is to submit **30 eligible, subsequently settled ES predictions** using the **same frozen Version 0.1 Logistic Regression model**, then evaluate predictive quality, matched-challenge external benchmarks, and a separately defined hypothetical investment strategy. Geopolitical/volatility analysis is exploratory.

**Rules that protect the experiment:**

1. Submit **at most one scored forecast per ES challenge**, before its official deadline. No retroactive forecasts or revisions after the outcome becomes known.
2. Do **not** retrain, change coefficients, replace the scaler, adjust features, tune the threshold, or modify the frozen source during the 30-forecast study. Preserve Notebooks 08–10 and the deployment's model identity.
3. Use **only the most recently completed and independently reviewed session** as the new day's input; do not inadvertently use any future or target-day settlement data.
4. Preserve existing Databento DBNs, candidate and reviewed audits, immutable snapshots, manifests, saved forecast JSONs, submission-attempt records, and receipts. Never overwrite them to make a forecast look better.
5. Make no duplicate Databento purchases. Quote first, approve an exact amount/hash, and download only when the required session is absent; reconcile ambiguous purchase attempts before trying again.
6. Record incorrect predictions alongside correct ones. Complete the predefined sample; do not selectively omit difficult days.
7. Do not confuse a **successful API submission** with an **accurate forecast** or with a **profitable trade**. These are three distinct measurements.
8. Keep private credentials and raw/licensed market data out of public Git history. The local evidence archive is not the same as the public research repository.

### Provider and code separation

| Component | Responsibility |
|---|---|
| Databento / Notebook 11 | Acquire and reconstruct source `GLBX.MDP3 / ohlcv-1m` data; review and export daily sessions |
| `carry_forward_es_reviews.py` | Verify previously audited rows and record human review of newly added sessions |
| Frozen Notebook 10 deployment | Calculate three-class probabilities from the verified session snapshot |
| `es_arena_daily.py` | Read the active Arena challenge, prepare a frozen-model forecast, and submit it once with explicit approval |
| Headline Arena | Accept, timestamp, score, and settle the official forecast |
| Notebooks 12–14 | Post-submission evaluation; **never** change an earlier forecast |

`es_arena_daily.py` is an Arena client, **not** the Databento purchase workflow. A successful `prepare` command never submits a forecast.

## 2. Current project status and file conventions

The following are historical checkpoints from the working experiment, **not guaranteed values for later runs**:

- Git branch: `experiment/headline-arena-es-daily`.
- Frozen deployment directory: `data/processed/es_arena_nb10_v0_1/deployment_ab4d5c14423352c7`.
- Frozen model file: `es_daily_lr_deployment_v0_1.joblib` within that directory.
- Training cutoff recorded for the deployment: **2026-09-30**; do not refit during this study.
- Most recently confirmed, completed input session for the Monday 2026-10-26 forecast: **Friday 2026-10-23**, contract `ESZ6`.
- Its immutable snapshot: `data/processed/es_arena_nb11_v0_1/snapshots/es_sessions_20261023_8cf115d4a9a4e666.csv` (60 daily rows in that snapshot).
- Its reviewed audit: `data/processed/es_arena_nb11_v0_1/audits/reviewed_ab9c0a6ade744429.csv`.
- Monday's authenticated ES challenge ID: `b0a7eb84-9b7d-422e-a85f-307ca8da23c9`; **never reuse on another day**.

The above records demonstrate the file naming conventions. On every later trading day, the session date, challenge ID, audit hash, and snapshot hash will change. **Never guess a new hash or ID.**

### Local paths to retain

```text
Market-Mood-Forecasting/
  es_arena_daily.py
  carry_forward_es_reviews.py
  notebooks/
    08_...ipynb                                # existing frozen historical pipeline
    09_...ipynb                                # existing frozen modeling
    10_es_arena_prospective_inference_v0_1.ipynb
    11_es_arena_incremental_data_v0_1.ipynb
    ES_ARENA_DAILY_RUNBOOK.md
    12_es_arena_prospective_evaluation_v0_1.ipynb       # evaluation notebook
    13_es_arena_paper_trading_roi_v0_1.ipynb            # planned
    14_es_arena_geopolitical_regimes_v0_1.ipynb         # planned
    ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_1.md   # methodology draft (freeze separately)
  data/raw/databento/...                    # keep local; licensed source records
  data/processed/es_arena_nb11_v0_1/
    audits/unreviewed_*.csv
    audits/reviewed_*.csv
    snapshots/es_sessions_YYYYMMDD_<hash>.csv
    snapshots/es_sessions_YYYYMMDD_<hash>.manifest.json
  data/processed/es_arena_live/
    forecasts/forecast_YYYYMMDD_<challenge_id>.json
    receipts/receipt_<challenge_id>.json
    ...                                     # submission-attempt / other local audit evidence
```

Some directory contents are local evidence, **not intended for public Git commits**. Future notebook filenames are a proposed convention and need not exist yet.

## 3. When to run and which session to use

- For the standard ES schedule, a trading session runs from **18:00 New York time on the previous calendar day** to **17:00 New York time on its settlement date**, with the normal last minute bar beginning at **16:59**. Always verify CME special hours, holidays, interruptions and the current contract.
- Normally submit **Monday through Friday**, only when the matching Arena challenge is offered. The Sunday-evening futures reopening starts **Monday's session**; it is not a separate Sunday prediction. After a weekend, Monday usually uses **Friday's completed session**.
- Headline Arena's usual US daily-futures submission deadline has been **10:00 New York time**. **Never assume the corresponding German clock time**: the US and Germany switch daylight-saving time on different dates. The live challenge deadline and `prepare` output are authoritative. For example, the 2026-10-26 API output showed `2026-10-26 14:00:00+00:00`, corresponding to **15:00 Berlin time** that day.
- The target ES futures session has already started by a typical German-midday submission. Our Version 0.1 model intentionally uses **previous completed sessions only**, not target-day intraday bars.
- If the platform does not offer a valid ES challenge, if the market is closed, or if required source data is incomplete, **record the reason and do not force a submission**.

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

A filename alone does **not** prove coverage. Verify the underlying requested UTC window and notebook audit output. If today's snapshot is already valid, skip to **Section 10**; do not purchase or rebuild it unnecessarily.

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

A successful submission is **not yet a correct prediction**. The final outcome and scoring appear only after the official settlement. Preserve model probabilities, source snapshot hash, challenge ID, submission time, response, receipt, and official outcome in the local evidence archive. If a screenshot or website label disagrees with local data, investigate rather than overwriting an existing record.

## 12. Audit the official target definition

For ES with a **±0.30%** band, a session's directional class is derived from its return relative to the published opening baseline:

- `bullish` if the return is above `+0.30%`;
- `bearish` if it is below `−0.30%`;
- `neutral` within the band, subject to the challenge's published boundary conventions.

Our session labels are constructed from Databento minute bars for the designated ET trading session. Arena's price sampling/resolution can differ slightly, especially near ±0.30%. **Never assume exact daily equality from a single matching example.**

In Notebook 12, reconcile the model's local open/close-based label with the **official settled Arena label** for each challenge, and record both return series, differences, threshold proximity, and any label disagreements. Use **official settled Arena outcomes** as ground truth for evaluating Arena submission accuracy. Never alter the frozen features or historical labels retroactively to conceal a mismatch.

## 13. Prospective research protocol — start building now, finalize later

**The primary observational target is 30 eligible settled ES forecasts from the frozen deployment.** Thirty is an initial diagnostic sample, **not** conclusive proof of skill or profitability. Record dates when the platform was closed, a forecast was missed, or an outcome is not yet settled; do not silently replace them with favorable days.

### Notebook 12 — `12_es_arena_prospective_evaluation_v0_1.ipynb`

**Develop the evaluation framework now** on the existing research branch, but declare the final evaluation method before the complete sample is observed. Clearly mark interim results as *preliminary*.

Required sections:

1. **Protocol and provenance:** frozen model fingerprint, observation window, eligibility, challenge/date/agent IDs, timestamped source data, receipts, raw vs. curated data lineage.
2. **Integrity:** duplicate challenges/submissions, one outcome per challenge, probability vector checks, deadline compliance, no post-outcome edits.
3. **Forecast quality:** accuracy, class prevalence, confusion matrix, per-class performance, average maximum probability/confidence, log loss when numerically safe, and **multiclass Brier score**.
4. **Calibration:** reliability diagrams/tables by class and uncertainty estimates; acknowledge that 30 forecasts produce very noisy calibration estimates.
5. **Predeclared baselines:** uniform 1/3 probabilities, **training-only** historical class frequencies, and any naive rule fixed before final evaluation. Do not estimate baseline priors using the evaluation outcomes.
6. **External Arena benchmarking:** same-challenge ES comparison versus other agents when legally and technically accessible; use identical settled challenge IDs, comparable valid timestamps and reported/scored outcomes. Show **matched sample size**. Separate other agents' all-market aggregate scores from ES-only matched performance. Do not assume a public agent listing exposes all probabilities.
7. **Target reconciliation:** Arena's official settlement/label versus locally reconstructed ES minute-bar label.
8. **Uncertainty and limitations:** bootstrap or other defensible intervals (with dependence and small-sample caveats), regime sensitivity, multiple comparisons, missing peer predictions and coverage bias.
9. **Results:** preliminary until the target cohort settles, final report afterward. Record adverse results plainly.

**Brier specification:** for a three-class one-hot outcome `y` and probabilities `p`, calculate `mean(sum((p - y)^2 across the three classes))`; if reporting Arena's own Brier score or skill metric, document its precise normalization/sign convention rather than assuming the numbers are identical.

### Notebook 13 — `13_es_arena_paper_trading_roi_v0_1.ipynb`

**Primary analysis: same-market ES economic utility.** The earlier draft suggested SPY/ETF fractional-share investing and hypothetical account balances of $100/$1,000/$10,000. **That design has been superseded for the primary study** to avoid changing instruments and conflating an account ROI with a contract's normalized return. The approved-for-review research design is documented in the separate **draft** [`ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_1.md`](ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_1.md). This link is to a protocol **draft**, not proof of Git commitment or prospective registration.

- Use the *exact* Arena ES futures contract and frozen saved classification (`bullish=long`, `bearish=short`, `neutral=flat`), without refitting or retrospectively tuning a confidence cutoff.
- Evaluate **post-acceptance tradeable windows**, **not** Arena's earlier recorded session open to resolver return. Proposed standard full-day window: **10:05–16:55 America/New_York**; verify market hours, source timestamp semantics and forecast receipt eligibility for every date.
- Compute gross and net ES points, per-contract hypothetical USD P&L, and **return on ES contract notional** under explicit execution-proxy and friction scenarios. **Do not report account ROI** without a fully specified account, margin, funding and liquidation model.
- Report neutral/flat days, missing execution data, time/contract mismatches, coverage, matched always-long and cash benchmarks, dispersion and limitations. Preserve negative results.
- October 20–28 trading results, whose outcomes were known by the October 28 methodology draft, are **historical exploratory QA**, not prospectively registered paper-trading returns. The future strategy's prospective cohort starts only after the actual protocol commit.

### Notebook 14 — `14_es_arena_geopolitical_regimes_v0_1.ipynb`

**Exploratory, descriptive and explicitly noncausal** volatility/event analysis, documented in the same [Notebook 13/14 research protocol](ES_ARENA_RESEARCH_PROTOCOL_NB13_NB14_v0_1.md):

- Use the most recent **VIX Index close published before each forecast**, not same-day information learned afterward; fixed draft buckets: low `<15`, medium `15–<25`, high `≥25`, and missing. Keep an optional ES realized-volatility robustness test strictly lagged.
- Build an independently sourced, dated event chronology. Preserve occurrence timestamps, first-publication timestamps, source URLs and `known_before_forecast` flags, explicitly separating news first available before versus after the submission.
- Compare Notebook 12 accuracy, unscaled multiclass Brier and forecast confidence across predeclared groups, with Notebook 13 economic outcomes only when the trade ledger is independently valid.
- Report sample sizes, class mix, missingness, small-sample uncertainty, competing macroeconomic drivers and multiple-comparison caveats. **Never claim geopolitical causality** or select event windows based on wrong predictions.

### Final integrated report

After the 30 **officially settled** forecasts: summarize protocol, sample coverage, model performance, matched Arena peers, Arena target alignment, hypothetical ROI, geopolitical sensitivity, negative findings, limitations and proposed Version 0.2 experiments. Do not change Version 0.1 to retroactively improve the 30-forecast record.

## 14. GitHub and PR plan — keep one research PR open

**Agreed workflow:** keep the existing branch `experiment/headline-arena-es-daily`; continue daily live forecasts; add Notebooks **12, 13 and 14** as they are developed; **do not merge the comprehensive research PR until the agreed 30-forecast study and final review are complete**. The original Gradio application stays unchanged. New commits pushed to the same source branch automatically update an already-open PR. A pushed branch is not itself proof that a PR exists or was merged.

### What belongs in public Git history

- Reviewed reusable `es_arena_daily.py` and `carry_forward_es_reviews.py` scripts and their safely documented dependencies.
- Current Notebook 11 code with **safe inert configuration** and **inspected notebook outputs**; frozen Notebooks 08–10 are not to be modified during the prospective study.
- This updated runbook, tests, public-methodology documentation, and the planned notebooks, with only suitably sanitized reproducibility examples.
- A final **sanitized** evaluation dataset or aggregated tables, **only** if platform terms, data licensing, and privacy/security checks permit distribution. Explain any private artifacts needed for complete reproduction.

### What should remain private/local

- `.env`, API tokens/keys, user registration details, private account/qualification JSON and secrets.
- Purchased/licensed Databento DBNs; raw source history and purchase/attempt logs.
- Human-review files, local snapshots/manifests, saved submission receipts, and any data whose publication rights or identifiers have not been checked.
- Unneeded one-off registration/qualification scripts until checked for dependencies and sensitive data. **Do not delete merely because filenames look similar.**

Before committing, restore Notebook 11's safe defaults:

```python
ENABLE_METADATA_QUOTE = False
ALLOW_PAID_DOWNLOAD = False
PAID_DOWNLOAD_CONFIRMATION = ""
APPROVED_QUOTE_SHA256 = ""
WRITE_UNREVIEWED_AUDIT_TEMPLATE = False
REVIEWED_AUDIT_CSV = ""
EXPORT_REVIEWED_SESSION_SNAPSHOT = False
EXPORT_CONFIRMATION = ""
```

Do not use `git add .`, `git clean -fd`, or `git reset --hard` as a shortcut for this cleanup. Review every staged file and notebook output for secrets and generated data.

### Safe CMD staging procedure

```bat
cd /d %USERPROFILE%\Market-Mood-Forecasting
git branch --show-current
git status --short
git ls-files -- .env
python -m py_compile es_arena_daily.py carry_forward_es_reviews.py
git diff --check

git add -- notebooks/11_es_arena_incremental_data_v0_1.ipynb
git add -- es_arena_daily.py
git add -- carry_forward_es_reviews.py
git add -- notebooks/ES_ARENA_DAILY_RUNBOOK.md

git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
```

**Stop and inspect `git diff --cached` before continuing.** Confirm no hardcoded credentials, private JSON, excessive notebook outputs, proprietary raw data, or accidental dependencies. If needed, unstage a specific file safely with `git restore --staged -- PATH_TO_FILE` and correct it locally. Review `.gitignore` separately for `.env`, Databento source, generated audits/snapshots/receipts, and Python/Jupyter caches; do not rewrite unrelated ignore rules.

Only after the staged changes have passed review:

```bat
git commit -m "Document and finalize safe ES daily forecasting workflow"
git push origin experiment/headline-arena-es-daily
git status
git log -1 --oneline
```

Commit Notebooks 12–14 separately as they become testable **on the same branch**. Preserve the local evidence store through the whole research period. Merging research into `main` later does **not** imply modifying the app's behavior.

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
- [ ] New result added to the prospectively collected record **when the official settlement becomes available**; no cherry-picking.

---

**Research principle:** A good experiment can show that a model is poorly calibrated, loses money, or performs worse than simpler baselines. Credibility comes from leakage controls, transparent prospective testing, matched comparisons, honest uncertainty, and immutable evidence—not from claiming every forecast is right.
