# Milestone 11 — Stage 9: Unique ES session dates across local CSV snapshots

**Date:** 2026-11-15. **Scope:** local, offline, retrospective structural audit, *not* model/dataset certification.

## Purpose and interpretation

Stage 8 found 58 ES CSV candidates containing 3,761 structurally inspected rows, many in successively expanding, overlapping `es_sessions_YYYYMMDD_*.csv` snapshots. These must not be summed into independent training observations. Stage 9 inspects ONLY CSVs whose filenames exactly match the `es_sessions_YYYYMMDD_<16 hex>.csv` pattern; other reviewed CSVs are listed as excluded and are not assumed to be market sessions. It counts distinct **literal `session_date` values**, total repeated occurrences, and exact whole-row disagreements for dates appearing in multiple snapshots. The Stage 8 report and selected CSVs are SHA-256 checked against their original pinned evidence. Every selected file must have a complete Stage 8 scan, be readable within caps, and have the required dates and UTC session boundary columns. A mismatch stops rather than silently choosing a corrected file.

`COMPLETE_WITH_LIMITATIONS` (exit 0) means a structural comparison finished with compatible schemas and no conflicting versions. It **does not** certify that the dates correspond to valid exchange sessions, that prices or returns are correct, that labels were released before model fitting, that a historical vendor vintage was available at a particular cutoff, or that data rights allow training, public sharing, or transfer. `REVIEW_REQUIRED` (exit 3) is an informative finding: conflicting rows or different headers; inspect the private report before training. Errors such as changed files, unsafe paths, malformed timestamps or mismatched hashes return exit 2. A snapshot with no eligible files returns exit 3.

All market CSV values remain local; no network access, prediction, model inference or training occurs. The report is private-only and uses SHA-256 hashes of selected rows, not raw data values. The selected and excluded filename lists remain private. The 19 historical accepted Arena forecasts and frozen M1 remain unchanged. Research counter is still 19/150 until independently verified official evidence changes it.

## CMD commands

Use an activated `.venv` in `C:\Users\melny\Market-Mood-Forecasting`, with Stage 8 committed. The expected Stage 8 Git SHA is `a4db9430944ed98d46da010315747651adbf4538`.

```bat
cd /d "%USERPROFILE%\Market-Mood-Forecasting"
set "M11ROOT=%USERPROFILE%\MarketMood_Private\phase_a_m10"
set "S8REPORT=%M11ROOT%\m11_future\source_inventories\m11_stage8_es_csv_review_20261115_A.json"
set "S8SHA=014ddbe666777d0ed5bc6bbb69d7e4c89f8557f1e6be48f5de05f4ceebedd671"
certutil -hashfile "%S8REPORT%" SHA256
python phase_b_m11_stage9_unique_sessions_v0_1.py audit-sessions --repo "%CD%" --private-root "%M11ROOT%" --stage8-report "%S8REPORT%" --expected-stage8-sha256 "%S8SHA%"
echo STAGE9_AUDIT_EXIT=%ERRORLEVEL%
```

To write a *new* private snapshot, only when the preview has been inspected:

```bat
set "S9REPORT=%M11ROOT%\m11_future\source_inventories\m11_stage9_unique_sessions_20261115_A.json"
if exist "%S9REPORT%" echo STOP - STAGE9 REPORT ALREADY EXISTS
```

Only if report does NOT already exist:

```bat
python phase_b_m11_stage9_unique_sessions_v0_1.py audit-sessions --repo "%CD%" --private-root "%M11ROOT%" --stage8-report "%S8REPORT%" --expected-stage8-sha256 "%S8SHA%" --snapshot-file "%S9REPORT%"
echo STAGE9_REPORT_EXIT=%ERRORLEVEL%
certutil -hashfile "%S9REPORT%" SHA256
```

**Safety:** Do not stage private report files, source CSV, DBN vendor data, model joblib, notebooks, or historical evidence. Check test and real-audit results before exact-three-file commit. Tests use synthetic temporary data, not live Arena. There is intentionally no training dataset export or automatic canonical row selection.
