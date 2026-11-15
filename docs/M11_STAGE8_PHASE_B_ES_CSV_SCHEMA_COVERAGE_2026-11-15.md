# M11 Stage 8 — Offline ES CSV schema and date-coverage inspection

**Date:** 2026-11-15. **Status:** candidate file inspection, NOT training-dataset certification.

## Evidence and protected boundaries

- Accepted forecast programme `ORIGINAL_150_TOTAL_DESCRIPTIVE` remains 19/150 as last verified in Stage 7; Stage 8 does not fetch settlements or update the count.
- Frozen M1 is the only official submitter. Stage 8 performs no inference, submission, journal changes, Jupyter modifications, model training or external network requests.
- Published Stage 5 seven-config draft registry and Stage 6 chronological contract remain SHA-256 pinned; Stage 7 module and the owner-adoption decision/19-settlement archive are checked via the existing Stage 7 validator.
- This validator takes the **private** Stage 7 snapshot file and its actual physical SHA-256 as input. Do not put that private snapshot into Git.

## What counts as inspected

A Stage 7 `ES` candidate is only directory-classified, not a verified ES instrument. The auditor distinguishes `PURCHASE_ATTEMPT_*.json` (purchase metadata), compressed `.dbn.zst` / `.dbn` (not parsed), CSV files, and other artifacts.

For candidate CSVs only, it checks that the file has not changed since the Stage 7 snapshot (when hashed), bounds the file bytes and scanned row count, checks duplicate/blank CSV headers and row widths, fingerprints column names and summarizes **literal date strings** for commonly named fields (e.g. `session_date` or `ts_event`). Date ranges are **not** exchange calendar coverage; time strings without explicit offsets are marked timezone-unverified. Unknown/numeric timestamp formats are not silently interpreted. Neither the presence of a `label` column nor a parsed date proves the label definition, point-in-time source availability, license, or economic meaning.

**Security:** Stage 7 inventory paths must resolve under approved ES directories inside the project; private files outside the designated inventory directory are not parsed. The audit does not read arbitrary non-ES files, DBN content, model joblibs or third-party credentials. Reports containing local file paths/column metadata must stay private.

**Audit result:** `PASS_WITH_LIMITATIONS` means CSV structural review completed to the configured caps and no detected structural issue requires stopping. `REVIEW_REQUIRED` means one or more selected CSVs were missing, changed, malformed, etc. Both retain `training_rows_certified=0`, historical vintage certification `false`, and model training `0`. Missing data or skipped large CSVs remain unresolved, not evidence of absence.

## Commands (Windows CMD)

Set the private inventory and its already-confirmed SHA-256:

```bat
cd /d "%USERPROFILE%\Market-Mood-Forecasting"
set "M11ROOT=%USERPROFILE%\MarketMood_Private\phase_a_m10"
set "S7REPORT=%M11ROOT%\m11_future\source_inventories\m11_stage7_local_inventory_20261115_A.json"
set "S7SHA=269d9587c328fe62202f559128ec9cb707ef975dd584d3804c15185680e05c4d"
certutil -hashfile "%S7REPORT%" SHA256
```

If the reported digest differs, stop. Then inspect **read-only**:

```bat
python phase_b_m11_stage8_es_csv_coverage_v0_1.py audit-inventory --repo "%CD%" --private-root "%M11ROOT%" --inventory-file "%S7REPORT%" --expected-inventory-sha256 "%S7SHA%"
echo STAGE8_SCAN_EXIT=%ERRORLEVEL%
```

The output must be reviewed rather than assuming PASS; source counts depend on your actual files. If `REVIEW_REQUIRED` with exit `3`, preserve files, investigate which CSV is malformed/missing/drifted, and do not modify archived snapshots. An inventory hash error exits `2`.

To write a **new, write-once private** report (no overwrite):

```bat
set "S8REPORT=%M11ROOT%\m11_future\source_inventories\m11_stage8_es_csv_review_20261115_A.json"
if exist "%S8REPORT%" echo STOP - STAGE8 REPORT ALREADY EXISTS
```

Only when the chosen path does **not** exist:

```bat
python phase_b_m11_stage8_es_csv_coverage_v0_1.py audit-inventory --repo "%CD%" --private-root "%M11ROOT%" --inventory-file "%S7REPORT%" --expected-inventory-sha256 "%S7SHA%" --snapshot-file "%S8REPORT%"
echo STAGE8_REPORT_EXIT=%ERRORLEVEL%
certutil -hashfile "%S8REPORT%" SHA256
```

Review only a **redacted summary** in the chat; do not paste private full paths, proprietary data, the full JSON or source content. Optional limited summary:

```bat
python -c "import json,os,pathlib; d=json.load(open(os.environ['S8REPORT'],encoding='utf-8')); print('STATUS:',d['status']); print('CATEGORIES:',d['category_counts']); print('CSV_STATUS:',d['csv_status_counts']); print('STRUCTURAL_ROWS_NOT_VALIDATED:',d['rows_structurally_inspected_not_validated']); [print(pathlib.Path(r['relative_path']).name, r['inspection_status'],r.get('rows_inspected'),r.get('date_column_observations',{})) for r in d['csv_details_private_only'][:20]]"
```

Do not treat CSV `rows_inspected` as valid training labels or independently evidenced market sessions. Stage 9 should review a **specific** verified CSV/manifest pair, its true session calendar, target construction, release-time provenance and licensing before constructing the first actual fold.
