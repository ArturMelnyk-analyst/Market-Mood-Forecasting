# Milestone 11 Stage 7 — Phase B local evidence inventory

**Date:** 2026-11-15. **Scope:** local data-discovery audit, not training, model readiness, a dataset certification, a new Arena prediction, or an amendment to the adopted ES opportunity protocol.

## Exact upstream controls

- Accepted-forecast programme remains `ORIGINAL_150_TOTAL_DESCRIPTIVE`: historical checkpoint 19/150, Phase A to accepted #30, Phase B accepted #31–90, Phase C #91–150.
- Adopted v1.0 protocol `6f089df837e5dcd1aa5fc198490e287a98fa7fd8`, future-effective 2026-11-15T12:00:00Z, SHA-256 `a0eec9b6929397338ee69ee9a17e48cd4807e006c27463f8bd6508af576dd63d`.
- Stage 5 seven-source draft SHA-256 `cf5f318f36fde97d6ea769667d184bb4ddb9ce25c3bea2515643a74a2dc7076d`.
- Stage 6 draft chronological source contract SHA-256 `bf7cbd2339753efb6a82a42bcb8e08e56262f24d6d5a395974b8aaf77ad61c06`.
- Stage 6 published commit `fad6162a314ba086b8843654c6f52e9613d7e8f2`.
- Private owner decision and 19-row archived historical snapshot are physically SHA-pinned; failures stop the audit.

## What the scanner actually does

It inspects only **named local directories** under the repository and the single archived historical-baseline directory under the existing private root. The fixed scan roots include local ES Databento raw data, NB11 reviewed snapshots/audits, the NB10 frozen-model deployment directory, and conventional GDELT, VIX, Google Trends, FRED/unemployment and processed Phase B locations. The scanner examines names, sizes, filesystem mtimes and SHA-256 for candidate files no larger than the requested size limit. It does not parse values, count trading sessions, discover vendor accounts, query market data APIs, inspect the content of model pickle files, or scan your whole home directory.

**Directory-name classification is not proof of source identity or completeness.** A downloaded historical file is not proof that the original vintage was available before an old forecast cutoff. Local filesystem mtime is not vendor publication time. Any externally obtained data must still undergo license, vintage, release-time, schema, target-definition and data-leakage review under Stages 5–6. ES-derived data alone do not establish all 7 configurations' training readiness. Missing recognized directories are reported as unknown/missing evidence, **not** proof that data do not exist elsewhere.

The report intentionally sets external vendor source certification to 0, training dataset rows validated to 0, and models trained to 0 even when candidate files are found. No new physical data, Jupyter notebook, frozen M1 model, Hugging Face, original 19-row historical snapshot or Arena prediction is modified.

## Windows CMD — read-only audit

```bat
cd /d "%USERPROFILE%\Market-Mood-Forecasting"
set "M11ROOT=%USERPROFILE%\MarketMood_Private\phase_a_m10"
python phase_b_m11_stage7_source_inventory_v0_1.py scan --repo "%CD%" --private-root "%M11ROOT%"
echo S7_SCAN_EXIT=%ERRORLEVEL%
```

Expected: `M11_STAGE7_LOCAL_INVENTORY: PASS`, `S7_SCAN_EXIT=0`. **Real file counts will depend on your machine; they cannot be predicted in advance.** If a protected private hash mismatches or a scan root is a symlink, stop and preserve evidence rather than adjust the expected hash.

## Optional write-once PRIVATE inventory snapshot

Create one *new* private report folder (do not overwrite an existing report):

```bat
set "S7REPORTDIR=%M11ROOT%\m11_future\source_inventories"
if not exist "%S7REPORTDIR%" mkdir "%S7REPORTDIR%"
set "S7REPORT=%S7REPORTDIR%\m11_stage7_local_inventory_20261115_A.json"
if exist "%S7REPORT%" echo STOP - PRIVATE INVENTORY ALREADY EXISTS
```

Only when the selected filename does not exist:

```bat
python phase_b_m11_stage7_source_inventory_v0_1.py scan --repo "%CD%" --private-root "%M11ROOT%" --snapshot-file "%S7REPORT%"
echo S7_SNAPSHOT_EXIT=%ERRORLEVEL%
certutil -hashfile "%S7REPORT%" SHA256
```

The inventory report contains **full local file paths and size metadata** and stays outside Git in `%M11ROOT%`; do not publish the JSON, licensed source files or proprietary model binary. Snapshot creation uses exclusive mode, so an existing filename is refused. A local report timestamp is not an independent external timestamp proof.

Review candidate names locally without pasting unrestricted absolute paths publicly:

```bat
python -c "import json,os,pathlib; d=json.load(open(os.environ['S7REPORT'],encoding='utf-8')); print('FILES:',d['discovered_file_count']); [print(r['source_bucket_by_directory_only'],pathlib.Path(r['path']).name,r['size_bytes'],r['sha256_status']) for r in d['file_metadata'][:40]]"
```

If all external buckets are zero, that only means the **named scanned directories** contained no matching candidate files. It does not certify that those feeds are inaccessible or unavailable. Model training must still wait for verified ES feature/label rows, reproducible point-in-time folds and reviewed external-source rights.

## Next engineering step after reading actual inventory

Use the **genuine output** to decide which source location and data format should be handled by the first Phase B data-loader. The next implementation should validate actual ES source schema, coverage, contract/date consistency, market-session calendar, neutral-class label definition, and historical publication/vintage claims before generating any training fold. Keep M1 frozen and official until the adopted protocol explicitly authorizes a change.
