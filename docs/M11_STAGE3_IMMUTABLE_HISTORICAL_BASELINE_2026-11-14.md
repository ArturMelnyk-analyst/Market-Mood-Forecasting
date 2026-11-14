# M11 Stage 3 — Immutable historical settlement baseline (v0.1)

**Prepared 2026-11-14. Code-only, offline, read-only audit.** This is a separately versioned implementation under the already **owner-adopted, future-effective** M11 ES protocol v1.0. It does not amend protocol v1.0, make adoption effective early, submit predictions, load the M1 model, change notebooks, mutate data, or connect to a network.

## Background and exact sources

The earlier `phase_a_m11_readiness_v0_1.py` pins the *live* 19-row `data/processed/es_arena_research_v0_1/arena_official_settlements.csv` to SHA-256 `093bc2aa57c7e87f073c2ff86b9fbe7ebd3cbb0b59d0491ee7d95b8c3aee68c4`. It is an archival pre-adoption checker; it will fail after a valid twentieth row is added to the live CSV. **Do not patch its source hashes or mistake its historical `DRAFT_NOT_ADOPTED` display for the current research status.** That output reflects the state of its historical audit version only.

Stage 3 instead reads the sealed *private* 2026-11-14 19-row snapshot at:

`%USERPROFILE%\MarketMood_Private\phase_a_m10\m11_future\historical_baseline_20261114\arena_official_settlements_19rows_20261114.csv`

It checks the snapshot's original physical SHA-256 `093bc2aa57c7e87f073c2ff86b9fbe7ebd3cbb0b59d0491ee7d95b8c3aee68c4` and the two original fixed historical reports:

- `%USERPROFILE%\MarketMood_Private\phase_a_m10\m10c_integrated_post_nov13_20261114_A.json` — `033b513f662524dc84579d9934a6593b79fd3b70eccd601116a19d82c8d1f920`.
- `%USERPROFILE%\MarketMood_Private\phase_a_m10\m10c_settlement_post_nov13_20261114_A.json` — `3f819eb2b422e02a95df5eb271b7dadf8770ee3f52223dc0709d854dce335be9`.

It reuses the existing M11 historical report-consistency assertions and confirms exactly 19 unique official challenge UUIDs with labels 8 bearish, 8 bullish and 3 neutral. It does **not** claim a complete historical eligible-opportunity census, official final-scored revision verification or independent confirmation that the model produced a particular historical forecast.

Stage 3 also validates the byte-pinned owner adoption record `m11_future\adoption\owner_adoption_decision_20261114_v1_0.json`, SHA-256 `90d797e773e50a023b59ff6b30cd9637fd2dcf8f5d432066b138d6b927111c25`, and the adopted v1.0 protocol text SHA-256 `a0eec9b6929397338ee69ee9a17e48cd4807e006c27463f8bd6508af576dd63d`. This confirms local record consistency; it does **not** independently certify the owner's local clock or GitHub publication time.

## Mutable live CSV policy

When run normally, Stage 3 reads the current live official settlement CSV but **does not pin the live CSV to the original 19-row hash**. It requires all 19 baseline challenge IDs to remain present and every field in each baseline row to remain text-identical. Additional columns in the live CSV are permitted, as long as all historical fields remain present. Extra unique official ES settlements with valid UUIDs and recognized labels are counted as **unclassified extra rows**. Their inclusion does not automatically prove an accepted forecast, submitted model, or prospectively eligible opportunity. Only genuine future evidence and subsequent reconciliation can change those counters.

If historical rows legitimately need correction due to official revisions, **do not edit the frozen 19-row evidence** to make this audit pass. Log the discrepancy, preserve revised sources separately, use a dated correction/amendment process and introduce a versioned comparison policy instead. This deliberately conservative checker flags any historical value change, even formatting changes; it does not infer that differences are fraudulent.

The `--baseline-only` option validates all pinned private historical/adoption sources without requiring or reading the mutable live CSV. This is useful to isolate a future live-import conflict.

## Windows CMD commands (from existing `.venv` repository root)

```bat
cd /d "%USERPROFILE%\Market-Mood-Forecasting"
set "M11ROOT=%USERPROFILE%\MarketMood_Private\phase_a_m10"
python -m unittest -v test_phase_a_m11_stage3_immutable_baseline_v0_1.py
python phase_a_m11_stage3_immutable_baseline_v0_1.py --repo "%CD%" --private-root "%M11ROOT%"
python phase_a_m11_stage3_immutable_baseline_v0_1.py --repo "%CD%" --private-root "%M11ROOT%" --baseline-only
```

The normal output should include `M11_STAGE3_IMMUTABLE_BASELINE: PASS`, `HISTORICAL_19_SETTLEMENTS: VERIFIED` and `LIVE_TOTAL_SETTLEMENT_ROWS: 19` at the November 14 checkpoint (or more if legitimate settlements have been imported). A nonzero exit is an audit blocker to be investigated. No execution writes a report file; use shell redirection to create a **separate** report only after deliberate user review, never redirect over raw evidence.

For stdout JSON inspection add `--json`; do not assume printed stdout automatically establishes independent timestamp certification.

## Study-specific interpretation

- The descriptive accepted-forecast programme target remains **150**, including the original 19 accepted-evidenced cases; 131 additional accepted forecasts were required at the original November 14 checkpoint.
- Phase boundaries remain 1–30, 31–90, 91–150 **accepted** forecasts; the prospectively confirmed opportunity census is a separate, uncapped denominator.
- Owner approval was saved locally at `2026-11-14T16:55:34Z` with approved prospective effective time `2026-11-15T12:00:00Z`. Nothing in this tool assumes that a weekend or weekday creates an official challenge.
- Stage 3's reading of the adopted record does not itself enroll challenges, certify new forecasts or compute Brier scores; all output counters are conservative.
- Do not change original Notebook 11, Notebook 12, frozen M1, Hugging Face, M10 source hashes, Stage 1, Stage 2, historical snapshot or adoption record.

## Tests and reproducibility

19 synthetic tests (separate temp directories, **no genuine Arena/private source files**) include positive baseline, 20th additional settlement, additional live CSV column, historical label/deadline mutation, missing historical ID, duplicate or malformed ID, immutable source tampering, changed owner design, false certification, baseline-only operation, missing live CSV and no-write evidence. New code and tests are dependency-free Python standard library and reuse existing, tracked `phase_a_m11_readiness_v0_1.py` without editing it. Before selective Git staging and normal push, run the root test suite and review any unrelated local modifications.
