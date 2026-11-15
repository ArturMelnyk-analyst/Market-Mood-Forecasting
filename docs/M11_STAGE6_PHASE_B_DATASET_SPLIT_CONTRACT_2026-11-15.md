# Milestone 11 Stage 6 — Draft dataset/split and point-in-time contract

**Date:** 2026-11-15. **Status: DRAFT, NOT FROZEN.** Research engineering only; this document does not amend the adopted opportunity protocol, train any model, approve any data vendor, certify point-in-time data availability, or establish an official ES opening.

## Adopted research boundary and inputs

- Adopted study design remains `ORIGINAL_150_TOTAL_DESCRIPTIVE`. Historical accepted checkpoint 19/150; future opportunity census and accepted forecast counts are separate.
- Adopted protocol is commit `6f089df837e5dcd1aa5fc198490e287a98fa7fd8` with prospective effective time 2026-11-15T12:00:00Z. Phase B is accepted positions 31–90 and must be prospectively frozen before the first verified eligible challenge opening following accepted #30. Phase C similarly follows #90.
- Stage 5 seven-source configuration registry is commit `52ed45faffc0cc932345aee9935d77ccadf49dad`, SHA-256 `cf5f318f36fde97d6ea769667d184bb4ddb9ce25c3bea2515643a74a2dc7076d`. This Stage 6 draft depends on those exact bytes. Six candidate configurations M2–M7, plus the matched ES-only training control, are the only permitted training subjects. M1 remains the **unchanged official frozen baseline**, and Stage 6 forbids M1 retraining.
- Do not add S&P 500 index data. Licensed ES and any other restricted inputs and joblib artifacts remain private.

## Why chronological folds and timestamped availability matter

Random splitting of overlapping daily time series can leak target or near-target information. Stage 6 requires nonoverlapping official challenge UUIDs; no duplicate target session across training, validation and test splits; strict chronological decision-time ordering between partitions; and no training/validation label used before its availability.

Two checkpoints separate model fitting, selection and evaluation:

1. **`fit_at_utc`** must be after all training labels were available but before the earliest validation decision. Do not use validation/test labels when fitting or computing feature normalizers.
2. **`selection_at_utc`** must be after validation labels were available but before the earliest test decision. The test split must never be used for tuning, model selection or feedback-driven revisions.

This validator inspects *supplied metadata*, not executable model training or the actual source history. A `PASS` result does not prove no train/test leakage in code. The **purge/embargo duration, exact fold dates/IDs, training labels and any hyperparameter tuning budget remain to be specified and approved** before Phase B's prospective model freeze. Its synthetic row minima (3/2/2) ensure only that tests have multiple samples; they are not scientific sample-size recommendations.

## Two different modes for data evidence

### `PROSPECTIVE_CAPTURED`

For genuinely captured observations, Stage 6 reuses Stage 5's source checks. Observation end, release/publication, availability and collection timestamps must all be known in UTC and no later than the forecast decision cutoff. The publication/availability/retrieval order must be internally consistent; a local source SHA, revision identifier and licensing status must be present, and target-date settlement leakage is forbidden. This is *metadata consistency*, not external certification.

### `HISTORICAL_RECONSTRUCTED`

For historical backtests, actual retrieval may happen after the historical forecast decision. The reconstructed manifest therefore permits later `retrieved_at_utc` **only in this explicitly marked mode**, while requiring the claimed original observation, publication and source availability times to precede the historical decision. Each source must remain marked `LOCAL_HISTORICAL_VINTAGE_REVIEW_REQUIRED`, and every successful audit still reports `historical_vintages_externally_certified: false`. A modern download or a local SHA is **not** sufficient to prove old vintages, publication times or account access. A separately documented vendor release calendar, archival original vintage and source-rights review are required before a scientifically defensible backtest.

### Manifest fields (future private data only)

`audit-folds` expects a **private** JSON file containing:

- `record_type: M11_STAGE6_CHRONOLOGICAL_FOLD_MANIFEST_V0_1`; `status: CANDIDATE_NOT_FROZEN`
- `model_id` one of M2–M7 or `MATCHED_TRAINING_ES_ONLY_UNWEIGHTED`
- `contract_sha256` actual physical Stage 6 draft contract hash
- `evidence_mode` one of the two modes above
- `fit_at_utc`, `selection_at_utc`
- `rows`: each has an authentic official `challenge_id` UUID; `decision_cutoff_utc`; `target_session` (`YYYY-MM-DD`); `split` (`train`, `validation`, `test`); `label_available_at_utc` (required for train/validation; test may be `null`); and `input_evidence` with exactly the sources from that model's registry entry.
- Each `input_evidence` item carries `source`, `observation_end_utc`, `source_publication_utc`, `source_available_utc`, `retrieved_at_utc`, `source_revision_or_vintage`, `provenance_sha256`, `licensing_restriction`, `includes_target_date_settlement: false`, and a mode-appropriate `asof_evidence_status`.

Real source evidence and genuine challenge UUIDs are necessary. **Do not create a fake/private real-work manifest merely to obtain a `PASS`.** Synthetic fixtures exist only in the test suite and are not written to the research journal.

## Commands

Validate this draft contract today (read-only):

```bat
python phase_b_m11_stage6_dataset_contract_audit_v0_1.py audit-contract --registry "phase_b\phase_b_seven_configuration_registry_DRAFT_v0_1.json" --contract "phase_b\phase_b_dataset_split_contract_DRAFT_v0_1.json"
```

Later, if privately collected actual and properly identified data are available (substitute genuine paths and hashes):

```bat
python phase_b_m11_stage6_dataset_contract_audit_v0_1.py audit-folds --registry "phase_b\phase_b_seven_configuration_registry_DRAFT_v0_1.json" --contract "phase_b\phase_b_dataset_split_contract_DRAFT_v0_1.json" --expected-contract-sha256 REAL_CONTRACT_SHA256 --fold-manifest "REAL_PRIVATE_FOLD_MANIFEST.json" --expected-fold-manifest-sha256 REAL_FOLD_MANIFEST_SHA256
```

The second command has placeholders and is **not** executable as printed.

## Explicit limitations

Stage 6 has no network access, no download or new vendor approval, no license verification, no model training, no official Arena submission, no creation of a candidate opportunity journal event and no physical data change. It does not define final purge rules or calculate actual folds. Even a successful later `audit-folds` only states that supplied metadata are internally consistent, not that external source timestamps, fairness of sample availability, historical vintages, hidden test non-use, model calibration or scientific significance are independently verified.

Next engineering work: source-by-source vendor/version/release-time inventory; immutable private evidence manifests; review actual historical time availability and settle purging/embargo/validation choice. Do that before model selection or Phase B freeze. Continue official M1 submissions only under the existing human-approved Arena SOP.
