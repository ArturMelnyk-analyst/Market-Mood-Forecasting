# Milestone 11 — Stage 5: Phase B seven-configuration research registry (DRAFT)

**Prepared 2026-11-15. Status: DRAFT_NOT_FROZEN.** This is a public, source-only engineering milestone, not a prospectively frozen Phase B design, an adopted amendment, a model-training report, a new Arena submission, or a claim that a data vendor is licensed/available. All external source statuses remain NOT_VALIDATED.

## 1. Applicable adopted study

The **adopted** protocol is `docs/M11_PROSPECTIVE_OPPORTUNITY_PROTOCOL_v1_0.md`, commit `6f089df837e5dcd1aa5fc198490e287a98fa7fd8`, adopted privately on 2026-11-14T16:55:34Z with future-effective time 2026-11-15T12:00:00Z. The approved design is `ORIGINAL_150_TOTAL_DESCRIPTIVE`: first 19 accepted historical forecasts count toward the 150 accepted-forecast target but are not retrospectively enrolled as prospective opportunities. Phase A 1–30, B 31–90, C 91–150 are **accepted-count** positions, not fixed observed opportunity counts. Keep all future eligible opportunity outcomes in a distinct operational ledger, including missed, ambiguous, unsuccessful and cancelled records.

### Actual approved boundaries

- Phase B feature, candidates, artifacts, training controls, information sets, missingness, evidence, primary finalists and selection rules must be frozen **before the first official eligible opening after accepted position #30 is verified**. If #30 itself is disputed, a real prospective transition decision is needed; do not assert Phase B begins regardless.
- Phase C finalist and confirmatory procedure must be frozen **before the first official eligible opening after accepted position #90**.
- Today’s versioned *draft* contains no claim of Phase B freeze and does not change these boundaries.

## 2. Seven model CONFIGURATIONS (not seven proven/trained algorithms)

| ID | Inputs | Status |
| --- | --- | --- |
| M1 | ES futures | Existing frozen official Logistic Regression baseline; keep original private model and training |
| M2 | ES + GDELT | Planned shadow |
| M3 | ES + VIX | Planned shadow |
| M4 | ES + GDELT + VIX | Planned shadow |
| M5 | M4 + Google Trends | Planned shadow |
| M6 | M4 + unemployment | Planned shadow |
| M7 | M4 + Google Trends + unemployment | Planned shadow |

The associated **matched-training ES-only unweighted control** is an additional diagnostic control, not an eighth primary candidate model. It must use the same chronological training window and target definition as the challenger so improvements from retraining can be distinguished from the incremental contribution of alternative inputs. No S&P 500 index-data feed is added.

M1 continues official Arena submissions during Phase B under the existing one-submission human-approval SOP. Shadow predictions are **not** Arena-scored predictions. Do not switch the official submitter without a separately authorized prospective decision and verified Arena attribution rules.

## 3. Prospective point-in-time feature controls

The audit for a prospective prediction checks *supplied metadata*, not vendor truth. For each required source and each actual decision time (UTC), retain:

- `observation_end_utc`: final observation included in a feature.
- `source_publication_utc`: first documented source release or publication. No assumption that observation time equals publication time.
- `source_available_utc`: verified time that the relevant release/vintage was usable.
- `retrieved_at_utc`: evidence of actual collection time, before the forecast cutoff (for prospectively captured data).
- `source_revision_or_vintage`: exact vintage or release identifier; especially important for unemployment and Google Trends, which may be released late or revised/re-normalized.
- `provenance_sha256`: hash of privately preserved evidence; a hash cannot independently attest origin or acquisition time.
- `licensing_restriction`: preserve licensed data privately; do not publish vendor output or private joblib.
- `includes_target_date_settlement=false`: the actual target settlement must never be used as a feature.

All four UTC timestamps must be no later than the actual forecast decision cutoff. `publication <= available <= retrieved` must hold. Unverified publication or availability means **not cleared for that forecast**. A present-day download of a past release does not certify prior availability for a historical backtest. Backtesting additionally requires genuine historical vintages/availability calendars and purged chronological validation where appropriate; Stage 5 does **not** certify backtests.

## 4. Scientific comparison plan requiring later freeze

- Primary reference: the **unchanged** M1. Metric: **unscaled multiclass Brier loss** for three ordered Arena classes on common settled challenge IDs and honest pre-deadline forecast evidence. Lower is better.
- **At most four preselected primary finalists including M1**; the other registered configurations are exploratory. The finalist list is **currently empty**, not chosen from 19 historical outcomes or upcoming test outcomes.
- Predeclare before Phase B: model algorithms and hyperparameters, exact preprocessing/features, allowed data vintages, chronological training/validation splits, matched ES-only retraining control, missing model/input policy, per-model coverage/paired N, improvement threshold, tie/inconclusive fallback, dependence-aware block bootstrap and multiple comparison handling. Thresholds and rules are intentionally `null` in this draft.
- Phase C is new-data confirmatory evaluation of the **frozen** selected candidate against unchanged M1, with no Phase C outcome-driven re-selection; if there is no defensible winner, retain M1.
- Six shadows on one challenge do **not** create six independent market observations. Track all failures and missing vectors, not only successful shadow runs.

## 5. Stage 5 deliverables and controls

Files installed in the public repo by the no-overwrite installer:

1. `phase_b/phase_b_seven_configuration_registry_DRAFT_v0_1.json`
2. `phase_b_m11_stage5_registry_audit_v0_1.py`
3. `test_phase_b_m11_stage5_registry_audit_v0_1.py`
4. `docs/M11_STAGE5_PHASE_B_SEVEN_CONFIGURATION_READINESS_2026-11-15.md`

The audit has modes:

```bat
python phase_b_m11_stage5_registry_audit_v0_1.py audit-registry --registry "phase_b\phase_b_seven_configuration_registry_DRAFT_v0_1.json"
```

For future *actual sourced evidence* (no synthetic manifest should be added to the research journal):

```bat
python phase_b_m11_stage5_registry_audit_v0_1.py audit-input-evidence --registry "phase_b\phase_b_seven_configuration_registry_DRAFT_v0_1.json" --manifest "PATH_TO_ACTUAL_PRIVATE_EVIDENCE_JSON" --expected-manifest-sha256 "PINNED_REAL_SHA256" --model-id "M2" --decision-utc "REAL_FORECAST_DECISION_UTC"
```

It checks syntactic/source/time consistency with the **provided** JSON and expected physical hash, not official Arena account eligibility, independent timestamps, data licenses, artifact execution, forecasts, data accuracy, completeness, training quality, model profitability or Phase B freeze. An attacker able to forge both evidence bytes and expected hash can pass this local gate; independent corroboration remains necessary.

## 6. What Stage 5 explicitly does NOT do

No forecasts created or submitted. No candidate fitted; no external data downloaded. No changes to M1, Notebooks 11/12, Hugging Face, the private 19-row historical snapshot, adoption evidence or Stage 2–4 tools. No additional accepted forecasts. No new prospective challenge enrollment. No public exposure of licensed data or the private model. No assertion of prospective Phase B readiness until a separate real and independently reviewable freeze occurs before the proper phase boundary.
