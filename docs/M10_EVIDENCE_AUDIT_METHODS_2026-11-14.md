# Milestone 10 — Evidence audit methodology (historical cohort)

**Date:** 2026-11-14. **Branch:** `experiment/headline-arena-es-daily`.

This document describes the 19 retained, acceptance-evidenced ES challenges already observed between 2026-10-20 and 2026-11-13. This is **not** an adopted prospective 150-eligible-opportunity protocol, and no historical observations have been reclassified as prospective Phase A.

## Evidence layers and reproducibility

- M10B: public `prediction-opportunity-audit` pinned response, 19 retained challenge UUIDs, no continuation cursor, incomplete historical heartbeat coverage, two coverage gaps; 19/19 historical public-opening, reader-delivery, and account-eligibility statuses **unknown**. Public observation does not imply personal access.
- M10A: authenticated own-prediction retained revision GET readbacks, captured separately and hash-pinned outside Git; 19/19 revisions have UTC `submitted_at` before the **deadline recorded in the later public snapshot**. 17 satisfy more-than-five-minute safety buffer; 2 do not (2026-10-20: 247.305982 seconds; 2026-11-06: 0.069164 seconds). Do not substitute parent `created_at` for missing server times; a future/current deadline snapshot does not itself certify historical deadline immutability.
- M10C: cross-source read-only reconciliation matched the 19 UUID sets; for 18 challenges, retained probability vectors **numerically** matched local frozen-M1 forecast file values and acceptance-receipt prediction IDs linked. For the earliest 2026-10-20 challenge there is no verified local vector or receipt: neither a match nor a mismatch may be claimed. Matching values and stored model hashes do **not** prove model execution.
- Official settlement CSV join: 18 local rows (8 bullish, 7 bearish, 3 neutral). Stored deadlines agree with the retained public snapshot for those 18. Five local rows have unknown settlement timestamps. The 2026-11-13 challenge lacks a row in the **local CSV**; do not infer that the server round was unresolved, or that missing evidence is a failure. The actual final scored revision identity is **not** certified by these sources.
- All raw authenticated responses, local full-precision vectors, receipts, and individual private summaries remain outside Git. These audit source files are pinned by SHA-256 in the source modules and in private manifests.

## Immutable source/checkpoint hashes (SHA-256 of actual bytes)

| Evidence | SHA-256 |
|---|---|
| Public opportunity audit snapshot | `5a4bc95d600dbd1af49cb566cebd4f91644ff62ede678f818b98f4ea071ca2ad` |
| M10B private checkpoint | `eba38ac15dd4b9d011f62dab58c51186d1b81505c97878395ef646bab2b06c8f` |
| M10A single private checkpoint | `43d136992bef1f2d7bff812ad5400bba60f00d1aa4dfd6c9b329b6ae5db78532` |
| M10A batch private checkpoint (Windows physical bytes) | `ca9bb6cd43d1be2d3015ad264287b3f978bd3388fc20f148c3c5b13fe1a43170` |
| M10C integrated private checkpoint | `6735ba058f873bbff09844978fc16263e34765878eaca0515985f25780d2afbb` |
| Local settlement CSV | `5aa88af67a3d5345dbd13a904290b7f755a50af9777e0190bf7bb5a51d2b351e` |
| M10C settlement private checkpoint | `251f1c267d06813cc92eddfe25f3650aa468f23bb84492a487ff177f2aaa1d74` |

The batch writer is now repaired **prospectively**: future summaries print SHA-256 of the exact bytes written, including on Windows. Do not overwrite the prior batch report, or change its recorded physical SHA-256 to the LF-normalized hash.

## Limitations and next research steps

- 19/19 applies only to observed retained, acceptance-evidenced challenge IDs; it is **not an exhaustive eligible-opportunity census**.
- Server-timestamp evidence does not certify frozen M1 execution, score attribution, account visibility, or settlement outcomes beyond those present in the local sources.
- Frozen M1, training cutoff, original Notebook 12 first-30 accepted cohort, historical predictions, original Hugging Face app and 150-opportunity protocol remain unchanged.
- For future prospective 30/60/60: publish and adopt a dated amendment, independently timestamp salted shadow commitments **before** deadlines, freeze challenger and missing-pair policy, and preserve prospective opportunity/reader observation evidence independently of prediction submission events.
- Before publishing the commit, rerun all 114 new/M10 tests and the existing project regression suite, check Git staging exactly, and commit only the reviewed code and methodological document.
