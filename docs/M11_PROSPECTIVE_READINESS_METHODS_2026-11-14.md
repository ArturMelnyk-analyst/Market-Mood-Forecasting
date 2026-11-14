# Milestone 11 — prospective ES opportunity-program readiness (2026-11-14)

**STATUS: CODE READINESS ONLY — CANDIDATE POLICY, NOT ADOPTED, NOT A PREREGISTRATION.**

## Historical boundary: do not contaminate the prospective trial

- Original Notebook 12 v0.3.2 measures **first 30 accepted forecasts** beginning 2026-10-20. It stays unchanged.
- Historical 19 observed Arena challenges and 19 official local settlements are pinned to the physical November 14 **post-November-13** M10C reports and settlement CSV. These are a historical evidence cohort, **not** the first 19 opportunities of any newly registered 150-opportunity experiment.
- The existing 2026-11-12 draft proposes **30/60/60 eligible ES daily challenges**. The post-start amendment cannot be backdated to 2026-10-20; historical coverage and phase ordinal remain unverified.
- The alternative 30/30/45/45 remains a distinct optional design and cannot be silently substituted. The research owner must make a dated **future-effective** adoption decision with a reviewed version, hash, actual publication evidence, and documented opening/first eligible challenge.

## M11 read-only module

`phase_a_m11_readiness_v0_1.py` checks the exact historical M10C physical hashes (integrated JSON, settlement JSON, official settlement CSV), source-hash linkage, 19 unique challenge identities and deadlines, all 19 local settlements, recorded 8/8/3 class distribution, empty review flags, and all *noncertification* assertions. It does not contact Arena or GitHub, stage Git files, send predictions, or load the M1 model.

The optional, private **candidate** JSONL opportunity ledger is validated separately. It does not automatically count or enroll a single observation in a 150-opportunity protocol, even if populated. Real adoption, independent publication timing, account eligibility, and opening/ordering evidence require human/external review; this tool never claims certification.

The candidate ledger is one JSON object per unique official ES daily challenge. Each record requires:

- `challenge_id` official UUID; `asset` = `ES`; `challenge_type` = `daily`; a frozen specific contract e.g. `ESZ6` or later valid ES delivery.
- Explicit UTC `open_at_utc`, `deadline_utc`, `first_observed_at_utc`; `opportunity_first_evidence_path` under the supplied private root, and its physical `opportunity_first_evidence_sha256`.
- Independent axes: `availability_status` (`CONFIRMED_OPEN`, `OBSERVED_BUT_TERMS_INCOMPLETE`, `CANCELLED_AFTER_OPEN`, `NEEDS_REVIEW`), `submission_status` (`ACCEPTED_SCORED`, `REJECTED_CONFIRMED`, `ATTEMPT_AMBIGUOUS`, `NO_ACCEPTED_RECORD_YET`, `MISSED_VERIFIED`, `UNKNOWN`), `outcome_status` (`OFFICIALLY_SETTLED`, `PENDING`, `CANCELLED_NO_OUTCOME`, `UNAVAILABLE`, `UNRESOLVED`), and `provenance_level` (`SOURCE_DISCOVERY_PREDEADLINE`, `OFFICIAL_ACCEPTED_WITH_TIME_LIMITATION`, `LEGACY_ACCEPTED`, `INCOMPLETE`, `UNRESOLVED`).
- `ACCEPTED_SCORED` requires a candidate `prediction_id` and `accepted_receipt_sha256` reference; it does not by itself certify the receipt identity, server acceptance time, or finally scored revision.
- `MISSED_VERIFIED` requires a candidate `miss_evidence_sha256` and explicit post-deadline `miss_reviewed_at_utc`; the submitted evidence and reviewer conclusion **still require independent manual scrutiny** and cannot be established by a SHA-256 alone.
- `OFFICIALLY_SETTLED` requires `official_label` and `official_settlement_evidence_sha256`. This validation does not itself retrieve/verify official Arena provenance.

A path under the private root and hash match is only evidence of **byte integrity**, not independent timing, account visibility, or truth of source claims. A server `Date` header is not a signed publication receipt. Future ledger phase ordinal remains unassigned.

## Operational order

1. Pin and preserve this draft and all M10C historical input evidence. Run the M11 offline read-only readiness audit.
2. Review the 2026-11-12 draft and 2026-11-13 M9 policy decisions, including 30/60/60 vs other phase splits; decide how to label retrospective history. Obtain Kopei's methodological feedback if possible.
3. Produce a separate **explicit adoption record** with genuine UTC date, reviewed text SHA-256, public source/independently checkable publication timing *if claimed*, and the first challenge opening strictly after adoption. The `M11` tool does **not** perform these steps or set `ADOPTED`.
4. Before every eligible future deadline, use the existing discovery capture, frozen ES-only M1 and five-minute operational guard. Preserve original sources and POST attempt/receipt even on failure. An unknown or ambiguous attempt is **not** a verified miss.
5. On settlement, import outcome separately into versioned derived data, keep the source and prior reports unchanged. Do not change Notebook 12 or Hugging Face.
6. Only after final policy adoption and future enrollment can a new opportunity-based scorecard count unique accepted, missed, failed, cancelled, pending and settled pairs toward **future 150**; no weekday-based invented IDs, no skipping losses to fill a quota.

## Source pins from verified local Windows logs

- M10 commit `3c35a3a13e017b87f1723ce50eccb858eeca41f3` (indicated by user; the M11 module records this as reference, not a Git server attestation).
- M10C integrated private JSON SHA-256 `033b513f662524dc84579d9934a6593b79fd3b70eccd601116a19d82c8d1f920`.
- M10C settlement private JSON SHA-256 `3f819eb2b422e02a95df5eb271b7dadf8770ee3f52223dc0709d854dce335be9`.
- Post-November-13 official-settlement CSV SHA-256 `093bc2aa57c7e87f073c2ff86b9fbe7ebd3cbb0b59d0491ee7d95b8c3aee68c4`.
- Post-November-13 peer CSV SHA-256 `9e278fcdfca84bb726042b6f91ffe7ac4258882fe402f29c72ae33002603e145`, **not consumed by M11 audit**; peers remain descriptive/comparison data.

**M11 acceptance criterion:** new synthetic tests pass and the real offline source-pin check prints `M11 PROSPECTIVE READINESS: OK` with 19/19 historical but **`DRAFT_NOT_ADOPTED`, `NEW_150_TRIAL_OPPORTUNITIES_ENROLLED: 0`**. This confirms readiness of the checks, not completion of adoption or independent timestamp commitments.
