# Phase A — ES Daily Opportunity Accounting Protocol Amendment v0.1 (prepared 2026-11-12)

**Status:** adoption-ready protocol amendment, **not yet publicly adopted or independently timestamped**. Its rules must be reviewed and committed before they can be described as prospectively fixed for later challenges. Prepared 2026-11-12; effective date is the documented date/time of adoption, not 2026-10-20. This document does not modify the frozen M1 deployment or Notebook 12.

## 1. Historical protocol versus proposed amendment

- **Existing published Notebook 12 v0.3.2:** evaluates the **first 30 accepted ES forecasts** from 2026-10-20. Keep this original descriptive cohort and its definition intact. Its existing optional `evaluation_calendar.csv` does not currently define the accepted cohort.
- **Proposed 150-opportunity research programme:** 30 Phase A + 60 Phase B + 60 Phase C **eligible ES daily challenge opportunities**, counting failures and missed operations rather than selectively extending phases until a desired number of successful submissions appears.
- This opportunity-based definition was clarified **after the experiment began**. It must therefore be documented as a **dated amendment**, not presented as prospectively preregistered on October 20. Report both populations and any discrepancy, with no selective removal.

## 2. Verified evidence checkpoint

As of 2026-11-12: the local candidate (`arena_opportunity_candidate_20261112.csv`, SHA-256 `620816c8161312508e7220c85a064776f60e6af4ae4acbeb1fd5a846b851a575`) contains 18 accepted ES challenge IDs, including one separately recovered legacy submission. Seventeen native records have corresponding forecast, attempt and receipt files with `counts_for_score=True`. The preserved October 20 challenge inspection (SHA-256 `d53ebca9bff1125a269f62a0965efb839f3833bc3d73edc0927f281f4d5afd47`) contains the original ES daily challenge. Two November 12 discovery captures occurred after that day's 15:00 UTC ES deadline and did **not** find ES in the active feed. None of this proves that the first 18 accepted records exhaust every eligible challenge opportunity. The contemporaneous official settlement snapshot contains 17 settled forecast outcomes; its contents may change in future imports.

## 3. Unit of analysis and evidence

The unit is the **distinct official Arena ES daily challenge ID**. One ID counts once regardless of revisions, number of models, discovery requests or forecast attempts. Do not create an opportunity solely from a weekday, trading date, market return, model forecast filename, or a newly generated local identifier.

Positive evidence of a challenge may come from (a) a captured official challenge description with matching ID, ES scope, daily type, terms and deadline; (b) an accepted official receipt tied to its challenge ID, which establishes an available scoring opportunity but not complete discovery coverage; or (c) other independently preserved official challenge or cancellation records whose provenance is documented. Preserve original source paths, hashes, timing evidence and ambiguity.

For new opportunities, capture the active feed **before** its submission deadline using the existing separate discovery logger. Record response timestamp/Date header, deadline, contract identity, status and any discrepancy with Arena UI. A server `Date` HTTP header is **not signed independent submission-time certification**. An empty feed proves only what was visible at that moment, not that no ES challenge existed earlier. Failed retrieval is not a successful empty feed.

## 4. Opportunity ledger and classifications

Track one challenge ID per row with:
- `challenge_id`, `asset`, `challenge_type`, `contract_symbol`, `settlement_date` (only if independently established), `deadline_utc`, `opportunity_first_evidence_path`, and its SHA-256.
- `availability_status`: `CONFIRMED_OPEN`, `OBSERVED_BUT_TERMS_INCOMPLETE`, `CANCELLED_AFTER_OPEN`, or `NEEDS_REVIEW`; separately track dates checked with no confirmed challenge as `UNKNOWN_COVERAGE`, without inventing an ID.
- `submission_status`: `ACCEPTED_SCORED`, `REJECTED_CONFIRMED`, `ATTEMPT_AMBIGUOUS`, `NO_ACCEPTED_RECORD_YET`, `MISSED_VERIFIED`, or `UNKNOWN`; make any `MISSED_VERIFIED` claim only after the deadline and with adequate challenge and server/operational evidence. Do **not** interpret a missing local receipt alone as a verified miss.
- `outcome_status`: `OFFICIALLY_SETTLED`, `PENDING`, `CANCELLED_NO_OUTCOME`, `UNAVAILABLE`, or `UNRESOLVED`.
- `provenance_level`: `SOURCE_DISCOVERY_PREDEADLINE`, `OFFICIAL_ACCEPTED_WITH_TIME_LIMITATION`, `LEGACY_ACCEPTED`, `INCOMPLETE`, or `UNRESOLVED` as applicable. Preserve server-acceptance timing as an **independent separate field**; never append UTC to an unzoned Arena `created_at`.

Do not collapse these dimensions into one label: a scored submission can be pending settlement, and a challenge can be observed without a submitted forecast.

## 5. Opportunity denominator and cancellation rule (amendment proposed for adoption)

For the proposed 30+60+60 *opportunity* programme, count each official ES daily challenge that became eligible/open for submission, once, **even if a model later fails, the submission is missed, or the challenge is later cancelled**. Mark cancelled-after-open separately as not evaluable for Brier loss. A challenge removed before ever opening for submissions is not an evidenced eligible opportunity. If the platform's actual cancellation rules conflict with this policy, record the discrepancy and issue a **prospective amendment before analysing outcomes**. No selectively replacing cancelled/missing opportunities with later ones to reach a preferred number of scored observations.

Assign each eligible ID chronologically using the documented official challenge opening/creation times, not based on the outcome, and use a stable, predeclared tie-break (challenge ID lexical order) for genuinely simultaneous eligible opens. Historical records with unknown ordering must remain flagged instead of being sorted as if reliably ordered. Freeze the boundary between A and B only after reviewing the opportunity evidence and documenting any unresolvable retrospective gaps. **Do not silently reclassify Notebook 12's 30 accepted records as 30 eligible opportunities.**

## 6. Evaluation and reporting

- Report denominators separately: eligible challenges, accepted scored submissions, confirmed misses/failures, pending/ambiguous attempts, cancellations, settled pairs, and independently UTC-supported timing subset.
- Use official Arena settlement labels as ground truth for Arena evaluation. Class probabilities remain native and full precision; multiclass Brier is primary. Provide classwise recalls, balanced accuracy, confusion matrix and simple baselines; no market-skill claims from a small preliminary cohort.
- Preserve the **original first-30-accepted** Notebook 12 analysis as a historical descriptive view. Produce a separately named **opportunity-based** report once evidence suffices. Explicitly disclose all post-start amendments.
- Never use later revisions to replace an earlier scored vector, fill missing challenges from weekdays, infer full opportunity coverage from an active-feed snapshot, or use unzoned server `created_at` as proof of UTC submission time.

## 7. Operational handoff

1. Before the official ES deadline, capture discovery evidence; confirm challenge ID, session/contract, neutral band and deadline.
2. Follow the frozen M1 preparation/submission workflow; preserve attempt and response even on failure, with a separately tested five-minute operational safety margin.
3. After the deadline, reconcile the discovery snapshot against attempts, receipts, and official Arena state. Do not convert unresolved to missed automatically.
4. On settlement, import official outcome separately and update derived **new** research reports without changing immutable source files.
5. Freeze Phase B data availability, training controls, comparison finalists, missingness policies, and selection criteria before its first eligible opportunity. Retain the original M1 reference.

## 8. Adoption and version control (new in v0.1)

**Adoption event:** The research owner may adopt this version by reviewing its provisions and making a version-specific Git commit before the first opportunity to which the amendment is prospectively applied. Record the resulting full commit SHA, UTC publication/receipt evidence (if available), and the first subsequently observed official challenge ID in an accompanying adoption note. Do not label a locally supplied Git author/committer timestamp as proof of external pre-deadline publication. Do not announce this document as already adopted merely because the file exists locally.

**Prospective scope:** Rules for recording future opportunities, failures, cancellations, and evidence availability apply only from the adoption event onward. Records from 2026-10-20 through adoption must continue to be disclosed as historical reconstruction, with the precise original evidence and limits. The amendment cannot independently establish a complete historical opportunity census or repair missing pre-deadline observations.

**Boundary rule:** The first 30 *eligible* Phase A challenges cannot be certified solely from the 18 accepted submissions. Until the pre-adoption opportunity ordering and coverage are resolved, mark Phase A ordinal assignments and the A/B boundary as **provisional**. Do not move to Phase B under a silently inferred "challenge #31" boundary. If an objective boundary cannot be reconstructed, openly document the unresolved gap and publish a **separately dated prospective boundary decision before the first Phase B opportunity**; preserve the original 30-accepted Notebook 12 cohort in parallel. Never choose a boundary based on observed predictive outcomes.

**Information and decision integrity:** Document any new ambiguity, challenge cancellation, outage, unclear contract, missing settlement, or submission-attempt uncertainty immediately. Use `UNKNOWN` or `NEEDS_REVIEW` when evidence is insufficient. Do not convert absence of a local file, a weekend, or an empty after-deadline active feed into evidence of a missed challenge. If a new rule is necessary, make a new numbered prospective amendment rather than overwriting this version.

**Privacy / public release:** The protocol, audit code, and synthetic tests are suitable candidates for review. Private source data, API credentials, full licensed Databento extracts, nonpublic submission artifacts, and raw authenticated server responses remain subject to existing repository exclusion and distribution restrictions. Verify `git status`, `git diff --cached`, and `.gitignore` before committing.

### Adoption record (complete only after actual publication)

| Field | Value |
|---|---|
| Prepared | 2026-11-12 |
| Adopted / published | **Not yet recorded** |
| Full adopting commit SHA | **Not yet recorded** |
| Independent publication-time evidence | **Not yet recorded** |
| First official challenge prospectively covered | **Not yet observed/recorded** |
| Historical opportunity census | **Not established** |
| Original Notebook 12 accepted-first-30 | **Unchanged** |

**Current confidence statement:** The existing 18 accepted records are supported. The complete historical eligible-opportunity census remains **unverified**. The read-only audit script improves accounting and makes future discrepancies visible; it does not solve retroactive independent time certification.
