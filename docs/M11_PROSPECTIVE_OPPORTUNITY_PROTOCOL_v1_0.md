# Market-Mood-Forecasting — ES Daily 150-Total Accepted-Forecast Protocol v1.0

**Prepared:** 2026-11-14

**Document status:** PROPOSED FOR OWNER REVIEW — **NOT ADOPTED; NOT A RETROSPECTIVE PREREGISTRATION**

**Candidate design:** `ORIGINAL_150_TOTAL_DESCRIPTIVE` — 150 accepted ES forecasts in all, including the original 19; accepted-count phases 30/60/60

**Effective date:** None has been adopted or certified. To be set in a separate genuine, future-effective adoption record.

**Research scope:** Headline Arena's official E-mini S&P 500 futures (ES) daily three-class probabilistic forecasting challenges.

**Pre-adoption design clarification:** This document retains the project's original overall target of 150 accepted forecasts including the first 19. An earlier published, unadopted proposal explored a separate `FRESH_150_OPPORTUNITIES` design; that alternative is not selected. This clarification predates any formal adoption and does not imply historical preregistration. Preserve the distinction when reporting the project's research provenance.

**Owner decision:** The design below must be reviewed and affirmatively adopted in a separate, future-effective record. Saving, hashing, committing, or publishing this file is not adoption.

## 1. Purpose and primary research questions

This research evaluates (1) the quality and calibration of pre-deadline probability forecasts for the official `bearish`, `neutral`, and `bullish` Arena ES daily outcomes; (2) the reliability of the operational forecasting pipeline; and (3), when prospectively and comparably evidenced, differences between a frozen baseline and candidate models. The study does not claim that predictive accuracy implies profitable trading or that the resulting sample represents all ES trading environments.

**Primary probabilistic endpoint:** Mean **unnormalized three-class multiclass Brier score** among observations with a verified, eligible, pre-deadline scored probability vector and an official settlement. For classes `k in {bearish, neutral, bullish}`, the per-challenge score is `sum_k (p_k - 1[y=k])^2` (lower is better; range 0–2). Do not confuse this with the alternative class-averaged Brier convention, which divides by three. Report the formula and number of evaluable observations every time.

**Co-primary operational reporting, separate from predictive skill:** Percentage of eligible challenges with a verified scored submission; confirmed missed/rejected counts; ambiguous attempts; cancellations; unobserved or unknown-coverage periods; and the fraction with independently supported pre-deadline timing. Operational failures are **not** assigned an invented Brier score.

Secondary descriptive measures: classwise recall, confusion matrix, balanced accuracy, accuracy, calibration by confidence range when sample size permits, comparison with explicitly computed uniform probabilities (`1/3` per class), and uncertainty intervals or exploratory paired comparisons where appropriate. Arena's own leaderboard score is reported separately and is not substituted for the fixed Brier definition.

## 2. Overall 150-target programme and historical-to-prospective boundary

**Single overall descriptive completion target:** 150 distinct official Arena ES daily challenge IDs with a supported accepted forecast, each counted once. On the 2026-11-14 historical checkpoint, **19 accepted-evidenced challenge IDs** (including the separately recovered October 20 legacy case) and **19 locally reconciled official settlements** exist. For the programme's accepted-forecast progress counter, these form **19 of 150**, leaving **131 additional accepted forecasts**. These figures do not certify every historical scored revision or every eligible challenge that might have been offered.

**This is not a fixed-150-opportunity study.** Future **eligible opportunities** are independently logged *even if no submission succeeds*. If some future challenges are missed, rejected, cancelled, or still ambiguous, more than 131 eligible future opportunities may occur before 131 additional accepted forecasts are obtained. Do not equate `131 future accepted required` with `131 future opportunities guaranteed`, and never hide failed or missed events to make the completion fraction look better.

**Two evidence populations within one overall 150-total programme:**

- **Historical accepted subset:** the first 19 accepted-evidenced challenges from the October 20–November 13 study period, assessed under the pre-amendment procedures. Retain them in the descriptive accepted-forecast count, segregate their evidence levels, and do **not** call them prospectively enrolled under this later protocol. The historical eligible-opportunity census and the final scored revision of some records are not independently certified.
- **Future prospectively documented subset:** all official ES challenges with an independently supported opening after the future-effective adoption time, including accepted, missed, rejected, ambiguous, and cancelled-after-open challenges in the *opportunity ledger*. Only future confirmed accepted challenge IDs increase the accepted-forecast progress count; prospectively counted opportunities and accepted forecasts have distinct denominators.
- **Interim after-November-13, before-adoption observations, if any:** do not backdate adoption. If any genuine accepted predictions occur during this interval, label them explicitly as a transition/pre-adoption subset; preserve them for descriptive totals but not prospective claims, and revise the checkpoint only on actual evidence.

The previously published Notebook 12 v0.3.2 preserves its **first 30 accepted ES forecasts** definition beginning 2026-10-20. This proposed protocol seeks to maintain that accepted-count convention for the 30/60/60 phase layout. It does **not** retrospectively turn the first 19 accepted forecasts into the first 19 confirmed *eligible opportunities*, nor prove all historical opportunities were observed.

**Non-retrospective scope:** A later adoption decision can freeze future evidence collection, missingness rules, comparisons and hold-out policies only for future opportunities. The project cannot claim the full 150 was preregistered on October 20 or that later procedures governed historical predictions. Report historical, transition, and post-adoption results separately before any pooled descriptive analysis.

## 3. Market, challenge, and fixed target

The research market is the actual ES futures contract specified in each **official** Arena ES daily challenge (for example, ESZ6 before a valid roll; later contracts only when the actual challenge uses them). An official contract roll is not automatically a protocol violation. This study does **not** add a separate S&P 500 cash-index price dataset.

For each challenge, preserve its exact official challenge UUID, challenge type and asset, named ES contract, official opening and deadline timestamps, settlement rules including the neutral/dead-zone threshold, and official post-resolution label. **Do not assume** the neutral band, contract, opening time, deadline, or market schedule is invariant across challenges; capture and validate the actual challenge terms.

Three-way probability order is fixed as **`[bearish, neutral, bullish]`**. Preserve the model's native full-precision probability vector, deterministic label ordering, model version, feature cutoff and a SHA-256 link to the frozen forecast. A UI-rounded percentage display is not a replacement for original probabilities. The vector must contain finite numbers within `[0,1]` summing to one within a prespecified numerical tolerance used by the existing audited pipeline. Any malformed vector is documented as a failure or review issue; it cannot be repaired using post-outcome information.

Official Arena outcomes, rather than a separately reconstructed ES return, are ground truth for Arena-specific scores. Disagreements or revised settlements are logged with versions and source hashes; never overwrite the original settlement evidence.

## 4. Accepted-forecast target versus prospective opportunity denominator

**Accepted-forecast programme unit:** one distinct official Arena ES daily challenge UUID with a documented accepted forecast for the research programme, counted **at most once** regardless of retries or multiple revisions. The 19 historical records are retained in this descriptive count with evidence limitations disclosed; prospective accepted status requires linked server evidence, as well as separately recording uncertainty about the final scored revision and timing. No locally generated ID or unverified draft vector counts as accepted.

**Prospective operational opportunity unit:** one distinct official Arena ES daily challenge that genuinely opened for the research account's eligible submissions after protocol adoption, **regardless of whether a scored forecast was accepted**. The opportunity ledger includes documented misses, confirmed rejected submissions, ambiguous attempts, late or unavailable inputs, and cancellations after opening. This denominator is not capped at 131 or 150. Public challenge visibility alone does not establish account-specific eligibility; record review uncertainty separately.

Evidence includes an archived official challenge description with actual ES daily scope, UUID, opening, deadline, contract/neutral-band terms; accepted official receipts where available; cancellation records; and genuine pre-deadline discovery captures. Preserve raw bytes, full hashes, UTC collection metadata and server evidence. A response HTTP `Date` header is *not* independent signed attestation of account visibility or forecast submission time.

**Never invent an opportunity** from weekdays, market returns, calendar holidays, empty feeds, or local filenames. Failed retrieval is not proof of an empty feed, and an empty after-deadline feed does not establish that a challenge never opened. Such dates belong in `UNKNOWN_COVERAGE` monitoring outside the confirmed one-ID opportunity ledger.

A challenge verified to have **opened and later been cancelled** remains in the future operational opportunity denominator with `CANCELLED_AFTER_OPEN`; if no official settlement exists, it is not evaluable for Brier. A challenge withdrawn before opening is not a confirmed opportunity. Neither cancellations nor missed/rejected forecasts inflate the accepted-forecast numerator. They remain visible in operational reporting. A future attempt that was accepted but whose scored revision remains uncertain can be counted as acceptance-evidenced only with a clearly stated evidence tier, not silently certified as a valid final-scored-vector observation.

## 5. Accepted-count phases and prospective transition safeguards

**Phase gates are based on the number of distinct accepted-evidenced forecasts in the 150-total programme, not on 150 eligible-opportunity ordinals.** This is an accepted-count design, not a fixed-150-eligible-opportunity design; report and distinguish both denominators. Maintain a separately ordered prospective opportunity census and report its coverage for each accepted-count phase.

| Accepted-forecast programme position | Phase | Planned role |
|---|---|---|
| 1–30 | A | Frozen M1 baseline; includes the historical 19 plus **11 further accepted forecasts** if their evidence is verified |
| 31–90 | B | Next 60 accepted forecasts, development/comparison period; all intervening eligible opportunities still logged |
| 91–150 | C | Final 60 accepted forecasts, locked hold-out evaluation of the selected finalist against M1 |

**At the November 14 checkpoint:** programme accepted progress `19/150`; Phase A accepted progress `19/30`; if those historical acceptances remain validated, Phase A needs `11` more accepted forecasts before the accepted-count milestone is reached. This does not imply only 11 more eligible opportunities will be observed. It does not claim Phase A had exactly 19 historical opportunities or that all 19 have independently certified finally scored revisions.

Chronologically order accepted challenge IDs using defensible actual evidence of submission/acceptance timing and challenge identity, without using settlement or outcome to reorder them. When acceptance ordering or final counted status remains ambiguous, record the ambiguity and treat the phase boundary as provisional rather than declaring a certified position. Do not substitute unzoned `created_at` for `submitted_at` UTC. Historical ordering gaps remain disclosed.

**Transition freeze rule:** Before the first *official eligible challenge opening after* a verified Phase A accepted milestone (accepted #30), prospectively freeze the Phase B experiment: feature/input lists, candidate models, matched training/control datasets, shadow evidence and model-selection rules. If the #30 milestone cannot be established with adequate evidence, do not start a claimed confirmatory Phase B; log challenges in a transition/review state until a dated prospective boundary decision is made. Similarly freeze the Phase C finalist, fallback and evaluation plan **before the first official eligible challenge opening after** accepted #90. Do not inspect Phase C outcomes to select the finalist.

**No concealed opportunity-based extension:** An accepted-count phase can span more than 30/60/60 eligible challenge opportunities if there are failures, by design. That is **not** a fixed-opportunity sample, and we must transparently report its actual length, completeness, losses and possible accepted-only selection bias. Never choose a boundary based on observed Brier or settlement results. If confirmation requires a strictly fixed number of eligible opportunities, preregister a distinct prospective fixed-opportunity sub-study rather than relabel this 150-total programme.

## 6. Models and prevention of information leakage

The **baseline** is the existing frozen, human-supervised ES daily M1/scikit-learn Logistic Regression workflow. Preserve its existing artifact, data provenance, original notebook definitions and Arena submission procedure. Do not silently retrain, recalibrate, change feature engineering, substitute an alternative instrument or modify a prior forecast while continuing to call the result the same frozen baseline.

Any M2/M3/M4 candidate (including possible VIX or other approved inputs) must receive a separate version, exact training/input cutoff, feature list and source-licensing description **before its prospective predictions**. Never use target-date settlement, post-deadline features, revised historical labels unavailable at the decision time, or future data in training or backtests. Train/validation splits must respect chronological availability and any publication lags. Keep licensed raw inputs and joblib artifacts private; publish redacted hashes/metadata only when allowed.

In Phase B, paired model comparisons may use only challenges for which **both** vectors have authentic, pre-deadline evidence, plus an official settlement. Report how many eligible opportunities occurred during the 60-accepted-forecast interval; never assume 60 complete settled pairs. A shadow model is not a Headline Arena scored submission merely because it has a local forecast. Independent pre-deadline commitments, where feasible, should identify the relevant challenge, model/version, probability vector or salted commitment and timestamped source without releasing licensed information.

**Candidate model list, hyperparameters, selection metric and tie-break are not yet approved here.** These must be explicitly fixed and published as a dated, prospective Phase B/Phase C decision document before the relevant phase; any choice made after inspecting Phase C outcomes is exploratory, not a valid confirmatory final test.

## 7. Submission procedure and time evidence

For each eligible future challenge:

1. Record a genuine official challenge and preserved raw evidence before its deadline where possible. Confirm the UUID, ES daily scope, contract, neutral-band terms and absolute UTC deadline.
2. Follow the existing ES-only M1 preparation and human-supervision SOP, including the **five-minute operational guard**. A forecast inside the guard is reported separately even if the server accepted it before the formal deadline.
3. Preserve the full-precision forecast, model version and hash, attempted API request, response, accepted receipt if any, and a readback of retained prediction revisions when available. Never place credentials or licensed inputs into public Git.
4. For retained revision timing use Arena's **server-side `submitted_at` UTC** when provided. A parent-level `created_at`, an unzoned string or a local timestamp alone cannot certify the scored revision's deadline compliance. A receipt/prediction ID is not by itself proof of which revision was finally scored.
5. After the deadline reconcile official challenge state, attempts, responses and receipts. Keep ambiguous attempts unresolved rather than classifying missing local receipts as confirmed misses.
6. After settlement, append a separately sourced outcome update. Do not modify earlier event files, submission receipts, original forecasts or previously hashed reports.

No extra live submission is authorized by this methodology document. The existing separate, explicit human-approval control for **one** scored Arena submission remains in force.

## 8. Candidate-ledger schema and status separation

Use the append-only Stage 2 private JSONL event journal with byte-pinned source files, event hashes, prior-journal hashes and explicit CREATE/UPDATE operations. Build separate versioned one-row-per-challenge snapshots. Keep private sources outside the Git repository.

Minimum stable fields are official `challenge_id`, `asset=ES`, `challenge_type=daily`, actual ES `contract_symbol`, explicit `open_at_utc`, `deadline_utc`, `first_observed_at_utc`, preserved `opportunity_first_evidence_path`, and its physical `opportunity_first_evidence_sha256`.

Maintain distinct classification axes, **not** a single combined status:

- **Availability:** `CONFIRMED_OPEN`, `OBSERVED_BUT_TERMS_INCOMPLETE`, `CANCELLED_AFTER_OPEN`, `NEEDS_REVIEW`.
- **Submission:** `ACCEPTED_SCORED`, `REJECTED_CONFIRMED`, `ATTEMPT_AMBIGUOUS`, `NO_ACCEPTED_RECORD_YET`, `MISSED_VERIFIED`, `UNKNOWN`.
- **Outcome:** `OFFICIALLY_SETTLED`, `PENDING`, `CANCELLED_NO_OUTCOME`, `UNAVAILABLE`, `UNRESOLVED`.
- **Provenance:** `SOURCE_DISCOVERY_PREDEADLINE`, `OFFICIAL_ACCEPTED_WITH_TIME_LIMITATION`, `LEGACY_ACCEPTED`, `INCOMPLETE`, `UNRESOLVED`.

`ACCEPTED_SCORED` requires a retained actual prediction ID and source-linked receipt, and remains subject to final scored-revision review. `MISSED_VERIFIED` requires a post-deadline reviewed evidence trail proving that an eligible opportunity existed and no accepted submission occurred; absence of a local receipt alone is insufficient. `OFFICIALLY_SETTLED` requires official settlement label and a source SHA-256. Journal contents and hashes establish local byte integrity, **not** independently authenticated source truth or timestamp.

Count unknown-coverage dates/failed discovery attempts in a separate monitoring log without assigning invented challenge IDs. A weekend or exchange holiday does not itself add one eligible opportunity.

## 9. Planned analysis and reporting denominators

**Overall descriptive completion counter:** `N_accepted_distinct_programme = 19_historical_accepted_evidenced + N_additional_accepted_after_checkpoint`, up to a planned 150. Count only distinct official ES daily challenge IDs; deduplicate retries/revisions and verify the 19 historical identities against preserved audit evidence. Report acceptance-evidence quality, pre/post-adoption partition and unresolved final-revision certification. If a historical acceptance is subsequently disproved, transparently revise the descriptive numerator with a dated audit note rather than preserving an unsupported `19`.

**Separate operational prospective denominator (not fixed to 150):** `N_eligible_verified_future` is the number of post-adoption official ES daily challenge IDs confirmed to have been open for submissions. Report `N_accepted_future`, `N_rejected_future`, `N_ambiguous_future`, `N_verified_missed_future`, `N_cancelled_after_open_future`, `N_pending_future`, and coverage gaps, independently of the 150 accepted-forecast completion counter.

For each accepted-count phase and separately for historical, transition, and post-adoption periods, report the *actual observed number of eligible opportunities*, accepted submissions, outcome-verified settled forecast pairs, and timing-supported subset. Do not inflate accepted-fraction reliability by using only accepted challenges as its opportunity denominator. Pre-deadline server-supported and five-minute-guard-compliant counts are different from raw local timestamps.

**Primary predictive measurement:** mean unnormalized three-class multiclass Brier loss on official-settled challenges with a verified scored vector and acceptable deadline evidence, displaying the *evaluable sample size* and evidence tier. Report both a descriptive pooled accepted-sample estimate (with historical caveats) and a **separately labeled prospective-only** estimate; do not describe the pooled 150 as a preregistered or fixed-opportunity sample. A missing/ambiguous/rejected/cancelled opportunity receives no invented Brier value. Include a uniform `[1/3,1/3,1/3]` baseline on the same evaluable sample, plus classwise recall, confusion matrix, balanced accuracy and calibration where sample sizes allow.

**Paired model comparisons:** use common challenge IDs, official labels and authentic pre-deadline evidence for every compared vector. Report both models' marginal coverage and paired-settled `N`. Selection of a minimum observed Brier is not in itself proof of superiority; report multiple comparisons, missingness and uncertainty transparently. The confirmation period begins only after an adequately documented prospective finalist freeze. Accepted-count sampling may overrepresent successful operating conditions; this selection effect must be acknowledged.

Results measure Arena forecast quality, not financial profitability or causal trading skill. Keep official Arena leaderboard points distinct from fixed research loss metrics.

## 10. Continuous evidence review and change control

Preserve old evidence, original Notebook 12 and the 19-row November 14 baseline separately. The original M11 readiness script intentionally pins the **19-row live settlement CSV**; after subsequent official imports, its original hash may no longer match. Preserve the sealed November 14 file and implement a **new version** of the historical-baseline reader rather than modifying old hashes or falsifying a pass.

Before future opportunities, review the applicable protocol version; preserve any discrepancy, outage, missed data, API ambiguity, platform-rule change or cancellation. An amendment is a separately dated, versioned document and Git commit effective **only prospectively**. It must say which terms changed and which observations fall under each version. No removal of unfavourable observations, unreported accepted-count phase extension through operational failures, hindsight-based model selection or undocumented denominator changes.

Publication of the protocol or an adoption JSON is not self-authenticating external timing proof. Preserve a public Git URL/commit and independent timestamp evidence where genuinely available, and report the strength/limits of that evidence. Keep raw proprietary data and sensitive account information private.

## 11. Adoption gate and software interface

**Design and historical distinction:** The selected candidate target is **150 total distinct accepted forecasts, including the original 19**, with 30/60/60 **accepted-count** phase boundaries. An alternative fixed-150-future-opportunity proposal was considered before adoption and is not the selected design. Historical forecasts are part of the descriptive target, not prospectively enrolled observations. No past model results, receipts, Notebook 12 analysis, or original audit hashes are modified by this pre-adoption design clarification.

**Adoption software interface:** the existing Stage 2 adoption-review tool allows `chosen_study_design = "ORIGINAL_150_TOTAL_DESCRIPTIVE"`. It also requires `historical_19_enrolled = false`, which means **the historical 19 are not enrolled as future prospectively registered journal opportunities**; it **does not mean they are excluded from the overall 150 accepted-forecast descriptive total**. Do not change that field to `true`: the tool deliberately rejects retroactive prospective enrollment. `planned_phase_lengths = [30,60,60]` now describes accepted-forecast phase counts, **not eligible-opportunity phase counts**. Preserve this meaning in all study materials.

**Before any prospective claim**, the owner must:

1. Explicitly review and choose `ORIGINAL_150_TOTAL_DESCRIPTIVE` as the 150-total accepted-forecast design, not the alternative `FRESH_150_OPPORTUNITIES` design.
2. Review whether the accepted-count phase design meets the intended use and acknowledge that future opportunity counts may exceed the remaining accepted forecast goal; separate the expected operational coverage and scored model evaluation.
3. Freeze, hash and commit this exact v1.0 protocol without staging private journal, original models, modified notebooks or historical evidence; push it and verify its public version history. Hashes and local dates alone are not independent publication attestations.
4. Create a *private* owner-reviewed candidate adoption JSON, with the actual v1.0 protocol physical SHA-256, genuine UTC review, a proposed UTC effective time **later than review and verified publication**, null first-future-challenge, and `independent_publication_time_certified = false` unless separate genuinely verified evidence warrants a later manual claim. Run the Stage 2 read-only candidate reviewer. **Reviewer PASS never auto-adopts.**
5. Make a **separate explicit affirmative adoption decision**, record real UTC time and effective time, and document the first subsequent official ES challenge's actual opening time. Do not backdate adoption to October or November 13–14. Any challenge that opens before effective adoption is a clearly labeled transition observation, not a prospectively enrolled one.
6. Freeze the applicable Phase B experiment *before the first relevant future opportunity after the verified #30 accepted milestone* and the Phase C finalist *before the first relevant opportunity after #90*. Do not retroactively claim Phase A historical rules were prospectively fixed or move phase gates to improve scores.

**Private ledger:** existing Stage 2 `CREATE`/`UPDATE` journal stays **future-only** and cannot be populated with the historical 19 IDs. Combine the separate pinned 19 accepted-evidenced historical identities with verified future accepted IDs **only in a new derived, clearly labeled descriptive progress report**, never by changing journal events or faking prospectively recorded events.

**Current status as prepared November 14, 2026:** **proposal, not adopted**; `19/150` accepted-evidenced historical target progress; `131` additional accepted forecasts required at this checkpoint; `0` post-adoption opportunities enrolled. The full historical opportunity census and independent scored-revision certification remain incomplete. A separate genuine adoption record and future-effective date are still required.
