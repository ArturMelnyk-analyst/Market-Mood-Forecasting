# Milestone 8 — RFC 3161 Multi-Authority Synthetic Timestamp Verification

**Project:** Market Mood Forecasting — Research Provenance  \
**Report date:** 16 November 2026 (UTC)  \
**Scope:** Public technical summary of a synthetic verification exercise  \
**Result:** `PASS_WITH_EXPLICIT_LIMITATIONS`

## Summary

The Milestone 8 synthetic exercise evaluated whether an offline, fail-closed verifier could validate RFC 3161 timestamp responses against a commitment SHA-256 digest, a conservative synthetic cutoff, and an independently selected certificate-authority (CA) trust bundle.

**The pre-specified acceptance threshold was met:** 4 of 6 timestamp responses validated successfully under the selected independent CA bundle; only 2 independently trusted responses were required. All 6 responses carried matching SHA-256 message imprints and signed timestamps earlier than the conservative synthetic evidence cutoff.

This result closes **synthetic verifier validation only**. It does **not** establish that a real Headline Arena forecast was independently committed or revealed before a live deadline.

## Acceptance results

| Check | Result | Assessment |
|---|---:|---|
| RFC 3161 responses examined | 6 | Recorded |
| Matching SHA-256 message imprints | 6/6 | PASS |
| Signed timestamps before conservative synthetic cutoff | 6/6 | PASS |
| Independently CA-trusted signatures/chains | 4/6 | PASS against minimum of 2 |
| Required independently trusted authorities | 2 | Threshold met |
| Verifier unit tests | 14/14 | PASS |
| Final classification | `PASS_WITH_EXPLICIT_LIMITATIONS` | Synthetic milestone complete |

### Per-authority outcomes

| Authority | Imprint | Timing gate | Verification under selected independent CA bundle |
|---|---|---|---|
| DigiCert | MATCH | PASS | PASS |
| IdenTrust | MATCH | PASS | PASS |
| GlobalSign | MATCH | PASS | PASS |
| Sectigo | MATCH | PASS | PASS |
| Microsoft | MATCH | PASS | NOT_TRUSTED_BY_SELECTED_BUNDLE |
| Apple | MATCH | PASS | NOT_TRUSTED_BY_SELECTED_BUNDLE |

The two untrusted outcomes are specific to the tested CA trust configuration; they are not presented as proof that those timestamp authorities or their certificates are universally invalid. Neither was counted toward the minimum trusted-authority requirement, and no embedded certificates were elevated to trust anchors.

## Verification approach

The private verification procedure used the OpenSSL RFC 3161 timestamp utilities and enforced these controls:

1. Verify the receipt structure and an expected receipt digest, then compare the declared SHA-256 commitment digest with the underlying public-file digest.
2. Decode and inspect all RFC 3161 responses, requiring a granted SHA-256 response, the correct message imprint, unique authority identifiers, and a signed time earlier than the conservative synthetic cutoff.
3. Verify each timestamp signature and certificate chain against an independently maintained CA bundle; embedded certificate material is not accepted as a new root of trust.
4. Count only independently trusted responses and reject the receipt if the specified minimum count is not reached.
5. Remain offline and read-only during verification; do not submit predictions or mutate underlying research evidence.

The synthetic timing policy included a five-minute safety offset and an additional sixty-second conservative margin. Positive and negative unit tests exercised success, digest mismatch, malformed responses, insufficient independent trust, timing boundaries, duplicate authorities, and source-file preservation.

## Evidence stewardship and reproducibility

Original synthetic receipts, private plans, source inputs, verification outputs, and cryptographic hashes are retained in an access-controlled research archive. The full internal evidence report records the exact artifact identifiers, pinned SHA-256 values, command-line reproduction procedure, and verification-environment limitations. These materials are not reproduced in this public summary.

The verification is reproducible from the original raw artifacts and trusted verifier under an explicitly recorded CA bundle. Trust counts can change when the CA bundle or its validation environment changes; new checks should be dated separately rather than altering the historical result.

## Limitations and non-claims

- **No live forecast attestation.** This exercise used a synthetic plan and synthetic deadline; it did not prove a forecast-specific commitment before an actual Headline Arena cutoff.
- **No Arena account eligibility proof.** Timestamp signatures do not establish official challenge opening time, account-specific visibility or scoring eligibility.
- **No online revocation check.** Historical certificate revocation status was not independently established through OCSP/CRL.
- **Bundle-specific trust.** Only four of the six submitted responses were independently trusted under the selected CA bundle.
- **No model-execution or market-data-vintage proof.** A timestamp of a digest alone does not prove model execution, data availability, forecast correctness or absence of leakage.
- **No public reveal.** A full real-world commitment/reveal demonstration remains separate future work.

## Decision

**Milestone 8 synthetic RFC 3161 receipt-verification exercise: COMPLETE, with explicit limitations (`PASS_WITH_EXPLICIT_LIMITATIONS`).**

Future work should test a forecast-specific commitment, independent timestamp and post-settlement reveal against an actual Arena challenge, while separately preserving server submission timestamps, official challenge-opportunity evidence and settlement records. That future work is not a prerequisite for closing the synthetic verification milestone documented here.
