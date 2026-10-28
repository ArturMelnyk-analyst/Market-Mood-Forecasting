"""Read-only SECONDARY peer audit for the Headline Arena ES study.

Main Brier comparison remains in Notebook 12; this script NEVER modifies it
or the canonical arena_peer_forecasts.csv. Assumptions: displayed whole
percentages use conventional nearest-percentage-point rounding. This can
be verified if the platform later publishes exact underlying vectors.

Usage from repo root:
  python arena_peer_secondary_audit.py
"""
from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "es_arena_research_v0_1"
AUDIT = DATA / "es_peer_source_audit_20261020_27.csv"
OFFICIAL = DATA / "arena_official_settlements.csv"
CLASSES = ("bearish", "neutral", "bullish")
OUR_AGENT = "agt_f7b9b0827875"


def brier(p: np.ndarray, truth: str) -> float:
    y = np.array([float(c == truth) for c in CLASSES])
    return float(np.sum((p - y) ** 2))


def rounded_brier_bounds(percentages: tuple[float, float, float], outcome: str):
    """Exact Brier min/max over rounded-probability boxes intersected with simplex.

    Max of convex squared loss occurs at a vertex. Min is interior or on a
    bounding face; for each fixed-coordinate face the remaining two-coordinate
    constrained minimizer has a closed-form projection.
    """
    centers = np.asarray(percentages, dtype=float) / 100.0
    lower = np.maximum(0.0, centers - 0.005)
    upper = np.minimum(1.0, centers + 0.005)
    y = np.asarray([float(c == outcome) for c in CLASSES])
    candidates = []

    def add(p):
        p = np.array(p, dtype=float)
        if (np.all(p >= lower - 1e-10) and np.all(p <= upper + 1e-10)
                and abs(p.sum() - 1.0) <= 1e-10):
            candidates.append(p)

    # Vertices: bind two independent coordinate constraints.
    for i, j in combinations(range(3), 2):
        k = ({0, 1, 2} - {i, j}).pop()
        for pi, pj in product((lower[i], upper[i]), (lower[j], upper[j])):
            p = np.zeros(3)
            p[i], p[j], p[k] = pi, pj, 1.0 - pi - pj
            add(p)

    # Minimum along each face: bind one coordinate; optimize the other two.
    for k in range(3):
        other = [i for i in range(3) if i != k]
        i, j = other
        for value in (lower[k], upper[k]):
            remaining = 1.0 - value
            feasible_lo = max(lower[i], remaining - upper[j])
            feasible_hi = min(upper[i], remaining - lower[j])
            if feasible_lo > feasible_hi + 1e-10:
                continue
            opt_i = np.clip((remaining + y[i] - y[j]) / 2.0,
                            feasible_lo, feasible_hi)
            p = np.zeros(3)
            p[k], p[i], p[j] = value, opt_i, remaining - opt_i
            add(p)

    # In case the one-hot outcome falls inside all displayed rounding ranges.
    add(y)
    if not candidates:
        raise ValueError("No valid sum-to-one vector is consistent with rounding")
    vals = [(brier(p, outcome), p) for p in candidates]
    mini = min(vals, key=lambda v: v[0])
    maxi = max(vals, key=lambda v: v[0])
    return mini, maxi


def main():
    for p in (AUDIT, OFFICIAL):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required source CSV: {p}")
    a = pd.read_csv(AUDIT, dtype={"challenge_id": str, "agent_id": str})
    o = pd.read_csv(OFFICIAL, dtype={"challenge_id": str, "official_label": str})
    assert not o.challenge_id.duplicated().any(), "Duplicate official challenge IDs"
    assert not a.duplicated(["challenge_id", "agent_id"]).any(), "Duplicate audit entries"
    assert a.loc[a.agent_id == OUR_AGENT].shape[0] == 6, "Expected six own entries"

    a = a.merge(o[["challenge_id", "official_label"]], on="challenge_id",
                how="left", validate="many_to_one")
    assert a.official_label.notna().all(), "Missing official outcomes"
    assert a.official_label.str.lower().isin(CLASSES).all()

    print("SECONDARY PEER EVIDENCE AUDIT — READ ONLY")
    print("=" * 55)
    print(f"Total challenge-page prediction cards: {len(a)}")
    print(f"Our own agent cards (excluded from peer CSV): {(a.agent_id == OUR_AGENT).sum()}")
    print("Counts by source-audit inclusion status:")
    print(a.groupby("benchmark_status").size().to_string())

    missing = a.loc[a.benchmark_status == "excluded_no_original_vector"].copy()
    assert len(missing) == 4, f"Audit changed: expected 4 no-vector agents, found {len(missing)}"
    missing["direction_correct"] = (
        missing.direction.str.lower() == missing.official_label.str.lower()
    )
    print("\nNO ORIGINAL THREE-CLASS VECTOR — DIRECTION ONLY")
    print(missing[["challenge_date", "agent_name", "direction", "official_label",
                   "direction_correct"]].to_string(index=False))
    print("Directional-only summary BY AGENT (no cross-agent pooling for ranking):")
    for agent, grp in missing.groupby("agent_name", sort=True):
        print(f"  {agent}: {int(grp.direction_correct.sum())}/{len(grp)} correct")
    print("Total evidence records without vectors:", len(missing))
    print("Not a Brier score, not a comparison on identical information sets.")

    rounded = a.loc[a.benchmark_status == "excluded_display_rounding_sum_101"].copy()
    assert len(rounded) == 1, f"Audit changed: expected one rounded 101% peer, got {len(rounded)}"
    r = rounded.iloc[0]
    assert r.agent_name == "KK-Forecaster" and r.challenge_date == "2026-10-23"
    pcts = tuple(float(r[f"display_{c}_pct"]) for c in CLASSES)
    assert round(sum(pcts), 6) == 101.0
    print("\n101% DISPLAYED VECTOR — ROUNDING-AWARE BRIER SENSITIVITY ONLY")
    print(f"Agent: {r.agent_name}  |  date: {r.challenge_date}  |  outcome: {r.official_label}")
    print("Displayed (bearish,neutral,bullish):", pcts)
    (lo_score, lo_p), (hi_score, hi_p) = rounded_brier_bounds(pcts, r.official_label.lower())
    print(f"Conditional Brier range: {lo_score:.6f} to {hi_score:.6f}")
    print("One minimum-loss possible vector:", np.round(lo_p, 6).tolist())
    print("One maximum-loss possible vector:", np.round(hi_p, 6).tolist())
    print("Assumes standard independent rounding to the nearest whole percentage point.")
    print("Neither bound is the authenticated original submission's exact Brier.")
    print("\nSAFEGUARD: No input CSV or notebook was modified.")


if __name__ == "__main__":
    main()
