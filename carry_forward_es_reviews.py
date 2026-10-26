"""Preserve manually reviewed ES audits only when ALL historical market fields match.

Use after Notebook 11 produced its NEW unreviewed_*.csv for newly added DBN data.
No market data is edited, and no new review is automatically approved.

Usage:
  python carry_forward_es_reviews.py --old <old_reviewed.csv> --new <new_unreviewed.csv>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REVIEW = ("coverage_reviewed", "reviewer", "review_notes")
# A new immutable source-file set has a new observation timestamp for all its rows.
# Thus old data_available_utc can legitimately change, but no other market field can.
ALLOWED_SOURCE_OBSERVATION_UPDATE = {"data_available_utc"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def confirm(prompt, expected):
    got = input(f"{prompt}\nType EXACTLY {expected}: ").strip()
    require(got == expected, "No manual confirmation; no reviewed CSV created")


def run(old_path, new_path):
    old_path = Path(old_path).resolve()
    new_path = Path(new_path).resolve()
    require(old_path.is_file() and new_path.is_file(), "Missing old/new audit CSV")
    require(new_path.name.startswith("unreviewed_") and new_path.suffix == ".csv",
            "New source must be a Notebook 11 immutable unreviewed template")
    dest = new_path.with_name("reviewed_" + new_path.name.removeprefix("unreviewed_"))
    require(not dest.exists(), "Target reviewed CSV already exists: do not overwrite")
    old = pd.read_csv(old_path, keep_default_na=False)
    candidate = pd.read_csv(new_path, keep_default_na=False)
    require("session_date" in old and "session_date" in candidate,
            "Missing settlement date")
    require(old.session_date.is_unique and candidate.session_date.is_unique,
            "Duplicate session dates")
    require(list(old.columns) == list(candidate.columns), "Audit schema differs")
    require(set(REVIEW).issubset(old.columns), "Old attestation columns absent")
    require(old.coverage_reviewed.astype(str).str.lower().eq("true").all(),
            "Previous review did not cover every source session")
    require(old.reviewer.astype(str).str.strip().ne("").all() and
            old.review_notes.astype(str).str.strip().ne("").all(),
            "Old reviewer/notes incomplete")
    candidate = candidate.sort_values("session_date").reset_index(drop=True)
    old = old.sort_values("session_date").reset_index(drop=True)
    old_dates = set(old.session_date)
    new_dates = set(candidate.session_date) - old_dates
    require(old_dates.issubset(set(candidate.session_date)), "Old reviewed session missing from new candidate")
    require(new_dates, "No new session to review")
    require(min(new_dates) > max(old_dates), "New unreviewed dates precede previously reviewed history")
    a = old.set_index("session_date")
    b = candidate.set_index("session_date")
    comparison_cols = [c for c in old.columns if c not in REVIEW + tuple(ALLOWED_SOURCE_OBSERVATION_UPDATE) + ("session_date",)]
    for col in comparison_cols:
        original = a.loc[old.session_date, col]
        current = b.loc[old.session_date, col]
        if pd.api.types.is_numeric_dtype(original) and not pd.api.types.is_bool_dtype(original):
            require(np.allclose(original.to_numpy(dtype=float), current.to_numpy(dtype=float),
                                atol=1e-8, rtol=0, equal_nan=True),
                    f"Previously attested market-data field changed: {col}. STOP.")
        else:
            require(original.astype(str).tolist() == current.astype(str).tolist(),
                    f"Previously attested market-data field changed: {col}. STOP.")
    old_available = pd.to_datetime(a.loc[old.session_date, "data_available_utc"], utc=True)
    new_available = pd.to_datetime(b.loc[old.session_date, "data_available_utc"], utc=True)
    require((new_available.array >= old_available.array).all(),
            "Source-set observation timestamp regressed. Investigate.")
    print(f"PASS: {len(old)} previously reviewed sessions preserve ALL historical OHLCV, "
          f"contract, coverage, eligibility, and diagnostics fields.")
    print("Source observation timestamps are allowed to advance for the new combined raw-file set.")
    print("New settlement dates awaiting YOUR manual inspection:", sorted(new_dates))
    confirm("Confirm that the prior audit evidence remains valid for these unchanged sessions.",
            "I_CONFIRMED_PRIOR_AUDITS")

    # Copy only the *prior* human attestations. Never auto-approve a new date.
    for day in old.session_date:
        mask = candidate.session_date == day
        candidate.loc[mask, "coverage_reviewed"] = True
        candidate.loc[mask, "reviewer"] = str(a.loc[day, "reviewer"])
        candidate.loc[mask, "review_notes"] = str(a.loc[day, "review_notes"])

    for day in sorted(new_dates):
        row = candidate.loc[candidate.session_date == day].iloc[0]
        print("\n" + "=" * 68)
        print("NEW, UNREVIEWED SESSION:", day)
        for field in ("session_start_utc", "session_end_utc", "open", "high", "low", "close",
                      "volume", "session_return", "raw_symbol", "bar_count",
                      "expected_bar_count", "bar_coverage_pct", "max_gap_minutes",
                      "target_eligible", "review_flags"):
            if field in row:
                print(f" {field}: {row[field]}")
        print("Open the raw minute records and inspect boundaries, gaps, symbols, and CME schedule.")
        print("If shortened/rolled/incomplete, explicitly document the reason and eligibility.")
        if str(row.get("review_flags", "")).strip():
            confirm("Nonempty flags require a separate source-minute investigation before attestation.",
                    "I_INSPECTED_FLAGGED_MINUTE_BARS")
        confirm(f"I personally reviewed ES minute-data evidence for {day}, not only summary counts.",
                "I_REVIEWED_THIS_SESSION")
        reviewer = input("Reviewer identifier (not blank): ").strip()
        notes = input("Evidence-based review notes (>=25 chars): ").strip()
        require(reviewer != "" and len(notes) >= 25,
                "Reviewer and substantive notes required; no CSV created")
        mask = candidate.session_date == day
        candidate.loc[mask, "coverage_reviewed"] = True
        candidate.loc[mask, "reviewer"] = reviewer
        candidate.loc[mask, "review_notes"] = notes

    require(candidate.coverage_reviewed.astype(str).str.lower().eq("true").all(),
            "Review completion failed")
    require(candidate.reviewer.astype(str).str.strip().ne("").all(), "Missing reviewer")
    require(candidate.review_notes.astype(str).str.strip().ne("").all(), "Missing notes")
    with dest.open("x", encoding="utf-8", newline="") as output:
        candidate.to_csv(output, index=False, float_format="%.12g", lineterminator="\n")
    print("\nREVIEWED CSV CREATED (original source unchanged):", dest)
    print("Review rows:", len(candidate), "new dates:", sorted(new_dates))
    print("Set REVIEWED_AUDIT_CSV to this new path and run Notebook 11 export independently.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()
    run(args.old, args.new)
