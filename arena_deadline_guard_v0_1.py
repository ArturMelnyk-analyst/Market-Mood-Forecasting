"""Fail-closed five-minute operational guard for ES Arena submissions.

This module only evaluates UTC clock/deadline values. It does not contact
Arena, inspect M1, modify forecasts, or write files.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

MINIMUM_LEAD = timedelta(minutes=5)


def _aware_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def require_safe_submission_time(deadline: datetime, *, now: datetime | None = None) -> datetime:
    """Reject preparation or POST when fewer than five minutes remain.

    The exact five-minute boundary is NOT allowed.  Pass `now` only when
    testing.  The default is the current, timezone-aware UTC system clock.

    Returns the UTC cutoff for optional logging; never modifies any state.
    """
    official_deadline = _aware_utc(deadline, "deadline")
    current_time = _aware_utc(datetime.now(timezone.utc) if now is None else now, "now")
    cutoff = official_deadline - MINIMUM_LEAD
    if current_time >= cutoff:
        raise RuntimeError(
            "Five-minute submission safety cutoff reached; no preparation/POST. "
            f"Current UTC: {current_time.isoformat()} | "
            f"Cutoff UTC: {cutoff.isoformat()} | "
            f"Official deadline UTC: {official_deadline.isoformat()}"
        )
    return cutoff
