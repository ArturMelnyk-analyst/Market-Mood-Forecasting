"""Offline tests for the Arena submission safety buffer and client wiring.

No live API calls, forecasts, receipt writes or model execution.
"""

from __future__ import annotations

import ast
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from arena_deadline_guard_v0_1 import MINIMUM_LEAD, require_safe_submission_time


DEADLINE = datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)


class DeadlineGuardTests(unittest.TestCase):
    def test_buffer_is_five_minutes(self):
        self.assertEqual(MINIMUM_LEAD, timedelta(minutes=5))

    def test_six_minutes_early_allowed(self):
        cutoff = require_safe_submission_time(DEADLINE, now=DEADLINE - timedelta(minutes=6))
        self.assertEqual(cutoff, DEADLINE - timedelta(minutes=5))

    def test_one_microsecond_before_cutoff_allowed(self):
        require_safe_submission_time(DEADLINE, now=DEADLINE - timedelta(minutes=5, microseconds=1))

    def test_exact_cutoff_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Five-minute submission safety cutoff"):
            require_safe_submission_time(DEADLINE, now=DEADLINE - timedelta(minutes=5))

    def test_inside_buffer_rejected(self):
        with self.assertRaises(RuntimeError):
            require_safe_submission_time(DEADLINE, now=DEADLINE - timedelta(minutes=4))

    def test_after_deadline_rejected(self):
        with self.assertRaises(RuntimeError):
            require_safe_submission_time(DEADLINE, now=DEADLINE + timedelta(seconds=1))

    def test_naive_deadline_rejected(self):
        with self.assertRaises(ValueError):
            require_safe_submission_time(DEADLINE.replace(tzinfo=None), now=DEADLINE)

    def test_naive_clock_rejected(self):
        with self.assertRaises(ValueError):
            require_safe_submission_time(DEADLINE, now=(DEADLINE - timedelta(minutes=6)).replace(tzinfo=None))

    def test_non_utc_aware_clock_converted(self):
        local_time = (DEADLINE - timedelta(minutes=6)).astimezone(timezone(timedelta(hours=1)))
        require_safe_submission_time(DEADLINE, now=local_time)


class ClientWiringTests(unittest.TestCase):
    """Static wiring checks: verify the guard cannot be omitted by accident.

    These complement timing unit tests; they do not replace an integration
    test of live authentication or submitted forecasts.
    """

    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parent / "es_arena_daily.py"
        cls.tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    def fn(self, name):
        return next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == name)

    @staticmethod
    def calls(fn):
        return [node for node in ast.walk(fn) if isinstance(node, ast.Call)]

    @staticmethod
    def called_name(node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return f"{node.func.value.id}.{node.func.attr}"
        return None

    def test_prepare_uses_safety_guard(self):
        calls = self.calls(self.fn("run_prepare"))
        self.assertTrue(any(self.called_name(c) == "require_safe_submission_time" for c in calls))

    def test_submit_guards_before_attempt_and_before_post(self):
        calls = self.calls(self.fn("run_submit"))
        guards = sorted(c.lineno for c in calls if self.called_name(c) == "require_safe_submission_time")
        attempts = sorted(c.lineno for c in calls if self.called_name(c) == "write_private_once"
                          and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == "attempt")
        posts = sorted(c.lineno for c in calls if self.called_name(c) == "requests.post")
        self.assertEqual(len(guards), 2, f"Expected two submit guards; found {guards}")
        self.assertEqual(len(attempts), 1, f"Expected attempt write; found {attempts}")
        self.assertEqual(len(posts), 1, f"Expected one POST; found {posts}")
        self.assertLess(guards[0], attempts[0])
        self.assertLess(attempts[0], guards[1])
        self.assertLess(guards[1], posts[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
