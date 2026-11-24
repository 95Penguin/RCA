"""Tests for operation-conditioned request-path latency decomposition."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.trace_paths import Span, entry_span, interval_union_length, service_latencies  # noqa: E402


class TracePathTest(unittest.TestCase):
    def test_interval_union_clips_and_does_not_double_count_parallel_children(self):
        self.assertEqual(
            interval_union_length([(5, 30), (20, 50), (80, 120)], 10, 100),
            60,
        )

    def test_exclusive_latency_subtracts_unmodeled_children(self):
        spans = [
            Span("t", "p", "", "frontendservice", "frontend", 0, 100),
            Span("t", "a", "p", "paymentservice", "pay", 10, 70),
            Span("t", "b", "p", "database", "query", 50, 90),
        ]
        values = service_latencies(spans, {"frontendservice", "paymentservice"})
        self.assertEqual(values["frontendservice"], (100, 20))
        self.assertEqual(values["paymentservice"], (60, 60))

    def test_entry_policy_matches_ob_and_tt(self):
        ob = Span("t", "1", "x", "frontendservice", "frontend", 0, 10)
        tt = Span("t", "2", "", "ts-auth-service", "POST /login", 1, 9)
        self.assertEqual(entry_span([ob], "ob"), ob)
        self.assertEqual(entry_span([tt], "tt"), tt)


if __name__ == "__main__":
    unittest.main()

