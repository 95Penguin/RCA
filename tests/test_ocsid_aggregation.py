"""Uniform mixes per-operation evidence, not service occurrence or traffic."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.aggregation import aggregate_operations, operation_presence
from ocsid.inference import attribute_requests
from ocsid.cpca import Span


class AggregationTests(unittest.TestCase):
    def test_absence_is_zero_and_counts_do_not_change_uniform(self):
        scores = {"rare": {"root": .9}, "busy": {"symptom": .3}}
        uniform, weights = aggregate_operations(scores, {"rare": 1, "busy": 100})
        self.assertEqual(uniform, {"root": .45, "symptom": .15})
        self.assertEqual(weights, {"rare": .5, "busy": .5})
        weighted, _ = aggregate_operations(scores, {"rare": 1, "busy": 100}, mode="request-weighted")
        self.assertGreater(weighted["symptom"], weighted["root"])

    def test_presence_is_independent_of_latency_and_zero_scores_still_observed(self):
        scores = {"a": {"root": .9, "shared": 0.}, "b": {"shared": .001}}
        self.assertEqual(operation_presence(scores), {"root": .5, "shared": 1.})

    def test_single_operation_modes_agree_and_missing_stays_empty(self):
        for mode in ("hard", "soft", "uniform", "request-weighted"):
            self.assertEqual(aggregate_operations({"a": {"root": .8}}, {"a": 1}, mode=mode)[0], {"root": .8})
            self.assertEqual(aggregate_operations({}, {}, mode=mode), ({}, {}))

    def test_uniform_is_default_in_live_inference(self):
        slow = [Span("s", "", "gateway", "POST /slow", 0., 1000.),
                Span("p", "s", "payment", "pay", 10., 900.)]
        fast = [Span("f", "", "gateway", "POST /fast", 0., 50.),
                Span("c", "f", "catalog", "read", 10., 20.)]
        result = attribute_requests([slow] + [fast] * 25,
                 {("payment", "pay"): 10., ("catalog", "read"): 20.}, {}, {}, "tt")
        self.assertAlmostEqual(result["scores"]["payment"], .445)
        self.assertEqual(set(result["operation_weights"].values()), {.5})
