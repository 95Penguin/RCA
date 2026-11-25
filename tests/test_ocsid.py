"""Tests for the OCSID model and the CPCA core."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.ocsid import OCSIDModel  # noqa: E402
from ocsid.cpca import Span, decompose_tree, hit_metrics, score_request  # noqa: E402
from ocsid.features import OP_FEATURE_NAMES, operation_feature_vector  # noqa: E402


class OperationFeatureTests(unittest.TestCase):
    def test_vector_dim_and_finite(self):
        row = {
            "p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 30.0, "mean_ms": 12.0,
            "count": 100.0, "span_count": 250.0, "n_services": 4.0,
            "error_count": 1.0, "baseline_p95_ms": 10.0,
            "baseline_count": 90.0, "roll_p95_ms": 10.0, "roll_count": 90.0,
        }
        values = operation_feature_vector(row)
        self.assertEqual(len(values), len(OP_FEATURE_NAMES))
        self.assertTrue(all(isinstance(value, float) for value in values))
        self.assertTrue(all(value == value for value in values))  # no NaN


class CPCATests(unittest.TestCase):
    def setUp(self):
        # frontend (root) -> checkout -> payment
        self.spans = [
            Span("f", "", "frontendservice", "frontend", 0.0, 1000.0),
            Span("c", "f", "checkoutservice", "POST /checkout", 20.0, 800.0),
            Span("p", "c", "paymentservice", "POST /pay", 30.0, 620.0),
        ]
        self.normal_local = {
            ("frontendservice", "frontend"): 0.0,
            ("checkoutservice", "POST /checkout"): 0.0,
            ("paymentservice", "POST /pay"): 50.0,
        }
        self.normal_edge = {
            ("frontendservice", "frontend", "checkoutservice", "POST /checkout"): (20.0, 180.0),
            ("checkoutservice", "POST /checkout", "paymentservice", "POST /pay"): (5.0, 5.0),
        }

    def test_decompose_tree(self):
        parsed = decompose_tree(self.spans, "ob")
        self.assertIsNotNone(parsed)
        _, _, root, local, pre, post = parsed
        self.assertEqual(root.sid, "f")
        self.assertAlmostEqual(local["f"], 0.0)
        self.assertAlmostEqual(pre["c"], 20.0)
        self.assertAlmostEqual(post["c"], 180.0)
        self.assertAlmostEqual(pre["p"], 10.0)
        self.assertAlmostEqual(post["p"], 170.0)

    def test_counterfactual_restore_payment(self):
        root, deltas, _ = score_request(
            self.spans, self.normal_local, self.normal_edge, "ob"
        )
        self.assertAlmostEqual(root.duration, 1000.0)
        self.assertAlmostEqual(deltas["paymentservice"], 740.0, places=6)
        self.assertAlmostEqual(deltas["checkoutservice"], 0.0, places=6)
        self.assertAlmostEqual(deltas["frontendservice"], 0.0, places=6)

    def test_hit_metrics(self):
        metrics = hit_metrics([1, 2, 3])
        self.assertEqual(metrics["events"], 3)
        self.assertAlmostEqual(metrics["hit_at_1"], 1 / 3)
        self.assertAlmostEqual(metrics["hit_at_3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1 + 0.5 + 1 / 3) / 3)


class OCSIDModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        model = OCSIDModel(
            feature_dim=4,
            operation_dim=3,
            feature_groups=[[0, 1], [2, 3]],
            hidden_dim=8,
            temporal_layers=1,
            graph_layers=1,
            operation_hidden=4,
            dropout=0.0,
        )
        x = torch.randn(2, 5, 3, 4)
        adjacency = torch.ones(2, 5, 3, 3)
        operations = torch.randn(2, 5, 4, 3)
        mask = torch.ones(2, 5, 4)
        out = model(x, adjacency, operations, mask)
        self.assertEqual(out["failure_logits"].shape, (2, 2))
        self.assertEqual(out["probability"].shape, (2,))
        self.assertEqual(out["operation_scores"].shape, (2, 4))
        self.assertNotIn("node_scores", out)
        self.assertEqual(out["node_embeddings"].shape, (2, 3, 8))
        self.assertTrue(torch.isfinite(out["probability"]).all())

    def test_aux_head_is_training_only(self):
        tensors = (
            torch.randn(2, 5, 3, 4),
            torch.ones(2, 5, 3, 3),
            torch.randn(2, 5, 4, 3),
            torch.ones(2, 5, 4),
        )
        with_aux = OCSIDModel(
            feature_dim=4, operation_dim=3, feature_groups=[[0, 1], [2, 3]],
            hidden_dim=8, temporal_layers=1, graph_layers=1,
            operation_hidden=4, dropout=0.0, aux_root=True,
        )
        self.assertIn("aux_root_logits", with_aux(*tensors))
        without_aux = OCSIDModel(
            feature_dim=4, operation_dim=3, feature_groups=[[0, 1], [2, 3]],
            hidden_dim=8, temporal_layers=1, graph_layers=1,
            operation_hidden=4, dropout=0.0, aux_root=False,
        )
        self.assertNotIn("aux_root_logits", without_aux(*tensors))
        self.assertNotIn("node_scores", without_aux(*tensors))


if __name__ == "__main__":
    unittest.main()
