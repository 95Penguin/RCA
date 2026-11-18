"""Tests for the modality feasibility experiment's feature grouping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.factory import build_model, feature_groups, named_feature_groups  # noqa: E402


class ModalitySelectionTests(unittest.TestCase):
    def test_entry_features_remain_trace_derived(self) -> None:
        columns = (
            "metric_cpu",
            "observed_cpu",
            "log_count",
            "log_observed",
            "trace_span_count",
            "entry_p95_ms",
            "observed_entry_p95",
        )
        named = named_feature_groups(columns)
        self.assertEqual(named["metric"], [0, 1])
        self.assertEqual(named["log"], [2, 3])
        self.assertEqual(named["trace"], [4, 5, 6])
        self.assertEqual(feature_groups(columns), [[0, 1], [2, 3], [4, 5, 6]])

    def test_dual_router_exposes_normalized_task_routes(self) -> None:
        columns = ("metric_cpu", "log_count", "trace_span_count")
        model = build_model(
            "ltfp-dual-router", feature_columns=columns, num_services=3
        ).model
        output = model(torch.randn(2, 5, 3, 3), torch.eye(3).repeat(2, 5, 1, 1))
        self.assertEqual(output["failure_modality_weights"].shape, (2, 3, 3))
        self.assertEqual(output["root_modality_weights"].shape, (2, 3, 3))
        torch.testing.assert_close(
            output["failure_modality_weights"].sum(dim=-1), torch.ones(2, 3)
        )
        self.assertIsNot(model.failure_router, model.root_router)


if __name__ == "__main__":
    unittest.main()
