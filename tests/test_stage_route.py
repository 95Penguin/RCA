"""Tests for DeST-inspired asymmetric temporal/spatial stage routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.factory import build_model  # noqa: E402
from models.ltfp import (  # noqa: E402
    LTFPStageRoute,
    LTFPStageRouteDirectionalResidual,
    LTFPStageRouteFutureQuery,
)


class StageRouteTests(unittest.TestCase):
    def _model(self) -> LTFPStageRoute:
        return LTFPStageRoute(
            6,
            feature_groups=((0, 1), (2, 3), (4, 5)),
            hidden_dim=8,
            dropout=0.0,
        ).eval()

    def test_factory_records_stage_routing_contract(self) -> None:
        built = build_model(
            "ltfp-stage-route",
            feature_columns=("metric_cpu", "log_count", "trace_span_count"),
            num_services=3,
        )
        self.assertEqual(built.checkpoint_config["architecture"], "ltfp-stage-route")
        self.assertEqual(built.checkpoint_config["spatial_residual_initialization"], 0.0)
        self.assertEqual(built.checkpoint_config["max_spatial_residual"], 0.5)

    def test_zero_gate_starts_from_strict_temporal_prediction(self) -> None:
        torch.manual_seed(7)
        model = self._model()
        features = torch.randn(2, 5, 3, 6)
        identity = torch.eye(3).repeat(2, 5, 1, 1)
        dense = torch.ones(2, 5, 3, 3)

        identity_output = model(features, identity)
        dense_output = model(features, dense)

        torch.testing.assert_close(
            identity_output["failure_logits"],
            identity_output["temporal_failure_logits"],
        )
        torch.testing.assert_close(
            identity_output["spatial_failure_logits"],
            torch.zeros_like(identity_output["spatial_failure_logits"]),
        )
        torch.testing.assert_close(
            identity_output["failure_logits"], dense_output["failure_logits"]
        )
        self.assertEqual(identity_output["node_scores"].shape, (2, 3))

    def test_spatial_gate_is_bounded_and_trainable(self) -> None:
        torch.manual_seed(11)
        model = self._model()
        model.spatial_failure_head.bias.data.fill_(1.0)
        model.spatial_residual_scale.data.fill_(10.0)
        output = model(
            torch.randn(2, 5, 3, 6),
            torch.ones(2, 5, 3, 3),
        )
        self.assertLessEqual(
            float(output["spatial_failure_logits"].detach().abs().max()),
            model.max_spatial_residual,
        )

        model.zero_grad(set_to_none=True)
        model.spatial_residual_scale.data.zero_()
        output = model(
            torch.randn(2, 5, 3, 6),
            torch.ones(2, 5, 3, 3),
        )
        output["failure_logits"][:, 1].sum().backward()
        self.assertIsNotNone(model.spatial_residual_scale.grad)
        self.assertNotEqual(float(model.spatial_residual_scale.grad), 0.0)

    def test_future_query_changes_only_the_capacity_matched_rca_context(self) -> None:
        torch.manual_seed(19)
        stage = self._model()
        query = LTFPStageRouteFutureQuery(
            6,
            feature_groups=((0, 1), (2, 3), (4, 5)),
            hidden_dim=8,
            dropout=0.0,
        ).eval()
        query.load_state_dict(stage.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in stage.parameters()),
            sum(parameter.numel() for parameter in query.parameters()),
        )

        features = torch.randn(2, 5, 3, 6)
        adjacency = torch.ones(2, 5, 3, 3)
        stage_output = stage(features, adjacency)
        query_output = query(features, adjacency)

        torch.testing.assert_close(
            stage_output["failure_logits"], query_output["failure_logits"]
        )
        torch.testing.assert_close(
            query_output["root_context_embeddings"],
            query_output["temporal_graph_state"],
        )
        self.assertFalse(torch.allclose(
            stage_output["root_context_embeddings"],
            query_output["root_context_embeddings"],
        ))

    def test_future_query_factory_records_temporal_root_context(self) -> None:
        built = build_model(
            "ltfp-stage-route-query",
            feature_columns=("metric_cpu", "log_count", "trace_span_count"),
            num_services=3,
        )
        self.assertEqual(
            built.checkpoint_config["root_context"],
            "pre-propagation temporal application state",
        )

    def test_directional_factory_records_conservative_rca_contract(self) -> None:
        built = build_model(
            "ltfp-stage-route-directional",
            feature_columns=("metric_cpu", "log_count", "trace_span_count"),
            num_services=3,
        )
        config = built.checkpoint_config
        self.assertEqual(config["directional_residual_initialization"], 0.0)
        self.assertEqual(config["directional_scope"], "RCA only")
        self.assertEqual(config["required_graph_direction"], "callee-to-caller")

    def test_zero_direction_gate_exactly_recovers_symmetric_stage_route(self) -> None:
        torch.manual_seed(23)
        stage = self._model()
        directional = LTFPStageRouteDirectionalResidual(
            6,
            feature_groups=((0, 1), (2, 3), (4, 5)),
            hidden_dim=8,
            dropout=0.0,
        ).eval()
        directional.load_state_dict(stage.state_dict(), strict=False)

        features = torch.randn(2, 5, 3, 6)
        directed_graph = torch.eye(3).repeat(2, 5, 1, 1)
        directed_graph[:, :, 0, 1] = 3.0
        directed_graph[:, :, 1, 2] = 2.0
        symmetric_graph = torch.maximum(
            directed_graph, directed_graph.transpose(2, 3)
        )

        expected = stage(features, symmetric_graph)
        actual = directional(features, directed_graph)
        torch.testing.assert_close(actual["failure_logits"], expected["failure_logits"])
        torch.testing.assert_close(actual["node_logits"], expected["node_logits"])
        torch.testing.assert_close(
            actual["directional_residual_gate"],
            torch.zeros_like(actual["directional_residual_gate"]),
        )

    def test_directional_residual_changes_only_rca_when_direction_flips(self) -> None:
        torch.manual_seed(29)
        model = LTFPStageRouteDirectionalResidual(
            6,
            feature_groups=((0, 1), (2, 3), (4, 5)),
            hidden_dim=8,
            dropout=0.0,
        ).eval()
        model.directional_residual_scale.data.fill_(2.0)
        features = torch.randn(2, 5, 3, 6)
        graph = torch.eye(3).repeat(2, 5, 1, 1)
        graph[:, :, 0, 1] = 4.0
        graph[:, :, 1, 2] = 2.0

        forward = model(features, graph)
        reverse = model(features, graph.transpose(2, 3))
        torch.testing.assert_close(
            forward["failure_logits"], reverse["failure_logits"]
        )
        self.assertFalse(torch.allclose(
            forward["node_logits"], reverse["node_logits"]
        ))

    def test_root_loss_can_open_zero_initialized_direction_gate(self) -> None:
        torch.manual_seed(31)
        model = LTFPStageRouteDirectionalResidual(
            6,
            feature_groups=((0, 1), (2, 3), (4, 5)),
            hidden_dim=8,
            dropout=0.0,
        )
        graph = torch.eye(3).repeat(2, 5, 1, 1)
        graph[:, :, 0, 1] = 3.0
        graph[:, :, 1, 2] = 2.0
        output = model(torch.randn(2, 5, 3, 6), graph)
        output["node_logits"][:, 0].sum().backward()
        gradient = model.directional_residual_scale.grad
        self.assertIsNotNone(gradient)
        self.assertNotEqual(float(gradient), 0.0)


if __name__ == "__main__":
    unittest.main()
