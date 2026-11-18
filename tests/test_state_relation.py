"""Safety and capacity controls for normal metric evidence."""

import sys
import unittest
from pathlib import Path

import torch
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.ltfp import LTFPBaseline
from models.coarse_fine import LTFPStateRelationEvidence
from training.normal_reference import configure_metric_reference


class StateRelationTests(unittest.TestCase):
    def make(self, relation):
        return LTFPStateRelationEvidence(
            4, 3, 5, (0, 1), (2, 3), relation,
            hidden_dim=8, feature_groups=((0, 1, 2, 3),), dropout=0.0,
        )

    def test_zero_residual_matches_baseline_and_root_gradient_is_protected(self):
        torch.manual_seed(7)
        baseline = LTFPBaseline(4, hidden_dim=8, feature_groups=((0, 1, 2, 3),), dropout=0.0).eval()
        model = self.make(True).eval()
        model.load_state_dict(baseline.state_dict(), strict=False)
        model.configure_normal_reference(torch.zeros(5, 3, 2), torch.ones(5, 3, 2))
        x = torch.randn(2, 5, 3, 4)
        x[..., 2:] = 1
        graph = torch.ones(2, 5, 3, 3)
        expected, actual = baseline(x, graph), model(x, graph)
        torch.testing.assert_close(actual["failure_logits"], expected["failure_logits"])
        torch.testing.assert_close(actual["node_scores"], expected["node_scores"])
        loss = -actual["node_logits"].log_softmax(-1)[:, 0].mean()
        loss.backward()
        self.assertIsNone(model.branches[0].project.weight.grad)
        self.assertIsNone(model.root_head[0].weight.grad)
        self.assertGreater(float(model.fine_root_head[-1].weight.grad.abs().sum()), 0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        updated = model(x, graph)
        cached = model.forward_evidence(x, graph, expected["node_logits"].detach())
        torch.testing.assert_close(updated["node_scores"], cached["node_scores"])
        torch.testing.assert_close(updated["failure_logits"], expected["failure_logits"])

    def test_control_has_same_capacity_and_masks_missing_deviation(self):
        local, relation = self.make(False), self.make(True)
        self.assertEqual(sum(p.numel() for p in local.parameters()), sum(p.numel() for p in relation.parameters()))
        local.configure_normal_reference(torch.zeros(5, 3, 2), torch.ones(5, 3, 2))
        x = torch.full((2, 5, 3, 4), 100.0)
        x[..., 2:] = 0
        _, deviations = local._local_metric_evidence(x)
        self.assertEqual(float(deviations.abs().sum()), 0)
        with self.assertRaises(ValueError):
            local.configure_normal_reference(torch.zeros(5, 3, 2), torch.zeros(5, 3, 2))

    def test_reference_uses_only_observed_healthy_training_windows(self):
        model = LTFPStateRelationEvidence(2, 1, 1, (0,), (1,), False, hidden_dim=8)
        dataset = SimpleNamespace(
            history=1, services=("a",), case_ids=["case"] * 5,
            failure_targets=np.array([0, 0, 0, 1, 0]),
            features=np.array([[[0, 1]], [[2, 1]], [[999, 0]], [[888, 1]], [[777, 1]]], dtype=np.float32),
        )
        configure_metric_reference(model, dataset, [1, 2, 3])
        self.assertEqual(float(model.normal_median.item()), 2.0)
        # Future/test values and unobserved training values cannot move the reference.
        dataset.features[4, 0, 0] = -10000
        dataset.features[2, 0, 0] = -999
        configure_metric_reference(model, dataset, [1, 2, 3])
        self.assertEqual(float(model.normal_median.item()), 2.0)

    def test_graph_control_only_permutes_sidecar_relations(self):
        model = self.make(True).eval()
        model.configure_normal_reference(torch.zeros(5, 3, 2), torch.ones(5, 3, 2))
        with torch.no_grad():
            model.fine_root_head[-1].weight.normal_()
        x = torch.randn(2, 5, 3, 4)
        x[..., 2:] = 1
        graph = torch.eye(3).expand(2, 5, 3, 3).clone()
        graph[:, :, 0, 1] = 5
        graph[:, :, 1, 0] = 5
        original = model(x, graph)
        model.relation_permutation.copy_(torch.tensor([2, 1, 0]))
        permuted = model(x, graph)
        torch.testing.assert_close(original["failure_logits"], permuted["failure_logits"])
        self.assertFalse(torch.allclose(original["node_logits"], permuted["node_logits"]))
        model.use_relation_graph = False
        local = model(x, graph)
        model.relation_permutation.copy_(torch.arange(3))
        torch.testing.assert_close(local["node_logits"], model(x, graph)["node_logits"])


if __name__ == "__main__":
    unittest.main()
