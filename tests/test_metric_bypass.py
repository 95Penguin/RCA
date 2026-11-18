"""Behavioral controls for explicit metric deviation calibration."""

import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.factory import build_model
from models.ltfp import LTFPBaseline


class MetricBypassTests(unittest.TestCase):
    columns = ("metric_cpu", "metric_mem", "observed_cpu", "observed_mem")

    def make(self, deviation):
        name = "metric-deviation-bypass" if deviation else "raw-metric-bypass"
        torch.manual_seed(3)
        model = build_model(f"ltfp-{name}-evidence", feature_columns=self.columns, num_services=3).model.eval()
        model.configure_normal_reference(torch.zeros(5, 3, 2), torch.ones(5, 3, 2))
        return model

    def test_zero_initialization_capacity_and_protected_gradient(self):
        raw, dev = self.make(False), self.make(True)
        self.assertEqual(sum(p.numel() for p in raw.bypass_head.parameters()),
                         sum(p.numel() for p in dev.bypass_head.parameters()))
        baseline = LTFPBaseline(4, feature_groups=[[0, 1, 2, 3]]).eval()
        baseline.load_state_dict({k:v for k,v in raw.state_dict().items() if k in baseline.state_dict()})
        x = torch.randn(2, 5, 3, 4)
        x[..., 2:] = 1
        graph = torch.ones(2, 5, 3, 3)
        expected = baseline(x, graph)
        for model in (raw, dev):
            output = model(x, graph)
            torch.testing.assert_close(output["node_scores"], expected["node_scores"])
            torch.testing.assert_close(output["failure_logits"], expected["failure_logits"])
            (-output["node_logits"].log_softmax(-1)[:, 0].mean()).backward()
            self.assertIsNone(model.branches[0].project.weight.grad)
            self.assertIsNone(model.root_head[0].weight.grad)
            self.assertGreater(float(model.bypass_head[-1].weight.grad.abs().sum()), 0)

    def test_raw_control_ignores_reference_and_missing_values(self):
        raw = self.make(False)
        with torch.no_grad():
            raw.bypass_head[-1].weight.normal_()
        x = torch.randn(2, 5, 3, 4)
        x[..., 2:] = 1
        graph = torch.ones(2, 5, 3, 3)
        logits = torch.randn(2, 3)
        a = raw.forward_evidence(x, graph, logits)["node_logits"]
        raw.configure_normal_reference(torch.ones(5, 3, 2)*50, torch.ones(5, 3, 2))
        torch.testing.assert_close(a, raw.forward_evidence(x, graph, logits)["node_logits"])
        x[..., 2:] = 0
        a = raw.forward_evidence(x, graph, logits)["node_logits"]
        x[..., :2] = 999
        torch.testing.assert_close(a, raw.forward_evidence(x, graph, logits)["node_logits"])

    def test_deviation_reference_reaches_output_and_cached_path_matches(self):
        model = self.make(True)
        with torch.no_grad():
            model.bypass_head[0].weight.fill_(0.01)
            model.bypass_head[0].bias.fill_(1)
            model.bypass_head[-1].weight.fill_(0.1)
        x = torch.randn(2, 5, 3, 4)
        x[..., 2:] = 1
        graph = torch.ones(2, 5, 3, 3)
        baseline = LTFPBaseline(4, feature_groups=[[0, 1, 2, 3]]).eval()
        baseline.load_state_dict({k:v for k,v in model.state_dict().items() if k in baseline.state_dict()})
        logits = baseline(x, graph)["node_logits"].detach()
        before = model.forward_evidence(x, graph, logits)
        torch.testing.assert_close(before["node_scores"], model(x, graph)["node_scores"])
        model.configure_normal_reference(torch.ones(5, 3, 2)*50, torch.ones(5, 3, 2))
        after = model.forward_evidence(x, graph, logits)
        self.assertFalse(torch.allclose(before["bypass_residual"], after["bypass_residual"]))


if __name__ == "__main__":
    unittest.main()
