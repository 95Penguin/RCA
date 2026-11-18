import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.service_risk import ServiceRiskInteraction


class ServiceRiskTests(unittest.TestCase):
    def test_zero_residual_and_bound_and_frozen_inputs(self):
        model = ServiceRiskInteraction(38)
        x = torch.randn(2, 3, 38, requires_grad=True)
        failure = torch.randn(2, 2, requires_grad=True)
        root = torch.randn(2, 3, requires_grad=True)
        out = model(x, failure, root)
        torch.testing.assert_close(out["failure_logits"], failure)
        torch.testing.assert_close(out["auxiliary_root_logits"], root)
        with torch.no_grad():
            model.system_residual[-1].bias.fill_(100)
        out = model(x, failure, root)
        self.assertLessEqual(float(out["system_residual"].detach().max()), 1.0)
        out["failure_logits"].sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(failure.grad)
        self.assertIsNone(root.grad)

    def test_auxiliary_root_supervision_shapes_shared_service_encoder(self):
        model = ServiceRiskInteraction(38)
        with torch.no_grad():
            model.root_auxiliary.weight.normal_(std=0.1)
        output = model(torch.randn(2, 3, 38), torch.randn(2, 2), torch.randn(2, 3))
        (-output["auxiliary_root_logits"].log_softmax(-1)[:,0].mean()).backward()
        self.assertGreater(float(model.service_encoder[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.system_residual[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
