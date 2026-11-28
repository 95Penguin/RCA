"""Regression contracts for corrected-v2 implementation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.ocsid import OCSIDModel
from ocsid.cpca import Span, decompose_tree, replay
from ocsid.training import _standardize


class ImplementationContracts(unittest.TestCase):
    def test_operation_gru_receives_each_operations_own_history(self):
        model = OCSIDModel(feature_dim=1, operation_dim=1, hidden_dim=4,
                           operation_hidden=2, temporal_layers=1, graph_layers=1,
                           dropout=0.0).eval()
        operations = torch.tensor([[[[0.], [100.]], [[1.], [101.]], [[2.], [102.]]]])
        captured = []
        hook = model.operation_encoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        with torch.no_grad():
            model(torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1),
                  operations, torch.ones(1, 3, 2))
        hook.remove()
        torch.testing.assert_close(captured[0].squeeze(-1),
                                   torch.tensor([[0., 1., 2.], [100., 101., 102.]]))

    def fixture(self):
        dataset = SimpleNamespace(features=np.zeros((10, 1, 1), dtype=np.float32),
                                  feature_columns=["observed_cpu"])
        x = np.ones((2, 5, 1, 1), dtype=np.float32)
        ops = np.array([[[[1.]], [[3.]]], [[[100.]], [[200.]]]], dtype=np.float32)
        return dataset, x, ops, np.ones((2, 2, 1), dtype=np.float32)

    def test_heldout_values_do_not_change_training_operation_features(self):
        dataset, x, ops, mask = self.fixture()
        first = _standardize(dataset, x, ops.copy(), mask, [4], [4, 9])[1][0]
        ops[1] *= 100
        second = _standardize(dataset, x, ops.copy(), mask, [4], [4, 9])[1][0]
        np.testing.assert_allclose(first, second)

    def test_binary_observation_flags_preserve_original_values(self):
        dataset, x, ops, mask = self.fixture()
        dataset.features[:5] = np.array([0., 0., 0., 0., 1.]).reshape(5, 1, 1)
        x[0] = dataset.features[:5]
        transformed, _ = _standardize(dataset, x, ops, mask, [4], [4, 9])
        np.testing.assert_array_equal(transformed, x)

    def test_serial_replay_without_intervention_conserves_duration(self):
        spans = [Span("r", "", "frontendservice", "frontend", 0., 100.),
                 Span("a", "r", "a", "op", 10., 20.),
                 Span("b", "r", "b", "op", 50., 20.)]
        _, children, root, local, pre, post = decompose_tree(spans, "ob")
        self.assertAlmostEqual(replay(root, children, local, pre, post, None, {}, {}),
                               root.duration)


if __name__ == "__main__":
    unittest.main()
