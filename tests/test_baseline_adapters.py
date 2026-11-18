"""Meaningful checks for external-model semantics, gradients and leakage."""
from __future__ import annotations

import os
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "artifacts/baselines/.dgl"))

import numpy as np
import torch

from baselines.adapters.data import HistoricalWindows, PreparedData
from baselines.adapters.frameworks import (ARTAuthor, EadroAuthor, LTFPAuthor,
                                          InstantOpsAdapter, permutation_scores)
from baselines.evaluation import binary_metrics, validation_threshold, summarize_predictions
from baselines.source import load_author_module, verify_sources


class BaselineAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, batch=3):
        torch.manual_seed(7)
        x = torch.randn(batch, 5, 3, 6)
        graph = torch.eye(3)[None, None].expand(batch, 5, -1, -1).clone()
        graph[:, :, 0, 1] = graph[:, :, 1, 0] = 1
        return x, graph

    def test_author_snapshot_hashes(self):
        self.assertEqual(len(verify_sources()), 13)

    def test_ltfp_wrapper_matches_author_forward(self):
        x, g = self.fixture()
        model = LTFPAuthor(6, {"metric": [0, 1], "log": [2, 3], "trace": [4, 5]},
                           8, 0.0, np.eye(3)).eval()
        actual = model(x, g)
        expected = model.model(x.permute(0, 2, 1, 3), g[:, -1])
        torch.testing.assert_close(actual["failure_logit"], expected["fault_logits"][:, 0])
        torch.testing.assert_close(actual["root_logits"], expected["root_logits"])

    @unittest.skipUnless(importlib.util.find_spec("dgl"), "Requires separate baseline DGL environment")
    def test_eadro_author_encoders_and_heads_receive_gradients(self):
        x, g = self.fixture()
        model = EadroAuthor(6, {"metric": [0, 1], "log": [2, 3]}, 3, 5, 8)
        model.trace_index = 4
        out = model(x, g)
        (out["failure_logit"].sum()+out["root_logits"].sum()).backward()
        for name in ("encoder.metric_model", "encoder.log_model", "encoder.trace_model", "encoder.status_model", "detecter", "localizer"):
            parameters = [p for n, p in model.model.named_parameters() if n.startswith(name)]
            self.assertTrue(parameters, name)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), name)
        model.eval()
        with torch.no_grad():
            full = model(x, g)
            single = model(x[:1], g[:1])
        torch.testing.assert_close(full["failure_logit"][:1], single["failure_logit"], atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(importlib.util.find_spec("dgl"), "Requires separate baseline DGL environment")
    def test_art_batch_independence_and_graph_layer_registration(self):
        x, g = self.fixture()
        model = ARTAuthor(6, 3, 8, 0.0).eval()
        full = model(x, g)
        single = model(x[:1], g[:1])
        torch.testing.assert_close(full["telemetry_prediction"][:1], single["telemetry_prediction"], atol=2e-6, rtol=2e-5)
        full["reconstruction_loss"].backward()
        names = dict(model.named_parameters())
        last = [p for name, p in names.items() if "GraphEncoder.convs.0" in name]
        self.assertTrue(last)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in last))

    def test_historical_adapter_excludes_target_telemetry(self):
        x, g = self.fixture(batch=1)
        features = np.concatenate((x[0].numpy(), np.ones((1, 3, 6))*10000))
        source = SimpleNamespace(features=features, history=5, adjacency=g[0],
                                 failure_targets=np.array([0, 0, 0, 0, 1, 0]),
                                 node_targets=np.array([[0, 0, 0]]*4+[[0, 1, 0], [0, 0, 0]], dtype=np.float32),
                                 timestamps=[str(i) for i in range(6)], _root_label_valid={"4": True})
        windows = HistoricalWindows(source, [4])
        before = windows[0]
        source.features[5] = -99999
        after = windows[0]
        self.assertEqual(set(before), {"x", "graph", "y", "root", "root_valid", "end"})
        torch.testing.assert_close(before["x"], after["x"])
        self.assertEqual(before["x"].shape[0], 5)

    def test_permutation_attribution_is_batch_independent_and_restores_mode(self):
        from models.instantops import InstantOpsGNNGRU
        x, g = self.fixture()
        model = InstantOpsAdapter(InstantOpsGNNGRU(6, 8, 8, dropout=0.0)).train()
        all_scores = permutation_scores(model, x, g, repeats=3, seed=1)
        one = permutation_scores(model, x[:1], g[:1], repeats=3, seed=1)
        self.assertTrue(model.training)
        torch.testing.assert_close(all_scores[:1], one, atol=1e-7, rtol=1e-5)

    def test_validation_threshold_ties_and_all_negative(self):
        labels, scores = np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.8, 0.5])
        self.assertEqual(validation_threshold(labels, scores), 0.8)
        self.assertEqual(binary_metrics(labels, scores, 0.8)["f1"], 1.0)
        t = validation_threshold(np.zeros(4), scores)
        self.assertGreater(t, scores.max())

    def test_real_re2_split_and_train_only_scaler(self):
        path = ROOT / "artifacts/re2_ob_panel_fair_v1"
        if not path.exists():
            self.skipTest("RE2 panel not installed")
        data = PreparedData(path, 0)
        d = data.dataset
        case_sets = [{d.case_ids[i] for i in ends} for ends in data.splits.values()]
        for i in range(3):
            for j in range(i):
                self.assertFalse(case_sets[i] & case_sets[j])
        train_times = set(data.input_fit_indices)
        test_times = {i for end in data.test for i in range(end-4, end+1)}
        self.assertFalse(train_times & test_times)
        self.assertTrue(data.normal_train)
        for end in data.train+data.validation+data.test:
            self.assertEqual(len(set(d.case_ids[end-4:end+1])), 1)
            target = d._time_indices[d.label_target_times[d.timestamps[end]]]
            self.assertEqual(d.timestamp_epochs[target]-d.timestamp_epochs[end], 60)


if __name__ == "__main__":
    unittest.main()
