"""Historical boundaries, missing-evidence policy and original-model gradients."""
from __future__ import annotations

import importlib.util
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "artifacts/baselines/.dgl"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/baselines/.matplotlib"))

import numpy as np
import pandas as pd
import torch

from baselines.adapters.raw import RawCaseStore, log_template
from baselines.adapters.events import EventFeatures
from baselines.specialists import root_metrics


class HistoricalSpecialistTests(unittest.TestCase):
    def test_selective_raw_io_preserves_used_observations(self):
        tables = {
            "simple_metrics.csv": [{"time": 100, "a_cpu": 1.5}],
            "traces.csv": [{"traceID": "t", "spanID": "s", "parentSpanID": "", "serviceName": "a",
                            "operationName": "op", "startTime": 100000000, "duration": 1000, "statusCode": 500}],
            "logs.csv": [{"timestamp": 100000000000, "container_name": "a", "message": "request 123", "level": "error"}],
        }
        case_id = "RE2-OB_a_cpu_1"
        data = SimpleNamespace(dataset=SimpleNamespace(case_ids=[case_id], services=["a"]))
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/"raw.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for name, rows in tables.items():
                    buffer = io.StringIO()
                    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                    output.writestr("RE2-OB/a_cpu/1/"+name, buffer.getvalue())
            full = RawCaseStore(archive, data)
            try:
                expected = full.case(case_id)
                for traces, logs in ((False, False), (True, False), (False, True)):
                    store = RawCaseStore(archive, data, include_traces=traces, include_logs=logs)
                    original_rows = store.rows
                    def read(case, name):
                        self.assertTrue(name != "traces.csv" or traces)
                        self.assertTrue(name != "logs.csv" or logs)
                        return original_rows(case, name)
                    store.rows = read
                    try:
                        actual = store.case(case_id)
                        pd.testing.assert_frame_equal(actual[0], expected[0])
                        self.assertEqual(actual[1], expected[1] if traces else [])
                        self.assertEqual(actual[2], expected[2] if logs else [])
                    finally:
                        store.close()
            finally:
                full.close()

    def fixture(self):
        store = RawCaseStore.__new__(RawCaseStore)
        store.dataset = SimpleNamespace(timestamp_epochs=[240], history=5, case_ids=["case"])
        store.data = SimpleNamespace(train=[0])
        store.service_index = {"a": 0, "b": 1}
        metrics = pd.DataFrame({"time": [0, 299, 300, 600], "a_cpu": [1., 2., 999., 9999.]})
        spans = [dict(trace="t", span="parent", parent="", service="a", operation="op",
                      start=299, available=301, latency=2., error=False),
                 dict(trace="t", span="child", parent="parent", service="b", operation="op",
                      start=280, available=281, latency=1., error=True)]
        logs = [dict(available=299, service="a", text="okay", level="info"),
                dict(available=300, service="a", text="future", level="error")]
        store.case = lambda _: (metrics, spans, logs)
        return store, metrics, spans, logs

    def test_history_cutoff_and_unfinished_span_exclusion(self):
        store, _, _, _ = self.fixture()
        metrics, spans, logs = store.window(0)
        self.assertEqual(metrics.time.tolist(), [0, 299])
        self.assertEqual([s["span"] for s in spans], ["child"])
        self.assertEqual([s["text"] for s in logs], ["okay"])

    def test_future_mutation_cannot_change_invocation_features(self):
        store, metrics, spans, logs = self.fixture()
        before = store.invocation_frame(0, {})
        metrics.loc[metrics.time >= 300, "a_cpu"] = -1e9
        spans[0]["service"] = "future-secret"
        logs[1]["text"] = "future-secret"
        after = store.invocation_frame(0, {})
        pd.testing.assert_frame_equal(before, after)
        # An unfinished/out-of-window parent is never read from the full case.
        self.assertEqual(after.iloc[0].source, "b")

    def test_reference_and_embedding_reject_nontraining_fit_windows(self):
        store, _, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "training windows only"):
            store.training_reference(fit_ends=[123])
        with self.assertRaisesRegex(ValueError, "training windows only"):
            EventFeatures(store, None, [123], ROOT, "hades")

    def test_fixed_log_parser_has_no_fitted_template_state(self):
        self.assertEqual(log_template("2024-01-01 12:03:04 request 123 deadbeef"),
                         log_template("2025-02-02 15:06:07 request 456 cafebabe"))

    def test_unavailable_rca_is_not_removed_from_denominator(self):
        d = SimpleNamespace(node_targets=np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
                            failure_targets=np.array([1, 1, 0]), timestamps=["a", "b", "c"],
                            _root_label_valid={"a": True, "b": True})
        result = root_metrics(SimpleNamespace(dataset=d), [0, 1, 2], [None, [0, 1, 0], None])
        self.assertEqual(result["attributed_positive_windows"], 2)
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["mrr"], .5)

    def test_ties_and_invalid_scores_are_explicit(self):
        d = SimpleNamespace(node_targets=np.array([[0, 1, 0]]), failure_targets=np.array([1]),
                            timestamps=["a"], _root_label_valid={"a": True})
        data = SimpleNamespace(dataset=d)
        result = root_metrics(data, [0], [[0, 0, 0]])
        self.assertEqual(result["mrr"], .5)
        self.assertAlmostEqual(result["tie_expected_mrr"], (1+.5+1/3)/3)
        with self.assertRaises(ValueError):
            root_metrics(data, [0], [[0, float("nan"), 0]])


class NeuralSpecialistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @unittest.skipUnless(importlib.util.find_spec("dgl"), "Requires baseline DGL environment")
    def test_hades_registered_time_encoder_gradients_and_batch_independence(self):
        from baselines.adapters.learned_specialists import HadesAuthor
        import models
        project_models = sys.modules["models"]
        torch.manual_seed(4)
        model = HadesAuthor(6, hidden=8).eval()
        metrics, logs = torch.randn(3, 5, 6), torch.randn(3, 5, 16, 32)
        full, single = model(metrics, logs), model(metrics[:1], logs[:1])
        self.assertEqual(single.shape, (1, 2))
        torch.testing.assert_close(full[:1], single, atol=2e-6, rtol=2e-5)
        full.square().mean().backward()
        parameters = [p for name, p in model.named_parameters() if "kpi_encoder.t_encoders" in name]
        self.assertTrue(parameters)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))
        self.assertIs(sys.modules["models"], project_models)

    def test_anofusion_heterogeneous_graph_batch_independence_and_gradients(self):
        from baselines.adapters.learned_specialists import AnoFusionAuthor
        rng = np.random.default_rng(1)
        features = rng.normal(size=(30, 3, 6)).astype(np.float32)
        data = SimpleNamespace(dataset=SimpleNamespace(features=features, history=5, services=list("abc"),
                               feature_columns=list("uvwxyz")), input_fit_indices=list(range(30)),
                               groups={"metric": [0, 1], "log": [2, 3], "trace": [4, 5]})
        torch.manual_seed(2)
        model = AnoFusionAuthor(data).eval()
        x = torch.randn(2, 5, 3, 6)
        full = model(x)
        torch.testing.assert_close(full["risk"][:1], model(x[:1])["risk"], atol=2e-6, rtol=2e-5)
        full["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    @unittest.skipUnless(importlib.util.find_spec("dgl"), "Requires baseline DGL environment")
    def test_diagfusion_both_heads_receive_gradients(self):
        import dgl
        from baselines.adapters.learned_specialists import DiagFusionAuthor
        graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
        for name in ("metric", "log", "trace"):
            graph.ndata[name] = torch.randn(3, 8)
        model = DiagFusionAuthor(8, 3, 2, 8)
        root, fault = model(dgl.batch([graph, graph]))
        loss = torch.nn.functional.cross_entropy(root, torch.tensor([0, 1]))
        loss += torch.nn.functional.cross_entropy(fault, torch.tensor([0, 1]))
        loss.backward()
        for head in (model.root, model.type_classifier):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))

    @unittest.skipUnless(importlib.util.find_spec("dgl"), "Requires baseline DGL environment")
    def test_tvdiag_same_label_augmented_objective_is_finite(self):
        import dgl
        from baselines.adapters.learned_specialists import TVDiagAuthor
        graphs = []
        for _ in range(2):
            graph = dgl.graph(([0, 1, 2, 0], [1, 2, 0, 0]), num_nodes=3)
            for name in ("metric", "log", "trace"):
                graph.ndata[name] = torch.randn(3, 8)
            graph.ndata["root"] = torch.tensor([1, 0, 0])
            graphs.append(graph)
        model = TVDiagAuthor(8, 2, 8)
        loss = model.objective(graphs, torch.tensor([0, 0]), torch.tensor([1, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    @unittest.skipUnless(importlib.util.find_spec("gluonts"), "Requires specialist GluonTS environment")
    def test_maat_training_gradient_seeded_forecast_and_observation_masks(self):
        from baselines.adapters.learned_specialists import MaatAuthor
        torch.manual_seed(11)
        model = MaatAuthor(3, diff_steps=3, samples=2, hidden=8)
        x, future = torch.randn(2, 5, 3), torch.randn(2, 3)
        observed = torch.ones_like(x); observed[0, 0, 0] = 0
        torch.testing.assert_close(model.arguments(x, observed)["past_observed_values"], observed)
        loss = model.loss(x, future, torch.ones_like(future), observed)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.train_net.parameters()))
        state = torch.random.get_rng_state().clone()
        a = model.forecast(x[:1], seed=7, observed=observed[:1])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        b = model.forecast(x[:1], seed=7, observed=observed[:1])
        torch.testing.assert_close(a, b)
        self.assertEqual(a.shape, (1, 2, 3))


class StatisticalSpecialistTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mlxtend"), "Requires specialist mining environment")
    def test_tracerca_singleton_optimization_preserves_original_ranking(self):
        from mlxtend.frequent_patterns import fpgrowth
        from baselines.adapters.statistical_rca import TraceRCAAuthor
        model = TraceRCAAuthor(["a", "b", "c"])
        frame = pd.DataFrame([("t1", "a", "b", True), ("t1", "b", "c", True),
                              ("t2", "a", "b", True), ("t3", "a", "c", False),
                              ("t4", "a", "c", False)], columns=["trace_id", "source", "target", "predict"])
        optimized = model(frame)
        globals_ = model.api.ItemsetHandler._frequent_pattern_on_abnormal.__globals__
        previous = globals_["fpgrowth"]
        try:
            globals_["fpgrowth"] = fpgrowth
            original = model(frame)
        finally:
            globals_["fpgrowth"] = previous
        np.testing.assert_array_equal(optimized, original)
        self.assertIsNone(model(frame.assign(predict=False)))

    @unittest.skipUnless(importlib.util.find_spec("statsmodels"), "Requires CIRCA dependencies")
    def test_circa_reference_and_history_are_loaded_as_exact_segments(self):
        from baselines.adapters.statistical_rca import CIRCAAuthor
        normal = pd.DataFrame({"a_cpu": np.arange(20.), "a_latency-90": np.arange(20.)})
        current = pd.DataFrame({"a_cpu": np.arange(100., 110.), "a_latency-90": np.arange(100., 110.)})
        model = CIRCAAuthor(["a"], set(), normal, "a")
        def inspect(case, current):
            self.assertEqual(case.train_window, len(normal))
            self.assertEqual(case.test_window, 10)
            data = case.load_data(model.graph, current)
            np.testing.assert_allclose(data[model.nodes["a_cpu"]], np.r_[normal.a_cpu, np.arange(100., 110.)])
            return []
        model.model = SimpleNamespace(analyze=inspect)
        model(normal, current)


if __name__ == "__main__":
    unittest.main()
