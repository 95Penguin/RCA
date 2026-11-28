"""Serial attribution invariants and unified two-output inference."""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.cpca import Span, decompose_tree, replay, score_request
from ocsid.clocks import align_span_clocks
from ocsid.inference import attribute_requests, predict_and_attribute
from models.ocsid import OCSIDModel


class V2Tests(unittest.TestCase):
    def test_randomized_replay_equals_linear_score(self):
        rng = random.Random(42)
        for policy in ("pre", "post"):
            for _ in range(30):
                spans = [Span("r", "", "frontendservice", "frontend", 0., 200.)]
                current = 10.
                for i in range(4):
                    duration = rng.uniform(5, 20)
                    spans.append(Span(str(i), "r", "s" + str(i % 2), "op", current, duration))
                    current += duration + rng.uniform(0, 10)
                by_id, children, root, local, pre, post = decompose_tree(spans, "ob", policy)
                self.assertAlmostEqual(replay(root, children, local, pre, post, None, {}, {}), 200.)
                self.assertTrue(all(v == 0 for v in score_request(spans, {}, {}, "ob", gap_policy=policy)[1].values()))
                nl = {(s.service, s.operation): rng.uniform(0, 40) for s in spans}
                ne = {(by_id[s.parent].service, by_id[s.parent].operation, s.service, s.operation):
                      (rng.uniform(0, 20), rng.uniform(0, 20)) for s in spans if s.parent}
                for mode in ("excess", "replace"):
                    scores = score_request(spans, nl, ne, "ob", mode, policy)[1]
                    for service, value in scores.items():
                        duration = replay(root, children, local, pre, post, service, nl, ne, mode)
                        self.assertAlmostEqual(value, max(200. - duration, 0.))

    def test_bounded_clock_alignment_and_official_score_equivalence(self):
        spans = [Span("r", "", "frontendservice", "frontend", 0., 10000.),
                 Span("c", "r", "payment", "pay", 3000., 7500.)]
        self.assertIsNone(decompose_tree(spans, "ob"))
        aligned = align_span_clocks(spans, 1000.)
        self.assertEqual(aligned[0], spans[0])
        self.assertEqual(aligned[1].start, 2500.)
        parsed = decompose_tree(spans, "ob", clock_tolerance_us=1000.)
        _, children, root, local, pre, post = parsed
        self.assertEqual(replay(root, children, local, pre, post, None, {}, {}), 10000.)
        nl = {("payment", "pay"): 500.}
        self.assertEqual(score_request(aligned, nl, {}, "ob")[1],
                         score_request(spans, nl, {}, "ob", clock_tolerance_us=1000.)[1])
        self.assertIsNone(align_span_clocks([spans[0],
            Span("c", "r", "payment", "pay", 3000., 12000.)], 1000.))

    def test_non_entry_requests_are_not_attributed(self):
        health = [Span("r", "", "emailservice", "/grpc.health.v1.Health/Check", 0., 100.)]
        self.assertIsNone(score_request(health, {}, {}, "ob"))

    def test_invalid_and_parallel_trees_are_rejected(self):
        root = Span("r", "", "frontendservice", "frontend", 0., 100.)
        for spans in ([root, root], [root, Span("o", "", "s", "op", 0., 20.)],
                      [root, Span("a", "r", "a", "op", 10., 50.),
                       Span("b", "r", "b", "op", 40., 40.)],
                      [root, Span("a", "r", "a", "op", 80., 30.)]):
            self.assertIsNone(decompose_tree(spans, "ob"))

    def test_operation_reordering_preserves_system_risk(self):
        torch.manual_seed(42)
        model = OCSIDModel(4, operation_dim=3, hidden_dim=8, temporal_layers=1,
                           graph_layers=1, operation_hidden=4, dropout=0.).eval()
        x, a = torch.randn(2, 5, 3, 4), torch.ones(2, 5, 3, 3)
        o, m = torch.randn(2, 5, 4, 3), torch.ones(2, 5, 4)
        perm = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            old, new = model(x, a, o, m), model(x, a, o[:, :, perm], m[:, :, perm])
        torch.testing.assert_close(old["operation_scores"][:, perm], new["operation_scores"])
        torch.testing.assert_close(old["probability"], new["probability"])

    def test_unified_api_needs_no_labels_and_restores_model_mode(self):
        model = OCSIDModel(2, operation_dim=1, hidden_dim=4, temporal_layers=1,
                           graph_layers=1, operation_hidden=2, dropout=0., aux_root=True)
        spans = [Span("r", "", "frontendservice", "frontend", 0., 100.),
                 Span("c", "r", "payment", "pay", 10., 80.)]
        kwargs = dict(requests=[spans], normal_local={("payment", "pay"): 10.},
                      normal_edge={}, normal_entry_p95={("frontendservice", "frontend"): 30.},
                      dataset="ob", min_op_requests=1)
        args = (model, torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2),
                torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1))
        out = predict_and_attribute(*args, threshold=0., **kwargs)
        self.assertTrue(model.training)
        self.assertEqual(out["attribution"]["ranking"][0], "payment")
        self.assertEqual(out["attribution"]["scores"]["payment"], .7)
        self.assertIsNone(predict_and_attribute(*args, threshold=1., **kwargs)["attribution"])
        self.assertTrue(attribute_requests([], {}, {}, {}, "ob")["abstained"])


if __name__ == "__main__":
    unittest.main()
