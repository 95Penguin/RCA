"""Partial traces may retain evidence only on the observed entry component."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.cpca import Span, entry_component, decompose_tree, score_request, replay
from ocsid.inference import attribute_requests


class PartialTraceTests(unittest.TestCase):
    def fixture(self):
        return [Span("r", "", "frontendservice", "frontend", 0., 100.),
                Span("c", "r", "payment", "pay", 10., 80.),
                Span("orphan", "missing", "irrelevant", "op", 20., 200.)]

    def test_orphans_are_omitted_without_inventing_edges(self):
        spans = self.fixture()
        self.assertIsNone(decompose_tree(spans, "ob"))
        self.assertEqual([s.sid for s in entry_component(spans, "ob")], ["r", "c"])
        parsed = decompose_tree(spans, "ob", tree_policy="entry-component")
        _, children, root, local, pre, post = parsed
        self.assertEqual(replay(root, children, local, pre, post, None, {}, {}), 100.)
        _, scores, _ = score_request(spans, {("payment", "pay"): 10.}, {}, "ob",
                                      tree_policy="entry-component")
        self.assertAlmostEqual(scores["payment"], 70.)
        self.assertNotIn("irrelevant", scores)

    def test_no_entry_or_multiple_entries_still_rejected(self):
        spans = self.fixture()
        self.assertIsNone(entry_component(spans[1:], "ob"))
        spans.append(Span("r2", "", "frontendservice", "frontend", 200., 100.))
        self.assertIsNone(entry_component(spans, "ob"))

    def test_min_count_filter_does_not_discard_slow_low_volume_business(self):
        slow = [Span("slow", "", "gateway", "POST /slow", 0., 1000.),
                Span("p", "slow", "payment", "pay", 10., 900.)]
        fast = [Span("fast", "", "gateway", "POST /healthy", 0., 50.),
                Span("c", "fast", "catalog", "read", 10., 20.)]
        refs = {("payment", "pay"): 10., ("catalog", "read"): 20.}
        entry_refs = {("gateway", "POST /slow"): 100., ("gateway", "POST /healthy"): 50.}
        requests = [slow, slow] + [fast] * 25
        old = attribute_requests(requests, refs, {}, entry_refs, "tt", min_op_requests=20)
        new = attribute_requests(requests, refs, {}, entry_refs, "tt")
        self.assertTrue(old["abstained"])
        self.assertEqual(new["ranking"][0], "payment")
        self.assertEqual(new["operation_request_counts"][("gateway", "POST /slow")], 2)

    def test_low_volume_operation_can_be_scored_without_labels(self):
        result = attribute_requests([self.fixture()], {("payment", "pay"): 10.}, {},
                 {("frontendservice", "frontend"): 20.}, "ob", min_op_requests=1,
                 tree_policy="entry-component")
        self.assertEqual(result["ranking"][0], "payment")
        self.assertAlmostEqual(result["scores"]["payment"], .7)
        self.assertEqual(result["partial_requests"], 1)
        self.assertEqual(result["omitted_spans"], 1)


if __name__ == "__main__":
    unittest.main()
