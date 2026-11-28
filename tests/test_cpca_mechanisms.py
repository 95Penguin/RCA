"""Mechanism controls isolate costs while keeping the same parsed request."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.cpca import Span, decompose_tree, score_request, score_decomposition


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.spans = [Span("f", "", "frontendservice", "frontend", 0., 1000.),
                      Span("c", "f", "checkout", "checkout", 20., 800.),
                      Span("p", "c", "payment", "pay", 30., 620.)]
        self.nl = {("payment", "pay"): 50.}
        self.ne = {("checkout", "checkout", "payment", "pay"): (5., 5.)}

    def test_full_local_and_edge_ledger(self):
        full = score_request(self.spans, self.nl, self.ne, "ob", "replace")[1]
        local = score_request(self.spans, self.nl, self.ne, "ob", "replace", components="local")[1]
        edge = score_request(self.spans, self.nl, self.ne, "ob", "replace", components="edge")[1]
        self.assertEqual(full["payment"], 740.)
        self.assertEqual(local["payment"], 570.)
        self.assertEqual(edge["payment"], 170.)

    def test_edge_owner_is_the_only_changed_rule(self):
        caller = score_request(self.spans, self.nl, self.ne, "ob", "replace", edge_owner="caller")[1]
        self.assertEqual(caller["payment"], 570.)
        self.assertEqual(caller["checkout"], 170.)

    def test_parsed_scoring_matches_public_request_api(self):
        parsed = decompose_tree(self.spans, "ob")
        for mode in ("replace", "excess"):
            self.assertEqual(score_decomposition(parsed, self.nl, self.ne, "ob", mode),
                             score_request(self.spans, self.nl, self.ne, "ob", mode))
