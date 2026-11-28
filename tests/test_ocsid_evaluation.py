"""The paper evaluator uses a fixed cohort and zero-fills absent services."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.evaluation import evaluate_attribution


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        cohort = {"folds": [{"services": ["root", "symptom", "other"], "test_predictions": [
            {"case": "c", "minute": t, "event": "e", "root": "root", "failure_target": 1}
            for t in [1, 2]]}]}
        audit = {"cases": {"c": {"case_id": "c", "windows": [
            {"minute": 1, "scores_by_method": {"cpca": {"root": .6, "symptom": 1.}}},
            {"minute": 2, "scores_by_method": {"cpca": {"root": .6}}}]}}}
        return audit, cohort

    def test_missing_service_is_zero_in_event_mean(self):
        audit, cohort = self.fixture()
        r = evaluate_attribution(audit, cohort)
        self.assertEqual(r["event"]["hit_at_1"], 1.)
        self.assertEqual(r["window"]["hit_at_1"], .5)

    def test_missing_windows_abstain_not_disappear(self):
        audit, cohort = self.fixture()
        audit["cases"]["c"]["windows"] = []
        r = evaluate_attribution(audit, cohort)
        self.assertEqual(r["positive_windows"], 2)
        self.assertEqual(r["event"]["mrr"], 0.)

    def test_worst_tie_and_inconsistent_root(self):
        audit, cohort = self.fixture()
        for w in audit["cases"]["c"]["windows"]:
            w["scores_by_method"]["cpca"] = {"symptom": 1.}
        self.assertEqual(evaluate_attribution(audit, cohort)["event_ranks"]["e"], 3)
        cohort["folds"][0]["test_predictions"][1]["root"] = "other"
        with self.assertRaises(ValueError):
            evaluate_attribution(audit, cohort)
