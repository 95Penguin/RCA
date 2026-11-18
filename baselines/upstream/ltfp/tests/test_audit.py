from __future__ import annotations

import unittest

import pandas as pd

from ltfp.audit import evaluate_two_reviewer_audit, stratified_audit_sample


class AuditTests(unittest.TestCase):
    def test_twenty_percent_sample_is_stratified_and_deterministic(self):
        records = pd.DataFrame(
            [
                {
                    "case_id": f"RE1-SS/root_{idx}/1",
                    "scenario_id": f"RE1-SS/root_{idx}",
                    "system": "sock-shop",
                    "candidate_failure_time": 100 + idx,
                    "candidate_event_observed": True,
                    "candidate_reason": "candidate",
                }
                for idx in range(10)
            ]
        )
        left = stratified_audit_sample(records, fraction=0.2, seed=7)
        right = stratified_audit_sample(records, fraction=0.2, seed=7)
        self.assertEqual(len(left), 2)
        pd.testing.assert_frame_equal(left, right)

    def test_quality_gate_uses_kappa_and_onset_difference(self):
        base = {
            "case_id": ["a", "b", "c", "d"],
            "reviewer": ["one"] * 4,
            "event_observed": [True, True, False, False],
            "onset_time": [100.0, 200.0, "", ""],
            "decision": ["accept"] * 4,
            "notes": [""] * 4,
        }
        other = dict(base)
        other["reviewer"] = ["two"] * 4
        other["onset_time"] = [115.0, 215.0, "", ""]
        report = evaluate_two_reviewer_audit(pd.DataFrame(base), pd.DataFrame(other))
        self.assertTrue(report["passes"])
        self.assertEqual(report["cohens_kappa"], 1.0)
        self.assertEqual(report["onset_difference_median_seconds"], 15.0)


if __name__ == "__main__":
    unittest.main()
