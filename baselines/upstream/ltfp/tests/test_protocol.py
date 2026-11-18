from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ltfp.protocol import (
    combine_topology_and_correlation,
    fold_roles,
    label_quality_gate,
    make_scenario_folds,
    validate_split_manifest,
)


class ProtocolTests(unittest.TestCase):
    def make_records(self) -> pd.DataFrame:
        rows = []
        for scenario_idx in range(10):
            for repetition in range(3):
                rows.append(
                    {
                        "case_id": f"system/root{scenario_idx}_fault/{repetition}",
                        "scenario_id": f"system/root{scenario_idx}_fault",
                        "system": "sock-shop",
                    }
                )
        return pd.DataFrame(rows)

    def test_scenario_fold_assignment_has_no_leakage(self):
        split = make_scenario_folds(self.make_records(), n_folds=5, seed=7)
        validate_split_manifest(split, n_folds=5)
        for fold in range(5):
            roles = fold_roles(split, fold)
            self.assertFalse(roles["train"] & roles["validation"])
            self.assertFalse(roles["train"] & roles["test"])
            by_case = dict(zip(split.case_id, split.scenario_id))
            role_scenarios = {key: {by_case[item] for item in values} for key, values in roles.items()}
            self.assertFalse(role_scenarios["train"] & role_scenarios["validation"])
            self.assertFalse(role_scenarios["train"] & role_scenarios["test"])

    def test_split_is_deterministic(self):
        left = make_scenario_folds(self.make_records(), n_folds=5, seed=7)
        right = make_scenario_folds(self.make_records(), n_folds=5, seed=7)
        pd.testing.assert_frame_equal(left, right)

    def test_correlation_is_symmetric_and_preserves_dependency(self):
        dependency = np.eye(3, dtype=np.float32)
        dependency[0, 1] = dependency[1, 0] = 1.0
        signal = np.asarray([[0, 0, 1], [1, 1, 0], [2, 2, 2], [3, 3, 1]], dtype=float)
        combined, correlation = combine_topology_and_correlation(dependency, signal, top_k=1)
        np.testing.assert_allclose(correlation, correlation.T)
        self.assertEqual(combined[0, 1], 1.0)
        self.assertTrue(np.all(np.diag(combined) == 1.0))

    def test_label_gate_derives_dataset_subset_from_case_id(self):
        rows = []
        for subset in ("RE1-OB", "RE3-OB", "RE1-SS", "RE2-SS"):
            for index in range(5):
                rows.append(
                    {
                        "case_id": f"{subset}/root_fault/{index}",
                        "system": "shared-system-name",
                        "failure_time": 100.0 if index < 3 else np.nan,
                    }
                )
        gate = label_quality_gate(pd.DataFrame(rows))
        self.assertEqual(gate["n_passing_subsets"], 4)
        self.assertEqual(gate["route_status"], "eligible_for_human_audit")


if __name__ == "__main__":
    unittest.main()
