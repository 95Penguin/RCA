from __future__ import annotations

import unittest

import numpy as np

from ltfp.dataset_v2 import FailureWindowDataset, FeatureSchema
from ltfp.evaluation_v2 import evaluate_events, evaluate_first_alert_roots


class FirstAlertEvaluationTests(unittest.TestCase):
    def dataset(self) -> FailureWindowDataset:
        n_windows = 6
        schema = FeatureSchema(
            version="test",
            services=("front-end", "orders"),
            modality_names=("metric",),
            modality_dims=(1,),
            features=(),
            max_features=1,
            fitted_case_ids=("RE1-SS/orders_cpu/1",),
        )
        end = np.asarray([0.0, 30.0, 60.0, 90.0, 120.0, 240.0])
        return FailureWindowDataset(
            X=np.zeros((n_windows, 2, 1, 1), dtype=np.float32),
            Y=np.asarray([[0], [0], [0], [0], [0], [1]], dtype=np.float32),
            label_mask=np.ones((n_windows, 1), dtype=bool),
            availability_mask=np.ones((n_windows, 2, 1), dtype=bool),
            feature_availability_mask=np.ones((n_windows, 2, 1, 1), dtype=bool),
            horizons=(60.0,),
            case_ids=("RE1-SS/orders_cpu/1",) * n_windows,
            scenario_ids=("RE1-SS/orders_cpu",) * n_windows,
            systems=("sock-shop",) * n_windows,
            end_times=end,
            inject_times=np.full(n_windows, 100.0),
            failure_times=np.full(n_windows, 300.0),
            censor_times=np.full(n_windows, 330.0),
            lead_times=300.0 - end,
            root_targets=np.ones(n_windows, dtype=int),
            root_active=end >= 100.0,
            root_labels=("orders",) * n_windows,
            schema=schema,
        )

    def test_earlier_false_alarm_cannot_be_rescued_by_later_timely_alarm(self) -> None:
        dataset = self.dataset()
        probabilities = np.asarray([[0.9], [0.9], [0.1], [0.9], [0.9], [0.9]])
        summary, cases = evaluate_events(
            dataset, np.arange(len(dataset.case_ids)), probabilities, threshold=0.5
        )
        self.assertEqual(summary[0]["event_true_positive"], 0)
        self.assertEqual(summary[0]["event_false_positive"], 1)
        self.assertEqual(summary[0]["event_false_negative"], 1)
        self.assertEqual(summary[0]["healthy_false_alert_episodes"], 2)
        self.assertAlmostEqual(summary[0]["false_alerts_per_hour"], 60.0)
        self.assertEqual(cases.iloc[0]["first_alert_time"], 120.0)
        self.assertFalse(bool(cases.iloc[0]["first_alert_within_horizon"]))

    def test_root_rank_is_taken_at_same_first_alarm(self) -> None:
        dataset = self.dataset()
        probabilities = np.asarray([[0.0], [0.0], [0.0], [0.0], [0.9], [0.9]])
        # The early first alarm ranks the true root second; the later timely
        # alarm ranks it first and must not replace the first alarm.
        root_logits = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ]
        )
        summary, cases = evaluate_first_alert_roots(
            dataset,
            np.arange(len(dataset.case_ids)),
            probabilities,
            root_logits,
            threshold=0.5,
            primary_horizon=60.0,
        )
        self.assertEqual(cases.iloc[0]["alert_time"], 120.0)
        self.assertEqual(cases.iloc[0]["rank"], 2)
        self.assertFalse(bool(cases.iloc[0]["alert_within_horizon"]))
        self.assertEqual(summary["conditional_top1"], 0.0)
        self.assertEqual(summary["joint_top3"], 0.0)

    def test_primary_horizon_must_be_declared_exactly(self) -> None:
        dataset = self.dataset()
        with self.assertRaisesRegex(ValueError, "not present"):
            evaluate_first_alert_roots(
                dataset,
                np.arange(len(dataset.case_ids)),
                np.zeros((len(dataset.case_ids), 1)),
                np.zeros((len(dataset.case_ids), 2)),
                threshold=0.5,
                primary_horizon=59.0,
            )


if __name__ == "__main__":
    unittest.main()
