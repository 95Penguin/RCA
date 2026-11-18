from __future__ import annotations

import unittest

import numpy as np

from ltfp.metrics import (
    assert_nested_probabilities,
    binary_metrics_v2,
    fit_shared_temperature,
    paired_scenario_bootstrap,
    sigmoid_temperature,
    tune_shared_threshold,
)


class MetricsV2Tests(unittest.TestCase):
    def test_shared_threshold_preserves_nested_alarms(self):
        y = np.asarray([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]])
        p = np.asarray([[0.1, 0.3, 0.8], [0.2, 0.7, 0.9], [0.6, 0.8, 0.95], [0.01, 0.02, 0.03]])
        threshold = tune_shared_threshold(y, p)
        alarms = (p >= threshold).astype(int)
        self.assertTrue(np.all(np.diff(alarms, axis=1) >= 0))
        assert_nested_probabilities(p)

    def test_temperature_is_positive_and_retains_monotonicity(self):
        logits = np.asarray([[-3.0, -1.0, 1.0], [-2.0, 0.0, 2.0], [0.5, 1.0, 3.0]])
        labels = np.asarray([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        temperature = fit_shared_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        assert_nested_probabilities(sigmoid_temperature(logits, temperature))

    def test_binary_metrics_include_calibration_and_imbalance(self):
        metrics = binary_metrics_v2(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.6, 0.9]), 0.5)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertAlmostEqual(metrics.mcc, 1.0)
        self.assertAlmostEqual(metrics.prevalence, 0.5)
        self.assertGreaterEqual(metrics.ece, 0.0)

    def test_bootstrap_samples_scenarios(self):
        result = paired_scenario_bootstrap(
            [0.7, 0.8, 0.6, 0.9],
            [0.5, 0.6, 0.4, 0.7],
            ["a", "a", "b", "b"],
            n_bootstrap=100,
            seed=7,
        )
        self.assertAlmostEqual(result["difference"], 0.2)
        self.assertEqual(result["n_scenarios"], 2)


if __name__ == "__main__":
    unittest.main()
