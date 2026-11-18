from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_ltfp_model as runner  # noqa: E402


class GraphStudyTests(unittest.TestCase):
    def test_lagged_activity_recovers_shifted_relation(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(12, 8))
        shifted = np.pad(source[:, :-2], ((0, 0), (2, 0)))
        noise = rng.normal(size=(12, 8))
        values = np.stack((source, shifted, noise), axis=1)[..., None]

        lagged = runner._lagged_activity_correlation(values, top_k=1, max_lag=2)

        self.assertGreater(lagged[0, 1], 0.95)
        self.assertGreater(lagged[1, 0], 0.95)

    def test_random_graph_preserves_each_row_edge_count_and_weights(self):
        correlation = np.asarray(
            [
                [0.0, 0.8, 0.7, 0.0],
                [0.6, 0.0, 0.0, 0.5],
                [0.4, 0.3, 0.0, 0.0],
                [0.0, 0.2, 0.1, 0.0],
            ],
            dtype=np.float32,
        )

        randomized = runner._randomize_correlation(correlation, seed=7)

        np.testing.assert_array_equal((correlation > 0).sum(axis=1), (randomized > 0).sum(axis=1))
        for row in range(len(correlation)):
            np.testing.assert_allclose(
                np.sort(correlation[row][correlation[row] > 0]),
                np.sort(randomized[row][randomized[row] > 0]),
            )
        self.assertTrue(np.all(np.diag(randomized) == 0))

    def test_graph_modes_retain_the_intended_sources(self):
        dependency = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        correlation = np.asarray([[0.0, 0.4], [0.2, 0.0]], dtype=np.float32)
        lagged = np.asarray([[0.0, 0.7], [0.6, 0.0]], dtype=np.float32)
        components = {
            "dependency": dependency,
            "correlation": correlation,
            "lagged_correlation": lagged,
            "random_correlation": correlation[:, ::-1],
        }

        np.testing.assert_allclose(
            runner.adjacency_for_mode(components, "dependency-only"), dependency
        )
        np.testing.assert_allclose(
            runner.adjacency_for_mode(components, "correlation-only"), correlation
        )
        np.testing.assert_allclose(
            runner.adjacency_for_mode(components, "hybrid-max"), np.maximum(dependency, correlation)
        )


if __name__ == "__main__":
    unittest.main()
