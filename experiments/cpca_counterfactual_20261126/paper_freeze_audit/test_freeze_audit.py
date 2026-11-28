import unittest

import numpy as np

from build_cache import cpca, parse_with_reason, reference_draws, score_cache


class Contracts(unittest.TestCase):
    def test_negative_logits_remain_available_and_event_order_preserved(self):
        from unify_baselines import translate_available_scores
        services=['a','b','c']
        scores=translate_available_scores([-3.,-1.,-2.],services)
        self.assertGreater(max(scores.values()),0)
        self.assertEqual(sorted(scores,key=scores.get,reverse=True),['b','c','a'])
        self.assertEqual(translate_available_scores([-3.,-1.,-2.],services,False),{})
        rows=[[-3.,-1.,-2.],[9.,1.,3.]]
        shifted=np.array([list(translate_available_scores(r,services).values()) for r in rows])
        np.testing.assert_array_equal(np.argsort(np.mean(rows,axis=0)),np.argsort(shifted.mean(axis=0)))
    def test_parser_matches_core_and_reasons(self):
        S = cpca.Span
        cases = [
            [S('a', '', 'frontendservice', 'frontend', 0, 100), S('b', 'a', 'x', 'x', 10, 30)],
            [S('a', '', 'frontendservice', 'frontend', 0, 100), S('b', 'a', 'x', 'x', 10, 80), S('c', 'a', 'y', 'y', 20, 70)],
            [S('a', '', 'frontendservice', 'frontend', 0, 100), S('a', '', 'x', 'x', 10, 30)],
            [S('a', '', 'frontendservice', 'frontend', 0, 100), S('b', 'a', 'x', 'x', 10, 30), S('c', 'z', 'y', 'y', 5, 30)],
        ]
        for spans in cases:
            actual = cpca.decompose_tree(spans, 'ob', clock_tolerance_us=1000., tree_policy='entry-component')
            got, _, _, _ = parse_with_reason(spans, 'ob')
            self.assertEqual(got, actual)

    def test_weighted_block_median_matches_literal_resampling(self):
        cache = {'n_blocks': np.array([3]), 'key_offsets': np.array([0, 4, 5]),
                 'normal_values': np.array([1., 2., 10., 20., 99.]),
                 'normal_blocks': np.array([0, 1, 1, 2, 2])}
        draws, _ = reference_draws(cache, 100, 42)
        counts = np.random.default_rng(42).multinomial(3, np.full(3, 1/3), size=100)
        for i, weights in enumerate(counts):
            expected = np.repeat(cache['normal_values'][:4], weights[cache['normal_blocks'][:4]])
            self.assertEqual(draws[i, 0], np.median(expected))
            if weights[2] == 0:
                self.assertTrue(np.isnan(draws[i, 1]))

    def test_clip_after_service_sum_and_operation_zero_fill(self):
        cache = {'selection': np.array([[1, 0], [1, 1]]), 'request_group': np.array([0, 1]),
                 'request_duration': np.array([200., 100.]), 'n_groups': np.array([2]),
                 'n_windows': np.array([1]), 'n_services': np.array([2]),
                 'component_key': np.array([0, 1, 2]), 'component_observed': np.array([100., 20., 60.]),
                 'component_pair': np.array([0, 0, 1]), 'pair_request': np.array([0, 1]),
                 'pair_group_service': np.array([0, 3]), 'group_window': np.array([0, 0])}
        result = score_cache(cache, np.array([30., 60., 10.]), 0)
        np.testing.assert_allclose(result, [[.075, .25]])
        # Service 0 has +70 and -40, which must be clipped after summing.
        np.testing.assert_allclose(score_cache(cache, np.array([30., 60., 10.]), 1), [[0., .5]])

    def test_internal_local_is_zero_under_current_gap_partition(self):
        S = cpca.Span
        spans = [S('a', '', 'frontendservice', 'frontend', 0, 100),
                 S('b', 'a', 'x', 'x', 10, 20), S('c', 'a', 'y', 'y', 50, 30)]
        parsed = cpca.decompose_tree(spans, 'ob')
        self.assertEqual(parsed[3]['a'], 0.)
        self.assertEqual(sum(parsed[3].values()) + sum(parsed[4].values()) + sum(parsed[5].values()), 100.)

    def test_no_cohort_requests_returns_empty_scores(self):
        cache = {'selection': np.zeros((0, 4), dtype=int), 'request_group': np.array([], dtype=int),
                 'request_duration': np.array([], dtype=float), 'n_groups': np.array([0]),
                 'n_windows': np.array([0]), 'n_services': np.array([2]),
                 'component_key': np.array([], dtype=int), 'component_observed': np.array([], dtype=float),
                 'component_pair': np.array([], dtype=int), 'pair_request': np.array([], dtype=int),
                 'pair_group_service': np.array([], dtype=int), 'group_window': np.array([], dtype=int)}
        self.assertEqual(score_cache(cache, np.array([]), 0).shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
