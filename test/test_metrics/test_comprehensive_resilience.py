# -*- coding: utf-8 -*-
"""
test_comprehensive_resilience.py

Unit tests for ComprehensiveResilienceExperiment class that implements
comprehensive resilience analysis including both node and hyperedge
removal experiments with TOPSIS-based selection and higher-order metrics.

This test suite validates the complete experimental framework for analyzing
hypergraph resilience under various attack scenarios and metric evaluation.
"""

import unittest
from unittest.mock import patch, MagicMock

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.comprehensive_resilience import (
    ComprehensiveResilienceExperiment
)


class TestComprehensiveResilienceExperiment(unittest.TestCase):
    """
    Test suite for ComprehensiveResilienceExperiment class.

    Validates comprehensive resilience analysis including both node and
    hyperedge removal experiments with advanced metrics evaluation.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.experiment = ComprehensiveResilienceExperiment()

        # Create test hypernetwork
        self.hn = Hypernetwork()
        self.hn.add_hyperedge(0, [1, 2, 3])
        self.hn.add_hyperedge(1, [2, 3, 4])
        self.hn.add_hyperedge(2, [4, 5, 6])
        self.hn.add_hyperedge(3, [1, 7, 8])

    def test_initialization(self):
        """Test proper initialization of experiment framework."""
        self.assertEqual(self.experiment.s, 1)
        self.assertEqual(self.experiment.m, 2)
        self.assertIsNotNone(self.experiment.traditional_metrics)
        self.assertIsNotNone(self.experiment.higher_order_metrics)
        self.assertIsNotNone(self.experiment.node_topsis)
        self.assertIsNotNone(self.experiment.hyperedge_topsis)

    def test_compute_all_metrics(self):
        """Test computation of all traditional metrics."""
        metrics = self.experiment._compute_all_metrics(self.hn)

        expected_metrics = [
            'hypergraph_connectivity',
            'hyperedge_connectivity',
            'redundancy_coefficient',
            'swalk_efficiency'
        ]

        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
            self.assertIsInstance(metrics[metric_name], float)

    def test_select_random_nodes(self):
        """Test random node selection."""
        nodes = self.experiment._select_random_nodes(self.hn, 25.0)  # 25%

        # Should select at least 1 node but not more than available
        self.assertGreaterEqual(len(nodes), 1)
        self.assertLessEqual(len(nodes), len(self.hn.nodes))

        # All selected nodes should exist in hypernetwork
        for node_id in nodes:
            self.assertIn(node_id, self.hn.nodes)

    def test_select_random_hyperedges(self):
        """Test random hyperedge selection."""
        hyperedges = self.experiment._select_random_hyperedges(
            self.hn, 50.0
        )  # 50%

        # Should select appropriate number of hyperedges
        expected_count = max(1, int(self.hn.size() * 0.5))
        self.assertEqual(len(hyperedges), min(expected_count, self.hn.size()))

        # All selected hyperedges should exist
        for he_id in hyperedges:
            self.assertIn(he_id, self.hn.edges)

    def test_select_top_nodes(self):
        """Test top node selection from ranked list."""
        ranked_nodes = [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)]
        selected = self.experiment._select_top_nodes(ranked_nodes, 50.0)  # 50%

        # Should select top 2 nodes (50% of 4)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, [1, 2])

    def test_select_bottom_nodes(self):
        """Test bottom node selection from ranked list."""
        ranked_nodes = [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)]
        selected = self.experiment._select_bottom_nodes(
            ranked_nodes, 50.0
        )  # 50%

        # Should select bottom 2 nodes (50% of 4)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, [3, 4])

    def test_select_top_hyperedges(self):
        """Test top hyperedge selection from ranked list."""
        ranked_hyperedges = [(0, 0.9), (1, 0.7), (2, 0.5), (3, 0.3)]
        selected = self.experiment._select_top_hyperedges(
            ranked_hyperedges, 25.0
        )  # 25%

        # Should select top 1 hyperedge (25% of 4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected, [0])

    def test_select_bottom_hyperedges(self):
        """Test bottom hyperedge selection from ranked list."""
        ranked_hyperedges = [(0, 0.9), (1, 0.7), (2, 0.5), (3, 0.3)]
        selected = self.experiment._select_bottom_hyperedges(
            ranked_hyperedges, 25.0
        )  # 25%

        # Should select bottom 1 hyperedge (25% of 4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected, [3])

    def test_average_metrics(self):
        """Test averaging of metrics across trials."""
        results = [
            {'metric1': 0.8, 'metric2': 0.6},
            {'metric1': 0.6, 'metric2': 0.4},
            {'metric1': 0.4, 'metric2': 0.8}
        ]

        averaged = self.experiment._average_metrics(results)

        self.assertAlmostEqual(averaged['metric1'], 0.6, places=6)
        self.assertAlmostEqual(averaged['metric2'], 0.6, places=6)

    def test_average_metrics_empty_results(self):
        """Test averaging with empty results list."""
        averaged = self.experiment._average_metrics([])
        self.assertEqual(averaged, {})

    @patch('random.sample')
    def test_test_random_node_removal(self, mock_sample):
        """Test random node removal experiment."""
        # Mock random selection to make test deterministic
        mock_sample.return_value = [1, 2]

        result = self.experiment._test_random_node_removal(
            self.hn, 25.0, trials=2
        )

        # Should return averaged metrics
        self.assertIsInstance(result, dict)

        # Should contain traditional metrics
        expected_metrics = [
            'hypergraph_connectivity',
            'hyperedge_connectivity',
            'redundancy_coefficient',
            'swalk_efficiency'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, result)

    @patch.object(ComprehensiveResilienceExperiment, '_compute_all_metrics')
    def test_test_topsis_node_removal(self, mock_compute):
        """Test TOPSIS-based node removal experiment."""
        # Mock metrics computation
        mock_compute.return_value = {'test_metric': 0.5}

        # Mock TOPSIS ranking
        with patch.object(
                self.experiment.node_topsis, 'rank_nodes'
        ) as mock_rank:
            mock_rank.return_value = [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)]

            result = self.experiment._test_topsis_node_removal(
                self.hn, 25.0, 'top'
            )

            self.assertIn('test_metric', result)
            mock_compute.assert_called()

    @patch('random.sample')
    def test_test_random_hyperedge_removal(self, mock_sample):
        """Test random hyperedge removal experiment."""
        # Mock random selection
        mock_sample.return_value = [0, 1]

        result = self.experiment._test_random_hyperedge_removal(
            self.hn, 50.0, trials=2
        )

        # Should return averaged metrics
        self.assertIsInstance(result, dict)

        # Should contain some metrics
        self.assertGreater(len(result), 0)

    @patch.object(ComprehensiveResilienceExperiment, '_compute_all_metrics')
    def test_test_topsis_hyperedge_removal(self, mock_compute):
        """Test TOPSIS-based hyperedge removal experiment."""
        # Mock metrics computation
        mock_compute.return_value = {'test_metric': 0.7}

        # Mock TOPSIS ranking for hyperedges
        with patch.object(
                self.experiment.hyperedge_topsis, 'rank_hyperedges'
        ) as mock_rank:
            mock_rank.return_value = [(0, 0.9), (1, 0.7), (2, 0.5), (3, 0.3)]

            result = self.experiment._test_topsis_hyperedge_removal(
                self.hn, 25.0, 'bottom'
            )

            self.assertIn('test_metric', result)
            mock_compute.assert_called()

    def test_run_node_removal_experiments(self):
        """Test complete node removal experiments."""
        with patch.multiple(
                self.experiment,
                _test_random_node_removal=MagicMock(
                    return_value={'metric': 0.5}
                ),
                _test_topsis_node_removal=MagicMock(
                    return_value={'metric': 0.6}
                )
        ):
            results = self.experiment._run_node_removal_experiments(
                self.hn, [5.0, 10.0], random_trials=2
            )

            # Should have all three strategies
            self.assertIn('random_removal', results)
            self.assertIn('topsis_top_removal', results)
            self.assertIn('topsis_bottom_removal', results)

            # Each strategy should have results for each percentage
            for strategy in results.values():
                self.assertIn(5.0, strategy)
                self.assertIn(10.0, strategy)

    def test_run_hyperedge_removal_experiments(self):
        """Test complete hyperedge removal experiments."""
        with patch.multiple(
                self.experiment,
                _test_random_hyperedge_removal=MagicMock(
                    return_value={'metric': 0.4}
                ),
                _test_topsis_hyperedge_removal=MagicMock(
                    return_value={'metric': 0.7}
                )
        ):
            results = self.experiment._run_hyperedge_removal_experiments(
                self.hn, [5.0, 10.0], random_trials=2
            )

            # Should have all three strategies
            self.assertIn('random_removal', results)
            self.assertIn('topsis_top_removal', results)
            self.assertIn('topsis_bottom_removal', results)

            # Each strategy should have results for each percentage
            for strategy in results.values():
                self.assertIn(5.0, strategy)
                self.assertIn(10.0, strategy)

    @patch.multiple(
        'hiper.metrics.comprehensive_resilience'
        '.ComprehensiveResilienceExperiment',
        _run_node_removal_experiments=MagicMock(
            return_value={'test': 'node_results'}
        ),
        _run_hyperedge_removal_experiments=MagicMock(
            return_value={'test': 'hyperedge_results'}
        ),
        _compute_all_metrics=MagicMock(
            return_value={'original_metric': 1.0}
        )
    )
    def test_run_node_and_hyperedge_removal_experiments(self):
        """Test complete comprehensive resilience analysis."""
        results = self.experiment.run_node_and_hyperedge_removal_experiments(
            self.hn, [5.0, 10.0], random_trials=3
        )

        # Should have all main sections
        expected_sections = [
            'original_metrics',
            'node_removal_experiments',
            'hyperedge_removal_experiments',
            'removal_percentages',
            'higher_order_analysis'
        ]

        for section in expected_sections:
            self.assertIn(section, results)

        # Check removal percentages
        self.assertEqual(results['removal_percentages'], [5.0, 10.0])

        # Check that original metrics are computed
        self.assertIn('original_metric', results['original_metrics'])

    def test_compute_all_metrics_handles_exceptions(self):
        """Test that metric computation handles exceptions gracefully."""
        # Create a mock metric that raises an exception
        failing_metric = MagicMock()
        failing_metric.compute.side_effect = Exception("Test exception")

        # Replace one metric with the failing one
        original_metric = self.experiment.traditional_metrics[
            'hypergraph_connectivity'
        ]
        self.experiment.traditional_metrics[
            'hypergraph_connectivity'
        ] = failing_metric

        try:
            metrics = self.experiment._compute_all_metrics(self.hn)

            # Should have the failing metric with value 0.0
            self.assertEqual(metrics['hypergraph_connectivity'], 0.0)

            # Other metrics should still work
            self.assertIn('redundancy_coefficient', metrics)

        finally:
            # Restore original metric
            self.experiment.traditional_metrics[
                'hypergraph_connectivity'
            ] = original_metric


if __name__ == '__main__':
    unittest.main()
