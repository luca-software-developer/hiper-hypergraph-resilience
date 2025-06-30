# -*- coding: utf-8 -*-
"""
test_average_hyperedge_cardinality.py

Unit tests for the AverageHyperedgeCardinality metric implementation.
Tests cover average cardinality computation, change analysis, distribution
analysis, and comprehensive statistics.
"""

import unittest

import numpy as np

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.average_hyperedge_cardinality import \
    AverageHyperedgeCardinality


class TestAverageHyperedgeCardinality(unittest.TestCase):
    """
    Tests AverageHyperedgeCardinality metric computation and analysis.
    Methods tested include cardinality computation, change analysis,
    distribution computation, and statistical analysis.
    """

    def setUp(self):
        """Create test fixtures with various hypergraph configurations."""
        self.ahc_metric = AverageHyperedgeCardinality()

        # Empty hypergraph
        self.hn_empty = Hypernetwork()

        # Single hyperedge
        self.hn_single = Hypernetwork()
        self.hn_single.add_hyperedge(0, [1, 2, 3])

        # Uniform cardinality hypergraph
        self.hn_uniform = Hypernetwork()
        self.hn_uniform.add_hyperedge(0, [1, 2])
        self.hn_uniform.add_hyperedge(1, [3, 4])
        self.hn_uniform.add_hyperedge(2, [5, 6])

        # Mixed cardinality hypergraph
        self.hn_mixed = Hypernetwork()
        self.hn_mixed.add_hyperedge(0, [1, 2, 3])  # Size 3
        self.hn_mixed.add_hyperedge(1, [2, 3, 4, 5])  # Size 4
        self.hn_mixed.add_hyperedge(2, [1, 2])  # Size 2

        # Large cardinality hypergraph
        self.hn_large = Hypernetwork()
        self.hn_large.add_hyperedge(0, [1, 2, 3, 4, 5, 6])
        self.hn_large.add_hyperedge(1, [7, 8, 9, 10])

        # After perturbation (smaller cardinalities)
        self.hn_smaller = Hypernetwork()
        self.hn_smaller.add_hyperedge(0, [1, 2])
        self.hn_smaller.add_hyperedge(1, [3, 4])

        # After perturbation (larger cardinalities)
        self.hn_larger = Hypernetwork()
        self.hn_larger.add_hyperedge(0, [1, 2, 3, 4, 5])
        self.hn_larger.add_hyperedge(1, [6, 7, 8, 9, 10, 11])

    def test_metric_initialization(self):
        """Test proper initialization of AverageHyperedgeCardinality metric."""
        self.assertEqual(self.ahc_metric.name, "Average Hyperedge Cardinality")
        self.assertEqual(self.ahc_metric.symbol, "AHC")

    def test_cardinality_empty_hypergraph(self):
        """Test average cardinality computation with empty hypergraph."""
        ahc = AverageHyperedgeCardinality.compute(self.hn_empty)
        self.assertEqual(ahc, 0.0)

    def test_cardinality_single_hyperedge(self):
        """Test average cardinality computation with single hyperedge."""
        ahc = AverageHyperedgeCardinality.compute(self.hn_single)
        self.assertEqual(ahc, 3.0)

    def test_cardinality_uniform_hypergraph(self):
        """Test average cardinality computation with uniform cardinalities."""
        ahc = AverageHyperedgeCardinality.compute(self.hn_uniform)
        expected = (2 + 2 + 2) / 3  # All hyperedges have size 2
        self.assertEqual(ahc, expected)

    def test_cardinality_mixed_hypergraph(self):
        """Test average cardinality computation with mixed cardinalities."""
        ahc = AverageHyperedgeCardinality.compute(self.hn_mixed)
        expected = (3 + 4 + 2) / 3  # Sizes: 3, 4, 2
        self.assertAlmostEqual(ahc, expected, places=6)

    def test_cardinality_large_hypergraph(self):
        """Test average cardinality computation with large cardinalities."""
        ahc = AverageHyperedgeCardinality.compute(self.hn_large)
        expected = (6 + 4) / 2  # Sizes: 6, 4
        self.assertEqual(ahc, expected)

    def test_compute_change_no_change(self):
        """Test change computation with no cardinality change."""
        change = self.ahc_metric.compute_change(self.hn_uniform,
                                                self.hn_uniform)
        self.assertEqual(change, 0.0)

    def test_compute_change_increase(self):
        """Test change computation with cardinality increase."""
        change = self.ahc_metric.compute_change(self.hn_smaller, self.hn_larger)
        self.assertGreater(change, 0.0)

    def test_compute_change_decrease(self):
        """Test change computation with cardinality decrease."""
        change = self.ahc_metric.compute_change(self.hn_larger, self.hn_smaller)
        self.assertLess(change, 0.0)

    def test_compute_change_empty_to_non_empty(self):
        """Test change computation from empty to non-empty hypergraph."""
        change = self.ahc_metric.compute_change(self.hn_empty, self.hn_mixed)
        expected = (3 + 4 + 2) / 3 - 0.0
        self.assertAlmostEqual(change, expected, places=6)

    def test_compute_change_non_empty_to_empty(self):
        """Test change computation from non-empty to empty hypergraph."""
        change = self.ahc_metric.compute_change(self.hn_mixed, self.hn_empty)
        expected = 0.0 - (3 + 4 + 2) / 3
        self.assertAlmostEqual(change, expected, places=6)

    def test_cardinality_distribution_empty(self):
        """Test cardinality distribution with empty hypergraph."""
        dist = AverageHyperedgeCardinality.get_cardinality_distribution(
            self.hn_empty)
        self.assertEqual(dist, {})

    def test_cardinality_distribution_uniform(self):
        """Test cardinality distribution with uniform cardinalities."""
        dist = AverageHyperedgeCardinality.get_cardinality_distribution(
            self.hn_uniform)
        expected = {2: 3}  # All 3 hyperedges have cardinality 2
        self.assertEqual(dist, expected)

    def test_cardinality_distribution_mixed(self):
        """Test cardinality distribution with mixed cardinalities."""
        dist = AverageHyperedgeCardinality.get_cardinality_distribution(
            self.hn_mixed)
        expected = {2: 1, 3: 1, 4: 1}  # One hyperedge each of sizes 2, 3, 4
        self.assertEqual(dist, expected)

    def test_statistics_empty_hypergraph(self):
        """Test statistics computation with empty hypergraph."""
        stats = AverageHyperedgeCardinality.get_statistics(self.hn_empty)

        expected = {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'total_hyperedges': 0
        }
        self.assertEqual(stats, expected)

    def test_statistics_single_hyperedge(self):
        """Test statistics computation with single hyperedge."""
        stats = AverageHyperedgeCardinality.get_statistics(self.hn_single)

        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['median'], 3.0)
        self.assertEqual(stats['std'], 0.0)
        self.assertEqual(stats['min'], 3)
        self.assertEqual(stats['max'], 3)
        self.assertEqual(stats['total_hyperedges'], 1)

    def test_statistics_uniform_hypergraph(self):
        """Test statistics computation with uniform cardinalities."""
        stats = AverageHyperedgeCardinality.get_statistics(self.hn_uniform)

        self.assertEqual(stats['mean'], 2.0)
        self.assertEqual(stats['median'], 2.0)
        self.assertEqual(stats['std'], 0.0)
        self.assertEqual(stats['min'], 2)
        self.assertEqual(stats['max'], 2)
        self.assertEqual(stats['total_hyperedges'], 3)

    def test_statistics_mixed_hypergraph(self):
        """Test statistics computation with mixed cardinalities."""
        stats = AverageHyperedgeCardinality.get_statistics(self.hn_mixed)

        # Cardinalities: [3, 4, 2]
        expected_mean = (3 + 4 + 2) / 3
        expected_median = 3.0  # Sorted: [2, 3, 4], median is 3
        expected_std = np.std([3, 4, 2])

        self.assertAlmostEqual(stats['mean'], expected_mean, places=6)
        self.assertEqual(stats['median'], expected_median)
        self.assertAlmostEqual(stats['std'], expected_std, places=6)
        self.assertEqual(stats['min'], 2)
        self.assertEqual(stats['max'], 4)
        self.assertEqual(stats['total_hyperedges'], 3)

    def test_statistics_data_types(self):
        """Test that statistics return correct data types."""
        stats = AverageHyperedgeCardinality.get_statistics(self.hn_mixed)

        self.assertIsInstance(stats['mean'], float)
        self.assertIsInstance(stats['median'], float)
        self.assertIsInstance(stats['std'], float)
        self.assertIsInstance(stats['min'], int)
        self.assertIsInstance(stats['max'], int)
        self.assertIsInstance(stats['total_hyperedges'], int)

    def test_cardinality_non_negative(self):
        """Test that cardinality values are always non-negative."""
        test_hypergraphs = [
            self.hn_empty, self.hn_single, self.hn_uniform,
            self.hn_mixed, self.hn_large
        ]

        for hn in test_hypergraphs:
            ahc = AverageHyperedgeCardinality.compute(hn)
            self.assertGreaterEqual(ahc, 0.0)

    def test_change_symmetry(self):
        """Test that change computation has correct symmetry."""
        change_forward = self.ahc_metric.compute_change(self.hn_uniform,
                                                        self.hn_mixed)
        change_backward = self.ahc_metric.compute_change(self.hn_mixed,
                                                         self.hn_uniform)

        # Changes should be opposite
        self.assertAlmostEqual(change_forward, -change_backward, places=6)

    def test_large_cardinality_handling(self):
        """Test handling of hypergraphs with large cardinalities."""
        # Create hypergraph with very large hyperedges
        hn_very_large = Hypernetwork()
        large_nodes = list(range(1, 1001))  # 1000 nodes
        hn_very_large.add_hyperedge(0, large_nodes)

        ahc = AverageHyperedgeCardinality.compute(hn_very_large)
        self.assertEqual(ahc, 1000.0)

    def test_string_representations(self):
        """Test string and repr methods."""
        str_repr = str(self.ahc_metric)
        expected_str = "Average Hyperedge Cardinality (AHC)"
        self.assertEqual(str_repr, expected_str)

        repr_str = repr(self.ahc_metric)
        expected_repr = (
            "AverageHyperedgeCardinality(name='Average Hyperedge Cardinality', "
            "symbol='AHC')")
        self.assertEqual(repr_str, expected_repr)


if __name__ == '__main__':
    unittest.main()
