# -*- coding: utf-8 -*-
"""
test_entropy_loss.py

Unit tests for the EntropyLoss metric implementation.
Tests cover entropy loss computation for various distribution types,
detailed analysis, and multiple distribution analysis.
"""

import math
import unittest

import numpy as np

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.entropy_loss import EntropyLoss


class TestEntropyLoss(unittest.TestCase):
    """
    Tests EntropyLoss metric computation and analysis.
    Methods tested include entropy loss computation for different distribution
    types, detailed analysis, and relative entropy computation.
    """

    def setUp(self):
        """Create test fixtures with various hypergraph configurations."""
        self.entropy_node_degree = EntropyLoss()
        self.entropy_hyperedge_size = EntropyLoss('hyperedge_size')
        self.entropy_hyperedge_degree = EntropyLoss('hyperedge_degree')

        # Empty hypergraph
        self.hn_empty = Hypernetwork()

        # Uniform degree distribution
        self.hn_uniform = Hypernetwork()
        self.hn_uniform.add_hyperedge(0, [1, 2])
        self.hn_uniform.add_hyperedge(1, [3, 4])
        self.hn_uniform.add_hyperedge(2, [5, 6])

        # Non-uniform degree distribution
        self.hn_nonuniform = Hypernetwork()
        self.hn_nonuniform.add_hyperedge(0, [1, 2, 3])
        self.hn_nonuniform.add_hyperedge(1, [1, 2])
        self.hn_nonuniform.add_hyperedge(2, [1])

        # After perturbation (more uniform)
        self.hn_more_uniform = Hypernetwork()
        self.hn_more_uniform.add_hyperedge(0, [1, 2])
        self.hn_more_uniform.add_hyperedge(1, [1, 2])

        # After perturbation (less uniform)
        self.hn_less_uniform = Hypernetwork()
        self.hn_less_uniform.add_hyperedge(0, [1, 2, 3, 4])
        self.hn_less_uniform.add_hyperedge(1, [1])

        # Single hyperedge
        self.hn_single = Hypernetwork()
        self.hn_single.add_hyperedge(0, [1, 2, 3])

    def test_metric_initialization(self):
        """Test proper initialization of EntropyLoss metric."""
        self.assertEqual(self.entropy_node_degree.name, "Entropy Loss")
        self.assertEqual(self.entropy_node_degree.symbol, "EL")
        self.assertEqual(self.entropy_node_degree.distribution_type,
                         'node_degree')
        self.assertEqual(self.entropy_node_degree.log_base, 2.0)

    def test_invalid_distribution_type(self):
        """Test initialization with invalid distribution type."""
        with self.assertRaises(ValueError):
            EntropyLoss('invalid_type')

    def test_entropy_loss_identical_hypergraphs(self):
        """Test entropy loss with identical hypergraphs."""
        loss = self.entropy_node_degree.compute(self.hn_uniform,
                                                self.hn_uniform)
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_entropy_loss_empty_hypergraphs(self):
        """Test entropy loss with empty hypergraphs."""
        loss = self.entropy_node_degree.compute(self.hn_empty, self.hn_empty)
        self.assertEqual(loss, 0.0)

    def test_node_degree_distribution_extraction(self):
        """Test extraction of node degree distribution."""
        dist = self.entropy_node_degree._get_node_degree_distribution(
            self.hn_nonuniform)

        # Node degrees: 1->3, 2->2, 3->1
        expected_degrees = [3, 2, 1]  # For nodes 1, 2, 3
        self.assertEqual(sorted(dist, reverse=True), expected_degrees)

    def test_hyperedge_size_distribution_extraction(self):
        """Test extraction of hyperedge size distribution."""
        dist = self.entropy_hyperedge_size._get_hyperedge_size_distribution(
            self.hn_nonuniform)

        # Hyperedge sizes: 0->3, 1->2, 2->1
        expected_sizes = [3, 2, 1]
        self.assertEqual(sorted(dist, reverse=True), expected_sizes)

    def test_hyperedge_degree_distribution_extraction(self):
        """Test extraction of hyperedge degree distribution."""
        dist = self.entropy_hyperedge_degree._get_hyperedge_degree_distribution(
            self.hn_nonuniform)

        # Each hyperedge's degree
        self.assertEqual(len(dist), 3)  # Three hyperedges
        for degree in dist:
            self.assertGreaterEqual(degree, 0)

    def test_entropy_with_different_log_base(self):
        """Test entropy computation with different logarithm base."""
        entropy_natural = EntropyLoss(log_base=math.e)

        # Compute same distribution with different bases
        loss_base2 = self.entropy_node_degree.compute(self.hn_uniform,
                                                      self.hn_nonuniform)
        loss_natural = entropy_natural.compute(self.hn_uniform,
                                               self.hn_nonuniform)

        # Should have same sign but different magnitude
        self.assertEqual(np.sign(loss_base2), np.sign(loss_natural))

    def test_compute_detailed_analysis(self):
        """Test detailed entropy loss analysis."""
        detailed = self.entropy_node_degree.compute_detailed(self.hn_uniform,
                                                             self.hn_nonuniform)

        # Check required keys in detailed analysis
        expected_keys = {
            'entropy_loss', 'entropy_before', 'entropy_after',
            'distribution_type',
            'log_base', 'values_before', 'values_after', 'unique_values_before',
            'unique_values_after', 'total_values_before', 'total_values_after',
            'frequency_counts_before', 'frequency_counts_after', 'mean_before',
            'mean_after', 'std_before', 'std_after'
        }
        self.assertEqual(set(detailed.keys()), expected_keys)

        # Check data types
        self.assertIsInstance(detailed['entropy_loss'], float)
        self.assertIsInstance(detailed['entropy_before'], float)
        self.assertIsInstance(detailed['entropy_after'], float)
        self.assertIsInstance(detailed['values_before'], list)
        self.assertIsInstance(detailed['values_after'], list)

    def test_compute_multiple_distributions(self):
        """Test entropy loss computation for multiple distribution types."""
        results = self.entropy_node_degree.compute_multiple_distributions(
            self.hn_uniform, self.hn_nonuniform)

        expected_types = ['node_degree', 'hyperedge_size', 'hyperedge_degree']
        self.assertEqual(set(results.keys()), set(expected_types))

        # Check that all results are floats or NaN
        for dist_type, loss in results.items():
            self.assertTrue(isinstance(loss, float) or np.isnan(loss))

    def test_relative_entropy_loss(self):
        """Test relative entropy loss computation."""
        rel_loss = self.entropy_node_degree.compute_relative_entropy_loss(
            self.hn_uniform, self.hn_nonuniform)

        # Should be between -inf and 1
        self.assertLessEqual(rel_loss, 1.0)

    def test_relative_entropy_loss_zero_initial(self):
        """Test relative entropy loss with zero initial entropy."""
        # Create hypergraph with zero entropy (all nodes have same degree)
        hn_zero_entropy = Hypernetwork()
        hn_zero_entropy.add_hyperedge(0, [1])
        hn_zero_entropy.add_hyperedge(1, [2])
        hn_zero_entropy.add_hyperedge(2, [3])

        rel_loss = self.entropy_node_degree.compute_relative_entropy_loss(
            hn_zero_entropy, hn_zero_entropy)
        self.assertEqual(rel_loss, 0.0)

    def test_hyperedge_size_entropy_computation(self):
        """Test entropy computation for hyperedge size distribution."""
        loss = self.entropy_hyperedge_size.compute(self.hn_uniform,
                                                   self.hn_nonuniform)

        # Should be a valid entropy loss value
        self.assertIsInstance(loss, float)

    def test_hyperedge_degree_entropy_computation(self):
        """Test entropy computation for hyperedge degree distribution."""
        loss = self.entropy_hyperedge_degree.compute(self.hn_uniform,
                                                     self.hn_nonuniform)

        # Should be a valid entropy loss value
        self.assertIsInstance(loss, float)

    def test_string_representations(self):
        """Test string and repr methods."""
        str_repr = str(self.entropy_node_degree)
        expected_str = "Entropy Loss (EL) - node_degree"
        self.assertEqual(str_repr, expected_str)

        repr_str = repr(self.entropy_node_degree)
        expected_repr = ("EntropyLoss(name='Entropy Loss', symbol='EL', "
                         "distribution_type='node_degree', log_base=2.0)")
        self.assertEqual(repr_str, expected_repr)


if __name__ == '__main__':
    unittest.main()
