# -*- coding: utf-8 -*-
"""
test_centrality_disruption.py

Unit tests for the CentralityDisruptionIndex metric implementation.
Tests cover centrality disruption computation, different centrality types,
and detailed analysis methods.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.centrality_disruption import CentralityDisruptionIndex


class TestCentralityDisruptionIndex(unittest.TestCase):
    """
    Tests CentralityDisruptionIndex metric computation and analysis.
    Methods tested include disruption computation for various centrality types,
    detailed analysis, and edge case handling.
    """

    def setUp(self):
        """Create test fixtures with various hypergraph configurations."""
        self.cdi_degree = CentralityDisruptionIndex()
        self.cdi_closeness = CentralityDisruptionIndex('closeness')
        self.cdi_betweenness = CentralityDisruptionIndex('betweenness')

        # Empty hypergraph
        self.hn_empty = Hypernetwork()

        # Single hyperedge
        self.hn_single = Hypernetwork()
        self.hn_single.add_hyperedge(0, [1, 2, 3])

        # Original hypergraph
        self.hn_original = Hypernetwork()
        self.hn_original.add_hyperedge(0, [1, 2, 3])
        self.hn_original.add_hyperedge(1, [2, 3, 4])
        self.hn_original.add_hyperedge(2, [3, 4, 5])

        # Perturbed hypergraph (node 1 removed)
        self.hn_perturbed = Hypernetwork()
        self.hn_perturbed.add_hyperedge(1, [2, 3, 4])
        self.hn_perturbed.add_hyperedge(2, [3, 4, 5])

        # Different structure
        self.hn_different = Hypernetwork()
        self.hn_different.add_hyperedge(0, [1, 2])
        self.hn_different.add_hyperedge(1, [3, 4])
        self.hn_different.add_hyperedge(2, [5, 6])

    def test_metric_initialization(self):
        """Test proper initialization of CentralityDisruptionIndex metric."""
        self.assertEqual(self.cdi_degree.name, "Centrality Disruption Index")
        self.assertEqual(self.cdi_degree.symbol, "CDI")
        self.assertEqual(self.cdi_degree.centrality_type, 'degree')

    def test_disruption_identical_hypergraphs(self):
        """Test disruption computation with identical hypergraphs."""
        cdi = self.cdi_degree.compute(self.hn_original, self.hn_original)
        self.assertEqual(cdi, 0.0)

    def test_disruption_empty_hypergraphs(self):
        """Test disruption computation with empty hypergraphs."""
        cdi = self.cdi_degree.compute(self.hn_empty, self.hn_empty)
        self.assertEqual(cdi, 0.0)

    def test_disruption_few_common_nodes(self):
        """Test disruption computation with insufficient common nodes."""
        hn1 = Hypernetwork()
        hn1.add_hyperedge(0, [1])

        hn2 = Hypernetwork()
        hn2.add_hyperedge(0, [1])

        cdi = self.cdi_degree.compute(hn1, hn2)
        self.assertEqual(cdi, 0.0)

    def test_disruption_node_removal(self):
        """Test disruption computation after node removal."""
        cdi = self.cdi_degree.compute(self.hn_original, self.hn_perturbed)
        self.assertGreater(cdi, 0.0)
        self.assertLessEqual(cdi, 1.0)

    def test_degree_centrality_computation(self):
        """Test degree centrality computation."""
        centralities = self.cdi_degree._compute_degree_centrality(
            self.hn_original)

        # Check that all nodes have centrality values
        expected_nodes = {1, 2, 3, 4, 5}
        self.assertEqual(set(centralities.keys()), expected_nodes)

        # Check centrality values are non-negative
        for centrality in centralities.values():
            self.assertGreaterEqual(centrality, 0.0)

    def test_closeness_centrality_computation(self):
        """Test closeness centrality computation."""
        centralities = self.cdi_closeness._compute_closeness_centrality(
            self.hn_original)

        # Check that all nodes have centrality values
        expected_nodes = {1, 2, 3, 4, 5}
        self.assertEqual(set(centralities.keys()), expected_nodes)

        # Check centrality values are non-negative
        for centrality in centralities.values():
            self.assertGreaterEqual(centrality, 0.0)

    def test_betweenness_centrality_computation(self):
        """Test betweenness centrality computation."""
        centralities = self.cdi_betweenness._compute_betweenness_centrality(
            self.hn_original)

        # Check that all nodes have centrality values
        expected_nodes = {1, 2, 3, 4, 5}
        self.assertEqual(set(centralities.keys()), expected_nodes)

        # Check centrality values are non-negative
        for centrality in centralities.values():
            self.assertGreaterEqual(centrality, 0.0)

    def test_custom_centrality_function(self):
        """Test disruption computation with custom centrality function."""

        def mock_centrality(hypergraph):
            """Return mock centralities"""
            return {node: 1.0 for node in hypergraph.nodes.keys()}

        cdi = self.cdi_degree.compute(self.hn_original, self.hn_perturbed,
                                      mock_centrality)
        self.assertEqual(cdi, 0.0)  # All centralities are the same

    def test_compute_detailed_analysis(self):
        """Test detailed centrality disruption analysis."""
        detailed = self.cdi_degree.compute_detailed(self.hn_original,
                                                    self.hn_perturbed)

        # Check required keys in detailed analysis
        expected_keys = {
            'ks_statistic', 'p_value', 'nodes_before', 'nodes_after',
            'common_nodes', 'centralities_before', 'centralities_after',
            'mean_before', 'mean_after', 'std_before', 'std_after'
        }
        self.assertEqual(set(detailed.keys()), expected_keys)

        # Check data types and bounds
        self.assertIsInstance(detailed['ks_statistic'], float)
        self.assertIsInstance(detailed['p_value'], float)
        self.assertIsInstance(detailed['nodes_before'], int)
        self.assertIsInstance(detailed['nodes_after'], int)
        self.assertIsInstance(detailed['common_nodes'], int)

        self.assertGreaterEqual(detailed['ks_statistic'], 0.0)
        self.assertLessEqual(detailed['ks_statistic'], 1.0)
        self.assertGreaterEqual(detailed['p_value'], 0.0)
        self.assertLessEqual(detailed['p_value'], 1.0)

    def test_node_adjacency_building(self):
        """Test building node adjacency from hyperedges."""
        adjacency = self.cdi_degree._build_node_adjacency(self.hn_original)

        # Check that all nodes are present
        expected_nodes = {1, 2, 3, 4, 5}
        self.assertEqual(set(adjacency.keys()), expected_nodes)

        # Check adjacency relationships
        self.assertIn(2, adjacency[1])  # Nodes 1 and 2 are in hyperedge 0
        self.assertIn(3, adjacency[2])  # Nodes 2 and 3 are in hyperedge 0 and 1

    def test_shortest_distances_computation(self):
        """Test shortest distance computation using BFS."""
        adjacency = self.cdi_degree._build_node_adjacency(self.hn_original)
        distances = self.cdi_degree._compute_shortest_distances(adjacency, 1)

        # Check that source has distance 0
        self.assertEqual(distances[1], 0)

        # Check that all distances are non-negative
        for distance in distances.values():
            self.assertGreaterEqual(distance, 0)

    def test_invalid_centrality_type(self):
        """Test handling of invalid centrality type."""
        cdi_invalid = CentralityDisruptionIndex('invalid_type')
        centralities = cdi_invalid._compute_centralities(self.hn_original)

        # Should default to degree centrality
        expected_nodes = {1, 2, 3, 4, 5}
        self.assertEqual(set(centralities.keys()), expected_nodes)

    def test_exception_handling_in_compute(self):
        """Test exception handling in compute method."""
        # Create hypergraphs that might cause issues
        hn_single_node = Hypernetwork()
        hn_single_node.add_hyperedge(0, [1])

        cdi = self.cdi_degree.compute(hn_single_node, self.hn_original)
        self.assertGreaterEqual(cdi, 0.0)
        self.assertLessEqual(cdi, 1.0)

    def test_string_representations(self):
        """Test string and repr methods."""
        str_repr = str(self.cdi_degree)
        expected_str = "Centrality Disruption Index (CDI) - degree centrality"
        self.assertEqual(str_repr, expected_str)

        repr_str = repr(self.cdi_degree)
        expected_repr = (
            "CentralityDisruptionIndex(name='Centrality Disruption Index', "
            "symbol='CDI', centrality_type='degree')")
        self.assertEqual(repr_str, expected_repr)


if __name__ == '__main__':
    unittest.main()
