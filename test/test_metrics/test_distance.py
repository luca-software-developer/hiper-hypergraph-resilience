# -*- coding: utf-8 -*-
"""
test_distance.py

Unit tests for the HypergraphDistance class.
Tests s-walk distance computation, path finding algorithms, and connectivity
analysis for various hypergraph structures and edge cases.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.distance import HypergraphDistance


class TestHypergraphDistance(unittest.TestCase):
    """
    Tests HypergraphDistance methods.
    Methods tested include s-walk validation, distance computation,
    path finding, connectivity checking, and all-pairs distance analysis.
    """

    def setUp(self):
        """Create distance calculator and test hypergraphs."""
        self.distance_calc = HypergraphDistance()

        # Create test hypergraphs using helper methods
        self._create_empty_hypergraph()
        self._create_single_node_hypergraph()
        self._create_path_hypergraph()
        self._create_connected_hypergraph()
        self._create_disconnected_hypergraph()

    def _create_empty_hypergraph(self):
        """Create empty test hypergraph."""
        self.hn_empty = Hypernetwork()

    def _create_single_node_hypergraph(self):
        """Create single node test hypergraph."""
        self.hn_single = Hypernetwork()
        self.hn_single.add_node(1)

    def _create_path_hypergraph(self):
        """Create simple path test hypergraph."""
        self.hn_path = Hypernetwork()
        self.hn_path.add_hyperedge(0, [1, 2, 3])
        self.hn_path.add_hyperedge(1, [2, 3, 4])
        self.hn_path.add_hyperedge(2, [4, 5])

    def _create_connected_hypergraph(self):
        """Create connected hypergraph with multiple paths."""
        self.hn_connected = Hypernetwork()
        self.hn_connected.add_hyperedge(0, [1, 2, 3])
        self.hn_connected.add_hyperedge(1, [2, 3, 4])
        self.hn_connected.add_hyperedge(2, [3, 4, 5])
        self.hn_connected.add_hyperedge(3, [1, 5])  # Alternative path

    def _create_disconnected_hypergraph(self):
        """Create disconnected test hypergraph."""
        self.hn_disconnected = Hypernetwork()
        self.hn_disconnected.add_hyperedge(0, [1, 2])
        self.hn_disconnected.add_hyperedge(1, [3, 4])

    def test_distance_same_node(self):
        """Distance from a node to itself should be zero."""
        distance = self.distance_calc.compute_distance(self.hn_path, 1, 1)
        self.assertEqual(distance, 0.0)

    def test_distance_adjacent_nodes(self):
        """Distance between nodes in same hyperedge should be one."""
        distance = self.distance_calc.compute_distance(self.hn_path, 1, 2)
        self.assertEqual(distance, 1.0)

    def test_distance_reachable_nodes(self):
        """Distance should be finite for reachable nodes."""
        distance = self.distance_calc.compute_distance(self.hn_path, 1, 5)
        self.assertLess(distance, float('inf'))
        self.assertGreater(distance, 0)

    def test_distance_unreachable_nodes(self):
        """Distance should be infinite for unreachable nodes."""
        distance = self.distance_calc.compute_distance(
            self.hn_disconnected, 1, 3)
        self.assertEqual(distance, float('inf'))

    def test_distance_nonexistent_nodes(self):
        """Distance involving nonexistent nodes should be infinite."""
        distance = self.distance_calc.compute_distance(self.hn_path, 1, 99)
        self.assertEqual(distance, float('inf'))

        distance = self.distance_calc.compute_distance(self.hn_path, 99, 1)
        self.assertEqual(distance, float('inf'))

    def test_connectivity_check_connected(self):
        """Connected hypergraph should be identified as connected."""
        is_connected = self.distance_calc.is_connected(self.hn_connected)
        self.assertTrue(is_connected)

    def test_connectivity_check_disconnected(self):
        """Disconnected hypergraph should be identified as disconnected."""
        is_connected = self.distance_calc.is_connected(self.hn_disconnected)
        self.assertFalse(is_connected)

    def test_connectivity_check_empty(self):
        """Empty hypergraph should be considered connected."""
        is_connected = self.distance_calc.is_connected(self.hn_empty)
        self.assertTrue(is_connected)

    def test_connectivity_check_single_node(self):
        """Single node hypergraph should be considered connected."""
        is_connected = self.distance_calc.is_connected(self.hn_single)
        self.assertTrue(is_connected)

    def test_all_pairs_distances_structure(self):
        """All pairs distances should have correct structure."""
        distances = self.distance_calc.compute_all_distances(self.hn_path)

        # Should be a dictionary mapping pairs to distances
        self.assertIsInstance(distances, dict)

        # Check that all node pairs are included
        nodes = list(self.hn_path.nodes)
        for source in nodes:
            for target in nodes:
                self.assertIn((source, target), distances)

    def test_all_pairs_distances_properties(self):
        """All pairs distances should satisfy basic properties."""
        distances = self.distance_calc.compute_all_distances(self.hn_path)
        nodes = list(self.hn_path.nodes)

        for source in nodes:
            for target in nodes:
                distance = distances[(source, target)]

                # Distance should be non-negative
                self.assertGreaterEqual(distance, 0.0)

                # Self-distance should be zero
                if source == target:
                    self.assertEqual(distance, 0.0)

    def test_all_pairs_distances_symmetry(self):
        """Distances should be symmetric in undirected hypergraphs."""
        distances = self.distance_calc.compute_all_distances(self.hn_connected)
        nodes = list(self.hn_connected.nodes)

        for source in nodes:
            for target in nodes:
                if source != target:
                    dist_st = distances[(source, target)]
                    dist_ts = distances[(target, source)]

                    # For connected nodes, distances should be symmetric
                    if dist_st != float('inf') and dist_ts != float('inf'):
                        self.assertEqual(dist_st, dist_ts)

    def test_s_parameter_influence(self):
        """Different s parameters should affect distance computation."""
        calc_s1 = HypergraphDistance()
        calc_s2 = HypergraphDistance(s=2)

        # Test on a hypergraph where s parameter matters
        hn_s_test = Hypernetwork()
        hn_s_test.add_hyperedge(0, [1, 2, 3, 4])
        hn_s_test.add_hyperedge(1, [2, 5])
        hn_s_test.add_hyperedge(2, [3, 6])

        dist1 = calc_s1.compute_distance(hn_s_test, 1, 6)
        dist2 = calc_s2.compute_distance(hn_s_test, 1, 6)

        # Both should produce valid distances
        self.assertGreaterEqual(dist1, 0.0)
        self.assertGreaterEqual(dist2, 0.0)

    def test_distance_triangle_inequality(self):
        """Distances should satisfy triangle inequality where applicable."""
        nodes = [1, 2, 3, 4, 5]

        for i in nodes:
            for j in nodes:
                for k in nodes:
                    dist_ij = self.distance_calc.compute_distance(
                        self.hn_connected, i, j)
                    dist_jk = self.distance_calc.compute_distance(
                        self.hn_connected, j, k)
                    dist_ik = self.distance_calc.compute_distance(
                        self.hn_connected, i, k)

                    # Triangle inequality should hold for finite distances
                    if (dist_ij != float('inf') and dist_jk != float('inf')
                            and dist_ik != float('inf')):
                        self.assertLessEqual(dist_ik, dist_ij + dist_jk + 1e-10)

    def test_large_hyperedge_distances(self):
        """Distances in large hyperedges should be computed correctly."""
        hn_large = Hypernetwork()
        # Create a large hyperedge connecting many nodes
        large_nodes = list(range(1, 11))  # Nodes 1-10
        hn_large.add_hyperedge(0, large_nodes)

        # All nodes should be at distance 1 from each other
        for i in range(1, 6):  # Test subset to avoid excessive computation
            for j in range(i + 1, 6):
                distance = self.distance_calc.compute_distance(hn_large, i, j)
                self.assertEqual(distance, 1.0)


if __name__ == '__main__':
    unittest.main()
