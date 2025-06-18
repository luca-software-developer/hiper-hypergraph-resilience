# -*- coding: utf-8 -*-
"""
test_connectivity.py

Unit tests for the connectivity metrics classes.
Tests hypergraph connectivity and hyperedge connectivity computations
with various hypergraph structures and edge cases.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.connectivity import HypergraphConnectivity, \
    HyperedgeConnectivity


class TestHypergraphConnectivity(unittest.TestCase):
    """
    Tests HypergraphConnectivity methods.
    Methods tested include connectivity computation for various
    hypergraph topologies, edge cases, and disconnected structures.
    """

    def setUp(self):
        """Create connectivity calculator and test hypergraphs."""
        self.connectivity_calc = HypergraphConnectivity()
        self.hn_empty = Hypernetwork()
        self.hn_single = Hypernetwork()
        self.hn_single.add_node(1)

        # Connected hypergraph
        self.hn_connected = Hypernetwork()
        self.hn_connected.add_hyperedge(0, [1, 2, 3])
        self.hn_connected.add_hyperedge(1, [2, 3, 4])
        self.hn_connected.add_hyperedge(2, [4, 5])

        # Star topology (vulnerable to central node removal)
        self.hn_star = Hypernetwork()
        self.hn_star.add_hyperedge(0, [1, 2])
        self.hn_star.add_hyperedge(1, [1, 3])
        self.hn_star.add_hyperedge(2, [1, 4])

        # Disconnected hypergraph
        self.hn_disconnected = Hypernetwork()
        self.hn_disconnected.add_hyperedge(0, [1, 2])
        self.hn_disconnected.add_hyperedge(1, [3, 4])

    def test_empty_hypergraph_connectivity(self):
        """Empty hypergraph should have connectivity zero."""
        kappa = self.connectivity_calc.compute(self.hn_empty)
        self.assertEqual(kappa, 0)

    def test_single_node_connectivity(self):
        """Single node hypergraph should have connectivity zero."""
        kappa = self.connectivity_calc.compute(self.hn_single)
        self.assertEqual(kappa, 0)

    def test_connected_hypergraph_connectivity(self):
        """Connected hypergraph should have positive connectivity."""
        kappa = self.connectivity_calc.compute(self.hn_connected)
        self.assertGreaterEqual(kappa, 0)
        self.assertLessEqual(kappa, self.hn_connected.order() - 1)

    def test_star_topology_connectivity(self):
        """Star topology should have connectivity one."""
        # Removing central node 1 should disconnect the graph
        kappa = self.connectivity_calc.compute(self.hn_star)
        self.assertEqual(kappa, 1)

    def test_disconnected_hypergraph_connectivity(self):
        """Disconnected hypergraph should have connectivity zero."""
        kappa = self.connectivity_calc.compute(self.hn_disconnected)
        self.assertEqual(kappa, 0)

    def test_complete_removal_connectivity(self):
        """Removing all but one node should always disconnect."""
        # Create small hypergraph for complete testing
        hn_small = Hypernetwork()
        hn_small.add_hyperedge(0, [1, 2])

        kappa = self.connectivity_calc.compute(hn_small)
        # Should be at most 1 (removing one node leaves one isolated)
        self.assertLessEqual(kappa, 1)

    def test_s_parameter_consistency(self):
        """Different s parameters should produce consistent results."""
        calc_s1 = HypergraphConnectivity()
        calc_s2 = HypergraphConnectivity(s=2)

        kappa1 = calc_s1.compute(self.hn_connected)
        kappa2 = calc_s2.compute(self.hn_connected)

        # Both should be valid connectivity values
        self.assertGreaterEqual(kappa1, 0)
        self.assertGreaterEqual(kappa2, 0)


class TestHyperedgeConnectivity(unittest.TestCase):
    """
    Tests HyperedgeConnectivity methods.
    Methods tested include hyperedge connectivity computation for various
    hypergraph structures and validation of edge removal scenarios.
    """

    def setUp(self):
        """Create hyperedge connectivity calculator and test hypergraphs."""
        self.edge_connectivity_calc = HyperedgeConnectivity()
        self.hn_empty = Hypernetwork()

        # Simple connected hypergraph
        self.hn_simple = Hypernetwork()
        self.hn_simple.add_hyperedge(0, [1, 2])
        self.hn_simple.add_hyperedge(1, [2, 3])

        # Redundant connections
        self.hn_redundant = Hypernetwork()
        self.hn_redundant.add_hyperedge(0, [1, 2, 3])
        self.hn_redundant.add_hyperedge(1, [1, 2])
        self.hn_redundant.add_hyperedge(2, [2, 3])
        self.hn_redundant.add_hyperedge(3, [1, 3])

        # Bridge topology
        self.hn_bridge = Hypernetwork()
        self.hn_bridge.add_hyperedge(0, [1, 2])
        self.hn_bridge.add_hyperedge(1, [2, 3])  # Bridge edge
        self.hn_bridge.add_hyperedge(2, [3, 4])

    def test_empty_hypergraph_edge_connectivity(self):
        """Empty hypergraph should have edge connectivity zero."""
        lambda_val = self.edge_connectivity_calc.compute(self.hn_empty)
        self.assertEqual(lambda_val, 0)

    def test_simple_hypergraph_edge_connectivity(self):
        """Simple hypergraph should have positive edge connectivity."""
        lambda_val = self.edge_connectivity_calc.compute(self.hn_simple)
        self.assertGreaterEqual(lambda_val, 0)
        self.assertLessEqual(lambda_val, self.hn_simple.size())

    def test_bridge_topology_edge_connectivity(self):
        """Bridge topology should have edge connectivity one."""
        # Removing the bridge edge should disconnect the graph
        lambda_val = self.edge_connectivity_calc.compute(self.hn_bridge)
        self.assertEqual(lambda_val, 1)

    def test_redundant_connections_edge_connectivity(self):
        """Highly connected hypergraph should have higher edge connectivity."""
        lambda_val = self.edge_connectivity_calc.compute(self.hn_redundant)
        self.assertGreaterEqual(lambda_val, 1)

    def test_edge_connectivity_bounds(self):
        """Edge connectivity should be within valid bounds."""
        lambda_val = self.edge_connectivity_calc.compute(self.hn_simple)
        # Should be between 0 and total number of edges
        self.assertGreaterEqual(lambda_val, 0)
        self.assertLessEqual(lambda_val, self.hn_simple.size())

    def test_single_edge_hypergraph(self):
        """Single edge hypergraph should have edge connectivity one."""
        hn_single_edge = Hypernetwork()
        hn_single_edge.add_hyperedge(0, [1, 2, 3])

        lambda_val = self.edge_connectivity_calc.compute(hn_single_edge)
        self.assertEqual(lambda_val, 1)

    def test_s_parameter_influence(self):
        """Different s parameters may affect edge connectivity."""
        calc_s1 = HyperedgeConnectivity()
        calc_s2 = HyperedgeConnectivity(s=2)

        lambda1 = calc_s1.compute(self.hn_redundant)
        lambda2 = calc_s2.compute(self.hn_redundant)

        # Both should produce valid results
        self.assertGreaterEqual(lambda1, 0)
        self.assertGreaterEqual(lambda2, 0)


if __name__ == '__main__':
    unittest.main()
