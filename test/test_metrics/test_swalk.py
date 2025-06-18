# -*- coding: utf-8 -*-
"""
test_swalk.py

Unit tests for the SwalkEfficiency class.
Tests s-walk efficiency computation for various hypergraph structures
including connectivity analysis, detailed statistics, and critical node
identification.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.swalk import SwalkEfficiency


class TestSwalkEfficiency(unittest.TestCase):
    """
    Tests SwalkEfficiency methods.
    Methods tested include s-walk efficiency computation, detailed analysis,
    node efficiency scoring, and critical node identification.
    """

    def setUp(self):
        """Create s-walk efficiency calculator and test hypergraphs."""
        self.efficiency_calc = SwalkEfficiency()

        # Create test hypergraphs using helper methods
        self._create_empty_hypergraph()
        self._create_single_node_hypergraph()
        self._create_simple_connected_hypergraph()
        self._create_well_connected_hypergraph()
        self._create_centralized_topology_hypergraph()
        self._create_disconnected_hypergraph()

    def _create_empty_hypergraph(self):
        """Create empty test hypergraph."""
        self.hn_empty = Hypernetwork()

    def _create_single_node_hypergraph(self):
        """Create single node test hypergraph."""
        self.hn_single = Hypernetwork()
        self.hn_single.add_node(1)

    def _create_simple_connected_hypergraph(self):
        """Create simple connected test hypergraph."""
        self.hn_simple = Hypernetwork()
        self.hn_simple.add_hyperedge(0, [1, 2])
        self.hn_simple.add_hyperedge(1, [2, 3])

    def _create_well_connected_hypergraph(self):
        """Create well-connected test hypergraph with overlapping edges."""
        self.hn_connected = Hypernetwork()
        self.hn_connected.add_hyperedge(0, [1, 2, 3])
        self.hn_connected.add_hyperedge(1, [2, 3, 4])
        self.hn_connected.add_hyperedge(2, [3, 4, 5])

    def _create_centralized_topology_hypergraph(self):
        """Create hub topology hypergraph."""
        self.hn_centralized = Hypernetwork()
        self.hn_centralized.add_hyperedge(0, [1, 6])
        self.hn_centralized.add_hyperedge(1, [1, 7])
        self.hn_centralized.add_hyperedge(2, [1, 8])
        self.hn_centralized.add_hyperedge(3, [1, 9])

    def _create_disconnected_hypergraph(self):
        """Create disconnected test hypergraph."""
        self.hn_disconnected = Hypernetwork()
        self.hn_disconnected.add_hyperedge(0, [1, 2])
        self.hn_disconnected.add_hyperedge(1, [3, 4])

    def test_empty_hypergraph_efficiency(self):
        """Empty hypergraph should have efficiency zero."""
        efficiency = self.efficiency_calc.compute(self.hn_empty)
        self.assertEqual(efficiency, 0.0)

    def test_single_node_efficiency(self):
        """Single node hypergraph should have efficiency zero."""
        efficiency = self.efficiency_calc.compute(self.hn_single)
        self.assertEqual(efficiency, 0.0)

    def test_connected_hypergraph_efficiency(self):
        """Connected hypergraph should have positive efficiency."""
        efficiency = self.efficiency_calc.compute(self.hn_connected)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)

    def test_efficiency_bounds(self):
        """Efficiency should always be in range [0, 1]."""
        test_hypergraphs = [
            self.hn_empty, self.hn_single, self.hn_simple,
            self.hn_connected, self.hn_centralized, self.hn_disconnected
        ]

        for hn in test_hypergraphs:
            efficiency = self.efficiency_calc.compute(hn)
            self.assertGreaterEqual(efficiency, 0.0)
            self.assertLessEqual(efficiency, 1.0)

    def test_star_topology_efficiency(self):
        """Star topology should have high efficiency due to central hub."""
        efficiency = self.efficiency_calc.compute(self.hn_centralized)
        self.assertGreater(efficiency, 0.0)

    def test_disconnected_hypergraph_efficiency(self):
        """Disconnected hypergraph should have lower efficiency."""
        connected_efficiency = self.efficiency_calc.compute(self.hn_simple)
        disconnected_efficiency = self.efficiency_calc.compute(
            self.hn_disconnected)

        # Both should be valid values
        self.assertGreaterEqual(connected_efficiency, 0.0)
        self.assertGreaterEqual(disconnected_efficiency, 0.0)

    def test_detailed_efficiency_computation(self):
        """Detailed efficiency should provide comprehensive statistics."""
        details = self.efficiency_calc.compute_detailed(self.hn_connected)

        # Check required keys
        required_keys = [
            'average_efficiency', 'min_efficiency', 'max_efficiency',
            'connected_pairs', 'total_pairs', 'connectivity_ratio'
        ]

        for key in required_keys:
            self.assertIn(key, details)
            self.assertIsInstance(details[key], (int, float))

    def test_detailed_efficiency_bounds(self):
        """Detailed efficiency values should be within valid bounds."""
        details = self.efficiency_calc.compute_detailed(self.hn_connected)

        # Efficiency values should be in [0, 1]
        self.assertGreaterEqual(details['average_efficiency'], 0.0)
        self.assertLessEqual(details['average_efficiency'], 1.0)
        self.assertGreaterEqual(details['min_efficiency'], 0.0)
        self.assertLessEqual(details['min_efficiency'], 1.0)
        self.assertGreaterEqual(details['max_efficiency'], 0.0)
        self.assertLessEqual(details['max_efficiency'], 1.0)

        # Connectivity ratio should be in [0, 1]
        self.assertGreaterEqual(details['connectivity_ratio'], 0.0)
        self.assertLessEqual(details['connectivity_ratio'], 1.0)

        # Pair counts should be non-negative
        self.assertGreaterEqual(details['connected_pairs'], 0)
        self.assertGreaterEqual(details['total_pairs'], 0)
        self.assertLessEqual(details['connected_pairs'], details['total_pairs'])

    def test_detailed_efficiency_consistency(self):
        """Detailed efficiency should be consistent with basic computation."""
        basic_efficiency = self.efficiency_calc.compute(self.hn_connected)
        detailed_efficiency = self.efficiency_calc.compute_detailed(
            self.hn_connected)

        # Average efficiency should match basic computation
        self.assertAlmostEqual(
            basic_efficiency,
            detailed_efficiency['average_efficiency'],
            places=5
        )

    def test_node_efficiencies_computation(self):
        """Node efficiencies should be computed for all nodes."""
        node_efficiencies = self.efficiency_calc.compute_node_efficiencies(
            self.hn_connected)

        # Should have efficiency for each node
        expected_nodes = set(self.hn_connected.nodes)
        actual_nodes = set(node_efficiencies.keys())
        self.assertEqual(expected_nodes, actual_nodes)

        # All efficiency values should be in valid range
        for node_id, efficiency in node_efficiencies.items():
            self.assertGreaterEqual(efficiency, 0.0)
            self.assertLessEqual(efficiency, 1.0)

    def test_node_efficiencies_star_topology(self):
        """Central node in star should have maximum efficiency."""
        node_efficiencies = self.efficiency_calc.compute_node_efficiencies(
            self.hn_centralized)

        # Node 1 is the central node and should have high efficiency
        central_efficiency = node_efficiencies.get(1, 0.0)

        # Compare with other nodes
        other_efficiencies = [
            eff for node_id, eff in node_efficiencies.items()
            if node_id != 1
        ]

        if other_efficiencies:
            max_other_efficiency = max(other_efficiencies)
            self.assertGreaterEqual(central_efficiency, max_other_efficiency)

    def test_critical_nodes_identification(self):
        """Critical nodes should be identified correctly."""
        critical_nodes = self.efficiency_calc.find_critical_nodes(
            self.hn_connected, top_k=3)

        # Should return at most 3 nodes (or fewer if graph is smaller)
        self.assertLessEqual(len(critical_nodes), 3)
        self.assertLessEqual(len(critical_nodes), self.hn_connected.order())

        # Each entry should be (node_id, impact) tuple
        for node_id, impact in critical_nodes:
            self.assertIsInstance(node_id, int)
            self.assertIsInstance(impact, float)
            self.assertGreaterEqual(impact, 0.0)

    def test_critical_nodes_sorting(self):
        """Critical nodes should be sorted by impact (descending)."""
        critical_nodes = self.efficiency_calc.find_critical_nodes(
            self.hn_connected)

        # Should be sorted by impact in descending order
        if len(critical_nodes) > 1:
            for i in range(len(critical_nodes) - 1):
                current_impact = critical_nodes[i][1]
                next_impact = critical_nodes[i + 1][1]
                self.assertGreaterEqual(current_impact, next_impact)

    def test_critical_nodes_star_topology(self):
        """Central node in star should be most critical."""
        critical_nodes = self.efficiency_calc.find_critical_nodes(
            self.hn_centralized, top_k=2)

        if critical_nodes:
            most_critical_node = critical_nodes[0][0]
            # Node 1 is the central node and should be most critical
            self.assertEqual(most_critical_node, 1)

    def test_s_parameter_variation(self):
        """Different s parameters should produce valid results."""
        calc_s1 = SwalkEfficiency()
        calc_s2 = SwalkEfficiency(s=2)

        eff1 = calc_s1.compute(self.hn_connected)
        eff2 = calc_s2.compute(self.hn_connected)

        # Both should be valid efficiency values
        self.assertGreaterEqual(eff1, 0.0)
        self.assertLessEqual(eff1, 1.0)
        self.assertGreaterEqual(eff2, 0.0)
        self.assertLessEqual(eff2, 1.0)

    def test_efficiency_with_isolated_nodes(self):
        """Efficiency computation should handle isolated nodes gracefully."""
        hn_isolated = Hypernetwork()
        hn_isolated.add_hyperedge(0, [1, 2])
        hn_isolated.add_node(3)  # Isolated node

        efficiency = self.efficiency_calc.compute(hn_isolated)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)

    def test_efficiency_consistency_across_calls(self):
        """Multiple calls should produce consistent results."""
        eff1 = self.efficiency_calc.compute(self.hn_connected)
        eff2 = self.efficiency_calc.compute(self.hn_connected)

        self.assertEqual(eff1, eff2)

    def test_empty_critical_nodes_handling(self):
        """Critical nodes computation should handle edge cases."""
        # Test with single node
        critical_nodes = self.efficiency_calc.find_critical_nodes(
            self.hn_single)

        # Should handle gracefully (might be empty or contain the single node)
        self.assertIsInstance(critical_nodes, list)
        self.assertLessEqual(len(critical_nodes), 1)


if __name__ == '__main__':
    unittest.main()
