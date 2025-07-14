# -*- coding: utf-8 -*-
"""
test_higher_order_cohesion.py

Unit tests for Higher-Order Cohesion Resilience (HOCR) and Largest Higher-Order
Component (LHC) metrics that analyze hypergraph resilience based on m-th order
components.

This test suite validates the computation of higher-order components and
associated resilience metrics following the mathematical definitions provided
in the research framework.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.higher_order_cohesion import HigherOrderCohesionMetrics


class TestHigherOrderCohesionMetrics(unittest.TestCase):
    """
    Test suite for HigherOrderCohesionMetrics class.

    Tests the computation of m-th order components and related resilience
    metrics including HOCR_m and LHC_m.
    """

    def setUp(self):
        """Set up test fixtures with sample hypernetworks."""
        # Create a simple test hypernetwork
        self.hn_simple = Hypernetwork()
        self.hn_simple.add_hyperedge(0, [1, 2, 3])
        self.hn_simple.add_hyperedge(1, [2, 3, 4])
        self.hn_simple.add_hyperedge(2, [4, 5, 6])
        self.hn_simple.add_hyperedge(3, [1, 7, 8])

        # Create a more complex hypernetwork with clear m-order structure
        self.hn_complex = Hypernetwork()
        self.hn_complex.add_hyperedge(0, [1, 2, 3, 4])  # Component 1
        self.hn_complex.add_hyperedge(1, [2, 3, 4,
                                          5])  # Component 1
        self.hn_complex.add_hyperedge(2, [3, 4, 5,
                                          6])  # Component 1
        self.hn_complex.add_hyperedge(3, [10, 11, 12])  # Component 2
        self.hn_complex.add_hyperedge(4, [11, 12,
                                          13])  # Component 2
        self.hn_complex.add_hyperedge(5,
                                      [20, 21])  # Isolated (too small for m=2)

        # Create an empty hypernetwork
        self.hn_empty = Hypernetwork()

    def test_initialization_valid_m(self):
        """Test proper initialization with valid m values."""
        metrics_m2 = HigherOrderCohesionMetrics()
        self.assertEqual(metrics_m2.m, 2)

        metrics_m3 = HigherOrderCohesionMetrics(m=3)
        self.assertEqual(metrics_m3.m, 3)

    def test_initialization_invalid_m(self):
        """Test that initialization fails with invalid m values."""
        with self.assertRaises(ValueError):
            HigherOrderCohesionMetrics(m=1)

        with self.assertRaises(ValueError):
            HigherOrderCohesionMetrics(m=0)

        with self.assertRaises(ValueError):
            HigherOrderCohesionMetrics(m=-1)

    def test_compute_mth_order_components_empty_hypernetwork(self):
        """Test m-th order components computation on empty hypernetwork."""
        metrics = HigherOrderCohesionMetrics()
        components = metrics.compute_mth_order_components(self.hn_empty)
        self.assertEqual(components, [])

    def test_compute_mth_order_components_simple_case(self):
        """Test m-th order components with m=2 on simple hypernetwork."""
        metrics = HigherOrderCohesionMetrics()
        components = metrics.compute_mth_order_components(self.hn_simple)

        # Expected: hyperedges 0 and 1 share nodes [2,3] (2 nodes >= m=2)
        # hyperedges 1 and 2 share node [4] (1 node < m=2)
        # hyperedge 3 is isolated (shares only 1 node with hyperedge 0)

        # Should have one component containing hyperedges 0 and 1
        self.assertEqual(len(components), 1)
        component = components[0]
        self.assertEqual(len(component), 2)
        self.assertIn(0, component)
        self.assertIn(1, component)

    def test_compute_mth_order_components_complex_case(self):
        """Test m-th order components with different values on hypernetwork."""
        # Test with m=2
        metrics_m2 = HigherOrderCohesionMetrics()
        components_m2 = metrics_m2.compute_mth_order_components(self.hn_complex)

        # Expected with m=2:
        # - Component 1: hyperedges 0, 1, 2 (they form a chain of 3+ nodes)
        # - Component 2: hyperedges 3, 4 (they share 2 nodes)
        # - hyperedge 5 is isolated (too small)

        self.assertEqual(len(components_m2), 2)

        # Find larger component (should be 3 hyperedges)
        larger_component = max(components_m2, key=len)
        smaller_component = min(components_m2, key=len)

        self.assertEqual(len(larger_component), 3)
        self.assertEqual(len(smaller_component), 2)

        # Test with m=3
        metrics_m3 = HigherOrderCohesionMetrics(m=3)
        components_m3 = metrics_m3.compute_mth_order_components(self.hn_complex)

        # Expected with m=3:
        # - Component 1: hyperedges 0, 1, 2 (they share 3+ nodes)
        # - hyperedges 3, 4 are now isolated (they share only 2 nodes < m=3)

        self.assertEqual(len(components_m3), 1)
        self.assertEqual(len(components_m3[0]), 3)

    def test_compute_hocr_m_complete_destruction(self):
        """Test HOCR_m computation when all structure is destroyed."""
        metrics = HigherOrderCohesionMetrics()
        hocr = metrics.compute_hocr_m(self.hn_simple, self.hn_empty)

        # Should be 0 / (original_total + 1) = 0 / (2 + 1) = 0
        # since the simple hypernetwork has one component with 2 hyperedges
        self.assertAlmostEqual(hocr, 0.0, places=6)

    def test_compute_lhc_m_complete_destruction(self):
        """Test LHC_m computation when all structure is destroyed."""
        metrics = HigherOrderCohesionMetrics()
        lhc = metrics.compute_lhc_m(self.hn_simple, self.hn_empty)

        # Should be 0 / (largest_original + 1) = 0 / (2 + 1) = 0
        self.assertAlmostEqual(lhc, 0.0, places=6)

    def test_analyze_component_distribution_empty(self):
        """Test component distribution analysis on empty hypernetwork."""
        metrics = HigherOrderCohesionMetrics()
        analysis = metrics.analyze_component_distribution(self.hn_empty)

        expected = {
            'num_components': 0,
            'total_hyperedges_in_components': 0,
            'largest_component_size': 0,
            'average_component_size': 0.0,
            'component_sizes': []
        }
        self.assertEqual(analysis, expected)

    def test_analyze_component_distribution_simple(self):
        """Test component distribution analysis on simple hypernetwork."""
        metrics = HigherOrderCohesionMetrics()
        analysis = metrics.analyze_component_distribution(self.hn_simple)

        # Should have one component with 2 hyperedges
        self.assertEqual(analysis['num_components'], 1)
        self.assertEqual(analysis['total_hyperedges_in_components'], 2)
        self.assertEqual(analysis['largest_component_size'], 2)
        self.assertEqual(analysis['average_component_size'], 2.0)
        self.assertEqual(analysis['component_sizes'], [2])

    def test_analyze_component_distribution_complex(self):
        """Test component distribution analysis on complex hypernetwork."""
        metrics = HigherOrderCohesionMetrics()
        analysis = metrics.analyze_component_distribution(self.hn_complex)

        # Should have 2 components: one with 3 hyperedges, one with 2
        self.assertEqual(analysis['num_components'], 2)
        self.assertEqual(analysis['total_hyperedges_in_components'], 5)
        self.assertEqual(analysis['largest_component_size'], 3)
        self.assertEqual(analysis['average_component_size'], 2.5)

        # Component sizes should be [3, 2] in some order
        component_sizes = sorted(analysis['component_sizes'], reverse=True)
        self.assertEqual(component_sizes, [3, 2])

    def test_different_m_values_behavior(self):
        """Test that different m values produce expected behavior."""
        hn = Hypernetwork()
        hn.add_hyperedge(0, [1, 2, 3, 4, 5])
        hn.add_hyperedge(1, [3, 4, 5, 6, 7])  # shares 3 nodes with he 0
        hn.add_hyperedge(2,
                         [5, 6, 7, 8])  # shares 3 nodes with he 1, 1 with he 0

        # With m=2: all hyperedges should be connected
        metrics_m2 = HigherOrderCohesionMetrics()
        components_m2 = metrics_m2.compute_mth_order_components(hn)
        self.assertEqual(len(components_m2), 1)
        self.assertEqual(len(components_m2[0]), 3)

        # With m=3: hyperedges 0,1 and 1,2 are connected
        metrics_m3 = HigherOrderCohesionMetrics(m=3)
        components_m3 = metrics_m3.compute_mth_order_components(hn)
        self.assertEqual(len(components_m3), 1)
        self.assertEqual(len(components_m3[0]), 3)

        # With m=4: no hyperedges share 4+ nodes, so no components
        metrics_m4 = HigherOrderCohesionMetrics(m=4)
        components_m4 = metrics_m4.compute_mth_order_components(hn)
        self.assertEqual(len(components_m4), 0)


if __name__ == '__main__':
    unittest.main()
