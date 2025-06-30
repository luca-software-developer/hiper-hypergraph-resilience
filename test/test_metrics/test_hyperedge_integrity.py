# -*- coding: utf-8 -*-
"""
test_hyperedge_integrity.py

Unit tests for the HyperedgeIntegrity metric implementation.
Tests cover integrity computation, edge cases, and change analysis.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.hyperedge_integrity import HyperedgeIntegrity


class TestHyperedgeIntegrity(unittest.TestCase):
    """
    Tests HyperedgeIntegrity metric computation and analysis.
    Methods tested include integrity computation, change analysis,
    and edge case handling for various hypergraph configurations.
    """

    def setUp(self):
        """Create test fixtures with various hypergraph configurations."""
        self.integrity_metric = HyperedgeIntegrity()

        # Empty hypergraph
        self.hn_empty = Hypernetwork()

        # Single hyperedge
        self.hn_single = Hypernetwork()
        self.hn_single.add_hyperedge(0, [1, 2, 3])

        # Multi-hyperedge original
        self.hn_original = Hypernetwork()
        self.hn_original.add_hyperedge(0, [1, 2, 3])
        self.hn_original.add_hyperedge(1, [2, 3, 4])
        self.hn_original.add_hyperedge(2, [4, 5, 6])

        # Partially perturbed (one hyperedge lost)
        self.hn_partial = Hypernetwork()
        self.hn_partial.add_hyperedge(0, [1, 2, 3])
        self.hn_partial.add_hyperedge(1, [2, 3, 4])

        # Completely perturbed (all hyperedges lost)
        self.hn_complete = Hypernetwork()

        # Expanded hypergraph (more hyperedges than original)
        self.hn_expanded = Hypernetwork()
        self.hn_expanded.add_hyperedge(0, [1, 2, 3])
        self.hn_expanded.add_hyperedge(1, [2, 3, 4])
        self.hn_expanded.add_hyperedge(2, [4, 5, 6])
        self.hn_expanded.add_hyperedge(3, [7, 8, 9])

    def test_metric_initialization(self):
        """Test proper initialization of HyperedgeIntegrity metric."""
        self.assertEqual(self.integrity_metric.name, "Hyperedge Integrity")
        self.assertEqual(self.integrity_metric.symbol, "HI")

    def test_integrity_no_original_provided(self):
        """Test computation when no original hypergraph is provided."""
        integrity = HyperedgeIntegrity.compute(self.hn_original)
        self.assertEqual(integrity, 1.0)

    def test_integrity_empty_original(self):
        """Test integrity computation with empty original hypergraph."""
        integrity = HyperedgeIntegrity.compute(self.hn_original, self.hn_empty)
        self.assertEqual(integrity, 1.0)

    def test_integrity_perfect_preservation(self):
        """Test integrity computation with perfect preservation."""
        integrity = HyperedgeIntegrity.compute(self.hn_original,
                                               self.hn_original)
        self.assertEqual(integrity, 1.0)

    def test_integrity_partial_loss(self):
        """Test integrity computation with partial hyperedge loss."""
        integrity = HyperedgeIntegrity.compute(self.hn_partial,
                                               self.hn_original)
        expected = 2.0 / 3.0  # 2 out of 3 hyperedges preserved
        self.assertAlmostEqual(integrity, expected, places=6)

    def test_integrity_complete_loss(self):
        """Test integrity computation with complete hyperedge loss."""
        integrity = HyperedgeIntegrity.compute(self.hn_complete,
                                               self.hn_original)
        self.assertEqual(integrity, 0.0)

    def test_integrity_expansion(self):
        """Test integrity computation when hyperedges are added."""
        integrity = HyperedgeIntegrity.compute(self.hn_expanded,
                                               self.hn_original)
        expected = 4.0 / 3.0  # 4 out of 3 original hyperedges
        expected_clamped = min(1.0, expected)  # Should be clamped to 1.0
        self.assertAlmostEqual(integrity, expected_clamped, places=6)

    def test_integrity_value_bounds(self):
        """Test that integrity values are properly bounded between 0 and 1."""
        # Test lower bound
        integrity_min = HyperedgeIntegrity.compute(self.hn_empty,
                                                   self.hn_original)
        self.assertGreaterEqual(integrity_min, 0.0)
        self.assertLessEqual(integrity_min, 1.0)

        # Test upper bound with expansion
        integrity_max = HyperedgeIntegrity.compute(self.hn_expanded,
                                                   self.hn_original)
        self.assertGreaterEqual(integrity_max, 0.0)
        self.assertLessEqual(integrity_max, 1.0)

    def test_compute_change_no_loss(self):
        """Test change computation with no integrity loss."""
        change = self.integrity_metric.compute_change(self.hn_original,
                                                      self.hn_original)
        self.assertEqual(change, 0.0)

    def test_compute_change_partial_loss(self):
        """Test change computation with partial integrity loss."""
        change = self.integrity_metric.compute_change(self.hn_original,
                                                      self.hn_partial)
        expected = (2.0 / 3.0) - 1.0  # From 1.0 to 2/3
        self.assertAlmostEqual(change, expected, places=6)

    def test_compute_change_complete_loss(self):
        """Test change computation with complete integrity loss."""
        change = self.integrity_metric.compute_change(self.hn_original,
                                                      self.hn_complete)
        self.assertEqual(change, -1.0)

    def test_compute_change_expansion(self):
        """Test change computation with hyperedge expansion."""
        change = self.integrity_metric.compute_change(self.hn_original,
                                                      self.hn_expanded)
        # From 1.0 to min(1.0, 4/3) = 1.0, so change = 0.0
        self.assertEqual(change, 0.0)

    def test_string_representations(self):
        """Test string and repr methods."""
        str_repr = str(self.integrity_metric)
        self.assertEqual(str_repr, "Hyperedge Integrity (HI)")

        repr_str = repr(self.integrity_metric)
        expected_repr = ("HyperedgeIntegrity(name='Hyperedge Integrity', "
                         "symbol='HI')")
        self.assertEqual(repr_str, expected_repr)


if __name__ == '__main__':
    unittest.main()
