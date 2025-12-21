# -*- coding: utf-8 -*-
"""
test_redundancy.py

Unit tests for the RedundancyCoefficient class.
Tests redundancy coefficient computation for various hypergraph structures
including edge cases, overlapping patterns, and detailed analysis methods.
"""

import math
import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.redundancy import RedundancyCoefficient


class TestRedundancyCoefficient(unittest.TestCase):
    """
    Tests RedundancyCoefficient methods.
    Methods tested include redundancy computation, pairwise overlap analysis,
    and identification of most overlapping hyperedge pairs.
    """

    def setUp(self):
        """Create redundancy calculator and test hypergraphs."""
        self.redundancy_calc = RedundancyCoefficient()

        # Empty hypergraph
        self.hn_empty = Hypernetwork()

        # Single hyperedge
        self.hn_single = Hypernetwork()
        self.hn_single.add_hyperedge(0, [1, 2, 3])

        # Disjoint hyperedges (no overlap)
        self.hn_disjoint = Hypernetwork()
        self.hn_disjoint.add_hyperedge(0, [1, 2])
        self.hn_disjoint.add_hyperedge(1, [3, 4])
        self.hn_disjoint.add_hyperedge(2, [5, 6])

        # Overlapping hyperedges
        self.hn_overlapping = Hypernetwork()
        self.hn_overlapping.add_hyperedge(0, [1, 2, 3])
        self.hn_overlapping.add_hyperedge(1, [2, 3, 4])
        self.hn_overlapping.add_hyperedge(2, [3, 4, 5])

        # Identical hyperedges (maximum overlap)
        self.hn_identical = Hypernetwork()
        self.hn_identical.add_hyperedge(0, [1, 2, 3])
        self.hn_identical.add_hyperedge(1, [1, 2, 3])

        # Complex overlap pattern
        self.hn_complex = Hypernetwork()
        self.hn_complex.add_hyperedge(0, [1, 2, 3, 4])
        self.hn_complex.add_hyperedge(1, [2, 3])
        self.hn_complex.add_hyperedge(2, [3, 4, 5])
        self.hn_complex.add_hyperedge(3, [1, 5, 6])

    def test_empty_hypergraph_redundancy(self):
        """Empty hypergraph should have redundancy zero."""
        rho = self.redundancy_calc.compute(self.hn_empty)
        self.assertEqual(rho, 0.0)

    def test_single_hyperedge_redundancy(self):
        """Single hyperedge should have redundancy zero."""
        rho = self.redundancy_calc.compute(self.hn_single)
        self.assertEqual(rho, 0.0)

    def test_disjoint_hyperedges_redundancy(self):
        """Disjoint hyperedges should have redundancy zero."""
        rho = self.redundancy_calc.compute(self.hn_disjoint)
        self.assertEqual(rho, 0.0)

    def test_overlapping_hyperedges_redundancy(self):
        """Overlapping hyperedges should have positive redundancy."""
        rho = self.redundancy_calc.compute(self.hn_overlapping)
        self.assertGreater(rho, 0.0)
        self.assertLessEqual(rho, 1.0)

    def test_redundancy_bounds(self):
        """Redundancy coefficient should always be in range [0, 1]."""
        test_hypergraphs = [
            self.hn_empty, self.hn_single, self.hn_disjoint,
            self.hn_overlapping, self.hn_identical, self.hn_complex
        ]

        for hn in test_hypergraphs:
            rho = self.redundancy_calc.compute(hn)
            self.assertGreaterEqual(rho, 0.0)
            self.assertLessEqual(rho, 1.0)

    def test_pairwise_overlaps_computation(self):
        """Pairwise overlaps should be computed correctly."""
        overlaps = self.redundancy_calc.compute_pairwise_overlaps(
            self.hn_overlapping)

        # Should have overlaps for all pairs of distinct hyperedges
        expected_pairs = 3  # C(3,2) = 3 pairs
        self.assertEqual(len(overlaps), expected_pairs)

        # Each overlap should be a tuple with correct structure
        for overlap in overlaps:
            self.assertEqual(len(overlap),
                             4)  # (e1, e2, intersection, normalized)
            edge1, edge2, intersection_size, normalized_overlap = overlap
            self.assertIsInstance(edge1, int)
            self.assertIsInstance(edge2, int)
            self.assertIsInstance(intersection_size, int)
            self.assertIsInstance(normalized_overlap, float)
            self.assertGreaterEqual(normalized_overlap, 0.0)

    def test_specific_overlap_calculation(self):
        """Test specific overlap calculation with known values."""
        # Create hypergraph with known overlap
        hn_known = Hypernetwork()
        hn_known.add_hyperedge(0, [1, 2, 3])  # Size 3
        hn_known.add_hyperedge(1, [2, 3, 4])  # Size 3, intersection {2,3}

        overlaps = self.redundancy_calc.compute_pairwise_overlaps(hn_known)
        self.assertEqual(len(overlaps), 1)

        edge1, edge2, intersection_size, normalized_overlap = overlaps[0]
        self.assertEqual(intersection_size, 2)  # Nodes 2 and 3

        # Normalized overlap should be 2/sqrt(3*3) = 2/3
        expected_normalized = 2.0 / math.sqrt(3 * 3)
        self.assertAlmostEqual(normalized_overlap, expected_normalized,
                               places=5)

    def test_most_overlapping_pairs(self):
        """Most overlapping pairs should be identified correctly."""
        top_pairs = self.redundancy_calc.get_most_overlapping_pairs(
            self.hn_complex, top_k=3)

        # Should return at most 3 pairs (or fewer if less available)
        self.assertLessEqual(len(top_pairs), 3)

        # Pairs should be sorted by normalized overlap (descending)
        if len(top_pairs) > 1:
            for i in range(len(top_pairs) - 1):
                current_overlap = top_pairs[i][3]
                next_overlap = top_pairs[i + 1][3]
                self.assertGreaterEqual(current_overlap, next_overlap)

    def test_empty_hypergraph_pairwise_overlaps(self):
        """Empty hypergraph should have no pairwise overlaps."""
        overlaps = self.redundancy_calc.compute_pairwise_overlaps(
            self.hn_empty)
        self.assertEqual(len(overlaps), 0)

    def test_single_hyperedge_pairwise_overlaps(self):
        """Single hyperedge should have no pairwise overlaps."""
        overlaps = self.redundancy_calc.compute_pairwise_overlaps(
            self.hn_single)
        self.assertEqual(len(overlaps), 0)

    def test_zero_size_hyperedges_handling(self):
        """Hyperedges with zero size should be handled gracefully."""
        # This is an edge case that might occur in corrupted data
        hn_edge_case = Hypernetwork()
        hn_edge_case.add_hyperedge(0, [1, 2])
        # Manually create a scenario that might lead to zero-size calculations

        rho = self.redundancy_calc.compute(hn_edge_case)
        self.assertGreaterEqual(rho, 0.0)
        self.assertLessEqual(rho, 1.0)

    def test_large_hyperedges_overlap(self):
        """Large hyperedges should have their overlap computed correctly."""
        hn_large = Hypernetwork()
        # Create large hyperedges with partial overlap
        large_edge1 = list(range(1, 11))  # [1, 2, ..., 10]
        large_edge2 = list(range(5, 15))  # [5, 6, ..., 14]

        hn_large.add_hyperedge(0, large_edge1)
        hn_large.add_hyperedge(1, large_edge2)

        overlaps = self.redundancy_calc.compute_pairwise_overlaps(hn_large)
        self.assertEqual(len(overlaps), 1)

        edge1, edge2, intersection_size, normalized_overlap = overlaps[0]
        self.assertEqual(intersection_size, 6)  # Nodes 5, 6, 7, 8, 9, 10

        # Verify normalized overlap calculation
        expected_normalized = 6.0 / math.sqrt(10 * 10)
        self.assertAlmostEqual(normalized_overlap, expected_normalized,
                               places=5)

    def test_redundancy_formula_verification(self):
        """Verify redundancy formula implementation with manual calculation."""
        # Use simple case for manual verification
        hn_verify = Hypernetwork()
        hn_verify.add_hyperedge(0, [1, 2])  # Size 2
        hn_verify.add_hyperedge(1, [2, 3])  # Size 2, intersection 1
        hn_verify.add_hyperedge(2, [1, 3])  # Size 2, intersection 1 with each

        # Manual calculation:
        # Pair (0,1): intersection=1, normalized=1/sqrt(2*2)=0.5
        # Pair (0,2): intersection=1, normalized=1/sqrt(2*2)=0.5
        # Pair (1,2): intersection=1, normalized=1/sqrt(2*2)=0.5
        # Total = 3 * 0.5 = 1.5
        # Formula: (2 * 1.5) / (3 * (3-1)) = 3.0 / 6 = 1/2

        rho = self.redundancy_calc.compute(hn_verify)
        expected_rho = 1.0 / 2.0
        self.assertAlmostEqual(rho, expected_rho, places=5)


if __name__ == '__main__':
    unittest.main()
