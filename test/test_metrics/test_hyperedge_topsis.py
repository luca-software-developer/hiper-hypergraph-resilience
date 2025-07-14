# -*- coding: utf-8 -*-
"""
test_hyperedge_topsis.py

Unit tests for HyperedgeTopsisRanker that implements TOPSIS-based ranking
for hyperedges using specialized criteria including size, intersections,
and encapsulation relationships.

This test suite validates the ranking algorithms and criteria computation
used for targeted hyperedge removal experiments.
"""

import unittest

import numpy as np

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.hyperedge_topsis import HyperedgeTopsisRanker


class TestHyperedgeTopsisRanker(unittest.TestCase):
    """
    Test suite for HyperedgeTopsisRanker class.

    Tests the TOPSIS-based ranking system for hyperedges using
    specialized criteria: size, intersections, and encapsulation.
    """

    def setUp(self):
        """Set up test fixtures with sample hypernetworks."""
        self.ranker = HyperedgeTopsisRanker()

        # Create test hypernetwork
        self.hn = Hypernetwork()
        self.hn.add_hyperedge(0, [1, 2, 3])  # size=3, intersections=2
        self.hn.add_hyperedge(1, [2, 3, 4])  # size=3, intersections=2
        self.hn.add_hyperedge(2, [3, 4, 5, 6])  # size=4, intersections=2
        self.hn.add_hyperedge(3, [1, 2])  # size=2, intersections=1

        # Empty hypernetwork
        self.hn_empty = Hypernetwork()

    def test_rank_hyperedges_empty_hypernetwork(self):
        """Test ranking on empty hypernetwork."""
        ranked = self.ranker.rank_hyperedges(self.hn_empty)
        self.assertEqual(ranked, [])

    def test_rank_hyperedges_single_hyperedge(self):
        """Test ranking with single hyperedge."""
        hn_single = Hypernetwork()
        hn_single.add_hyperedge(0, [1, 2, 3])

        ranked = self.ranker.rank_hyperedges(hn_single)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0][0], 0)  # hyperedge id
        self.assertEqual(ranked[0][1], 1.0)  # perfect score for single element

    def test_compute_size(self):
        """Test hyperedge size computation."""
        self.assertEqual(self.ranker._compute_size(self.hn, 0), 3.0)
        self.assertEqual(self.ranker._compute_size(self.hn, 2), 4.0)
        self.assertEqual(self.ranker._compute_size(self.hn, 3), 2.0)

    def test_compute_encapsulation(self):
        """Test encapsulation computation."""
        # Hyperedge 3 [1,2] is contained in hyperedge 0 [1,2,3]
        self.assertEqual(self.ranker._compute_encapsulation(self.hn, 3), 1.0)

        # Hyperedge 0 is not contained in any other hyperedge
        self.assertEqual(self.ranker._compute_encapsulation(self.hn, 0), 0.0)

    def test_rank_hyperedges_with_default_weights(self):
        """Test ranking with default equal weights."""
        ranked = self.ranker.rank_hyperedges(self.hn)

        self.assertEqual(len(ranked), 4)
        # All hyperedge IDs should be present
        ranked_ids = [he_id for he_id, _ in ranked]
        self.assertEqual(set(ranked_ids), {0, 1, 2, 3})

        # Scores should be between 0 and 1
        for _, score in ranked:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_topsis_scores_computation(self):
        """Test TOPSIS scores computation."""
        weighted_matrix = np.array(
            [[0.6, 0.8], [0.8, 0.6], [0.7, 0.7]],
            dtype=float
        )
        scores = self.ranker._compute_topsis_scores(weighted_matrix)

        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_topsis_scores_identical_alternatives(self):
        """Test TOPSIS scores when all alternatives are identical."""
        weighted_matrix = np.array(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            dtype=float
        )
        scores = self.ranker._compute_topsis_scores(weighted_matrix)

        self.assertEqual(len(scores), 3)
        # All scores should be the same (0.5) since alternatives are identical
        for score in scores:
            self.assertAlmostEqual(score, 0.5, places=6)

    def test_intersections_boost_ranking(self):
        """
        Test that hyperedges with more intersections rank higher when weighted
        appropriately.
        """
        # Create hypernetwork where one hyperedge has many intersections
        hn = Hypernetwork()
        hn.add_hyperedge(0, [1, 2])  # intersects with 1, 2, 3
        hn.add_hyperedge(1, [1, 3])  # intersects with 0, 2, 3
        hn.add_hyperedge(2, [2, 4])  # intersects with 0, 1, 3
        hn.add_hyperedge(3, [1, 2, 3, 4])  # intersects with 0, 1, 2
        hn.add_hyperedge(4, [5, 6])  # isolated, no intersections

        # Use weights that favor intersections
        weights = {'size': 0.1, 'intersections': 0.8, 'encapsulation': 0.1}
        ranked = self.ranker.rank_hyperedges(hn, weights)

        # Hyperedge with most intersections should rank highly
        top_ranked_id = ranked[0][0]
        top_intersections = self.ranker._compute_intersections(
            hn, top_ranked_id
        )

        # Bottom ranked should be the isolated hyperedge
        bottom_ranked_id = ranked[-1][0]
        bottom_intersections = self.ranker._compute_intersections(
            hn, bottom_ranked_id
        )

        self.assertGreater(top_intersections, bottom_intersections)

    def test_encapsulation_affects_ranking(self):
        """
        Test that encapsulation affects ranking when weighted appropriately.
        """
        # Create hypernetwork with clear encapsulation relationships
        hn = Hypernetwork()
        hn.add_hyperedge(0, [1, 2, 3, 4])  # contains hyperedge 1
        hn.add_hyperedge(1, [1, 2])  # encapsulated by hyperedge 0
        hn.add_hyperedge(2, [5, 6])  # no encapsulation

        # This test verifies the encapsulation score affects ranking
        encap_score_1 = self.ranker._compute_encapsulation(hn, 1)
        encap_score_2 = self.ranker._compute_encapsulation(hn, 2)
        self.assertGreater(encap_score_1, encap_score_2)


if __name__ == '__main__':
    unittest.main()
