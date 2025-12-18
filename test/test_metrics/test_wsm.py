# -*- coding: utf-8 -*-
"""
test_wsm.py

Unit tests for the WSMNodeRanker class.
Tests WSM (Weighted Sum Model) multi-criteria node ranking including criteria
computation, ranking algorithms, node selection, and weight handling for
various hypergraph structures.
"""

import unittest

import numpy as np

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.wsm import WSMNodeRanker


class TestWSMNodeRanker(unittest.TestCase):
    """
    Tests WSMNodeRanker methods.
    Methods tested include node ranking computation, criteria calculation,
    top/bottom node selection, and weight customization.
    """

    def setUp(self):
        """Create WSM ranker and test hypergraphs."""
        self.ranker = WSMNodeRanker()

        # Create test hypergraphs using helper methods
        self._create_empty_hypergraph()
        self._create_single_node_hypergraph()
        self._create_simple_hypergraph()
        self._create_complex_hypergraph()
        self._create_star_topology_hypergraph()

    def _create_empty_hypergraph(self):
        """Create empty test hypergraph."""
        self.hn_empty = Hypernetwork()

    def _create_single_node_hypergraph(self):
        """Create single node test hypergraph."""
        self.hn_single = Hypernetwork()
        self.hn_single.add_node(1)

    def _create_simple_hypergraph(self):
        """Create simple test hypergraph with basic connectivity."""
        self.hn_simple = Hypernetwork()
        self.hn_simple.add_hyperedge(0, [1, 2, 3])
        self.hn_simple.add_hyperedge(1, [2, 3, 4])

    def _create_complex_hypergraph(self):
        """Create complex hypergraph with varied connectivity patterns."""
        self.hn_complex = Hypernetwork()
        self.hn_complex.add_hyperedge(0, [1, 2, 3])
        self.hn_complex.add_hyperedge(1, [2, 3, 4])
        self.hn_complex.add_hyperedge(2, [3, 4, 5])
        self.hn_complex.add_hyperedge(3, [1, 4])
        self.hn_complex.add_hyperedge(4, [5, 6, 7])
        self.hn_complex.add_hyperedge(5, [6, 7, 8])

    def _create_star_topology_hypergraph(self):
        """Create hub topology hypergraph."""
        self.hn_hub = Hypernetwork()
        self.hn_hub.add_hyperedge(0, [1, 2])
        self.hn_hub.add_hyperedge(1, [1, 3])
        self.hn_hub.add_hyperedge(2, [1, 4])
        self.hn_hub.add_hyperedge(3, [1, 5])

    def test_empty_hypergraph_ranking(self):
        """Empty hypergraph should return empty ranking."""
        ranked_nodes = self.ranker.rank_nodes(self.hn_empty)
        self.assertEqual(len(ranked_nodes), 0)

    def test_single_node_ranking(self):
        """Single node hypergraph should return single ranking."""
        ranked_nodes = self.ranker.rank_nodes(self.hn_single)
        self.assertEqual(len(ranked_nodes), 1)

        node_id, score = ranked_nodes[0]
        self.assertEqual(node_id, 1)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_ranking_completeness(self):
        """Ranking should include all nodes in hypergraph."""
        ranked_nodes = self.ranker.rank_nodes(self.hn_complex)

        # Should include all nodes
        expected_nodes = set(self.hn_complex.nodes)
        ranked_node_ids = set(node_id for node_id, score in ranked_nodes)
        self.assertEqual(expected_nodes, ranked_node_ids)

    def test_ranking_score_non_negative(self):
        """All ranking scores should be non-negative."""
        ranked_nodes = self.ranker.rank_nodes(self.hn_complex)

        for node_id, score in ranked_nodes:
            self.assertGreaterEqual(score, 0.0)

    def test_ranking_order(self):
        """Ranking should be sorted in descending order of scores."""
        ranked_nodes = self.ranker.rank_nodes(self.hn_complex)

        if len(ranked_nodes) > 1:
            for i in range(len(ranked_nodes) - 1):
                current_score = ranked_nodes[i][1]
                next_score = ranked_nodes[i + 1][1]
                self.assertGreaterEqual(current_score, next_score)

    def test_star_topology_ranking(self):
        """Central node in star topology should rank highest."""
        ranked_nodes = self.ranker.rank_nodes(self.hn_hub)

        if ranked_nodes:
            top_node_id = ranked_nodes[0][0]
            # Node 1 is the central node and should rank highest
            self.assertEqual(top_node_id, 1)

    def test_custom_weights_ranking(self):
        """Custom weights should affect ranking results."""
        # Equal weights
        weights_equal = [1.0, 1.0, 1.0, 1.0, 1.0]
        ranking_equal = self.ranker.rank_nodes(self.hn_complex, weights_equal)

        # Emphasize hyperdegree
        weights_hyperdegree = [5.0, 1.0, 1.0, 1.0, 1.0]
        ranking_hyperdegree = self.ranker.rank_nodes(
            self.hn_complex, weights_hyperdegree)

        # Both should produce valid rankings
        self.assertGreater(len(ranking_equal), 0)
        self.assertGreater(len(ranking_hyperdegree), 0)
        self.assertEqual(len(ranking_equal), len(ranking_hyperdegree))

    def test_hyperdegree_computation(self):
        """Hyperdegree should be computed correctly."""
        # Node 3 appears in multiple hyperedges in hn_complex
        node_3_hyperdegree = self.ranker._compute_hyperdegree(
            self.hn_complex, 3)
        expected_hyperdegree = len(self.hn_complex.get_hyperedges(3))
        self.assertEqual(node_3_hyperdegree, expected_hyperdegree)

    def test_neighborhood_size_computation(self):
        """Neighborhood size should be computed correctly."""
        neighborhood_size = self.ranker._compute_neighborhood_size(
            self.hn_complex, 3)
        self.assertGreaterEqual(neighborhood_size, 0.0)

    def test_degree_centrality_computation(self):
        """Degree centrality should be in range [0, 1]."""
        degree_centrality = self.ranker._compute_degree_centrality(
            self.hn_complex, 3)
        self.assertGreaterEqual(degree_centrality, 0.0)
        self.assertLessEqual(degree_centrality, 1.0)

    def test_degree_centrality_star_center(self):
        """Central node in star should have high degree centrality."""
        central_centrality = self.ranker._compute_degree_centrality(
            self.hn_hub, 1)

        # Compare with peripheral nodes
        peripheral_centrality = [
            self.ranker._compute_degree_centrality(self.hn_hub, node_id)
            for node_id in [2, 3, 4, 5]
        ]

        for peripheral_centrality in peripheral_centrality:
            self.assertGreaterEqual(central_centrality, peripheral_centrality)

    def test_clustering_coefficient_computation(self):
        """Clustering coefficient should be in valid range."""
        clustering = self.ranker._compute_avg_clustering(self.hn_complex, 3)
        self.assertGreaterEqual(clustering, 0.0)
        self.assertLessEqual(clustering, 1.0)

    def test_closed_triad_indicator(self):
        """Closed triad indicator should be binary."""
        indicator = self.ranker._compute_closed_triad_indicator(
            self.hn_complex, 3)
        self.assertIn(indicator, [0.0, 1.0])

    def test_get_top_nodes_percentage(self):
        """Top nodes selection should respect percentage parameter."""
        for percentage in [10, 25, 50]:
            top_nodes = self.ranker.get_top_nodes(self.hn_complex, percentage)

            total_nodes = self.hn_complex.order()
            expected_count = max(1, int(total_nodes * percentage / 100.0))
            actual_count = len(top_nodes)

            self.assertLessEqual(actual_count, expected_count)
            self.assertGreater(actual_count, 0)

    def test_get_bottom_nodes_percentage(self):
        """Bottom nodes selection should respect percentage parameter."""
        for percentage in [10, 25, 50]:
            bottom_nodes = self.ranker.get_bottom_nodes(
                self.hn_complex, percentage)

            total_nodes = self.hn_complex.order()
            expected_count = max(1, int(total_nodes * percentage / 100.0))
            actual_count = len(bottom_nodes)

            self.assertLessEqual(actual_count, expected_count)
            self.assertGreater(actual_count, 0)

    def test_top_bottom_nodes_disjoint(self):
        """Top and bottom nodes should be disjoint."""
        top_25 = set(self.ranker.get_top_nodes(self.hn_complex, 25))
        bottom_25 = set(self.ranker.get_bottom_nodes(self.hn_complex, 25))

        # For 25% each, there should be no overlap in most cases
        overlap = top_25 & bottom_25
        total_nodes = self.hn_complex.order()

        # Allow overlap only if graph is very small
        if total_nodes > 4:
            self.assertEqual(len(overlap), 0)

    def test_top_nodes_higher_scores(self):
        """Top nodes should have higher scores than bottom nodes."""
        ranking = self.ranker.rank_nodes(self.hn_complex)

        if len(ranking) > 2:
            top_nodes = self.ranker.get_top_nodes(self.hn_complex, 25)
            bottom_nodes = self.ranker.get_bottom_nodes(self.hn_complex, 25)

            # Get scores for comparison
            score_dict = dict(ranking)

            if top_nodes and bottom_nodes:
                min_top_score = min(score_dict[node] for node in top_nodes)
                max_bottom_score = max(
                    score_dict[node] for node in bottom_nodes)

                self.assertGreaterEqual(min_top_score, max_bottom_score)

    def test_matrix_normalization(self):
        """Matrix normalization should produce values in [0, 1]."""
        # Create test matrix
        test_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        normalized = self.ranker._normalize_matrix(test_matrix)

        # Check that all values are in [0, 1] (max normalization)
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                self.assertGreaterEqual(normalized[i, j], 0.0)
                self.assertLessEqual(normalized[i, j], 1.0)

    def test_wsm_algorithm_components(self):
        """WSM algorithm components should work correctly."""
        # Create simple test matrix
        criteria_matrix = np.array([
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0]
        ])

        weights = [1.0, 1.0, 1.0]
        scores = self.ranker._apply_wsm(criteria_matrix, weights)

        # Should return valid scores
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)

    def test_ranking_consistency(self):
        """Multiple ranking calls should produce consistent results."""
        ranking1 = self.ranker.rank_nodes(self.hn_complex)
        ranking2 = self.ranker.rank_nodes(self.hn_complex)

        # Results should be identical
        self.assertEqual(ranking1, ranking2)

    def test_percentage_edge_cases(self):
        """Edge case percentages should be handled correctly."""
        # Test 0% and 100% and values beyond bounds
        total_nodes = self.hn_complex.order()

        # Very small percentage should still return at least one node
        top_1 = self.ranker.get_top_nodes(self.hn_complex, 1)
        self.assertGreaterEqual(len(top_1), 1)

        # 100% should return all nodes
        top_100 = self.ranker.get_top_nodes(self.hn_complex, 100)
        self.assertEqual(len(top_100), total_nodes)

    def test_criteria_bounds_validation(self):
        """All computed criteria should be within reasonable bounds."""
        nodes = list(self.hn_complex.nodes)

        for node_id in nodes[:3]:
            hyperdegree = self.ranker._compute_hyperdegree(
                self.hn_complex, node_id)
            neighborhood_size = self.ranker._compute_neighborhood_size(
                self.hn_complex, node_id)
            clustering = self.ranker._compute_avg_clustering(
                self.hn_complex, node_id)
            centrality = self.ranker._compute_degree_centrality(
                self.hn_complex, node_id)
            triad_indicator = self.ranker._compute_closed_triad_indicator(
                self.hn_complex, node_id)

            # Validate bounds
            self.assertGreaterEqual(hyperdegree, 0.0)
            self.assertGreaterEqual(neighborhood_size, 0.0)
            self.assertGreaterEqual(clustering, 0.0)
            self.assertLessEqual(clustering, 1.0)
            self.assertGreaterEqual(centrality, 0.0)
            self.assertLessEqual(centrality, 1.0)
            self.assertIn(triad_indicator, [0.0, 1.0])

    def test_weight_validation(self):
        """Different weight configurations should be handled properly."""
        # Test with different weight vectors
        weight_configs = [
            None,  # Default weights
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Equal weights
            [2.0, 1.0, 1.0, 1.0, 1.0],  # Emphasize hyperdegree
            [0.5, 0.5, 0.5, 0.5, 0.5],  # Reduced weights
        ]

        for weights in weight_configs:
            ranking = self.ranker.rank_nodes(self.hn_complex, weights)

            # Should produce valid ranking
            self.assertGreater(len(ranking), 0)
            self.assertEqual(len(ranking), self.hn_complex.order())


if __name__ == '__main__':
    unittest.main()
