# -*- coding: utf-8 -*-
"""
topsis.py

Implements TOPSIS (Technique for Order Preference by Similarity to Ideal
Solution) method for ranking hypergraph nodes based on multiple criteria.
Based on the approach described in the PIF-MN paper.
"""

import math
from typing import List, Tuple

from hiper.core.hypernetwork import Hypernetwork

try:
    import numpy as np
except ImportError:
    # Fallback implementation without numpy
    np = None


class TopsisNodeRanker:
    """
    Ranks hypergraph nodes using TOPSIS multi-criteria decision method.

    Criteria used (as per PIF-MN paper):
    1. Number of hyperedges containing the node (hyperdegree)
    2. Number of nodes in the neighborhood
    3. Average clustering coefficient of the neighborhood
    4. Degree centrality of the node
    5. Binary indicator for participation in closed triads
    """

    def __init__(self):
        """Initialize TOPSIS ranker."""
        self.criteria_names = [
            'hyperdegree',
            'neighborhood_size',
            'avg_clustering',
            'degree_centrality',
            'closed_triad_indicator'
        ]

    def rank_nodes(self, hypernetwork: Hypernetwork,
                   weights: List[float] = None) -> List[Tuple[int, float]]:
        """
        Rank nodes using TOPSIS method.

        Args:
            hypernetwork: Target hypergraph.
            weights: Weights for criteria (default: equal weights).

        Returns:
            List of (node_id, topsis_score) tuples, sorted by score descending.
        """
        nodes = list(hypernetwork.nodes)
        if not nodes:
            return []

        if weights is None:
            weights = [1.0] * len(self.criteria_names)

        # Compute criteria matrix
        criteria_matrix = self._compute_criteria_matrix(hypernetwork, nodes)

        # Apply TOPSIS algorithm
        scores = self._apply_topsis(criteria_matrix, weights)

        # Create ranked list
        ranked_nodes = list(zip(nodes, scores))
        ranked_nodes.sort(key=lambda x: x[1], reverse=True)

        return ranked_nodes

    def _compute_criteria_matrix(self, hypernetwork: Hypernetwork,
                                 nodes: List[int]) -> List[List[float]]:
        """
        Compute the criteria matrix for all nodes.

        Args:
            hypernetwork: Target hypergraph.
            nodes: List of node IDs.

        Returns:
            Matrix where rows are nodes and columns are criteria.
        """
        n_nodes = len(nodes)
        n_criteria = len(self.criteria_names)

        if np is not None:
            matrix = np.zeros((n_nodes, n_criteria))
        else:
            matrix = [[0.0 for _ in range(n_criteria)] for _ in range(n_nodes)]

        for i, node_id in enumerate(nodes):
            values = [
                self._compute_hyperdegree(hypernetwork, node_id),
                self._compute_neighborhood_size(hypernetwork, node_id),
                self._compute_avg_clustering(hypernetwork, node_id),
                self._compute_degree_centrality(hypernetwork, node_id),
                self._compute_closed_triad_indicator(hypernetwork, node_id)
            ]

            for j, value in enumerate(values):
                if np is not None:
                    matrix[i, j] = value
                else:
                    matrix[i][j] = value

        return matrix

    @staticmethod
    def _compute_hyperdegree(hypernetwork: Hypernetwork,
                             node_id: int) -> float:
        """Compute number of hyperedges containing the node."""
        return float(len(hypernetwork.get_hyperedges(node_id)))

    @staticmethod
    def _compute_neighborhood_size(hypernetwork: Hypernetwork,
                                   node_id: int) -> float:
        """Compute size of node's neighborhood."""
        neighbors = set(hypernetwork.get_neighbors(node_id))

        # Extended neighborhood: neighbors of neighbors
        extended_neighbors = set(neighbors)
        for neighbor in neighbors:
            extended_neighbors.update(hypernetwork.get_neighbors(neighbor))

        # Remove the node itself
        extended_neighbors.discard(node_id)

        return float(len(extended_neighbors))

    def _compute_avg_clustering(self, hypernetwork: Hypernetwork,
                                node_id: int) -> float:
        """Compute average clustering coefficient of node's neighborhood."""
        neighbors = hypernetwork.get_neighbors(node_id)

        if len(neighbors) < 2:
            return 0.0

        clustering_sum = 0.0
        valid_neighbors = 0

        for neighbor in neighbors:
            clustering = self._compute_local_clustering(hypernetwork, neighbor)
            clustering_sum += clustering
            valid_neighbors += 1

        return clustering_sum / valid_neighbors if valid_neighbors > 0 else 0.0

    @staticmethod
    def _compute_local_clustering(hypernetwork: Hypernetwork,
                                  node_id: int) -> float:
        """Compute local clustering coefficient for a node."""
        neighbors = set(hypernetwork.get_neighbors(node_id))
        k = len(neighbors)

        if k < 2:
            return 0.0

        # Count edges between neighbors (approximation for hypergraphs)
        connections = 0
        neighbors_list = list(neighbors)

        for i, n1 in enumerate(neighbors_list):
            for n2 in neighbors_list[i + 1:]:
                # Check if n1 and n2 are connected
                if n2 in hypernetwork.get_neighbors(n1):
                    connections += 1

        max_connections = k * (k - 1) // 2
        return connections / max_connections if max_connections > 0 else 0.0

    @staticmethod
    def _compute_degree_centrality(hypernetwork: Hypernetwork,
                                   node_id: int) -> float:
        """Compute degree centrality of the node."""
        degree = len(hypernetwork.get_neighbors(node_id))
        max_degree = hypernetwork.order() - 1

        return degree / max_degree if max_degree > 0 else 0.0

    @staticmethod
    def _compute_closed_triad_indicator(hypernetwork: Hypernetwork,
                                        node_id: int) -> float:
        """Binary indicator for participation in closed triads."""
        neighbors = list(hypernetwork.get_neighbors(node_id))

        # Check all pairs of neighbors
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                # Check if n1 and n2 are connected (forming a triad)
                if n2 in hypernetwork.get_neighbors(n1):
                    return 1.0

        return 0.0

    def _apply_topsis(self, criteria_matrix, weights: List[float]) -> List[
        float]:
        """
        Apply TOPSIS algorithm to criteria matrix.

        Args:
            criteria_matrix: Matrix of criteria values.
            weights: Weights for criteria.

        Returns:
            TOPSIS scores for each alternative.
        """
        # Normalize the decision matrix
        normalized_matrix = self._normalize_matrix(criteria_matrix)

        # Apply weights
        weighted_matrix = self._apply_weights(normalized_matrix, weights)

        # Determine ideal and negative-ideal solutions
        ideal_solution, negative_ideal = self._find_ideal_solutions(
            weighted_matrix)

        # Calculate distances to ideal solutions
        distances_to_ideal = []
        distances_to_negative = []

        n_rows = len(weighted_matrix) if isinstance(weighted_matrix, list) \
            else weighted_matrix.shape[0]

        for i in range(n_rows):
            row = weighted_matrix[i]
            dist_ideal = math.sqrt(sum((row[j] - ideal_solution[j]) ** 2
                                       for j in range(len(ideal_solution))))
            if np is not None:
                dist_negative = math.sqrt(sum((row[j] - negative_ideal[j]) ** 2
                                              for j in
                                              range(len(negative_ideal))))
            else:
                dist_negative = math.sqrt(sum((row[j] - negative_ideal[j]) ** 2
                                              for j in
                                              range(len(negative_ideal))))

            distances_to_ideal.append(dist_ideal)
            distances_to_negative.append(dist_negative)

        # Calculate TOPSIS scores
        scores = []
        for i in range(len(distances_to_ideal)):
            if distances_to_ideal[i] + distances_to_negative[i] == 0:
                score = 0.0
            else:
                score = (distances_to_negative[i] /
                         (distances_to_ideal[i] + distances_to_negative[i]))
            scores.append(score)

        return scores

    @staticmethod
    def _normalize_matrix(matrix):
        """
        Normalize criteria matrix using vector normalization.

        Args:
            matrix: Raw criteria matrix.

        Returns:
            Normalized matrix.
        """
        if np is not None:
            normalized = matrix.copy()
            for j in range(matrix.shape[1]):
                column_norm = math.sqrt(sum(matrix[i, j] ** 2
                                            for i in range(matrix.shape[0])))
                if column_norm > 0:
                    normalized[:, j] = matrix[:, j] / column_norm
            return normalized
        else:
            # Pure Python implementation
            n_rows = len(matrix)
            n_cols = len(matrix[0]) if n_rows > 0 else 0
            normalized = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

            for j in range(n_cols):
                # Calculate column norm
                column_norm = math.sqrt(sum(matrix[i][j] ** 2
                                            for i in range(n_rows)))

                # Normalize column
                if column_norm > 0:
                    for i in range(n_rows):
                        normalized[i][j] = matrix[i][j] / column_norm

            return normalized

    @staticmethod
    def _apply_weights(matrix, weights: List[float]):
        """Apply weights to normalized matrix."""
        if np is not None:
            return matrix * np.array(weights)
        else:
            n_rows = len(matrix)
            n_cols = len(matrix[0]) if n_rows > 0 else 0
            weighted = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

            for i in range(n_rows):
                for j in range(n_cols):
                    weighted[i][j] = matrix[i][j] * weights[j]

            return weighted

    @staticmethod
    def _find_ideal_solutions(matrix):
        """Find ideal and negative-ideal solutions."""
        if np is not None:
            ideal_solution = [max(matrix[i][j] for i in range(len(matrix)))
                              for j in range(len(matrix[0]))]
            negative_ideal = [min(matrix[i][j] for i in range(len(matrix)))
                              for j in range(len(matrix[0]))]
        else:
            n_rows = len(matrix)
            n_cols = len(matrix[0]) if n_rows > 0 else 0

            ideal_solution = []
            negative_ideal = []

            for j in range(n_cols):
                column_values = [matrix[i][j] for i in range(n_rows)]
                ideal_solution.append(max(column_values))
                negative_ideal.append(min(column_values))

        return ideal_solution, negative_ideal

    def get_top_nodes(self, hypernetwork: Hypernetwork,
                      percentage: float) -> List[int]:
        """
        Get top percentage of nodes by TOPSIS ranking.

        Args:
            hypernetwork: Target hypergraph.
            percentage: Percentage of top nodes to return (0-100).

        Returns:
            List of top node IDs.
        """
        ranked_nodes = self.rank_nodes(hypernetwork)
        n_nodes = len(ranked_nodes)
        n_top = max(1, int(n_nodes * percentage / 100.0))

        return [node_id for node_id, _ in ranked_nodes[:n_top]]

    def get_bottom_nodes(self, hypernetwork: Hypernetwork,
                         percentage: float) -> List[int]:
        """
        Get bottom percentage of nodes by TOPSIS ranking.

        Args:
            hypernetwork: Target hypergraph.
            percentage: Percentage of bottom nodes to return (0-100).

        Returns:
            List of bottom node IDs.
        """
        ranked_nodes = self.rank_nodes(hypernetwork)
        n_nodes = len(ranked_nodes)
        n_bottom = max(1, int(n_nodes * percentage / 100.0))

        return [node_id for node_id, _ in ranked_nodes[-n_bottom:]]
