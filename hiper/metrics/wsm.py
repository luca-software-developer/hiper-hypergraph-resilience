# -*- coding: utf-8 -*-
"""
wsm.py

Implements WSM (Weighted Sum Model) method for ranking hypergraph nodes
based on multiple criteria.

WSM is a simpler MCDM method that uses weighted normalization and summation
to rank alternatives. It is expected to provide comparable but potentially
less sophisticated results compared to TOPSIS.
"""

from typing import List, Tuple

from hiper.core.hypernetwork import Hypernetwork

try:
    import numpy as np
except ImportError:
    # Fallback implementation without numpy
    np = None


class WSMNodeRanker:
    """
    Ranks hypergraph nodes using WSM (Weighted Sum Model) multi-criteria
    decision method.

    Criteria used (same as TOPSIS for comparability):
    1. Number of hyperedges containing the node (hyperdegree)
    2. Number of nodes in the neighborhood
    3. Average clustering coefficient of the neighborhood
    4. Degree centrality of the node
    5. Binary indicator for participation in closed triads
    """

    def __init__(self):
        """Initialize WSM ranker."""
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
        Rank nodes using WSM method.

        Args:
            hypernetwork: Target hypergraph.
            weights: Weights for criteria (default: equal weights).

        Returns:
            List of (node_id, wsm_score) tuples, sorted by score descending.
        """
        nodes = list(hypernetwork.nodes)
        if not nodes:
            return []

        if weights is None:
            weights = [1.0] * len(self.criteria_names)

        # Compute criteria matrix
        criteria_matrix = self._compute_criteria_matrix(hypernetwork, nodes)

        # Apply WSM algorithm
        scores = self._apply_wsm(criteria_matrix, weights)

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
        """Compute hypergraph clustering coefficient for a node."""
        # Get all hyperedges containing this node
        node_hyperedges = hypernetwork.get_hyperedges(node_id)

        if len(node_hyperedges) < 2:
            return 0.0

        # For hypergraph clustering, we measure how many pairs of hyperedges
        # containing this node also share other common nodes
        shared_connections = 0
        total_pairs = 0

        hyperedges_list = list(node_hyperedges)
        for i, edge1 in enumerate(hyperedges_list):
            for edge2 in hyperedges_list[i + 1:]:
                total_pairs += 1

                # Get nodes in both hyperedges (excluding the central node)
                nodes1 = set(hypernetwork.get_nodes(edge1)) - {node_id}
                nodes2 = set(hypernetwork.get_nodes(edge2)) - {node_id}

                # If they share additional nodes, they form a clustered structure
                if len(nodes1.intersection(nodes2)) > 0:
                    shared_connections += 1

        return shared_connections / total_pairs if total_pairs > 0 else 0.0

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

    def _apply_wsm(self, criteria_matrix, weights: List[float]) -> List[float]:
        """
        Apply WSM algorithm to criteria matrix.

        WSM steps:
        1. Normalize the decision matrix
        2. Apply weights
        3. Sum weighted values for each alternative

        Args:
            criteria_matrix: Matrix of criteria values.
            weights: Weights for criteria.

        Returns:
            WSM scores for each alternative.
        """
        # Normalize the decision matrix
        normalized_matrix = self._normalize_matrix(criteria_matrix)

        # Apply weights and sum
        scores = []
        n_rows = len(normalized_matrix) if isinstance(normalized_matrix, list) \
            else normalized_matrix.shape[0]

        for i in range(n_rows):
            row = normalized_matrix[i]
            # Weighted sum
            weighted_sum = sum(row[j] * weights[j]
                               for j in range(len(weights)))
            scores.append(weighted_sum)

        return scores

    @staticmethod
    def _normalize_matrix(matrix):
        """
        Normalize criteria matrix using max normalization.

        For WSM, we use max normalization: x_ij_norm = x_ij / max(x_j)
        This is simpler than TOPSIS's vector normalization.

        Args:
            matrix: Raw criteria matrix.

        Returns:
            Normalized matrix.
        """
        if np is not None:
            normalized = np.zeros_like(matrix, dtype=float)
            for j in range(matrix.shape[1]):
                column = matrix[:, j]
                max_val = np.max(column)
                if max_val > 0:
                    normalized[:, j] = column / max_val
                else:
                    normalized[:, j] = column
            return normalized
        else:
            # Pure Python implementation
            n_rows = len(matrix)
            n_cols = len(matrix[0]) if n_rows > 0 else 0
            normalized = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

            for j in range(n_cols):
                # Find max value in column
                max_val = max(matrix[i][j] for i in range(n_rows))

                # Normalize column
                if max_val > 0:
                    for i in range(n_rows):
                        normalized[i][j] = matrix[i][j] / max_val

            return normalized

    def get_top_nodes(self, hypernetwork: Hypernetwork,
                      percentage: float) -> List[int]:
        """
        Get top percentage of nodes by WSM ranking.

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
        Get bottom percentage of nodes by WSM ranking.

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
