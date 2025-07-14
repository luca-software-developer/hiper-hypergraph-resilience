# -*- coding: utf-8 -*-
"""
hyperedge_topsis.py

TOPSIS-based ranking system for hyperedges using specialized criteria.

Evaluates hyperedges based on size, number of intersections with other
hyperedges, and encapsulation relationships for targeted removal experiments.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np

from hiper.core.hypernetwork import Hypernetwork


class HyperedgeTopsisRanker:
    """
    TOPSIS-based ranking system for hyperedges using specialized criteria.

    Evaluates hyperedges based on size, number of intersections with other
    hyperedges, and encapsulation relationships.
    """

    def __init__(self):
        """Initialize the TOPSIS ranker for hyperedges."""
        pass

    def rank_hyperedges(
            self,
            hypernetwork: Hypernetwork,
            weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank hyperedges using TOPSIS method with specialized criteria.

        Args:
            hypernetwork: Target hypergraph for analysis.
            weights: Optional weights for criteria. Default: equal weights.
                   Expected keys: 'size', 'intersections', 'encapsulation'

        Returns:
            List of (hyperedge_id, topsis_score) tuples, sorted by score
            descending.
        """
        if not hypernetwork.edges:
            return []

        # Set default weights if not provided
        if weights is None:
            weights = {
                'size': 1.0,
                'intersections': 1.0,
                'encapsulation': 1.0
            }

        # Compute criteria for all hyperedges
        criteria_matrix = []
        hyperedge_ids = list(hypernetwork.edges.keys())

        for he_id in hyperedge_ids:
            size = self._compute_size(hypernetwork, he_id)
            intersections = self._compute_intersections(hypernetwork, he_id)
            encapsulation = self._compute_encapsulation(hypernetwork, he_id)

            criteria_matrix.append([size, intersections, encapsulation])

        criteria_matrix = np.array(criteria_matrix)

        if len(criteria_matrix) == 0:
            return []

        if len(criteria_matrix) == 1:
            return [(hyperedge_ids[0], 1.0)]

        # Normalize criteria matrix
        normalized_matrix = self._normalize_matrix(criteria_matrix)

        # Apply weights
        weight_vector = np.array([
            weights['size'],
            weights['intersections'],
            weights['encapsulation']
        ])
        weighted_matrix = normalized_matrix * weight_vector

        # Compute TOPSIS scores
        scores = self._compute_topsis_scores(weighted_matrix)

        # Create ranked list
        ranked_hyperedges = list(zip(hyperedge_ids, scores))
        ranked_hyperedges.sort(key=lambda x: x[1], reverse=True)

        return ranked_hyperedges

    @staticmethod
    def _compute_size(hypernetwork: Hypernetwork, hyperedge_id: int) -> float:
        """Compute the size (cardinality) of a hyperedge."""
        return float(len(hypernetwork.edges[hyperedge_id]))

    @staticmethod
    def _compute_intersections(
            hypernetwork: Hypernetwork,
            hyperedge_id: int
    ) -> float:
        """
        Compute the number of non-empty intersections with other hyperedges.
        """
        target_hyperedge = set(hypernetwork.edges[hyperedge_id])
        intersections = 0

        for other_id, other_hyperedge in hypernetwork.edges.items():
            if other_id != hyperedge_id:
                if target_hyperedge.intersection(set(other_hyperedge)):
                    intersections += 1

        return float(intersections)

    @staticmethod
    def _compute_encapsulation(
            hypernetwork: Hypernetwork,
            hyperedge_id: int
    ) -> float:
        """
        Compute encapsulation score.

        Encapsulation is defined as the number of hyperedges that contain
        the target hyperedge (i.e., are supersets of it).
        """
        target_hyperedge = set(hypernetwork.edges[hyperedge_id])
        encapsulation_count = 0

        for other_id, other_hyperedge in hypernetwork.edges.items():
            if other_id != hyperedge_id:
                other_set = set(other_hyperedge)
                if target_hyperedge.issubset(other_set):
                    encapsulation_count += 1

        return float(encapsulation_count)

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """Normalize criteria matrix using vector normalization."""
        normalized = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            column = matrix[:, j]
            norm = np.linalg.norm(column)
            if norm > 0:
                normalized[:, j] = column / norm
            else:
                normalized[:, j] = column

        return normalized

    @staticmethod
    def _compute_topsis_scores(weighted_matrix: np.ndarray) -> List[float]:
        """Compute TOPSIS scores for the weighted normalized matrix."""
        # All criteria are beneficial (higher is better)
        ideal_solution = np.max(weighted_matrix, axis=0)
        anti_ideal_solution = np.min(weighted_matrix, axis=0)

        scores = []
        for i in range(weighted_matrix.shape[0]):
            alternative = weighted_matrix[i]

            # Distance to ideal
            d_ideal = np.linalg.norm(alternative - ideal_solution)

            # Distance to anti-ideal
            d_anti_ideal = np.linalg.norm(alternative - anti_ideal_solution)

            # TOPSIS score
            if d_ideal + int(d_anti_ideal) > 0:
                score = d_anti_ideal / (d_ideal + int(d_anti_ideal))
            else:
                score = 0.5  # If both distances are 0, assign neutral score

            scores.append(score)

        return scores
