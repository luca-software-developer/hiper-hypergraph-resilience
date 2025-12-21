# -*- coding: utf-8 -*-
"""
redundancy.py

Implements the redundancy coefficient ρ(H) for hypergraphs.
"""

import math
from typing import List

from hiper.core.hypernetwork import Hypernetwork


class RedundancyCoefficient:
    """
    Computes the redundancy coefficient ρ(H).

    The redundancy coefficient measures the average degree of normalized
    overlap between pairs of hyperedges in the hypergraph.

    Formula: :math:`\\rho(H) = \\frac{2}{|E|(|E|-1)} \\sum_{\\substack{e_1, e_2 \\in E \\\\ e_1 \\neq e_2 \\\\ e_1 < e_2}} \\frac{|e_1 \\cap e_2|}{\\sqrt{|e_1| \\cdot |e_2|}}`

    The sum is over all unordered pairs of distinct hyperedges.
    """

    @staticmethod
    def compute(hypernetwork: Hypernetwork) -> float:
        """
        Compute the redundancy coefficient.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            Redundancy coefficient value ρ(H) in range [0, 1].
        """
        edges = list(hypernetwork.edges)
        m = len(edges)

        if m <= 1:
            return 0.0

        total_overlap = 0.0
        pair_count = 0

        # Compute normalized overlap for all pairs of distinct hyperedges
        for i, e1 in enumerate(edges):
            nodes_e1 = set(hypernetwork.get_nodes(e1))
            size_e1 = len(nodes_e1)

            for e2 in edges[i + 1:]:
                nodes_e2 = set(hypernetwork.get_nodes(e2))
                size_e2 = len(nodes_e2)

                # Compute intersection size
                intersection_size = len(nodes_e1 & nodes_e2)

                # Compute normalized overlap
                if size_e1 > 0 and size_e2 > 0:
                    normalized_overlap = (intersection_size /
                                          math.sqrt(size_e1 * size_e2))
                    total_overlap += normalized_overlap
                    pair_count += 1

        # Apply formula: ρ(H) = 2/(|E|(|E|-1)) * Σ(normalized_overlaps)
        # We sum over unordered pairs (i < j), hence the factor of 2
        if pair_count > 0:
            return (2 * total_overlap) / (m * (m - 1))
        else:
            return 0.0

    @staticmethod
    def compute_pairwise_overlaps(hypernetwork: Hypernetwork) -> List[
        tuple]:
        """
        Compute detailed pairwise overlap information.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            List of tuples (edge1_id, edge2_id, intersection_size,
            normalized_overlap).
        """
        edges = list(hypernetwork.edges)
        overlaps = []

        for i, e1 in enumerate(edges):
            nodes_e1 = set(hypernetwork.get_nodes(e1))
            size_e1 = len(nodes_e1)

            for e2 in edges[i + 1:]:
                nodes_e2 = set(hypernetwork.get_nodes(e2))
                size_e2 = len(nodes_e2)

                intersection_size = len(nodes_e1 & nodes_e2)

                if size_e1 > 0 and size_e2 > 0:
                    normalized_overlap = (intersection_size /
                                          math.sqrt(size_e1 * size_e2))
                else:
                    normalized_overlap = 0.0

                overlaps.append((e1, e2, intersection_size, normalized_overlap))

        return overlaps

    def get_most_overlapping_pairs(self, hypernetwork: Hypernetwork,
                                   top_k: int = 5) -> List[tuple]:
        """
        Get the k most overlapping hyperedge pairs.

        Args:
            hypernetwork: Target hypergraph.
            top_k: Number of top pairs to return.

        Returns:
            List of top k overlapping pairs sorted by normalized overlap.
        """
        overlaps = self.compute_pairwise_overlaps(hypernetwork)

        # Sort by normalized overlap (descending)
        overlaps.sort(key=lambda x: x[3], reverse=True)

        return overlaps[:top_k]
