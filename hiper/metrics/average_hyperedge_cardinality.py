# -*- coding: utf-8 -*-
"""
average_hyperedge_cardinality.py

This module implements the Average Hyperedge Cardinality (AHC) metric for
measuring the average size of hyperedges in a hypergraph after perturbation.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hiper.core import Hypernetwork


class AverageHyperedgeCardinality:
    """
    Computes the Average Hyperedge Cardinality (AHC) metric.

    The Average Hyperedge Cardinality measures the mean size of hyperedges
    in the hypergraph after perturbation:

    AHC = (1/|E'|) * Σ|e| for e ∈ E'

    where E' is the set of hyperedges after perturbation and |e| is the
    cardinality (size) of hyperedge e.

    This metric provides insight into how perturbations affect the
    size distribution of hyperedges, which can impact the hypergraph's
    functional properties.
    """

    def __init__(self):
        """Initialize the Average Hyperedge Cardinality metric."""
        self.name = "Average Hyperedge Cardinality"
        self.symbol = "AHC"

    @staticmethod
    def compute(hypergraph: 'Hypernetwork') -> float:
        """
        Compute the Average Hyperedge Cardinality metric.

        Args:
            hypergraph: The hypergraph to analyze.

        Returns:
            The average hyperedge cardinality.

        Raises:
            ValueError: If the hypergraph has no hyperedges.
        """
        if hypergraph.size() == 0:
            # No hyperedges present
            return 0.0

        total_cardinality = 0
        hyperedge_count = 0

        # Sum the cardinalities of all hyperedges
        for hyperedge_id in hypergraph.edges:
            hyperedge_nodes = hypergraph.get_nodes(hyperedge_id)
            total_cardinality += len(hyperedge_nodes)
            hyperedge_count += 1

        if hyperedge_count == 0:
            return 0.0

        return total_cardinality / hyperedge_count

    def compute_change(self,
                       before_hypergraph: 'Hypernetwork',
                       after_hypergraph: 'Hypernetwork') -> float:
        """
        Compute the change in average hyperedge cardinality.

        Args:
            before_hypergraph: Hypergraph state before perturbation.
            after_hypergraph: Hypergraph state after perturbation.

        Returns:
            The change in average cardinality (can be positive or negative).
        """
        before_ahc = self.compute(before_hypergraph)
        after_ahc = self.compute(after_hypergraph)

        return after_ahc - before_ahc

    @staticmethod
    def get_cardinality_distribution(hypergraph: 'Hypernetwork') -> dict:
        """
        Get the distribution of hyperedge cardinalities.

        Args:
            hypergraph: The hypergraph to analyze.

        Returns:
            Dictionary mapping cardinality values to their frequencies.
        """
        cardinality_counts = {}

        for hyperedge_id in hypergraph.edges:
            hyperedge_nodes = hypergraph.get_nodes(hyperedge_id)
            cardinality = len(hyperedge_nodes)

            cardinality_counts[cardinality] = cardinality_counts.get(
                cardinality, 0) + 1

        return cardinality_counts

    @staticmethod
    def get_statistics(hypergraph: 'Hypernetwork') -> dict:
        """
        Get comprehensive cardinality statistics.

        Args:
            hypergraph: The hypergraph to analyze.

        Returns:
            Dictionary containing mean, median, std, min, max cardinalities.
        """
        if hypergraph.size() == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0,
                'total_hyperedges': 0
            }

        cardinalities = []
        for hyperedge_id in hypergraph.edges:
            hyperedge_nodes = hypergraph.get_nodes(hyperedge_id)
            cardinalities.append(len(hyperedge_nodes))

        cardinalities = np.array(cardinalities)

        return {
            'mean': float(np.mean(cardinalities)),
            'median': float(np.median(cardinalities)),
            'std': float(np.std(cardinalities)),
            'min': int(np.min(cardinalities)),
            'max': int(np.max(cardinalities)),
            'total_hyperedges': len(cardinalities)
        }

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} ({self.symbol})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"AverageHyperedgeCardinality(name='{self.name}', "
                f"symbol='{self.symbol}')")
