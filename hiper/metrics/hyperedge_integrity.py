# -*- coding: utf-8 -*-
"""
hyperedge_integrity.py

This module implements the Hyperedge Integrity (HI) metric for measuring
the resilience of hypergraphs by calculating the ratio of hyperedges
preserved after perturbation.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hiper.core import Hypernetwork


class HyperedgeIntegrity:
    """
    Computes the Hyperedge Integrity (HI) metric.

    The Hyperedge Integrity measures the fraction of hyperedges that remain
    in the hypergraph after a perturbation:

    ``HI = |E'| / |E|``

    where E is the original set of hyperedges and E' is the set of hyperedges
    after perturbation.

    A value of 1.0 indicates perfect integrity (no hyperedges lost),
    while 0.0 indicates complete loss of all hyperedges.
    """

    def __init__(self):
        """Initialize the Hyperedge Integrity metric."""
        self.name = "Hyperedge Integrity"
        self.symbol = "HI"

    @staticmethod
    def compute(perturbed_hypergraph: 'Hypernetwork',
                original_hypergraph: Optional['Hypernetwork'] = None) -> float:
        """
        Compute the Hyperedge Integrity metric.

        Args:
            perturbed_hypergraph: The hypergraph after perturbation.
            original_hypergraph: The original hypergraph before perturbation.
                                If None, assumes the metric is computed on the
                                current state only.

        Returns:
            The hyperedge integrity value in [0, 1].

        Raises:
            ValueError: If original_hypergraph is None when required.
            ZeroDivisionError: If the original hypergraph has no hyperedges.
        """
        if original_hypergraph is None:
            # If no original provided, treat current as reference (HI = 1.0)
            return 1.0

        original_size = original_hypergraph.size()
        if original_size == 0:
            # Edge case: if original had no hyperedges, define HI = 1.0
            return 1.0

        perturbed_size = perturbed_hypergraph.size()

        # Compute the ratio
        integrity = perturbed_size / original_size

        return min(1.0, max(0.0, integrity))  # Clamp to [0, 1]

    def compute_change(self,
                       before_hypergraph: 'Hypernetwork',
                       after_hypergraph: 'Hypernetwork') -> float:
        """
        Compute the change in hyperedge integrity.

        Args:
            before_hypergraph: Hypergraph state before perturbation.
            after_hypergraph: Hypergraph state after perturbation.

        Returns:
            The change in integrity (negative values indicate loss).
        """
        before_integrity = self.compute(before_hypergraph, before_hypergraph)
        after_integrity = self.compute(after_hypergraph, before_hypergraph)

        return after_integrity - before_integrity

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} ({self.symbol})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HyperedgeIntegrity(name='{self.name}', symbol='{self.symbol}')"
