# -*- coding: utf-8 -*-
"""
entropy_loss.py

This module implements the Entropy Loss (EL) metric for measuring
how perturbations affect the entropy of various distributions in a hypergraph.
"""

from collections import Counter
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from hiper.core import Hypernetwork


class EntropyLoss:
    """
    Computes the Entropy Loss (EL) metric.

    The Entropy Loss measures how much entropy is lost in a distribution
    after a perturbation:

    :math:`\\text{EL} = H(P^{\\text{pre}}) - H(P^{\\text{post}})`

    where:
    - :math:`H(P) = -\\sum p_i \\log(p_i)` is the Shannon entropy
    - :math:`P^{\\text{pre}}` is the distribution before perturbation
    - :math:`P^{\\text{post}}` is the distribution after perturbation

    Positive values indicate entropy loss (reduced diversity),
    negative values indicate entropy gain (increased diversity),
    and zero indicates no change in entropy.

    The metric can be applied to various distributions such as:
    - Node degree distribution
    - Hyperedge size distribution
    - Hyperedge degree distribution
    """

    def __init__(self, distribution_type: str = 'node_degree',
                 log_base: float = 2.0):
        """
        Initialize the Entropy Loss metric.

        Args:
            distribution_type: Type of distribution to analyze
            log_base: Base for logarithm in entropy calculation
        """
        self.name = "Entropy Loss"
        self.symbol = "EL"
        self.distribution_type = distribution_type
        self.log_base = log_base

        valid_types = ['node_degree', 'hyperedge_size', 'hyperedge_degree']
        if distribution_type not in valid_types:
            raise ValueError(f"distribution_type must be one of {valid_types}")

    def compute(self,
                before_hypergraph: 'Hypernetwork',
                after_hypergraph: 'Hypernetwork') -> float:
        """
        Compute the Entropy Loss metric.

        Args:
            before_hypergraph: The hypergraph before perturbation.
            after_hypergraph: The hypergraph after perturbation.

        Returns:
            The entropy loss value (positive = loss, negative = gain).
        """
        # Extract distributions
        dist_before = self._extract_distribution(before_hypergraph)
        dist_after = self._extract_distribution(after_hypergraph)

        # Compute entropies
        entropy_before = self._compute_shannon_entropy(dist_before)
        entropy_after = self._compute_shannon_entropy(dist_after)

        # Return entropy loss (before - after)
        return entropy_before - entropy_after

    def _extract_distribution(self, hypergraph: 'Hypernetwork') -> List[int]:
        """Extract the specified distribution from the hypergraph."""
        if self.distribution_type == 'node_degree':
            return self._get_node_degree_distribution(hypergraph)
        elif self.distribution_type == 'hyperedge_size':
            return self._get_hyperedge_size_distribution(hypergraph)
        elif self.distribution_type == 'hyperedge_degree':
            return self._get_hyperedge_degree_distribution(hypergraph)
        else:
            raise ValueError(
                f"Unknown distribution type: {self.distribution_type}")

    @staticmethod
    def _get_node_degree_distribution(hypergraph: 'Hypernetwork') -> List[int]:
        """Get the degree of each node (number of hyperedges it belongs to)."""
        degrees = []
        for node in hypergraph.nodes.keys():
            degree = len(hypergraph.get_hyperedges(node))
            degrees.append(degree)
        return degrees

    @staticmethod
    def _get_hyperedge_size_distribution(hypergraph: 'Hypernetwork') -> List[
        int]:
        """Get the size of each hyperedge (number of nodes it contains)."""
        sizes = []
        for he_id in hypergraph.edges.keys():
            size = len(hypergraph.get_nodes(he_id))
            sizes.append(size)
        return sizes

    @staticmethod
    def _get_hyperedge_degree_distribution(hypergraph: 'Hypernetwork') -> List[
        int]:
        """
        Get the degree of each hyperedge (number of other hyperedges it shares
        nodes with).
        """
        degrees = []
        hyperedge_ids = list(hypergraph.edges.keys())

        for he_id in hyperedge_ids:
            he_nodes = set(hypergraph.get_nodes(he_id))
            degree = 0

            # Count hyperedges that share at least one node
            for other_he_id in hyperedge_ids:
                if other_he_id != he_id:
                    other_nodes = set(hypergraph.get_nodes(other_he_id))
                    if he_nodes.intersection(other_nodes):
                        degree += 1

            degrees.append(degree)
        return degrees

    def _compute_shannon_entropy(self, distribution: List[int]) -> float:
        """
        Compute Shannon entropy of a distribution.

        Args:
            distribution: List of values representing the distribution.

        Returns:
            Shannon entropy value.
        """
        if not distribution:
            return 0.0

        # Count frequencies
        counts = Counter(distribution)
        total = len(distribution)

        # Compute probabilities and entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log(probability) / np.log(
                    self.log_base)

        return entropy

    def compute_detailed(self,
                         before_hypergraph: 'Hypernetwork',
                         after_hypergraph: 'Hypernetwork') -> dict:
        """
        Compute detailed entropy analysis.

        Args:
            before_hypergraph: The hypergraph before perturbation.
            after_hypergraph: The hypergraph after perturbation.

        Returns:
            Dictionary containing detailed entropy analysis.
        """
        # Extract distributions
        dist_before = self._extract_distribution(before_hypergraph)
        dist_after = self._extract_distribution(after_hypergraph)

        # Compute statistics
        entropy_before = self._compute_shannon_entropy(dist_before)
        entropy_after = self._compute_shannon_entropy(dist_after)
        entropy_loss = entropy_before - entropy_after

        # Distribution statistics
        counts_before = Counter(dist_before)
        counts_after = Counter(dist_after)

        return {
            'entropy_loss': entropy_loss,
            'entropy_before': entropy_before,
            'entropy_after': entropy_after,
            'distribution_type': self.distribution_type,
            'log_base': self.log_base,
            'values_before': dist_before,
            'values_after': dist_after,
            'unique_values_before': len(counts_before),
            'unique_values_after': len(counts_after),
            'total_values_before': len(dist_before),
            'total_values_after': len(dist_after),
            'frequency_counts_before': dict(counts_before),
            'frequency_counts_after': dict(counts_after),
            'mean_before': np.mean(dist_before) if dist_before else 0.0,
            'mean_after': np.mean(dist_after) if dist_after else 0.0,
            'std_before': np.std(dist_before) if dist_before else 0.0,
            'std_after': np.std(dist_after) if dist_after else 0.0
        }

    def compute_multiple_distributions(self,
                                       before_hypergraph: 'Hypernetwork',
                                       after_hypergraph: 'Hypernetwork') \
            -> dict:
        """
        Compute entropy loss for multiple distribution types.

        Args:
            before_hypergraph: The hypergraph before perturbation.
            after_hypergraph: The hypergraph after perturbation.

        Returns:
            Dictionary containing entropy loss for each distribution type.
        """
        distribution_types = ['node_degree', 'hyperedge_size',
                              'hyperedge_degree']
        results = {}

        original_type = self.distribution_type

        for dist_type in distribution_types:
            self.distribution_type = dist_type
            try:
                entropy_loss = self.compute(before_hypergraph, after_hypergraph)
                results[dist_type] = entropy_loss
            except (ValueError, ZeroDivisionError):
                results[dist_type] = np.nan

        # Restore original distribution type
        self.distribution_type = original_type

        return results

    def compute_relative_entropy_loss(self,
                                      before_hypergraph: 'Hypernetwork',
                                      after_hypergraph: 'Hypernetwork') \
            -> float:
        """
        Compute relative entropy loss (normalized by initial entropy).

        Args:
            before_hypergraph: The hypergraph before perturbation.
            after_hypergraph: The hypergraph after perturbation.

        Returns:
            Relative entropy loss in [0, 1] or negative for entropy gain.
        """
        dist_before = self._extract_distribution(before_hypergraph)
        dist_after = self._extract_distribution(after_hypergraph)

        entropy_before = self._compute_shannon_entropy(dist_before)
        entropy_after = self._compute_shannon_entropy(dist_after)

        if entropy_before == 0:
            return 0.0 if entropy_after == 0 else -float('inf')

        return (entropy_before - entropy_after) / entropy_before

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} ({self.symbol}) - {self.distribution_type}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"EntropyLoss(name='{self.name}', symbol='{self.symbol}', "
                f"distribution_type='{self.distribution_type}', "
                f"log_base={self.log_base})")
