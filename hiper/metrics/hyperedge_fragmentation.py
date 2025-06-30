# -*- coding: utf-8 -*-
"""
hyperedge_fragmentation.py

This module implements the Hyperedge Fragmentation Index (HFI) metric for
measuring how much hyperedges have been fragmented by perturbations.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hiper.core import Hypernetwork


class HyperedgeFragmentationIndex:
    """
    Computes the Hyperedge Fragmentation Index (HFI) metric.

    The Hyperedge Fragmentation Index measures how much the original hyperedges
    have been fragmented by a perturbation:

    HFI = 1 - (1/|E|) * Σ(|e ∩ V'|/|e|) for e ∈ E

    where:
    - E is the original set of hyperedges
    - V' is the set of nodes after perturbation
    - e ∩ V' is the intersection of original hyperedge e with remaining nodes
    - |e| is the original size of hyperedge e

    A value of 0.0 indicates no fragmentation (all hyperedges intact),
    while 1.0 indicates complete fragmentation (all hyperedges destroyed).
    """

    def __init__(self):
        """Initialize the Hyperedge Fragmentation Index metric."""
        self.name = "Hyperedge Fragmentation Index"
        self.symbol = "HFI"

    @staticmethod
    def compute(perturbed_hypergraph: 'Hypernetwork',
                original_hypergraph: 'Hypernetwork') -> float:
        """
        Compute the Hyperedge Fragmentation Index metric.

        Args:
            perturbed_hypergraph: The hypergraph after perturbation.
            original_hypergraph: The original hypergraph before perturbation.

        Returns:
            The hyperedge fragmentation index value in [0, 1].

        Raises:
            ValueError: If original_hypergraph has no hyperedges.
        """
        if original_hypergraph.size() == 0:
            # No original hyperedges to fragment
            return 0.0

        # Get the set of remaining nodes after perturbation
        remaining_nodes = set(perturbed_hypergraph.nodes.keys())

        total_preservation_ratio = 0.0
        hyperedge_count = 0

        # For each original hyperedge, calculate preservation ratio
        for hyperedge_id in original_hypergraph.edges.keys():
            original_nodes = set(
                original_hypergraph.get_nodes(hyperedge_id))

            if len(original_nodes) == 0:
                # Skip empty hyperedges
                continue

            # Find intersection with remaining nodes
            preserved_nodes = original_nodes.intersection(remaining_nodes)

            # Calculate preservation ratio for this hyperedge
            preservation_ratio = len(preserved_nodes) / len(original_nodes)
            total_preservation_ratio += preservation_ratio
            hyperedge_count += 1

        if hyperedge_count == 0:
            return 0.0

        # Average preservation ratio
        avg_preservation = total_preservation_ratio / hyperedge_count

        # Fragmentation index is 1 minus preservation
        fragmentation_index = 1.0 - avg_preservation

        return max(0.0, min(1.0, fragmentation_index))  # Clamp to [0, 1]

    @staticmethod
    def compute_detailed(perturbed_hypergraph: 'Hypernetwork',
                         original_hypergraph: 'Hypernetwork') -> dict:
        """
        Compute detailed fragmentation statistics.

        Args:
            perturbed_hypergraph: The hypergraph after perturbation.
            original_hypergraph: The original hypergraph before perturbation.

        Returns:
            Dictionary containing detailed fragmentation analysis.
        """
        if original_hypergraph.size() == 0:
            return {
                'fragmentation_index': 0.0,
                'total_hyperedges': 0,
                'intact_hyperedges': 0,
                'partially_fragmented': 0,
                'completely_destroyed': 0,
                'preservation_ratios': []
            }

        remaining_nodes = set(perturbed_hypergraph.nodes.keys())
        preservation_ratios = []
        intact_count = 0
        partial_count = 0
        destroyed_count = 0

        for hyperedge_id in original_hypergraph.edges.keys():
            original_nodes = set(
                original_hypergraph.get_nodes(hyperedge_id))

            if len(original_nodes) == 0:
                continue

            preserved_nodes = original_nodes.intersection(remaining_nodes)
            preservation_ratio = len(preserved_nodes) / len(original_nodes)
            preservation_ratios.append(preservation_ratio)

            if preservation_ratio == 1.0:
                intact_count += 1
            elif preservation_ratio == 0.0:
                destroyed_count += 1
            else:
                partial_count += 1

        total_hyperedges = len(preservation_ratios)
        avg_preservation = (np.mean(preservation_ratios)
                            if preservation_ratios else 0.0)
        fragmentation_index = 1.0 - avg_preservation

        return {
            'fragmentation_index': fragmentation_index,
            'total_hyperedges': total_hyperedges,
            'intact_hyperedges': intact_count,
            'partially_fragmented': partial_count,
            'completely_destroyed': destroyed_count,
            'preservation_ratios': preservation_ratios,
            'avg_preservation_ratio': avg_preservation
        }

    @staticmethod
    def get_hyperedge_fragmentation_map(perturbed_hypergraph: 'Hypernetwork',
                                        original_hypergraph: 'Hypernetwork'
                                        ) -> dict:
        """
        Get fragmentation status for each individual hyperedge.

        Args:
            perturbed_hypergraph: The hypergraph after perturbation.
            original_hypergraph: The original hypergraph before perturbation.

        Returns:
            Dictionary mapping hyperedge IDs to their fragmentation status.
        """
        remaining_nodes = set(perturbed_hypergraph.nodes.keys())
        fragmentation_map = {}

        for hyperedge_id in original_hypergraph.edges.keys():
            original_nodes = set(
                original_hypergraph.get_nodes(hyperedge_id))

            if len(original_nodes) == 0:
                fragmentation_map[hyperedge_id] = {
                    'status': 'empty',
                    'preservation_ratio': 0.0,
                    'original_size': 0,
                    'preserved_size': 0
                }
                continue

            preserved_nodes = original_nodes.intersection(remaining_nodes)
            preservation_ratio = len(preserved_nodes) / len(original_nodes)

            if preservation_ratio == 1.0:
                status = 'intact'
            elif preservation_ratio == 0.0:
                status = 'destroyed'
            else:
                status = 'fragmented'

            fragmentation_map[hyperedge_id] = {
                'status': status,
                'preservation_ratio': preservation_ratio,
                'original_size': len(original_nodes),
                'preserved_size': len(preserved_nodes),
                'original_nodes': original_nodes,
                'preserved_nodes': preserved_nodes
            }

        return fragmentation_map

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} ({self.symbol})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"HyperedgeFragmentationIndex(name='{self.name}', "
                f"symbol='{self.symbol}')")
