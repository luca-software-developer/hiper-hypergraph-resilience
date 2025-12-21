# -*- coding: utf-8 -*-
"""
swalk.py

Implements s-walk efficiency metrics for hypergraphs.
"""

from typing import Dict

from hiper.core.hypernetwork import Hypernetwork
from .distance import HypergraphDistance


class SwalkEfficiency:
    """
    Computes average s-walk efficiency ℰs(H).

    The s-walk efficiency measures how efficiently nodes can reach each
    other via s-walks, computed as the average of inverse distances.

    Formula: :math:`\mathcal{E}_s(H) = \\frac{1}{|V|(|V|-1)} \\sum \\frac{1}{d_H^s(u,v)}`
    for all pairs :math:`u,v \\in V` with :math:`u \\neq v`.
    """

    def __init__(self, s: int = 1):
        """
        Initialize with s-walk parameter.

        Args:
            s: Parameter for s-walk computation.
        """
        self.s = s
        self.distance_calculator = HypergraphDistance(s)

    def compute(self, hypernetwork: Hypernetwork) -> float:
        """
        Compute average s-walk efficiency.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            S-walk efficiency value ℰs(H) in range [0, 1].
        """
        nodes = list(hypernetwork.nodes)
        n = len(nodes)

        if n <= 1:
            return 0.0

        total_efficiency = 0.0
        pair_count = 0

        # Compute efficiency for all distinct node pairs
        for i, u in enumerate(nodes):
            for v in nodes:
                if u != v:
                    distance = self.distance_calculator.compute_distance(
                        hypernetwork, u, v)

                    if distance != float('inf') and distance > 0:
                        efficiency = 1.0 / distance
                    else:
                        efficiency = 0.0

                    total_efficiency += efficiency
                    pair_count += 1

        # Apply formula
        if pair_count > 0:
            return total_efficiency / pair_count
        else:
            return 0.0

    def compute_detailed(self, hypernetwork: Hypernetwork) -> Dict[str, float]:
        """
        Compute detailed s-walk efficiency statistics.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            Dictionary with efficiency statistics including average, min, max.
        """
        nodes = list(hypernetwork.nodes)
        n = len(nodes)

        if n <= 1:
            return {
                'average_efficiency': 0.0,
                'min_efficiency': 0.0,
                'max_efficiency': 0.0,
                'connected_pairs': 0,
                'total_pairs': 0,
                'connectivity_ratio': 0.0
            }

        efficiencies = []
        connected_pairs = 0
        total_pairs = 0

        for i, u in enumerate(nodes):
            for v in nodes:
                if u != v:
                    distance = self.distance_calculator.compute_distance(
                        hypernetwork, u, v)

                    if distance != float('inf') and distance > 0:
                        efficiency = 1.0 / distance
                        connected_pairs += 1
                    else:
                        efficiency = 0.0

                    efficiencies.append(efficiency)
                    total_pairs += 1

        average_efficiency = sum(efficiencies) / len(efficiencies)
        min_efficiency = min(efficiencies)
        max_efficiency = max(efficiencies)
        connectivity_ratio = connected_pairs / total_pairs \
            if total_pairs > 0 else 0.0

        return {
            'average_efficiency': average_efficiency,
            'min_efficiency': min_efficiency,
            'max_efficiency': max_efficiency,
            'connected_pairs': connected_pairs,
            'total_pairs': total_pairs,
            'connectivity_ratio': connectivity_ratio
        }

    def compute_node_efficiencies(self, hypernetwork: Hypernetwork) -> Dict[
        int, float]:
        """
        Compute individual node efficiency scores.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            Dictionary mapping node IDs to their efficiency scores.
        """
        nodes = list(hypernetwork.nodes)
        node_efficiencies = {}

        for source in nodes:
            total_efficiency = 0.0
            reachable_count = 0

            for target in nodes:
                if source != target:
                    distance = self.distance_calculator.compute_distance(
                        hypernetwork, source, target)

                    if distance != float('inf') and distance > 0:
                        total_efficiency += 1.0 / distance
                        reachable_count += 1

            # Average efficiency for this node
            if reachable_count > 0:
                node_efficiencies[source] = total_efficiency / reachable_count
            else:
                node_efficiencies[source] = 0.0

        return node_efficiencies

    def find_critical_nodes(self, hypernetwork: Hypernetwork,
                            top_k: int = 5) -> list:
        """
        Find nodes whose removal would most impact s-walk efficiency.

        Args:
            hypernetwork: Target hypergraph.
            top_k: Number of critical nodes to return.

        Returns:
            List of (node_id, efficiency_impact) tuples.
        """
        original_efficiency = self.compute(hypernetwork)
        impacts = []

        for node_id in hypernetwork.nodes:
            # Create copy without this node
            test_hn = Hypernetwork()

            for edge_id in hypernetwork.edges:
                original_nodes = hypernetwork.get_nodes(edge_id)
                remaining_nodes = [n for n in original_nodes if n != node_id]

                if len(remaining_nodes) >= 1:
                    test_hn.add_hyperedge(edge_id, remaining_nodes)

            # Compute efficiency without this node
            new_efficiency = self.compute(test_hn)
            impact = original_efficiency - new_efficiency
            impacts.append((node_id, impact))

        # Sort by impact (descending)
        impacts.sort(key=lambda x: x[1], reverse=True)

        return impacts[:top_k]
