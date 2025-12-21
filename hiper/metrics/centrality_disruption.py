# -*- coding: utf-8 -*-
"""
centrality_disruption.py

This module implements the Centrality Disruption Index (CDI) metric for
measuring how perturbations affect the centrality distribution of nodes
in a hypergraph.
"""

from typing import TYPE_CHECKING, Optional, Dict, Callable

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from hiper.core import Hypernetwork


class CentralityDisruptionIndex:
    """
    Computes the Centrality Disruption Index (CDI) metric.

    The Centrality Disruption Index measures how much a perturbation affects
    the centrality distribution of nodes using the Kolmogorov-Smirnov test:

    :math:`\\text{CDI} = \\text{KS}(\\{C^{\\text{pre}}_v\\}, \\{C^{\\text{post}}_v\\})`

    where:
    - :math:`C^{\\text{pre}}_v` are centrality values before perturbation
    - :math:`C^{\\text{post}}_v` are centrality values after perturbation
    - KS is the Kolmogorov-Smirnov distance between distributions

    The metric returns a value in [0, 1] where:
    - 0.0 indicates no disruption (identical distributions)
    - 1.0 indicates maximum disruption (completely different distributions)
    """

    def __init__(self, centrality_type: str = 'degree'):
        """
        Initialize the Centrality Disruption Index metric.

        Args:
            centrality_type: Type of centrality to use ('degree', 'closeness',
                           'betweenness', or custom function).
        """
        self.name = "Centrality Disruption Index"
        self.symbol = "CDI"
        self.centrality_type = centrality_type

    def compute(self,
                before_hypergraph: 'Hypernetwork',
                after_hypergraph: 'Hypernetwork',
                centrality_function: Optional[Callable] = None) -> float:
        """
        Compute the Centrality Disruption Index metric.

        Args:
            before_hypergraph: The hypergraph before perturbation.
            after_hypergraph: The hypergraph after perturbation.
            centrality_function: Custom centrality function. If None, uses
                                built-in centrality based on centrality_type.

        Returns:
            The centrality disruption index value in [0, 1].
        """
        # Compute centralities before and after
        centralities_before = self._compute_centralities(
            before_hypergraph, centrality_function)
        centralities_after = self._compute_centralities(
            after_hypergraph, centrality_function)

        # Get common nodes for fair comparison
        common_nodes = set(centralities_before.keys()).intersection(
            set(centralities_after.keys()))

        if len(common_nodes) < 2:
            # Not enough nodes for meaningful comparison
            return 1.0 if len(centralities_before) != len(
                centralities_after) else 0.0

        # Extract centrality values for common nodes
        values_before = [centralities_before[node] for node in common_nodes]
        values_after = [centralities_after[node] for node in common_nodes]

        # Handle edge cases
        if len(values_before) == 0 or len(values_after) == 0:
            return 1.0

        # Compute Kolmogorov-Smirnov statistic
        try:
            ks_statistic, _ = stats.ks_2samp(values_before, values_after)
            return min(1.0, max(0.0, ks_statistic))
        except (ValueError, RuntimeError):
            # Fallback to normalized absolute difference of means
            mean_before = np.mean(values_before)
            mean_after = np.mean(values_after)
            max_val = max(max(values_before), max(values_after))
            if max_val == 0:
                return 0.0
            return min(1.0, abs(mean_before - mean_after) / max_val)

    def _compute_centralities(self,
                              hypergraph: 'Hypernetwork',
                              centrality_function: Optional[Callable] = None
                              ) -> Dict[int, float]:
        """
        Compute centrality values for all nodes in the hypergraph.

        Args:
            hypergraph: The hypergraph to analyze.
            centrality_function: Custom centrality function.

        Returns:
            Dictionary mapping node IDs to centrality values.
        """
        if centrality_function is not None:
            return centrality_function(hypergraph)

        if self.centrality_type == 'degree':
            return self._compute_degree_centrality(hypergraph)
        elif self.centrality_type == 'closeness':
            return self._compute_closeness_centrality(hypergraph)
        elif self.centrality_type == 'betweenness':
            return self._compute_betweenness_centrality(hypergraph)
        else:
            # Default to degree centrality
            return self._compute_degree_centrality(hypergraph)

    @staticmethod
    def _compute_degree_centrality(hypergraph: 'Hypernetwork') -> Dict[
        int, float]:
        """Compute degree centrality for all nodes."""
        centralities = {}

        for node in hypergraph.nodes.keys():
            # Count how many hyperedges contain this node
            degree = len(hypergraph.get_hyperedges(node))
            centralities[node] = float(degree)

        return centralities

    def _compute_closeness_centrality(self, hypergraph: 'Hypernetwork'
                                      ) -> Dict[int, float]:
        """Compute closeness centrality based on hypergraph distances."""
        centralities = {}
        nodes = list(hypergraph.nodes.keys())

        if len(nodes) <= 1:
            return {node: 1.0 for node in nodes}

        # Build adjacency based on hyperedge co-membership
        adjacency = self._build_node_adjacency(hypergraph)

        for node in nodes:
            distances = self._compute_shortest_distances(adjacency, node)
            if distances:
                total_distance = sum(distances.values())
                if total_distance > 0:
                    centralities[node] = (len(distances) - 1) / total_distance
                else:
                    centralities[node] = 1.0
            else:
                centralities[node] = 0.0

        return centralities

    @staticmethod
    def _compute_betweenness_centrality(hypergraph: 'Hypernetwork'
                                        ) -> Dict[int, float]:
        """
        Compute betweenness centrality.
        """
        centralities = {}
        nodes = list(hypergraph.nodes.keys())

        for node in nodes:
            centralities[node] = 0.0

        # Betweenness based on hyperedge participation
        for node in nodes:
            node_hyperedges = hypergraph.get_hyperedges(node)

            # Count how many node pairs this node connects
            connected_pairs = 0
            for he_id in node_hyperedges:
                he_nodes = hypergraph.get_nodes(he_id)
                other_nodes = [n for n in he_nodes if n != node]
                pairs_count = len(other_nodes) * (len(other_nodes) - 1) // 2
                connected_pairs += pairs_count

            centralities[node] = float(connected_pairs)

        return centralities

    @staticmethod
    def _build_node_adjacency(hypergraph: 'Hypernetwork') -> Dict[int, set]:
        """Build node adjacency graph based on hyperedge co-membership."""
        adjacency = {node: set() for node in hypergraph.nodes.keys()}

        for he_id in hypergraph.edges.keys():
            he_nodes = hypergraph.get_nodes(he_id)

            # Connect all pairs in the hyperedge
            for i, node1 in enumerate(he_nodes):
                for node2 in he_nodes[i + 1:]:
                    adjacency[node1].add(node2)
                    adjacency[node2].add(node1)

        return adjacency

    @staticmethod
    def _compute_shortest_distances(adjacency: Dict[int, set],
                                    source: int) -> Dict[int, int]:
        """Compute the shortest distances from source using BFS."""
        distances = {source: 0}
        queue = [source]
        visited = {source}

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]

            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def compute_detailed(self,
                         before_hypergraph: 'Hypernetwork',
                         after_hypergraph: 'Hypernetwork',
                         centrality_function: Optional[Callable] = None
                         ) -> dict:
        """
        Compute detailed centrality disruption analysis.

        Args:
            before_hypergraph: The hypergraph before perturbation.
            after_hypergraph: The hypergraph after perturbation.
            centrality_function: Custom centrality function.

        Returns:
            Dictionary containing detailed analysis results.
        """
        centralities_before = self._compute_centralities(
            before_hypergraph, centrality_function)
        centralities_after = self._compute_centralities(
            after_hypergraph, centrality_function)

        common_nodes = set(centralities_before.keys()).intersection(
            set(centralities_after.keys()))

        values_before = [centralities_before[node] for node in common_nodes]
        values_after = [centralities_after[node] for node in common_nodes]

        ks_statistic = 0.0
        p_value = 1.0

        if len(values_before) >= 2 and len(values_after) >= 2:
            try:
                ks_statistic, p_value = stats.ks_2samp(values_before,
                                                       values_after)
            except (ValueError, RuntimeError):
                pass

        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'nodes_before': len(centralities_before),
            'nodes_after': len(centralities_after),
            'common_nodes': len(common_nodes),
            'centralities_before': centralities_before,
            'centralities_after': centralities_after,
            'mean_before': np.mean(values_before) if values_before else 0.0,
            'mean_after': np.mean(values_after) if values_after else 0.0,
            'std_before': np.std(values_before) if values_before else 0.0,
            'std_after': np.std(values_after) if values_after else 0.0
        }

    def __str__(self) -> str:
        """String representation of the metric."""
        return (f"{self.name} ({self.symbol}) - "
                f"{self.centrality_type} centrality")

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CentralityDisruptionIndex(name='{self.name}', "
                f"symbol='{self.symbol}', "
                f"centrality_type='{self.centrality_type}')")
