# -*- coding: utf-8 -*-
"""
connectivity.py

Implements hypergraph and hyperedge connectivity metrics.
"""

from typing import Set

from hiper.core.hypernetwork import Hypernetwork
from .distance import HypergraphDistance


class HypergraphConnectivity:
    """
    Computes hypergraph connectivity κ(H).

    The connectivity κ(H) is the minimum number of nodes that must be
    removed to disconnect the hypergraph or reduce it to a single isolated node.
    """

    def __init__(self, s: int = 1):
        """
        Initialize with s-walk parameter.

        Args:
            s: Parameter for s-walk connectivity computation.
        """
        self.s = s
        self.distance_calculator = HypergraphDistance(s)

    def compute(self, hypernetwork: Hypernetwork) -> int:
        """
        Compute hypergraph connectivity.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            Connectivity value κ(H).
        """
        if hypernetwork.order() <= 1:
            return 0

        # Try removing increasing numbers of nodes
        nodes = list(hypernetwork.nodes)
        n = len(nodes)

        for k in range(n):
            if self._can_disconnect_with_k_nodes(hypernetwork, k):
                return k

        return n - 1  # Worst case: remove all but one node

    def _can_disconnect_with_k_nodes(self, hypernetwork: Hypernetwork,
                                     k: int) -> bool:
        """
        Check if removing k nodes can disconnect the hypergraph.

        Args:
            hypernetwork: Target hypergraph.
            k: Number of nodes to remove.

        Returns:
            True if k nodes can disconnect the hypergraph.
        """
        if k == 0:
            return not self.distance_calculator.is_connected(hypernetwork)

        nodes = list(hypernetwork.nodes)

        # Try all combinations of k nodes to remove
        from itertools import combinations

        for nodes_to_remove in combinations(nodes, k):
            # Create copy and remove nodes
            test_hn = self._copy_without_nodes(hypernetwork,
                                               set(nodes_to_remove))

            # Check if disconnected or too small
            if (test_hn.order() <= 1 or
                    not self.distance_calculator.is_connected(test_hn)):
                return True

        return False

    @staticmethod
    def _copy_without_nodes(hypernetwork: Hypernetwork,
                            nodes_to_remove: Set[int]) -> Hypernetwork:
        """
        Create copy of hypergraph without specified nodes.

        Args:
            hypernetwork: Original hypergraph.
            nodes_to_remove: Set of node IDs to remove.

        Returns:
            New hypergraph without the specified nodes.
        """
        new_hn = Hypernetwork()

        # Copy all hyperedges, excluding removed nodes
        for edge_id in hypernetwork.edges:
            original_nodes = hypernetwork.get_nodes(edge_id)
            remaining_nodes = [n for n in original_nodes
                               if n not in nodes_to_remove]

            if len(remaining_nodes) >= 1:  # Keep non-empty hyperedges
                new_hn.add_hyperedge(edge_id, remaining_nodes)

        return new_hn


class HyperedgeConnectivity:
    """
    Computes hyperedge connectivity λ(H).

    The hyperedge connectivity λ(H) is the minimum number of hyperedges
    that must be removed to disconnect the hypergraph.
    """

    def __init__(self, s: int = 1):
        """
        Initialize with s-walk parameter.

        Args:
            s: Parameter for s-walk connectivity computation.
        """
        self.s = s
        self.distance_calculator = HypergraphDistance(s)

    def compute(self, hypernetwork: Hypernetwork) -> int:
        """
        Compute hyperedge connectivity.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            Hyperedge connectivity value λ(H).
        """
        if hypernetwork.size() == 0:
            return 0

        # Try removing increasing numbers of hyperedges
        edges = list(hypernetwork.edges)
        m = len(edges)

        for k in range(m + 1):
            if self._can_disconnect_with_k_edges(hypernetwork, k):
                return k

        return m  # Worst case: remove all edges

    def _can_disconnect_with_k_edges(self, hypernetwork: Hypernetwork,
                                     k: int) -> bool:
        """
        Check if removing k hyperedges can disconnect the hypergraph.

        Args:
            hypernetwork: Target hypergraph.
            k: Number of hyperedges to remove.

        Returns:
            True if k hyperedges can disconnect the hypergraph.
        """
        if k == 0:
            return not self.distance_calculator.is_connected(hypernetwork)

        edges = list(hypernetwork.edges)

        # Try all combinations of k edges to remove
        from itertools import combinations

        for edges_to_remove in combinations(edges, k):
            # Create copy and remove edges
            test_hn = self._copy_without_edges(hypernetwork,
                                               set(edges_to_remove))

            # Check if disconnected
            if not self.distance_calculator.is_connected(test_hn):
                return True

        return False

    @staticmethod
    def _copy_without_edges(hypernetwork: Hypernetwork,
                            edges_to_remove: Set[int]) -> Hypernetwork:
        """
        Create copy of hypergraph without specified hyperedges.

        Args:
            hypernetwork: Original hypergraph.
            edges_to_remove: Set of edge IDs to remove.

        Returns:
            New hypergraph without the specified hyperedges.
        """
        new_hn = Hypernetwork()

        # Copy all hyperedges except those to be removed
        for edge_id in hypernetwork.edges:
            if edge_id not in edges_to_remove:
                nodes = hypernetwork.get_nodes(edge_id)
                new_hn.add_hyperedge(edge_id, nodes)

        return new_hn