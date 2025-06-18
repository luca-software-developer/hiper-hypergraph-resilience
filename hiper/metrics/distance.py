# -*- coding: utf-8 -*-
"""
distance.py

Implements hypergraph distance computations and path finding algorithms.
Provides s-walk and hyperpath functionality.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional

from hiper.core.hypernetwork import Hypernetwork


class HypergraphDistance:
    """
    Computes distances and paths in hypergraphs using s-walk semantics.

    An s-walk is a sequence of alternating nodes and hyperedges where each
    hyperedge intersects at least s previously visited nodes in the walk.
    """

    def __init__(self, s: int = 1):
        """
        Initialize with parameter s for s-walk computation.

        Args:
            s: Minimum intersection requirement for s-walks.
        """
        self.s = s

    def is_valid_s_walk(self, hypernetwork: Hypernetwork,
                        walk: List[Tuple[int, int]]) -> bool:
        """
        Check if a walk is a valid s-walk.

        Args:
            hypernetwork: Target hypergraph.
            walk: List of (node_id, edge_id) pairs representing the walk.

        Returns:
            True if the walk satisfies s-walk conditions.
        """
        if not walk:
            return True

        visited_nodes = set()

        for i, (node_id, edge_id) in enumerate(walk):
            # Check if node is in the hyperedge
            edge_nodes = set(hypernetwork.get_nodes(edge_id))
            if node_id not in edge_nodes:
                return False

            visited_nodes.add(node_id)

            # Check s-walk condition: hyperedge must intersect at least
            # min(s, i+1) previously visited nodes
            required_intersection = min(self.s, i + 1)
            actual_intersection = len(edge_nodes & visited_nodes)

            if actual_intersection < required_intersection:
                return False

        return True

    def find_shortest_s_walk(self, hypernetwork: Hypernetwork,
                             source: int, target: int) -> Optional[List[int]]:
        """
        Find shortest s-walk between two nodes using BFS.

        Args:
            hypernetwork: Target hypergraph.
            source: Source node ID.
            target: Target node ID.

        Returns:
            List of edge IDs forming shortest s-walk, or None if no path.
        """
        if source == target:
            return []

        if (source not in hypernetwork.nodes or
                target not in hypernetwork.nodes):
            return None

        # BFS queue: (current_node, visited_nodes, edge_path)
        queue = deque([(source, [source], [])])
        visited = set()

        while queue:
            current_node, visited_nodes, edge_path = queue.popleft()

            # Create state key for cycle detection
            state_key = (current_node, tuple(sorted(visited_nodes)))
            if state_key in visited:
                continue
            visited.add(state_key)

            # Explore all hyperedges containing current node
            for edge_id in hypernetwork.get_hyperedges(current_node):
                edge_nodes = set(hypernetwork.get_nodes(edge_id))

                # Check s-walk condition
                visited_set = set(visited_nodes)
                required_intersection = min(self.s, len(visited_nodes))
                actual_intersection = len(edge_nodes & visited_set)

                if actual_intersection < required_intersection:
                    continue

                # Explore all nodes in this hyperedge
                for next_node in edge_nodes:
                    if next_node == target:
                        return edge_path + [edge_id]

                    if next_node not in visited_set:
                        new_visited = visited_nodes + [next_node]
                        new_path = edge_path + [edge_id]
                        queue.append((next_node, new_visited, new_path))

        return None

    def compute_distance(self, hypernetwork: Hypernetwork,
                         source: int, target: int) -> float:
        """
        Compute s-walk distance between two nodes.

        Args:
            hypernetwork: Target hypergraph.
            source: Source node ID.
            target: Target node ID.

        Returns:
            Length of shortest s-walk, or infinity if no path exists.
        """
        path = self.find_shortest_s_walk(hypernetwork, source, target)
        if path is None:
            return float('inf')
        return len(path)

    def compute_all_distances(self, hypernetwork: Hypernetwork) -> Dict[
        Tuple[int, int], float]:
        """
        Compute all pairwise s-walk distances.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            Dictionary mapping (source, target) pairs to distances.
        """
        distances = {}
        nodes = list(hypernetwork.nodes)

        for i, source in enumerate(nodes):
            for target in nodes[i + 1:]:
                # Compute both directions as hypergraphs may be asymmetric
                dist_st = self.compute_distance(hypernetwork, source, target)
                dist_ts = self.compute_distance(hypernetwork, target, source)

                distances[(source, target)] = dist_st
                distances[(target, source)] = dist_ts

        # Add self-distances
        for node in nodes:
            distances[(node, node)] = 0.0

        return distances

    def is_connected(self, hypernetwork: Hypernetwork) -> bool:
        """
        Check if hypergraph is connected using s-walk semantics.

        Args:
            hypernetwork: Target hypergraph.

        Returns:
            True if all node pairs are connected by s-walks.
        """
        nodes = list(hypernetwork.nodes)
        if len(nodes) <= 1:
            return True

        # Check if all nodes are reachable from first node
        first_node = nodes[0]
        for other_node in nodes[1:]:
            if self.compute_distance(hypernetwork, first_node,
                                     other_node) == float('inf'):
                return False

        return True
