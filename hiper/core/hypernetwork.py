# -*- coding: utf-8 -*-
"""
hypernetwork.py

High-performance Hypernetwork using dict-of-sets representation.
"""
from typing import Dict, List, Set, Tuple


class Hypernetwork:
    """
    Hypernetwork implemented with two dicts:
    - nodes_map: node_id -> set of edge_ids
    - edges_map: edge_id -> set of node_ids

    All operations are optimized to O(1) amortized where possible.
    """

    def __init__(self) -> None:
        self.nodes_map: Dict[int, Set[int]] = {}
        self.edges_map: Dict[int, Set[int]] = {}

    def lightweight_copy(self) -> 'Hypernetwork':
        """
        Create a lightweight copy of the hypernetwork.

        Returns:
            A new Hypernetwork instance with copied structure
        """
        new_hn = Hypernetwork()
        new_hn.nodes_map = {
            nid: edges.copy() for nid, edges in self.nodes_map.items()
        }
        new_hn.edges_map = {
            eid: nodes.copy() for eid, nodes in self.edges_map.items()
        }
        return new_hn

    def add_node(self, node_id: int) -> None:
        """Add a node if not present."""
        self.nodes_map.setdefault(node_id, set())

    def remove_node(self, node_id: int) -> None:
        """Remove a node and its incidences."""
        edge_ids = self.nodes_map.pop(node_id, None)
        if not edge_ids:
            return
        for eid in list(edge_ids):
            self.edges_map[eid].discard(node_id)
            if not self.edges_map[eid]:
                self.edges_map.pop(eid, None)

    def add_hyperedge(self, edge_id: int, members: List[int]) -> None:
        """Add a hyperedge linking given node IDs."""
        if edge_id in self.edges_map:
            return
        member_set = set(members)
        self.edges_map[edge_id] = member_set
        for nid in member_set:
            self.nodes_map.setdefault(nid, set()).add(edge_id)

    def add_node_to_hyperedge(self, edge_id: int, node_id: int) -> None:
        """Add a node to an existing hyperedge."""
        if edge_id not in self.edges_map:
            return
        if node_id in self.edges_map[edge_id]:
            return
        self.edges_map[edge_id].add(node_id)
        self.nodes_map.setdefault(node_id, set()).add(edge_id)

    def remove_node_from_hyperedge(self, edge_id: int, node_id: int) -> None:
        """Remove a node from a given hyperedge."""
        if edge_id not in self.edges_map:
            return
        if node_id not in self.edges_map[edge_id]:
            return
        self.edges_map[edge_id].remove(node_id)
        self.nodes_map[node_id].remove(edge_id)
        if not self.edges_map[edge_id]:
            self.edges_map.pop(edge_id, None)
        if not self.nodes_map.get(node_id):
            self.nodes_map.pop(node_id, None)

    def remove_hyperedge(self, edge_id: int) -> None:
        """Remove a hyperedge and its incidences (nodes remain)."""
        node_ids = self.edges_map.pop(edge_id, None)
        if not node_ids:
            return
        for nid in node_ids:
            self.nodes_map[nid].discard(edge_id)

    def get_nodes(self, edge_id: int) -> List[int]:
        """List node IDs in a given hyperedge."""
        return list(self.edges_map.get(edge_id, []))

    def get_hyperedges(self, node_id: int) -> List[int]:
        """List hyperedge IDs containing a given node."""
        return list(self.nodes_map.get(node_id, []))

    def get_neighbors(self, node_id: int) -> List[int]:
        """List neighbors of a given node."""
        neighbors: Set[int] = set()
        for eid in self.nodes_map.get(node_id, []):
            neighbors |= (self.edges_map[eid] - {node_id})
        return list(neighbors)

    def degree(self, node_id: int) -> int:
        """Return degree (number of neighbors) of a node."""
        return len(self.get_neighbors(node_id))

    def hyperdegree(self, node_id: int) -> int:
        """Return hyperdegree (number of hyperedges) of a node."""
        return len(self.nodes_map.get(node_id, []))

    def order(self) -> int:
        """Return number of nodes."""
        return len(self.nodes_map)

    def size(self) -> int:
        """Return number of hyperedges."""
        return len(self.edges_map)

    def avg_deg(self) -> float:
        """Return average node degree."""
        n = self.order()
        return 0.0 if n == 0 else sum(
            self.degree(nid) for nid in self.nodes_map) / n

    def avg_hyperdegree(self) -> float:
        """Return average hyperdegree of nodes."""
        n = self.order()
        return 0.0 if n == 0 else sum(
            self.hyperdegree(nid) for nid in self.nodes_map) / n

    def hyperedge_size(self, edge_id: int) -> int:
        """Return size (number of nodes) of a hyperedge."""
        return len(self.edges_map.get(edge_id, []))

    def avg_hyperedge_size(self) -> float:
        """Return average hyperedge size."""
        m = self.size()
        return 0.0 if m == 0 else sum(
            self.hyperedge_size(eid) for eid in self.edges_map) / m

    def line_graph(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Return line graph (edge_ids, intersecting hyperedges)."""
        edge_ids = list(self.edges_map)
        lg: List[Tuple[int, int]] = []
        for i, e1 in enumerate(edge_ids):
            for e2 in edge_ids[i + 1:]:
                if self.edges_map[e1] & self.edges_map[e2]:
                    lg.append((e1, e2))
        return edge_ids, lg

    def print_info(self) -> None:
        """Print basic hypernetwork metrics."""
        print(f"Order: {self.order()}")
        print(f"Size: {self.size()}")
        print(f"Avg degree: {self.avg_deg():.2f}")
        print(f"Avg hyperdegree: {self.avg_hyperdegree():.2f}")
        print(f"Avg hyperedge size: {self.avg_hyperedge_size():.2f}")

    @property
    def nodes(self) -> Dict[int, Set[int]]:
        """Expose mapping of node IDs to hyperedge sets."""
        return self.nodes_map

    @property
    def edges(self) -> Dict[int, Set[int]]:
        """Expose mapping of hyperedge IDs to node sets."""
        return self.edges_map
