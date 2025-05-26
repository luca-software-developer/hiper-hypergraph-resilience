# -*- coding: utf-8 -*-
"""
test_hypernetwork.py

Unit tests for the Hypernetwork class.
Tests addition and removal of nodes and edges.
Tests incidence operations, queries, metrics, and line graph.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork


class TestHypernetwork(unittest.TestCase):
    """
    Tests Hypernetwork methods.
    Methods tested include node and edge management,
    incidence operations, query functions, metrics, and line graph.
    """

    def setUp(self):
        """Create a fresh Hypernetwork for each test."""
        self.hn = Hypernetwork()

    def test_add_existing_node(self):
        """Adding the same node twice does not change the count."""
        # Add node 1 twice
        self.hn.add_node(1)
        self.hn.add_node(1)
        # Expect one node
        self.assertEqual(self.hn.order(), 1)

    def test_add_and_remove_node(self):
        """Node addition and removal update the network."""
        # Add node
        self.hn.add_node(1)
        self.assertIn(1, self.hn.nodes)
        self.assertEqual(self.hn.order(), 1)
        # Remove node
        self.hn.remove_node(1)
        self.assertNotIn(1, self.hn.nodes)
        self.assertEqual(self.hn.order(), 0)

    def test_remove_node_cascade(self):
        """Removing the only node in an edge removes the edge."""
        # Create edge 1 with [1]
        self.hn.add_hyperedge(1, [1])
        self.assertIn(1, self.hn.edges)
        # Remove node
        self.hn.remove_node(1)
        self.assertNotIn(1, self.hn.nodes)
        self.assertNotIn(1, self.hn.edges)

    def test_remove_node_partial(self):
        """Removing a shared node keeps nonempty edges."""
        # Create edges with shared node
        self.hn.add_hyperedge(10, [1, 2])
        self.hn.add_hyperedge(11, [1, 3])
        # Remove node
        self.hn.remove_node(1)
        # Edges remain
        self.assertIn(10, self.hn.edges)
        self.assertIn(11, self.hn.edges)
        # Node removed
        self.assertNotIn(1, self.hn.nodes)

    def test_add_hyperedge_duplicates(self):
        """Adding an edge with existing ID is ignored."""
        # Initial edge
        self.hn.add_hyperedge(20, [1, 2, 3])
        # Duplicate ID
        self.hn.add_hyperedge(20, [4, 5])
        # Members remain unchanged
        self.assertCountEqual(self.hn.get_nodes(20), [1, 2, 3])

    def test_query_nonexistent(self):
        """Queries on invalid IDs return empty lists."""
        self.assertEqual(self.hn.get_nodes(999), [])
        self.assertEqual(self.hn.get_hyperedges(999), [])

    def test_add_node_to_hyperedge(self):
        """Adding a node to an existing edge works."""
        # Create edge
        self.hn.add_hyperedge(30, [1, 2])
        # Add node
        self.hn.add_node_to_hyperedge(30, 3)
        self.assertCountEqual(self.hn.get_nodes(30), [1, 2, 3])

    def test_add_to_nonexistent_edge(self):
        """Adding to a nonexistent edge does nothing."""
        self.hn.add_node_to_hyperedge(999, 1)
        self.assertEqual(self.hn.size(), 0)

    def test_remove_node_from_edge_partial(self):
        """Removing a node from an edge removes incidence."""
        # Create edge
        self.hn.add_hyperedge(40, [1, 2, 3])
        # Remove node
        self.hn.remove_node_from_hyperedge(40, 1)
        # Node removed
        self.assertNotIn(1, self.hn.nodes)
        # Edge retains other nodes
        self.assertIn(40, self.hn.edges)
        self.assertCountEqual(self.hn.get_nodes(40), [2, 3])

    def test_remove_nonmember_or_edge(self):
        """Invalid removal calls do nothing."""
        self.hn.add_hyperedge(41, [1, 2])
        self.hn.remove_node_from_hyperedge(41, 3)
        self.assertCountEqual(self.hn.get_nodes(41), [1, 2])
        self.hn.remove_node_from_hyperedge(999, 1)
        self.assertIn(1, self.hn.nodes)

    def test_remove_node_from_edge_cascade(self):
        """Removing sole node from edge removes the edge."""
        self.hn.add_hyperedge(42, [1])
        self.hn.remove_node_from_hyperedge(42, 1)
        self.assertNotIn(42, self.hn.edges)
        self.assertNotIn(1, self.hn.nodes)

    def test_remove_hyperedge(self):
        """Removing an edge does not remove its nodes."""
        self.hn.add_hyperedge(50, [1, 2])
        self.hn.remove_hyperedge(50)
        self.assertNotIn(50, self.hn.edges)
        self.assertIn(1, self.hn.nodes)

    def test_neighbors_degrees_invalid(self):
        """Neighbors and degrees for invalid node return defaults."""
        self.assertEqual(self.hn.get_neighbors(999), [])
        self.assertEqual(self.hn.degree(999), 0)
        self.assertEqual(self.hn.hyperdegree(999), 0)

    def test_neighbors_and_degrees(self):
        """Neighbor and degree metrics."""
        self.hn.add_hyperedge(60, [1, 2])
        self.hn.add_hyperedge(61, [2, 3])
        self.assertCountEqual(self.hn.get_neighbors(2), [1, 3])
        self.assertEqual(self.hn.degree(2), 2)
        self.assertEqual(self.hn.hyperdegree(2), 2)

    def test_averages_empty_and_single(self):
        """Average metrics for empty and single-edge network."""
        self.assertEqual(self.hn.avg_deg(), 0)
        self.assertEqual(self.hn.avg_hyperdegree(), 0)
        self.assertEqual(self.hn.avg_hyperedge_size(), 0)
        self.hn.add_hyperedge(70, [1, 2, 3])
        self.assertEqual(self.hn.avg_deg(), 2)
        self.assertEqual(self.hn.avg_hyperdegree(), 1)
        self.assertEqual(self.hn.avg_hyperedge_size(), 3)

    def test_averages_multiple(self):
        """Average metrics for multiple-edge network."""
        # Create edges: 80->[1,2], 81->[2,3,4]
        self.hn.add_hyperedge(80, [1, 2])
        self.hn.add_hyperedge(81, [2, 3, 4])
        # Node degrees: 1->1, 2->3, 3->2, 4->2
        # => avg_deg=(1+3+2+2)/4=2.0
        self.assertAlmostEqual(self.hn.avg_deg(), 2.0)
        # Node hyperdegrees: 1->1, 2->2, 3->1, 4->1
        # => avg_hyperdegree=(1+2+1+1)/4=1.25
        self.assertAlmostEqual(self.hn.avg_hyperdegree(), 1.25)
        # Hyperedge sizes: 2 and 3 => avg_hyperedge_size=(2+3)/2=2.5
        self.assertAlmostEqual(self.hn.avg_hyperedge_size(), 2.5)

    def test_hyperedge_size_invalid(self):
        """hyperedge_size returns correct counts or zero."""
        self.hn.add_hyperedge(90, [1, 2, 3])
        self.assertEqual(self.hn.hyperedge_size(90), 3)
        self.assertEqual(self.hn.hyperedge_size(999), 0)

    def test_line_graph_empty_and_full(self):
        """Line graph for empty and full-overlap networks."""
        nodes, edges = self.hn.line_graph()
        self.assertEqual(nodes, [])
        self.assertEqual(edges, [])
        self.hn.add_hyperedge(100, [1, 2])
        self.hn.add_hyperedge(101, [1, 2])
        nodes, edges = self.hn.line_graph()
        self.assertCountEqual(nodes, [100, 101])
        self.assertIn((100, 101), edges)

    def test_line_graph_partial(self):
        """Line graph for disjoint networks."""
        self.hn.add_hyperedge(110, [1, 2])
        self.hn.add_hyperedge(111, [3, 4])
        nodes, edges = self.hn.line_graph()
        self.assertCountEqual(nodes, [110, 111])
        self.assertEqual(edges, [])


if __name__ == '__main__':
    unittest.main()
