# -*- coding: utf-8 -*-
"""
test_attack.py

Comprehensive unit tests for the attack module of the hiper.simulation package.
Tests cover all attack types, state management, error handling, and result
tracking.
"""

import unittest
from unittest.mock import Mock

from hiper.core.hypernetwork import Hypernetwork
from hiper.simulation.attack import (
    Attack, AddNodeAttack, RemoveNodeAttack,
    AddHyperedgeAttack, RemoveHyperedgeAttack
)


class TestAttackBase(unittest.TestCase):
    """Test cases for the abstract Attack base class functionality."""

    def setUp(self):
        """Set up test fixtures with mock attack implementation."""

        # Create concrete implementation for testing abstract base
        class MockAttack(Attack):
            def execute(self, hypernetwork):
                self._executed = True
                self._result = {'mock': 'result'}
                return True

            def describe(self):
                return "Mock attack for testing"

        self.mock_attack = MockAttack("test_attack_001")

    def test_attack_initialization(self):
        """Verify proper initialization of attack base class."""
        self.assertEqual(self.mock_attack.attack_id, "test_attack_001")
        self.assertFalse(self.mock_attack.is_executed())
        self.assertEqual(self.mock_attack.get_result(), {})

    def test_attack_execution_state_tracking(self):
        """Test that execution state is properly tracked."""
        hypernetwork = Hypernetwork()

        # Verify initial state
        self.assertFalse(self.mock_attack.is_executed())

        # Execute attack
        result = self.mock_attack.execute(hypernetwork)

        # Verify state changes
        self.assertTrue(result)
        self.assertTrue(self.mock_attack.is_executed())
        self.assertEqual(self.mock_attack.get_result(), {'mock': 'result'})

    def test_attack_reset_functionality(self):
        """Test reset functionality restores initial state."""
        hypernetwork = Hypernetwork()

        # Execute attack
        self.mock_attack.execute(hypernetwork)
        self.assertTrue(self.mock_attack.is_executed())
        self.assertNotEqual(self.mock_attack.get_result(), {})

        # Reset attack
        self.mock_attack.reset()
        self.assertFalse(self.mock_attack.is_executed())
        self.assertEqual(self.mock_attack.get_result(), {})


class TestAddNodeAttack(unittest.TestCase):
    """Test cases for AddNodeAttack implementation."""

    def setUp(self):
        """Set up test fixtures with hypernetwork and attack instances."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [2, 3, 4])

        self.attack = AddNodeAttack("add_node_test", 10)

    def test_add_node_attack_initialization(self):
        """Verify proper initialization of AddNodeAttack."""
        self.assertEqual(self.attack.attack_id, "add_node_test")
        self.assertEqual(self.attack.node_id, 10)
        self.assertEqual(self.attack.describe(), "Add node 10")

    def test_add_new_node_success(self):
        """Test successful addition of new node to hypernetwork."""
        initial_order = self.hypernetwork.order()

        result = self.attack.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.attack.is_executed())

        attack_result = self.attack.get_result()
        self.assertEqual(attack_result['node_id'], 10)
        self.assertEqual(attack_result['initial_order'], initial_order)
        self.assertEqual(attack_result['final_order'], initial_order + 1)
        self.assertEqual(attack_result['nodes_added'], 1)

    def test_add_existing_node_handling(self):
        """Test behavior when adding node that already exists."""
        # Add node first
        self.hypernetwork.add_node(10)
        initial_order = self.hypernetwork.order()

        result = self.attack.execute(self.hypernetwork)

        self.assertTrue(result)

        attack_result = self.attack.get_result()
        self.assertEqual(attack_result['initial_order'], initial_order)
        self.assertEqual(attack_result['final_order'], initial_order)
        self.assertEqual(attack_result['nodes_added'], 0)

    def test_add_node_error_handling(self):
        """Test error handling during node addition."""
        # Mock hypernetwork to raise exception
        mock_hypernetwork = Mock()
        mock_hypernetwork.order.side_effect = Exception("Test error")

        result = self.attack.execute(mock_hypernetwork)

        self.assertFalse(result)
        attack_result = self.attack.get_result()
        self.assertIn('error', attack_result)
        self.assertEqual(attack_result['error'], "Test error")


class TestRemoveNodeAttack(unittest.TestCase):
    """Test cases for RemoveNodeAttack implementation."""

    def setUp(self):
        """Set up test fixtures with populated hypernetwork."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [2, 3, 4])
        self.hypernetwork.add_hyperedge(2, [3, 4, 5])

        self.attack = RemoveNodeAttack("remove_node_test", 3)

    def test_remove_node_attack_initialization(self):
        """Verify proper initialization of RemoveNodeAttack."""
        self.assertEqual(self.attack.attack_id, "remove_node_test")
        self.assertEqual(self.attack.node_id, 3)
        self.assertEqual(self.attack.describe(), "Remove node 3")

    def test_remove_existing_node_with_hyperedges(self):
        """Test removal of node that affects multiple hyperedges."""
        initial_order = self.hypernetwork.order()
        affected_edges = self.hypernetwork.get_hyperedges(3)

        result = self.attack.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.attack.is_executed())

        attack_result = self.attack.get_result()
        self.assertEqual(attack_result['node_id'], 3)
        self.assertTrue(attack_result['node_existed'])
        self.assertEqual(attack_result['initial_order'], initial_order)
        self.assertEqual(attack_result['final_order'], initial_order - 1)
        self.assertEqual(attack_result['nodes_removed'], 1)
        self.assertEqual(attack_result['edges_affected'], len(affected_edges))
        self.assertGreaterEqual(attack_result['edges_removed'], 0)

    def test_remove_nonexistent_node(self):
        """Test removal of node that does not exist."""
        initial_order = self.hypernetwork.order()

        nonexistent_attack = RemoveNodeAttack("remove_nonexistent", 99)
        result = nonexistent_attack.execute(self.hypernetwork)

        self.assertTrue(result)

        attack_result = nonexistent_attack.get_result()
        self.assertEqual(attack_result['node_id'], 99)
        self.assertFalse(attack_result['node_existed'])
        self.assertEqual(attack_result['initial_order'], initial_order)
        self.assertEqual(attack_result['final_order'], initial_order)
        self.assertEqual(attack_result['nodes_removed'], 0)
        self.assertEqual(attack_result['edges_affected'], 0)

    def test_remove_isolated_node(self):
        """Test removal of node not connected to any hyperedges."""
        self.hypernetwork.add_node(20)

        isolated_attack = RemoveNodeAttack("remove_isolated", 20)
        result = isolated_attack.execute(self.hypernetwork)

        self.assertTrue(result)

        attack_result = isolated_attack.get_result()
        self.assertTrue(attack_result['node_existed'])
        self.assertEqual(attack_result['nodes_removed'], 1)
        self.assertEqual(attack_result['edges_affected'], 0)
        self.assertEqual(attack_result['edges_removed'], 0)


class TestAddHyperedgeAttack(unittest.TestCase):
    """Test cases for AddHyperedgeAttack implementation."""

    def setUp(self):
        """Set up test fixtures with hypernetwork."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])

        self.attack = AddHyperedgeAttack("add_edge_test", 5, [4, 5, 6])

    def test_add_hyperedge_attack_initialization(self):
        """Verify proper initialization of AddHyperedgeAttack."""
        self.assertEqual(self.attack.attack_id, "add_edge_test")
        self.assertEqual(self.attack.edge_id, 5)
        self.assertEqual(self.attack.members, [4, 5, 6])
        self.assertEqual(self.attack.describe(),
                         "Add hyperedge 5 with members [4, 5, 6]")

    def test_add_new_hyperedge_success(self):
        """Test successful addition of new hyperedge."""
        initial_size = self.hypernetwork.size()

        result = self.attack.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.attack.is_executed())

        attack_result = self.attack.get_result()
        self.assertEqual(attack_result['edge_id'], 5)
        self.assertEqual(attack_result['members'], [4, 5, 6])
        self.assertEqual(attack_result['initial_size'], initial_size)
        self.assertEqual(attack_result['final_size'], initial_size + 1)
        self.assertEqual(attack_result['edges_added'], 1)
        self.assertGreaterEqual(attack_result['nodes_added'], 0)

    def test_add_hyperedge_with_existing_nodes(self):
        """Test adding hyperedge using existing nodes."""
        attack_existing = AddHyperedgeAttack("add_existing", 6, [1, 2, 4])

        result = attack_existing.execute(self.hypernetwork)

        self.assertTrue(result)

        attack_result = attack_existing.get_result()
        # Should add fewer new nodes since 1, 2 already exist
        self.assertLessEqual(attack_result['nodes_added'], 1)

    def test_add_duplicate_hyperedge(self):
        """Test behavior when adding hyperedge with existing ID."""
        existing_attack = AddHyperedgeAttack("add_duplicate", 0, [7, 8, 9])

        result = existing_attack.execute(self.hypernetwork)

        self.assertTrue(result)

        attack_result = existing_attack.get_result()
        # Should not add new hyperedge if ID already exists
        self.assertEqual(attack_result['edges_added'], 0)

    def test_members_list_independence(self):
        """Test that modifying original members list does not affect attack."""
        original_members = [7, 8, 9]
        attack = AddHyperedgeAttack("independence_test", 10, original_members)

        # Modify original list
        original_members.append(10)

        # Attack should still use original members
        self.assertEqual(attack.members, [7, 8, 9])


class TestRemoveHyperedgeAttack(unittest.TestCase):
    """Test cases for RemoveHyperedgeAttack implementation."""

    def setUp(self):
        """Set up test fixtures with populated hypernetwork."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [2, 3, 4])
        self.hypernetwork.add_hyperedge(2, [4, 5, 6])

        self.attack = RemoveHyperedgeAttack("remove_edge_test", 1)

    def test_remove_hyperedge_attack_initialization(self):
        """Verify proper initialization of RemoveHyperedgeAttack."""
        self.assertEqual(self.attack.attack_id, "remove_edge_test")
        self.assertEqual(self.attack.edge_id, 1)
        self.assertEqual(self.attack.describe(), "Remove hyperedge 1")

    def test_remove_existing_hyperedge(self):
        """Test successful removal of existing hyperedge."""
        initial_size = self.hypernetwork.size()
        members = self.hypernetwork.get_nodes(1)

        result = self.attack.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.attack.is_executed())

        attack_result = self.attack.get_result()
        self.assertEqual(attack_result['edge_id'], 1)
        self.assertTrue(attack_result['edge_existed'])
        self.assertEqual(attack_result['members'], members)
        self.assertEqual(attack_result['initial_size'], initial_size)
        self.assertEqual(attack_result['final_size'], initial_size - 1)
        self.assertEqual(attack_result['edges_removed'], 1)

    def test_remove_nonexistent_hyperedge(self):
        """Test removal of hyperedge that does not exist."""
        initial_size = self.hypernetwork.size()

        nonexistent_attack = RemoveHyperedgeAttack("remove_nonexistent", 99)
        result = nonexistent_attack.execute(self.hypernetwork)

        self.assertTrue(result)

        attack_result = nonexistent_attack.get_result()
        self.assertEqual(attack_result['edge_id'], 99)
        self.assertFalse(attack_result['edge_existed'])
        self.assertEqual(attack_result['members'], [])
        self.assertEqual(attack_result['initial_size'], initial_size)
        self.assertEqual(attack_result['final_size'], initial_size)
        self.assertEqual(attack_result['edges_removed'], 0)

    def test_hyperedge_removal_preserves_nodes(self):
        """Verify that removing hyperedge does not remove nodes."""
        initial_order = self.hypernetwork.order()

        result = self.attack.execute(self.hypernetwork)
        final_order = self.hypernetwork.order()

        self.assertTrue(result)
        # Nodes should be preserved even if hyperedge is removed
        self.assertEqual(final_order, initial_order)


class TestAttackErrorHandling(unittest.TestCase):
    """Test cases for error handling across all attack types."""

    def test_attack_execution_with_invalid_hypernetwork(self):
        """Test attack behavior with invalid hypernetwork object."""
        attack = RemoveNodeAttack("test_invalid", 1)
        invalid_hypernetwork = "not a hypernetwork"

        result = attack.execute(invalid_hypernetwork)  # type: ignore
        self.assertFalse(result)

        attack_result = attack.get_result()
        self.assertIn('error', attack_result)

    def test_concurrent_attack_execution(self):
        """Test behavior when attack is executed multiple times."""
        hypernetwork = Hypernetwork()
        attack = AddNodeAttack("concurrent_test", 1)

        # First execution
        result1 = attack.execute(hypernetwork)
        self.assertTrue(result1)

        # Second execution without reset
        result2 = attack.execute(hypernetwork)
        self.assertTrue(result2)

        # Both executions should be recorded
        self.assertTrue(attack.is_executed())


if __name__ == '__main__':
    unittest.main(verbosity=2)
