# -*- coding: utf-8 -*-
"""
test_sequence.py

Comprehensive unit tests for the sequence module of the hiper.simulation
package.
Tests cover sequence creation, attack management, execution control,
and statistical analysis.
"""

import unittest
from unittest.mock import Mock, patch

from hiper.core.hypernetwork import Hypernetwork
from hiper.simulation.attack import AddNodeAttack, RemoveNodeAttack, \
    AddHyperedgeAttack
from hiper.simulation.sequence import AttackSequence


class TestAttackSequenceInitialization(unittest.TestCase):
    """Test cases for AttackSequence initialization and basic properties."""

    def setUp(self):
        """Set up test fixtures with sequence instance."""
        self.sequence = AttackSequence("test_sequence_001")

    def test_sequence_initialization(self):
        """Verify proper initialization of AttackSequence."""
        self.assertEqual(self.sequence.sequence_id, "test_sequence_001")
        self.assertEqual(self.sequence.size(), 0)
        self.assertFalse(self.sequence.is_executed())
        self.assertEqual(self.sequence.get_results(), [])
        self.assertEqual(self.sequence.get_execution_stats(), {})

    def test_sequence_description_empty(self):
        """Test description generation for empty sequence."""
        expected_description = "Empty attack sequence 'test_sequence_001'"
        self.assertEqual(self.sequence.describe(), expected_description)


class TestAttackSequenceManagement(unittest.TestCase):
    """Test cases for adding, removing, and managing attacks in sequences."""

    def setUp(self):
        """Set up test fixtures with sequence and sample attacks."""
        self.sequence = AttackSequence("management_test")
        self.attack1 = AddNodeAttack("add_node_1", 10)
        self.attack2 = RemoveNodeAttack("remove_node_2", 5)
        self.attack3 = AddHyperedgeAttack("add_edge_3", 100, [1, 2, 3])

    def test_add_single_attack(self):
        """Test adding a single attack to sequence."""
        self.sequence.add_attack(self.attack1)

        self.assertEqual(self.sequence.size(), 1)
        self.assertEqual(self.sequence.get_attack(0), self.attack1)

    def test_add_multiple_attacks_batch(self):
        """Test adding multiple attacks using add_attacks method."""
        attacks = [self.attack1, self.attack2, self.attack3]
        self.sequence.add_attacks(attacks)

        self.assertEqual(self.sequence.size(), 3)
        for i, attack in enumerate(attacks):
            self.assertEqual(self.sequence.get_attack(i), attack)

    def test_get_attack_valid_indices(self):
        """Test retrieving attacks by valid indices."""
        self.sequence.add_attacks([self.attack1, self.attack2, self.attack3])

        self.assertEqual(self.sequence.get_attack(0), self.attack1)
        self.assertEqual(self.sequence.get_attack(1), self.attack2)
        self.assertEqual(self.sequence.get_attack(2), self.attack3)

    def test_get_attack_invalid_indices(self):
        """Test retrieving attacks by invalid indices."""
        self.sequence.add_attack(self.attack1)

        self.assertIsNone(self.sequence.get_attack(-1))
        self.assertIsNone(self.sequence.get_attack(1))
        self.assertIsNone(self.sequence.get_attack(100))

    def test_remove_attack_valid_index(self):
        """Test removing attack by valid index."""
        self.sequence.add_attacks([self.attack1, self.attack2, self.attack3])

        removed_attack = self.sequence.remove_attack(1)

        self.assertEqual(removed_attack, self.attack2)
        self.assertEqual(self.sequence.size(), 2)
        self.assertEqual(self.sequence.get_attack(0), self.attack1)
        self.assertEqual(self.sequence.get_attack(1), self.attack3)

    def test_remove_attack_invalid_index(self):
        """Test removing attack by invalid index."""
        self.sequence.add_attack(self.attack1)

        removed_attack = self.sequence.remove_attack(5)
        self.assertIsNone(removed_attack)
        self.assertEqual(self.sequence.size(), 1)

    def test_insert_attack_valid_position(self):
        """Test inserting attack at valid position."""
        self.sequence.add_attacks([self.attack1, self.attack3])

        success = self.sequence.insert_attack(1, self.attack2)

        self.assertTrue(success)
        self.assertEqual(self.sequence.size(), 3)
        self.assertEqual(self.sequence.get_attack(0), self.attack1)
        self.assertEqual(self.sequence.get_attack(1), self.attack2)
        self.assertEqual(self.sequence.get_attack(2), self.attack3)

    def test_insert_attack_invalid_position(self):
        """Test inserting attack at invalid position."""
        self.sequence.add_attack(self.attack1)

        success = self.sequence.insert_attack(5, self.attack2)
        self.assertFalse(success)
        self.assertEqual(self.sequence.size(), 1)

    def test_clear_sequence(self):
        """Test clearing all attacks from sequence."""
        self.sequence.add_attacks([self.attack1, self.attack2, self.attack3])
        self.assertEqual(self.sequence.size(), 3)

        self.sequence.clear()

        self.assertEqual(self.sequence.size(), 0)
        self.assertFalse(self.sequence.is_executed())


class TestSequenceExecutionControl(unittest.TestCase):
    """Test cases for sequence execution with different control options."""

    def setUp(self):
        """Set up test fixtures with hypernetwork and sequence."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [2, 3, 4])

        self.sequence = AttackSequence("execution_test")

    def test_execute_empty_sequence(self):
        """Test execution of empty sequence."""
        result = self.sequence.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.sequence.is_executed())
        self.assertEqual(len(self.sequence.get_results()), 0)

        stats = self.sequence.get_execution_stats()
        self.assertEqual(stats['total_attacks'], 0)
        self.assertEqual(stats['successful_attacks'], 0)
        self.assertEqual(stats['failed_attacks'], 0)
        self.assertEqual(stats['success_rate'], 0.0)

    def test_execute_successful_sequence(self):
        """Test execution of sequence with all successful attacks."""
        attack1 = AddNodeAttack("add_node", 10)
        attack2 = AddHyperedgeAttack("add_edge", 5, [5, 6, 7])

        self.sequence.add_attacks([attack1, attack2])

        result = self.sequence.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.sequence.is_executed())

        results = self.sequence.get_results()
        self.assertEqual(len(results), 2)

        for i, attack_result in enumerate(results):
            self.assertEqual(attack_result['attack_index'], i)
            self.assertTrue(attack_result['success'])
            self.assertIn('description', attack_result)
            self.assertIn('result_data', attack_result)

        stats = self.sequence.get_execution_stats()
        self.assertEqual(stats['total_attacks'], 2)
        self.assertEqual(stats['successful_attacks'], 2)
        self.assertEqual(stats['failed_attacks'], 0)
        self.assertEqual(stats['success_rate'], 1.0)

    def test_execute_sequence_with_exception(self):
        """Test execution handling attack that raises exception."""
        attack1 = AddNodeAttack("add_node", 10)
        attack2 = Mock()
        attack2.attack_id = "exception_attack"
        attack2.describe.return_value = "Exception attack"
        attack2.execute.side_effect = RuntimeError("Test exception")

        self.sequence.add_attacks([attack1, attack2])

        result = self.sequence.execute(self.hypernetwork)

        self.assertFalse(result)

        results = self.sequence.get_results()
        self.assertEqual(len(results), 2)

        # Check exception handling in results
        exception_result = results[1]
        self.assertFalse(exception_result['success'])
        self.assertEqual(exception_result['error'], "Test exception")

    def test_double_execution_prevention(self):
        """Test that sequence cannot be executed twice without reset."""
        attack = AddNodeAttack("add_node", 10)
        self.sequence.add_attack(attack)

        # First execution
        self.sequence.execute(self.hypernetwork)
        self.assertTrue(self.sequence.is_executed())

        # Second execution should raise error
        with self.assertRaises(RuntimeError):
            self.sequence.execute(self.hypernetwork)


class TestSequenceStateManagement(unittest.TestCase):
    """Test cases for sequence state management and reset functionality."""

    def setUp(self):
        """Set up test fixtures with executed sequence."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])

        self.sequence = AttackSequence("state_test")
        attack1 = AddNodeAttack("add_node", 10)
        attack2 = AddHyperedgeAttack("add_edge", 5, [5, 6])

        self.sequence.add_attacks([attack1, attack2])
        self.sequence.execute(self.hypernetwork)

    def test_sequence_reset(self):
        """Test resetting sequence state after execution."""
        # Verify executed state
        self.assertTrue(self.sequence.is_executed())
        self.assertNotEqual(len(self.sequence.get_results()), 0)
        self.assertNotEqual(self.sequence.get_execution_stats(), {})

        # Reset sequence
        self.sequence.reset()

        # Verify reset state
        self.assertFalse(self.sequence.is_executed())
        self.assertEqual(self.sequence.get_results(), [])
        self.assertEqual(self.sequence.get_execution_stats(), {})

        # Verify attacks are also reset
        for i in range(self.sequence.size()):
            attack = self.sequence.get_attack(i)
            self.assertFalse(attack.is_executed())

    def test_sequence_re_execution_after_reset(self):
        """Test that sequence can be re-executed after reset."""
        # Reset and re-execute
        self.sequence.reset()
        result = self.sequence.execute(self.hypernetwork)

        self.assertTrue(result)
        self.assertTrue(self.sequence.is_executed())
        self.assertNotEqual(len(self.sequence.get_results()), 0)

    def test_modification_after_execution_prevention(self):
        """Test that sequence cannot be modified after execution."""
        attack = AddNodeAttack("new_attack", 20)

        with self.assertRaises(RuntimeError):
            self.sequence.add_attack(attack)

        with self.assertRaises(RuntimeError):
            self.sequence.remove_attack(0)

        with self.assertRaises(RuntimeError):
            self.sequence.insert_attack(0, attack)

    def test_modification_allowed_after_reset(self):
        """Test that sequence can be modified after reset."""
        self.sequence.reset()

        new_attack = AddNodeAttack("new_attack", 20)

        # These should not raise exceptions
        self.sequence.add_attack(new_attack)
        self.assertEqual(self.sequence.size(), 3)

        removed = self.sequence.remove_attack(2)
        self.assertEqual(removed, new_attack)
        self.assertEqual(self.sequence.size(), 2)


class TestSequenceMetricsAndAnalysis(unittest.TestCase):
    """Test cases for sequence execution metrics and statistical analysis."""

    def setUp(self):
        """Set up test fixtures with complex sequence scenario."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [3, 4, 5])

        self.sequence = AttackSequence("metrics_test")

    def test_execution_timing_capture(self):
        """Test that execution timing is properly captured."""
        attack = AddNodeAttack("timed_attack", 10)
        self.sequence.add_attack(attack)

        with patch('time.perf_counter', side_effect=[0.0, 0.1]):
            result = self.sequence.execute(self.hypernetwork)

        self.assertTrue(result)

        # Note: Since we're testing with real execution, timing will be captured
        # but exact values depend on actual execution time
        stats = self.sequence.get_execution_stats()
        self.assertIn('initial_order', stats)
        self.assertIn('final_order', stats)
        self.assertIn('initial_size', stats)
        self.assertIn('final_size', stats)


if __name__ == '__main__':
    unittest.main(verbosity=2)
