# -*- coding: utf-8 -*-
"""
test_simulator.py

Comprehensive unit tests for the simulator module of the hiper.simulation
package.
Tests cover simulation orchestration, state management, backup functionality,
and reporting.
"""

import unittest
from unittest.mock import Mock, patch

from hiper.core.hypernetwork import Hypernetwork
from hiper.simulation.attack import AddNodeAttack, RemoveNodeAttack, \
    AddHyperedgeAttack
from hiper.simulation.sequence import AttackSequence
from hiper.simulation.simulator import HypernetworkSimulator


class TestSimulatorInitialization(unittest.TestCase):
    """Test cases for HypernetworkSimulator initialization and configuration."""

    def setUp(self):
        """Set up test fixtures with simulator instance."""
        self.simulator = HypernetworkSimulator("test_simulator_001")

    def test_simulator_initialization(self):
        """Verify proper initialization of HypernetworkSimulator."""
        self.assertEqual(self.simulator.simulator_id, "test_simulator_001")
        self.assertEqual(len(self.simulator.simulation_history), 0)
        self.assertIsNone(self.simulator.current_hypernetwork)
        self.assertEqual(self.simulator.baseline_stats, {})

    def test_set_hypernetwork_without_backup(self):
        """Test setting hypernetwork without creating backup."""
        hypernetwork = Hypernetwork()
        hypernetwork.add_hyperedge(0, [1, 2, 3])

        self.simulator.set_hypernetwork(hypernetwork, create_backup=False)

        self.assertEqual(self.simulator.current_hypernetwork, hypernetwork)
        self.assertEqual(self.simulator.baseline_stats, {})

    def test_set_hypernetwork_with_backup(self):
        """Test setting hypernetwork with backup creation."""
        hypernetwork = Hypernetwork()
        hypernetwork.add_hyperedge(0, [1, 2, 3])
        hypernetwork.add_hyperedge(1, [2, 3, 4])

        self.simulator.set_hypernetwork(hypernetwork)

        self.assertEqual(self.simulator.current_hypernetwork, hypernetwork)

        baseline = self.simulator.baseline_stats
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline['label'], 'baseline')
        self.assertEqual(baseline['order'], hypernetwork.order())
        self.assertEqual(baseline['size'], hypernetwork.size())
        self.assertIsInstance(baseline['capture_timestamp'], float)


class TestSingleAttackSimulation(unittest.TestCase):
    """Test cases for simulating individual attacks."""

    def setUp(self):
        """Set up test fixtures with configured simulator."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [2, 3, 4])

        self.simulator = HypernetworkSimulator("attack_sim_test")
        self.simulator.set_hypernetwork(self.hypernetwork)

    def test_simulate_attack_without_hypernetwork(self):
        """Test simulation failure when no hypernetwork is set."""
        simulator = HypernetworkSimulator("no_network_test")
        attack = AddNodeAttack("test_attack", 10)

        with self.assertRaises(RuntimeError) as context:
            simulator.simulate_attack(attack)

        self.assertIn("No hypernetwork set", str(context.exception))

    def test_simulate_successful_attack_without_restoration(self):
        """Test successful attack simulation without network restoration."""
        attack = AddNodeAttack("add_node_permanent", 15)
        initial_order = self.hypernetwork.order()

        result = self.simulator.simulate_attack(attack, restore_after=False)

        self.assertTrue(result['success'])
        self.assertFalse(result['restored'])

        # Verify network was modified
        self.assertEqual(self.hypernetwork.order(), initial_order + 1)

        # Verify changes were tracked
        changes = result['changes']
        self.assertEqual(changes['order_change'], 1)
        self.assertEqual(changes['size_change'], 0)

    def test_simulate_failed_attack(self):
        """Test simulation of attack that fails."""
        # Create mock attack that fails
        mock_attack = Mock()
        mock_attack.attack_id = "failing_attack"
        mock_attack.describe.return_value = "Mock failing attack"
        mock_attack.execute.return_value = False
        mock_attack.get_result.return_value = {'error': 'Mock failure'}

        result = self.simulator.simulate_attack(mock_attack)

        self.assertFalse(result['success'])
        self.assertEqual(result['attack_result'], {'error': 'Mock failure'})

    def test_simulate_attack_timing_measurement(self):
        """Test that attack execution timing is properly measured."""
        attack = AddNodeAttack("timed_attack", 20)

        with patch('time.perf_counter', side_effect=[0.0, 0.1]):
            result = self.simulator.simulate_attack(attack)

        self.assertIsInstance(result['execution_time_seconds'], float)
        self.assertGreaterEqual(result['execution_time_seconds'], 0.0)

    def test_simulate_attack_statistics_capture(self):
        """Test comprehensive statistics capture during attack simulation."""
        attack = RemoveNodeAttack("remove_test", 2)

        result = self.simulator.simulate_attack(attack)

        # Verify pre-attack statistics
        pre_stats = result['pre_attack_stats']
        self.assertEqual(pre_stats['label'], 'pre_attack')
        self.assertIsInstance(pre_stats['order'], int)
        self.assertIsInstance(pre_stats['size'], int)
        self.assertIsInstance(pre_stats['avg_degree'], float)

        # Verify post-attack statistics
        post_stats = result['post_attack_stats']
        self.assertEqual(post_stats['label'], 'post_attack')

        # Verify changes calculation
        changes = result['changes']
        expected_order_change = post_stats['order'] - pre_stats['order']
        self.assertEqual(changes['order_change'], expected_order_change)


class TestSequenceSimulation(unittest.TestCase):
    """Test cases for simulating attack sequences."""

    def setUp(self):
        """Set up test fixtures with configured simulator and sequence."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [3, 4, 5])

        self.simulator = HypernetworkSimulator("sequence_sim_test")
        self.simulator.set_hypernetwork(self.hypernetwork)

        self.sequence = AttackSequence("test_sequence")
        self.sequence.add_attack(AddNodeAttack("add_node", 10))
        self.sequence.add_attack(
            AddHyperedgeAttack("add_edge", 5, [10, 11, 12]))
        self.sequence.add_attack(RemoveNodeAttack("remove_node", 1))

    def test_simulate_sequence_without_hypernetwork(self):
        """Test sequence simulation failure when no hypernetwork is set."""
        simulator = HypernetworkSimulator("no_network_seq_test")

        with self.assertRaises(RuntimeError) as context:
            simulator.simulate_sequence(self.sequence)

        self.assertIn("No hypernetwork set", str(context.exception))

    def test_simulate_sequence_without_restoration(self):
        """Test sequence simulation without network restoration."""
        result = self.simulator.simulate_sequence(self.sequence,
                                                  restore_after=False)

        self.assertFalse(result['restored'])

        # Network should be modified
        changes = result['changes']
        self.assertIsInstance(changes['order_change'], int)
        self.assertIsInstance(changes['size_change'], int)

    def test_simulate_sequence_with_stop_on_failure(self):
        """Test sequence simulation with stop_on_failure option."""
        # Create sequence with failing attack
        failing_sequence = AttackSequence("failing_sequence")
        failing_sequence.add_attack(AddNodeAttack("success", 20))

        mock_attack = Mock()
        mock_attack.attack_id = "failing_attack"
        mock_attack.describe.return_value = "Mock failure"
        mock_attack.execute.return_value = False
        mock_attack.get_result.return_value = {'failed': True}
        failing_sequence.add_attack(mock_attack)

        failing_sequence.add_attack(AddNodeAttack("should_not_execute", 21))

        result = self.simulator.simulate_sequence(failing_sequence,
                                                  stop_on_failure=True)

        self.assertFalse(result['success'])

        # Verify execution stopped early
        exec_stats = result['execution_stats']
        self.assertTrue(exec_stats['stopped_on_failure'])
        self.assertLess(exec_stats['executed_attacks'],
                        exec_stats['total_attacks'])

    def test_simulate_sequence_timing_and_statistics(self):
        """Test timing measurement and statistics capture for sequences."""
        result = self.simulator.simulate_sequence(self.sequence)

        # Verify timing
        self.assertIsInstance(result['execution_time_seconds'], float)
        self.assertGreaterEqual(result['execution_time_seconds'], 0.0)

        # Verify comprehensive statistics
        self.assertIn('pre_sequence_stats', result)
        self.assertIn('post_sequence_stats', result)
        self.assertIn('changes', result)

        # Verify sequence-specific data
        self.assertEqual(result['sequence_description'],
                         self.sequence.describe())
        self.assertIsInstance(result['sequence_results'], list)
        self.assertIsInstance(result['execution_stats'], dict)


class TestSimulatorBackupAndRestore(unittest.TestCase):
    """Test cases for simulator backup and restore functionality."""

    def setUp(self):
        """Set up test fixtures for backup testing."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])
        self.hypernetwork.add_hyperedge(1, [2, 3, 4])
        self.hypernetwork.add_hyperedge(2, [4, 5, 6])

        self.simulator = HypernetworkSimulator("backup_test")
        self.simulator.set_hypernetwork(self.hypernetwork)

    def test_hypernetwork_backup_creation(self):
        """Test creation of hypernetwork backup."""
        backup = self.simulator._create_hypernetwork_backup(self.hypernetwork)

        # Verify backup is separate object
        self.assertIsNot(backup, self.hypernetwork)

        # Verify backup has same structure
        self.assertEqual(backup.order(), self.hypernetwork.order())
        self.assertEqual(backup.size(), self.hypernetwork.size())

        # Verify all edges are copied
        for edge_id in self.hypernetwork.edges.keys():
            original_nodes = set(self.hypernetwork.get_nodes(edge_id))
            backup_nodes = set(backup.get_nodes(edge_id))
            self.assertEqual(original_nodes, backup_nodes)

    def test_backup_independence(self):
        """Test that backup is independent of original hypernetwork."""
        backup = self.simulator._create_hypernetwork_backup(self.hypernetwork)

        # Modify original
        self.hypernetwork.add_node(100)
        self.hypernetwork.add_hyperedge(10, [100, 101, 102])

        # Verify backup unchanged
        self.assertNotEqual(backup.order(), self.hypernetwork.order())
        self.assertNotEqual(backup.size(), self.hypernetwork.size())
        self.assertNotIn(10, backup.edges)


class TestSimulatorReporting(unittest.TestCase):
    """Test cases for simulator reporting and analysis functionality."""

    def setUp(self):
        """Set up test fixtures with simulation history."""
        self.hypernetwork = Hypernetwork()
        self.hypernetwork.add_hyperedge(0, [1, 2, 3])

        self.simulator = HypernetworkSimulator("reporting_test")
        self.simulator.set_hypernetwork(self.hypernetwork)

        # Execute several simulations to create history
        attack1 = AddNodeAttack("attack_1", 10)
        attack2 = RemoveNodeAttack("attack_2", 1)
        sequence = AttackSequence("test_sequence")
        sequence.add_attack(AddHyperedgeAttack("seq_attack", 5, [11, 12]))

        self.simulator.simulate_attack(attack1)
        self.simulator.simulate_attack(attack2)
        self.simulator.simulate_sequence(sequence)

    def test_get_simulation_history(self):
        """Test retrieval of complete simulation history."""
        history = self.simulator.get_simulation_history()

        self.assertEqual(len(history), 3)

        # Verify history contains expected simulations
        attack_sims = [sim for sim in history if
                       sim['simulation_type'] == 'single_attack']
        sequence_sims = [sim for sim in history if
                         sim['simulation_type'] == 'attack_sequence']

        self.assertEqual(len(attack_sims), 2)
        self.assertEqual(len(sequence_sims), 1)

    def test_get_baseline_stats(self):
        """Test retrieval of baseline statistics."""
        baseline = self.simulator.get_baseline_stats()

        self.assertIsNotNone(baseline)
        self.assertEqual(baseline['label'], 'baseline')
        self.assertIsInstance(baseline['order'], int)
        self.assertIsInstance(baseline['size'], int)

    def test_clear_simulation_history(self):
        """Test clearing simulation history."""
        self.assertEqual(len(self.simulator.simulation_history), 3)

        self.simulator.clear_history()

        self.assertEqual(len(self.simulator.simulation_history), 0)
        self.assertEqual(len(self.simulator.get_simulation_history()), 0)

    def test_generate_summary_report_empty_history(self):
        """Test report generation with empty simulation history."""
        empty_simulator = HypernetworkSimulator("empty_test")
        empty_simulator.set_hypernetwork(self.hypernetwork)

        report = empty_simulator.generate_summary_report()

        self.assertIn('message', report)
        self.assertEqual(report['message'], 'No simulations have been executed')

    def test_generate_comprehensive_summary_report(self):
        """Test generation of comprehensive summary report."""
        report = self.simulator.generate_summary_report()

        # Verify report structure
        expected_keys = [
            'simulator_id', 'baseline_stats', 'total_simulations',
            'single_attacks', 'attack_sequences', 'successful_simulations',
            'success_rate', 'timing_stats', 'attack_type_distribution',
            'generation_timestamp'
        ]

        for key in expected_keys:
            self.assertIn(key, report)

        # Verify report content
        self.assertEqual(report['simulator_id'], 'reporting_test')
        self.assertEqual(report['total_simulations'], 3)
        self.assertEqual(report['single_attacks'], 2)
        self.assertEqual(report['attack_sequences'], 1)
        self.assertIsInstance(report['success_rate'], float)

        # Verify timing statistics
        timing_stats = report['timing_stats']
        self.assertIn('average_execution_time', timing_stats)
        self.assertIn('min_execution_time', timing_stats)
        self.assertIn('max_execution_time', timing_stats)

        # Verify attack type distribution
        attack_types = report['attack_type_distribution']
        self.assertIsInstance(attack_types, dict)
        self.assertIn('AddNodeAttack', attack_types)
        self.assertIn('RemoveNodeAttack', attack_types)

    def test_statistics_capture_accuracy(self):
        """Test accuracy of captured hypernetwork statistics."""
        stats = self.simulator._capture_hypernetwork_stats(self.hypernetwork,
                                                           "test_label")

        self.assertEqual(stats['label'], "test_label")
        self.assertEqual(stats['order'], self.hypernetwork.order())
        self.assertEqual(stats['size'], self.hypernetwork.size())
        self.assertAlmostEqual(stats['avg_degree'], self.hypernetwork.avg_deg(),
                               places=6)
        self.assertAlmostEqual(stats['avg_hyperdegree'],
                               self.hypernetwork.avg_hyperdegree(), places=6)
        self.assertAlmostEqual(stats['avg_hyperedge_size'],
                               self.hypernetwork.avg_hyperedge_size(), places=6)
        self.assertIsInstance(stats['capture_timestamp'], float)

    def test_changes_calculation_accuracy(self):
        """Test accuracy of changes calculation between states."""
        pre_stats = {
            'order': 5, 'size': 3, 'avg_degree': 2.4,
            'avg_hyperdegree': 1.8, 'avg_hyperedge_size': 3.0
        }
        post_stats = {
            'order': 7, 'size': 4, 'avg_degree': 2.1,
            'avg_hyperdegree': 1.7, 'avg_hyperedge_size': 2.8
        }

        changes = self.simulator._calculate_changes(pre_stats, post_stats)

        self.assertEqual(changes['order_change'], 2)
        self.assertEqual(changes['size_change'], 1)
        self.assertAlmostEqual(changes['avg_degree_change'], -0.3, places=1)
        self.assertAlmostEqual(changes['avg_hyperdegree_change'], -0.1,
                               places=1)
        self.assertAlmostEqual(changes['avg_hyperedge_size_change'], -0.2,
                               places=1)


class TestSimulatorErrorHandling(unittest.TestCase):
    """Test cases for simulator error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures for error testing."""
        self.simulator = HypernetworkSimulator("error_test")

    def test_simulation_without_hypernetwork_set(self):
        """Test error handling when no hypernetwork is set."""
        attack = AddNodeAttack("test", 1)
        sequence = AttackSequence("test_seq")

        with self.assertRaises(RuntimeError):
            self.simulator.simulate_attack(attack)

        with self.assertRaises(RuntimeError):
            self.simulator.simulate_sequence(sequence)

    def test_report_generation_with_minimal_data(self):
        """Test report generation with minimal simulation data."""
        hypernetwork = Hypernetwork()
        hypernetwork.add_hyperedge(0, [1])
        self.simulator.set_hypernetwork(hypernetwork)

        # Single quick simulation
        attack = AddNodeAttack("minimal", 2)
        self.simulator.simulate_attack(attack)

        report = self.simulator.generate_summary_report()

        # Should handle minimal data gracefully
        self.assertEqual(report['total_simulations'], 1)
        self.assertIsInstance(report['success_rate'], float)
        self.assertIsInstance(report['timing_stats']['average_execution_time'],
                              float)


if __name__ == '__main__':
    unittest.main(verbosity=2)
