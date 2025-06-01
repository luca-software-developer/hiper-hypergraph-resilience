# -*- coding: utf-8 -*-
"""
simulator.py

Defines the HypernetworkSimulator class for orchestrating attack simulations.
"""

import time
from typing import Dict, List, Any, Optional

from hiper.core.hypernetwork import Hypernetwork
from .attack import Attack
from .sequence import AttackSequence


class HypernetworkSimulator:
    """
    Orchestrates and manages hypernetwork attack simulations.

    The simulator provides high-level functionality for running attack
    simulations, analyzing results, and managing multiple scenarios.
    It can work with both individual attacks and attack sequences.
    """

    def __init__(self, simulator_id: str) -> None:
        """
        Initialize the hypernetwork simulator.

        Args:
            simulator_id: Unique identifier for this simulator instance.
        """
        self.simulator_id = simulator_id
        self.simulation_history: List[Dict[str, Any]] = []
        self.current_hypernetwork: Optional[Hypernetwork] = None
        self.baseline_stats: Optional[Dict[str, Any]] = {}

    def set_hypernetwork(self, hypernetwork: Hypernetwork,
                         create_backup: bool = True) -> None:
        """
        Set the target hypernetwork for simulations.

        Args:
            hypernetwork: The hypernetwork to use for simulations.
            create_backup: If True, creates a backup of the original state.
        """
        self.current_hypernetwork = hypernetwork

        if create_backup:
            self.baseline_stats = self._capture_hypernetwork_stats(hypernetwork,
                                                                   "baseline")

    def simulate_attack(self, attack: Attack, restore_after: bool = True) -> \
            Dict[str, Any]:
        """
        Simulate a single attack on the current hypernetwork.

        Args:
            attack: The attack to simulate.
            restore_after: If True, restore the hypernetwork to its original
            state after the attack.

        Returns:
            Dictionary containing simulation results.

        Raises:
            RuntimeError: If no hypernetwork has been set.
        """
        if self.current_hypernetwork is None:
            raise RuntimeError(
                "No hypernetwork set. Use set_hypernetwork() first.")

        # Create backup if restoration is requested
        backup_hypernetwork = None
        if restore_after:
            backup_hypernetwork = self._create_hypernetwork_backup(
                self.current_hypernetwork)

        # Capture pre-attack state
        pre_attack_stats = self._capture_hypernetwork_stats(
            self.current_hypernetwork, "pre_attack")

        # Execute attack with timing
        start_time = time.perf_counter()
        success = attack.execute(self.current_hypernetwork)
        execution_time = time.perf_counter() - start_time

        # Capture post-attack state
        post_attack_stats = self._capture_hypernetwork_stats(
            self.current_hypernetwork, "post_attack")

        # Prepare simulation result
        simulation_result = {
            'simulation_type': 'single_attack',
            'attack_id': attack.attack_id,
            'attack_type': type(attack).__name__,
            'attack_description': attack.describe(),
            'success': success,
            'execution_time_seconds': execution_time,
            'pre_attack_stats': pre_attack_stats,
            'post_attack_stats': post_attack_stats,
            'attack_result': attack.get_result(),
            'changes': self._calculate_changes(pre_attack_stats,
                                               post_attack_stats),
            'timestamp': time.time(),
            'restored': restore_after
        }

        # Restore hypernetwork if requested
        if restore_after and backup_hypernetwork is not None:
            self.current_hypernetwork = backup_hypernetwork

        # Add to simulation history
        self.simulation_history.append(simulation_result)

        return simulation_result

    def simulate_sequence(self, sequence: AttackSequence,
                          restore_after: bool = True,
                          stop_on_failure: bool = False) -> Dict[str, Any]:
        """
        Simulate an attack sequence on the current hypernetwork.

        Args:
            sequence: The attack sequence to simulate.
            restore_after: If True, restore the hypernetwork to its original
            state after the sequence.
            stop_on_failure: If True, stop execution when an attack fails.

        Returns:
            Dictionary containing simulation results.

        Raises:
            RuntimeError: If no hypernetwork has been set.
        """
        if self.current_hypernetwork is None:
            raise RuntimeError(
                "No hypernetwork set. Use set_hypernetwork() first.")

        # Create backup if restoration is requested
        backup_hypernetwork = None
        if restore_after:
            backup_hypernetwork = self._create_hypernetwork_backup(
                self.current_hypernetwork)

        # Capture pre-sequence state
        pre_sequence_stats = self._capture_hypernetwork_stats(
            self.current_hypernetwork, "pre_sequence")

        # Execute sequence with timing
        start_time = time.perf_counter()
        success = sequence.execute(self.current_hypernetwork, stop_on_failure)
        execution_time = time.perf_counter() - start_time

        # Capture post-sequence state
        post_sequence_stats = self._capture_hypernetwork_stats(
            self.current_hypernetwork, "post_sequence")

        # Prepare simulation result
        simulation_result = {
            'simulation_type': 'attack_sequence',
            'sequence_id': sequence.sequence_id,
            'sequence_description': sequence.describe(),
            'sequence_size': sequence.size(),
            'success': success,
            'execution_time_seconds': execution_time,
            'pre_sequence_stats': pre_sequence_stats,
            'post_sequence_stats': post_sequence_stats,
            'sequence_results': sequence.get_results(),
            'execution_stats': sequence.get_execution_stats(),
            'changes': self._calculate_changes(pre_sequence_stats,
                                               post_sequence_stats),
            'timestamp': time.time(),
            'restored': restore_after
        }

        # Restore hypernetwork if requested
        if restore_after and backup_hypernetwork is not None:
            self.current_hypernetwork = backup_hypernetwork

        # Add to simulation history
        self.simulation_history.append(simulation_result)

        return simulation_result

    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete simulation history.

        Returns:
            List of all simulation results.
        """
        return self.simulation_history.copy()

    def get_baseline_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get the baseline statistics of the original hypernetwork.

        Returns:
            Baseline statistics dictionary, or None if not available.
        """
        return self.baseline_stats.copy() if self.baseline_stats else None

    def clear_history(self) -> None:
        """Clear the simulation history."""
        self.simulation_history.clear()

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all simulations.

        Returns:
            Dictionary containing summary statistics and analysis.
        """
        if not self.simulation_history:
            return {'message': 'No simulations have been executed'}

        # Basic counts
        total_simulations = len(self.simulation_history)
        single_attacks = sum(1 for sim in self.simulation_history
                             if sim['simulation_type'] == 'single_attack')
        sequences = sum(1 for sim in self.simulation_history
                        if sim['simulation_type'] == 'attack_sequence')

        # Success rates
        successful_simulations = sum(
            1 for sim in self.simulation_history if sim['success'])
        success_rate = successful_simulations / total_simulations \
            if total_simulations > 0 else 0.0

        # Timing statistics
        execution_times = [sim['execution_time_seconds'] for sim in
                           self.simulation_history]
        avg_execution_time = sum(execution_times) / len(
            execution_times) if execution_times else 0.0
        min_execution_time = min(execution_times) if execution_times else 0.0
        max_execution_time = max(execution_times) if execution_times else 0.0

        # Attack type distribution
        attack_types = {}
        for sim in self.simulation_history:
            if sim['simulation_type'] == 'single_attack':
                attack_type = sim['attack_type']
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            else:
                # For sequences, count individual attacks
                for result in sim['sequence_results']:
                    attack_type = result['attack_type']
                    attack_types[attack_type] = attack_types.get(attack_type,
                                                                 0) + 1

        return {
            'simulator_id': self.simulator_id,
            'baseline_stats': self.baseline_stats,
            'total_simulations': total_simulations,
            'single_attacks': single_attacks,
            'attack_sequences': sequences,
            'successful_simulations': successful_simulations,
            'success_rate': success_rate,
            'timing_stats': {
                'average_execution_time': avg_execution_time,
                'min_execution_time': min_execution_time,
                'max_execution_time': max_execution_time
            },
            'attack_type_distribution': attack_types,
            'generation_timestamp': time.time()
        }

    @staticmethod
    def _capture_hypernetwork_stats(hypernetwork: Hypernetwork,
                                    label: str) -> Dict[str, Any]:
        """
        Capture comprehensive statistics of a hypernetwork state.

        Args:
            hypernetwork: The hypernetwork to analyze.
            label: Label for this capture.

        Returns:
            Dictionary containing hypernetwork statistics.
        """
        return {
            'label': label,
            'order': hypernetwork.order(),
            'size': hypernetwork.size(),
            'avg_degree': hypernetwork.avg_deg(),
            'avg_hyperdegree': hypernetwork.avg_hyperdegree(),
            'avg_hyperedge_size': hypernetwork.avg_hyperedge_size(),
            'capture_timestamp': time.time()
        }

    @staticmethod
    def _create_hypernetwork_backup(hypernetwork: Hypernetwork) -> Hypernetwork:
        """
        Create a deep copy backup of a hypernetwork.

        Args:
            hypernetwork: The hypernetwork to back up.

        Returns:
            A deep copy of the hypernetwork.
        """
        backup = Hypernetwork()

        # Copy all hyperedges with their members
        for edge_id, node_set in hypernetwork.edges.items():
            backup.add_hyperedge(edge_id, list(node_set))

        return backup

    @staticmethod
    def _calculate_changes(pre_stats: Dict[str, Any],
                           post_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the changes between two hypernetwork states.

        Args:
            pre_stats: Statistics before the operation.
            post_stats: Statistics after the operation.

        Returns:
            Dictionary containing calculated changes.
        """
        return {
            'order_change': post_stats['order'] - pre_stats['order'],
            'size_change': post_stats['size'] - pre_stats['size'],
            'avg_degree_change': post_stats['avg_degree'] - pre_stats[
                'avg_degree'],
            'avg_hyperdegree_change': post_stats['avg_hyperdegree'] - pre_stats[
                'avg_hyperdegree'],
            'avg_hyperedge_size_change': post_stats['avg_hyperedge_size'] -
                                         pre_stats['avg_hyperedge_size']
        }
