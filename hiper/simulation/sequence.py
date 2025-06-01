# -*- coding: utf-8 -*-
"""
sequence.py

Defines the AttackSequence class for managing multiple attacks.
"""

from typing import Dict, List, Any, Optional

from hiper.core.hypernetwork import Hypernetwork
from .attack import Attack


class AttackSequence:
    """
    Manages and executes a sequence of attacks on a hypernetwork.

    An attack sequence represents a series of operations that can be applied
    to a hypernetwork in order. It provides functionality to execute all attacks
    track results, and handle failures.
    """

    def __init__(self, sequence_id: str) -> None:
        """
        Initialize the attack sequence.

        Args:
            sequence_id: Unique identifier for this sequence.
        """
        self.sequence_id = sequence_id
        self.attacks: List[Attack] = []
        self._executed = False
        self._results: List[Dict[str, Any]] = []
        self._execution_stats: Dict[str, Any] = {}

    def add_attack(self, attack: Attack) -> None:
        """
        Add an attack to the sequence.

        Args:
            attack: The attack to add to the sequence.
        """
        if self._executed:
            raise RuntimeError(
                "Cannot add attacks to an already executed sequence")
        self.attacks.append(attack)

    def add_attacks(self, attacks: List[Attack]) -> None:
        """
        Add multiple attacks to the sequence.

        Args:
            attacks: List of attacks to add to the sequence.
        """
        for attack in attacks:
            self.add_attack(attack)

    def execute(self, hypernetwork: Hypernetwork,
                stop_on_failure: bool = False) -> bool:
        """
        Execute all attacks in the sequence on the given hypernetwork.

        Args:
            hypernetwork: The target hypernetwork.
            stop_on_failure: If True, stop execution when an attack fails.

        Returns:
            True if all attacks were successful, False otherwise.
        """
        if self._executed:
            raise RuntimeError(
                "Sequence has already been executed. Use reset() first.")

        self._results.clear()
        successful_attacks = 0
        failed_attacks = 0

        # Capture initial state
        initial_order = hypernetwork.order()
        initial_size = hypernetwork.size()

        for i, attack in enumerate(self.attacks):
            try:
                success = attack.execute(hypernetwork)
                attack_result = {
                    'attack_index': i,
                    'attack_id': attack.attack_id,
                    'attack_type': type(attack).__name__,
                    'success': success,
                    'description': attack.describe(),
                    'result_data': attack.get_result()
                }
                self._results.append(attack_result)

                if success:
                    successful_attacks += 1
                else:
                    failed_attacks += 1
                    if stop_on_failure:
                        break

            except Exception as e:
                failed_attacks += 1
                attack_result = {
                    'attack_index': i,
                    'attack_id': attack.attack_id,
                    'attack_type': type(attack).__name__,
                    'success': False,
                    'description': attack.describe(),
                    'error': str(e),
                    'result_data': {}
                }
                self._results.append(attack_result)

                if stop_on_failure:
                    break

        # Capture final state
        final_order = hypernetwork.order()
        final_size = hypernetwork.size()

        # Store execution statistics
        self._execution_stats = {
            'total_attacks': len(self.attacks),
            'successful_attacks': successful_attacks,
            'failed_attacks': failed_attacks,
            'executed_attacks': successful_attacks + failed_attacks,
            'success_rate': successful_attacks / len(
                self.attacks) if self.attacks else 0.0,
            'initial_order': initial_order,
            'final_order': final_order,
            'initial_size': initial_size,
            'final_size': final_size,
            'order_change': final_order - initial_order,
            'size_change': final_size - initial_size,
            'stopped_on_failure': stop_on_failure and failed_attacks > 0
        }

        self._executed = True
        return failed_attacks == 0

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get detailed results of all executed attacks.

        Returns:
            List of dictionaries containing attack results.
        """
        return self._results.copy()

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get overall execution statistics.

        Returns:
            Dictionary containing execution statistics.
        """
        return self._execution_stats.copy()

    def is_executed(self) -> bool:
        """Check if the sequence has been executed."""
        return self._executed

    def reset(self) -> None:
        """
        Reset the sequence state to allow re-execution.

        This resets both the sequence and all individual attacks.
        """
        self._executed = False
        self._results.clear()
        self._execution_stats.clear()

        for attack in self.attacks:
            attack.reset()

    def clear(self) -> None:
        """
        Clear all attacks from the sequence and reset state.
        """
        self.attacks.clear()
        self.reset()

    def size(self) -> int:
        """Get the number of attacks in the sequence."""
        return len(self.attacks)

    def describe(self) -> str:
        """
        Return a human-readable description of the sequence.

        Returns:
            String description of the attack sequence.
        """
        if not self.attacks:
            return f"Empty attack sequence '{self.sequence_id}'"

        attack_descriptions = [attack.describe() for attack in self.attacks]
        return ((f"Attack sequence '{self.sequence_id}' "
                 f"with {len(self.attacks)} attacks:\n  - ")
                + ";\n  - ".join(attack_descriptions))

    def get_attack(self, index: int) -> Optional[Attack]:
        """
        Get an attack by its index in the sequence.

        Args:
            index: Index of the attack to retrieve.

        Returns:
            The attack at the given index, or None if index is invalid.
        """
        if 0 <= index < len(self.attacks):
            return self.attacks[index]
        return None

    def remove_attack(self, index: int) -> Optional[Attack]:
        """
        Remove and return an attack by its index.

        Args:
            index: Index of the attack to remove.

        Returns:
            The removed attack, or None if index is invalid.
        """
        if self._executed:
            raise RuntimeError("Cannot modify an already executed sequence")

        if 0 <= index < len(self.attacks):
            return self.attacks.pop(index)
        return None

    def insert_attack(self, index: int, attack: Attack) -> bool:
        """
        Insert an attack at a specific position in the sequence.

        Args:
            index: Position where to insert the attack.
            attack: The attack to insert.

        Returns:
            True if the attack was inserted successfully.
        """
        if self._executed:
            raise RuntimeError("Cannot modify an already executed sequence")

        if 0 <= index <= len(self.attacks):
            self.attacks.insert(index, attack)
            return True
        return False
