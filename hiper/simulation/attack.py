# -*- coding: utf-8 -*-
"""
attack.py

Defines the base Attack class and specific attack implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from hiper.core.hypernetwork import Hypernetwork


class Attack(ABC):
    """
    Abstract base class for hypernetwork attacks.

    An attack represents a single operation that can be applied to
    a hypernetwork, such as adding or removing nodes or hyperedges.
    """

    def __init__(self, attack_id: str) -> None:
        """
        Initialize the attack with a unique identifier.

        Args:
            attack_id: Unique identifier for this attack.
        """
        self.attack_id = attack_id
        self._executed = False
        self._result: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, hypernetwork: Hypernetwork) -> bool:
        """
        Execute the attack on the given hypernetwork.

        Args:
            hypernetwork: The target hypernetwork.

        Returns:
            True if the attack was successful, False otherwise.
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """
        Return a human-readable description of the attack.

        Returns:
            String description of the attack.
        """
        pass

    def is_executed(self) -> bool:
        """Check if the attack has been executed."""
        return self._executed

    def get_result(self) -> Dict[str, Any]:
        """Get the result of the attack execution."""
        return self._result.copy()

    def reset(self) -> None:
        """Reset the attack state to allow re-execution."""
        self._executed = False
        self._result.clear()


class AddNodeAttack(Attack):
    """
    Attack that adds a single node to the hypernetwork.
    """

    def __init__(self, attack_id: str, node_id: int) -> None:
        """
        Initialize the add node attack.

        Args:
            attack_id: Unique identifier for this attack.
            node_id: ID of the node to add.
        """
        super().__init__(attack_id)
        self.node_id = node_id

    def execute(self, hypernetwork: Hypernetwork) -> bool:
        """
        Execute the node addition attack.

        Args:
            hypernetwork: The target hypernetwork.

        Returns:
            True if the node was added successfully.
        """
        try:
            initial_order = hypernetwork.order()
            hypernetwork.add_node(self.node_id)
            final_order = hypernetwork.order()

            self._executed = True
            self._result = {
                'node_id': self.node_id,
                'initial_order': initial_order,
                'final_order': final_order,
                'nodes_added': final_order - initial_order
            }
            return True

        except Exception as e:
            self._result = {'error': str(e)}
            return False

    def describe(self) -> str:
        """Return description of the add node attack."""
        return f"Add node {self.node_id}"


class RemoveNodeAttack(Attack):
    """
    Attack that removes a single node from the hypernetwork.
    """

    def __init__(self, attack_id: str, node_id: int) -> None:
        """
        Initialize the remove node attack.

        Args:
            attack_id: Unique identifier for this attack.
            node_id: ID of the node to remove.
        """
        super().__init__(attack_id)
        self.node_id = node_id

    def execute(self, hypernetwork: Hypernetwork) -> bool:
        """
        Execute the node removal attack.

        Args:
            hypernetwork: The target hypernetwork.

        Returns:
            True if the node was removed successfully.
        """
        try:
            initial_order = hypernetwork.order()
            initial_size = hypernetwork.size()

            # Check if node exists before removal
            node_exists = self.node_id in hypernetwork.nodes
            affected_edges = hypernetwork.get_hyperedges(
                self.node_id) if node_exists else []

            hypernetwork.remove_node(self.node_id)

            final_order = hypernetwork.order()
            final_size = hypernetwork.size()

            self._executed = True
            self._result = {
                'node_id': self.node_id,
                'node_existed': node_exists,
                'initial_order': initial_order,
                'final_order': final_order,
                'initial_size': initial_size,
                'final_size': final_size,
                'nodes_removed': initial_order - final_order,
                'edges_affected': len(affected_edges),
                'edges_removed': initial_size - final_size
            }
            return True

        except Exception as e:
            self._result = {'error': str(e)}
            return False

    def describe(self) -> str:
        """Return description of the remove node attack."""
        return f"Remove node {self.node_id}"


class AddHyperedgeAttack(Attack):
    """
    Attack that adds a single hyperedge to the hypernetwork.
    """

    def __init__(self, attack_id: str, edge_id: int,
                 members: list[int]) -> None:
        """
        Initialize the add hyperedge attack.

        Args:
            attack_id: Unique identifier for this attack.
            edge_id: ID of the hyperedge to add.
            members: List of node IDs that form the hyperedge.
        """
        super().__init__(attack_id)
        self.edge_id = edge_id
        self.members = list(
            members)  # Make a copy to avoid external modifications

    def execute(self, hypernetwork: Hypernetwork) -> bool:
        """
        Execute the hyperedge addition attack.

        Args:
            hypernetwork: The target hypernetwork.

        Returns:
            True if the hyperedge was added successfully.
        """
        try:
            initial_size = hypernetwork.size()
            initial_order = hypernetwork.order()

            hypernetwork.add_hyperedge(self.edge_id, self.members)

            final_size = hypernetwork.size()
            final_order = hypernetwork.order()

            self._executed = True
            self._result = {
                'edge_id': self.edge_id,
                'members': self.members.copy(),
                'initial_size': initial_size,
                'final_size': final_size,
                'initial_order': initial_order,
                'final_order': final_order,
                'edges_added': final_size - initial_size,
                'nodes_added': final_order - initial_order
            }
            return True

        except Exception as e:
            self._result = {'error': str(e)}
            return False

    def describe(self) -> str:
        """Return description of the add hyperedge attack."""
        return f"Add hyperedge {self.edge_id} with members {self.members}"


class RemoveHyperedgeAttack(Attack):
    """
    Attack that removes a single hyperedge from the hypernetwork.
    """

    def __init__(self, attack_id: str, edge_id: int) -> None:
        """
        Initialize the remove hyperedge attack.

        Args:
            attack_id: Unique identifier for this attack.
            edge_id: ID of the hyperedge to remove.
        """
        super().__init__(attack_id)
        self.edge_id = edge_id

    def execute(self, hypernetwork: Hypernetwork) -> bool:
        """
        Execute the hyperedge removal attack.

        Args:
            hypernetwork: The target hypernetwork.

        Returns:
            True if the hyperedge was removed successfully.
        """
        try:
            initial_size = hypernetwork.size()

            # Check if edge exists and get its members before removal
            edge_exists = self.edge_id in hypernetwork.edges
            members = hypernetwork.get_nodes(
                self.edge_id) if edge_exists else []

            hypernetwork.remove_hyperedge(self.edge_id)

            final_size = hypernetwork.size()

            self._executed = True
            self._result = {
                'edge_id': self.edge_id,
                'edge_existed': edge_exists,
                'members': members,
                'initial_size': initial_size,
                'final_size': final_size,
                'edges_removed': initial_size - final_size
            }
            return True

        except Exception as e:
            self._result = {'error': str(e)}
            return False

    def describe(self) -> str:
        """Return description of the remove hyperedge attack."""
        return f"Remove hyperedge {self.edge_id}"
