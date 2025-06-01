# -*- coding: utf-8 -*-
"""
simulation package

Provides classes for simulating attacks on hypernetworks, including individual
attacks, attack sequences, and comprehensive simulation management.

This package allows users to:
- Define and execute individual attacks (node/hyperedge addition/removal)
- Create and manage sequences of multiple attacks
- Run comprehensive simulations with performance monitoring
- Analyze and report on attack impacts

Classes:
    Attack: Abstract base class for all attack types
    AddNodeAttack: Attack for adding nodes to hypernetworks
    RemoveNodeAttack: Attack for removing nodes from hypernetworks
    AddHyperedgeAttack: Attack for adding hyperedges to hypernetworks
    RemoveHyperedgeAttack: Attack for removing hyperedges from hypernetworks
    AttackSequence: Manager for sequences of multiple attacks
    HypernetworkSimulator: High-level orchestrator for attack simulations

Example usage:
    from hiper.simulation import (
        AddNodeAttack, RemoveNodeAttack,
        AttackSequence, HypernetworkSimulator
    )
    from hiper.core.hypernetwork import Hypernetwork

    # Create a hypernetwork
    hn = Hypernetwork()
    hn.add_hyperedge(0, [1, 2, 3])

    # Create attacks
    add_attack = AddNodeAttack("add_node_4", 4)
    remove_attack = RemoveNodeAttack("remove_node_1", 1)

    # Create and execute sequence of attacks
    sequence = AttackSequence("test_sequence")
    sequence.add_attack(add_attack)
    sequence.add_attack(remove_attack)

    # Run simulation
    simulator = HypernetworkSimulator("test_sim")
    simulator.set_hypernetwork(hn)
    result = simulator.simulate_sequence(sequence)
"""

from .attack import (
    Attack,
    AddNodeAttack,
    RemoveNodeAttack,
    AddHyperedgeAttack,
    RemoveHyperedgeAttack
)

from .sequence import AttackSequence

from .simulator import HypernetworkSimulator

__all__ = [
    'Attack',
    'AddNodeAttack',
    'RemoveNodeAttack',
    'AddHyperedgeAttack',
    'RemoveHyperedgeAttack',
    'AttackSequence',
    'HypernetworkSimulator'
]

__version__ = '1.0.0'
