# -*- coding: utf-8 -*-
"""
test_simulation

Unit tests for the attack simulation framework and security analysis tools.
This package validates the comprehensive attack simulation capabilities
including individual attacks, attack sequences, and simulation management
for evaluating hypergraph resilience under various adversarial scenarios.

The test suite covers attack definition and execution, sequence orchestration,
simulation result analysis, and performance monitoring for security-focused
resilience evaluation. Testing ensures reliable execution of attack scenarios
and accurate measurement of their impact on hypergraph structural properties.

Test Modules:
    test_attack: Individual attack implementation and execution
    test_sequence: Attack sequence management and orchestration
    test_simulator: High-level simulation framework and coordination
    test_attack_types: Specific attack type implementations and validation
"""

from .test_attack import (
    TestAttackBase,
    TestAddNodeAttack,
    TestRemoveNodeAttack,
    TestAddHyperedgeAttack,
    TestRemoveHyperedgeAttack,
    TestAttackErrorHandling
)
from .test_sequence import (
    TestAttackSequenceInitialization,
    TestAttackSequenceManagement,
    TestSequenceExecutionControl,
    TestSequenceStateManagement,
    TestSequenceMetricsAndAnalysis
)
from .test_simulator import (
    TestSimulatorInitialization,
    TestSingleAttackSimulation,
    TestSequenceSimulation,
    TestSimulatorBackupAndRestore,
    TestSimulatorReporting,
    TestSimulatorErrorHandling
)

__all__ = [
    # Attack test classes
    'TestAttackBase',
    'TestAddNodeAttack',
    'TestRemoveNodeAttack',
    'TestAddHyperedgeAttack',
    'TestRemoveHyperedgeAttack',
    'TestAttackErrorHandling',

    # Sequence test classes
    'TestAttackSequenceInitialization',
    'TestAttackSequenceManagement',
    'TestSequenceExecutionControl',
    'TestSequenceStateManagement',
    'TestSequenceMetricsAndAnalysis',

    # Simulator test classes
    'TestSimulatorInitialization',
    'TestSingleAttackSimulation',
    'TestSequenceSimulation',
    'TestSimulatorBackupAndRestore',
    'TestSimulatorReporting',
    'TestSimulatorErrorHandling'
]

__version__ = '1.0.0'
__description__ = 'Attack simulation framework test suite'
