# -*- coding: utf-8 -*-
"""
test

Unit tests for the hiper library, organized in packages:
- test_core: tests for core hypernetwork data structures
- test_metrics: tests for resilience metrics and analysis algorithms
- test_simulation: tests for the attack simulation framework
"""

# Import main test classes from each package
from .test_core import TestHypernetwork

from .test_metrics import (
    TestHypergraphConnectivity,
    TestHyperedgeConnectivity,
    TestHypergraphDistance,
    TestRedundancyCoefficient,
    TestSwalkEfficiency,
    TestTopsisNodeRanker
)

from .test_simulation import (
    TestAttackBase,
    TestAddNodeAttack,
    TestRemoveNodeAttack,
    TestAddHyperedgeAttack,
    TestRemoveHyperedgeAttack,
    TestAttackErrorHandling,
    TestAttackSequenceInitialization,
    TestAttackSequenceManagement,
    TestSequenceExecutionControl,
    TestSequenceStateManagement,
    TestSequenceMetricsAndAnalysis,
    TestSimulatorInitialization,
    TestSingleAttackSimulation,
    TestSequenceSimulation,
    TestSimulatorBackupAndRestore,
    TestSimulatorReporting,
    TestSimulatorErrorHandling
)

__all__ = [
    # Core
    'TestHypernetwork',

    # Metrics
    'TestHypergraphConnectivity',
    'TestHyperedgeConnectivity',
    'TestHypergraphDistance',
    'TestRedundancyCoefficient',
    'TestSwalkEfficiency',
    'TestTopsisNodeRanker',

    # Simulation: attacks
    'TestAttackBase',
    'TestAddNodeAttack',
    'TestRemoveNodeAttack',
    'TestAddHyperedgeAttack',
    'TestRemoveHyperedgeAttack',
    'TestAttackErrorHandling',

    # Simulation: sequences
    'TestAttackSequenceInitialization',
    'TestAttackSequenceManagement',
    'TestSequenceExecutionControl',
    'TestSequenceStateManagement',
    'TestSequenceMetricsAndAnalysis',

    # Simulation: simulator
    'TestSimulatorInitialization',
    'TestSingleAttackSimulation',
    'TestSequenceSimulation',
    'TestSimulatorBackupAndRestore',
    'TestSimulatorReporting',
    'TestSimulatorErrorHandling'
]

__version__ = '1.0.0'
__description__ = 'Comprehensive test suite for the hiper library'
