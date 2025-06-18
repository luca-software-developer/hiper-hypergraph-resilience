# -*- coding: utf-8 -*-
"""
test_metrics

Unit tests for hypergraph resilience metrics and analysis algorithms.
This package provides comprehensive validation of the metrics framework
including connectivity measures, redundancy coefficients, efficiency
calculations, and multi-criteria node ranking systems based on formal
mathematical definitions.

The test suite validates implementations of resilience metrics including
hypergraph connectivity, hyperedge connectivity, redundancy coefficients,
and s-walk efficiency measures. Additional testing covers TOPSIS-based node
ranking and experimental frameworks for comprehensive resilience analysis.

Test Modules:
    test_connectivity: Hypergraph and hyperedge connectivity metrics
    test_redundancy: Redundancy coefficient computation and analysis
    test_swalk: s-walk efficiency and distance-based metrics
    test_distance: Hypergraph distance computation and path finding
    test_topsis: TOPSIS multi-criteria node ranking algorithms
"""

from .test_connectivity import TestHypergraphConnectivity, \
    TestHyperedgeConnectivity
from .test_distance import TestHypergraphDistance
from .test_redundancy import TestRedundancyCoefficient
from .test_swalk import TestSwalkEfficiency
from .test_topsis import TestTopsisNodeRanker

__all__ = [
    'TestHypergraphConnectivity',
    'TestHyperedgeConnectivity',
    'TestRedundancyCoefficient',
    'TestSwalkEfficiency',
    'TestHypergraphDistance',
    'TestTopsisNodeRanker'
]

__version__ = '1.0.0'
__description__ = 'Hypergraph resilience metrics test suite'
