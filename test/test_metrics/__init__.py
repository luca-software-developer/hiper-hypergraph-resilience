# -*- coding: utf-8 -*-
"""
test_metrics

Unit tests for hypergraph resilience metrics and analysis algorithms.
This package provides comprehensive validation of the metrics framework
including connectivity measures, redundancy coefficients, efficiency
calculations, multi-criteria node ranking systems, perturbation
resilience metrics, higher-order cohesion analysis, and comprehensive
experimental frameworks.

The test suite validates implementations of resilience metrics including
hypergraph connectivity, hyperedge connectivity, redundancy coefficients,
s-walk efficiency measures, perturbation-based metrics for analyzing
hypergraph integrity, fragmentation, centrality disruption, entropy loss,
average hyperedge cardinality, higher-order cohesion metrics, TOPSIS-based
ranking systems, and comprehensive resilience experiment frameworks.

Test Modules:
    test_connectivity: Hypergraph and hyperedge connectivity metrics
    test_redundancy: Redundancy coefficient computation and analysis
    test_swalk: s-walk efficiency and distance-based metrics
    test_distance: Hypergraph distance computation and path finding
    test_topsis: TOPSIS multi-criteria node ranking algorithms
    test_hyperedge_integrity: Hyperedge preservation after perturbations
    test_centrality_disruption: Centrality distribution disruption analysis
    test_hyperedge_fragmentation: Hyperedge fragmentation measurement
    test_entropy_loss: Entropy-based diversity loss assessment
    test_average_hyperedge_cardinality: Average hyperedge size analysis
    test_higher_order_cohesion: Higher-order cohesion metrics validation
    test_hyperedge_topsis: TOPSIS-based hyperedge ranking validation
    test_comprehensive_resilience: Comprehensive experiment framework testing
"""

from .test_connectivity import TestHypergraphConnectivity, \
    TestHyperedgeConnectivity
from .test_distance import TestHypergraphDistance
from .test_redundancy import TestRedundancyCoefficient
from .test_swalk import TestSwalkEfficiency
from .test_topsis import TestTopsisNodeRanker
from .test_hyperedge_integrity import TestHyperedgeIntegrity
from .test_centrality_disruption import TestCentralityDisruptionIndex
from .test_hyperedge_fragmentation import TestHyperedgeFragmentationIndex
from .test_entropy_loss import TestEntropyLoss
from .test_average_hyperedge_cardinality import TestAverageHyperedgeCardinality
from .test_higher_order_cohesion import TestHigherOrderCohesionMetrics
from .test_hyperedge_topsis import TestHyperedgeTopsisRanker
from .test_comprehensive_resilience import TestComprehensiveResilienceExperiment

__all__ = [
    'TestHypergraphConnectivity',
    'TestHyperedgeConnectivity',
    'TestRedundancyCoefficient',
    'TestSwalkEfficiency',
    'TestHypergraphDistance',
    'TestTopsisNodeRanker',
    'TestHyperedgeIntegrity',
    'TestCentralityDisruptionIndex',
    'TestHyperedgeFragmentationIndex',
    'TestEntropyLoss',
    'TestAverageHyperedgeCardinality',
    'TestHigherOrderCohesionMetrics',
    'TestHyperedgeTopsisRanker',
    'TestComprehensiveResilienceExperiment'
]

__version__ = '1.0.0'
__description__ = 'Hypergraph resilience metrics test suite'
