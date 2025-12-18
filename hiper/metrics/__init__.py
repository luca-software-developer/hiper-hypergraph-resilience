# -*- coding: utf-8 -*-
"""
metrics package

Provides resilience metrics for hypergraphs, including connectivity measures,
redundancy coefficients, efficiency metrics, structural integrity measures,
and comprehensive resilience analysis frameworks.

The metrics package implements:
- Hypergraph connectivity (κ(H))
- Hyperedge connectivity (λ(H))
- Redundancy coefficient (ρ(H))
- Average s-walk efficiency (Es(H))
- Hyperedge integrity (HI)
- Average hyperedge cardinality (AHC)
- Hyperedge fragmentation index (HFI)
- Centrality disruption index (CDI)
- Entropy loss (EL)
- Higher-order cohesion resilience (HOCR_m)
- Largest higher-order component resilience (LHC_m)
- Comprehensive resilience experiments with TOPSIS-based strategies

These metrics enable comprehensive analysis of hypergraph resilience under
various attack scenarios including node and hyperedge removal with both
random and targeted strategies.

Classes:
    HypergraphConnectivity: Computes minimum nodes to disconnect hypergraph
    HyperedgeConnectivity: Computes minimum hyperedges to disconnect graph
    RedundancyCoefficient: Measures normalized overlap between hyperedges
    SwalkEfficiency: Computes average efficiency of s-walk distances
    HyperedgeIntegrity: Measures fraction of hyperedges preserved
    AverageHyperedgeCardinality: Computes average size of hyperedges
    HyperedgeFragmentationIndex: Measures hyperedge fragmentation
    CentralityDisruptionIndex: Measures centrality distribution changes
    EntropyLoss: Measures entropy changes in distributions
    HigherOrderCohesionMetrics: Computes HOCR_m and LHC_m metrics
    HyperedgeTopsisRanker: TOPSIS-based ranking for hyperedges
    ComprehensiveResilienceExperiment: Complete experimental framework
    TopsisNodeRanker: TOPSIS-based ranking for nodes
    ResilienceExperiment: Basic resilience experiment framework

Example usage:
    from hiper.core import Hypernetwork
    from hiper.metrics import (
        HypergraphConnectivity,
        RedundancyCoefficient,
        SwalkEfficiency,
        HigherOrderCohesionMetrics,
        HyperedgeTopsisRanker,
        ComprehensiveResilienceExperiment
    )

    # Create hypergraph
    hn = Hypernetwork()
    hn.add_hyperedge(0, [1, 2, 3])
    hn.add_hyperedge(1, [2, 3, 4])

    # Compute traditional metrics
    kappa = HypergraphConnectivity().compute(hn)
    rho = RedundancyCoefficient().compute(hn)
    efficiency = SwalkEfficiency(s=1).compute(hn)

    # Compute higher-order cohesion metrics
    ho_metrics = HigherOrderCohesionMetrics(m=2)
    components = ho_metrics.compute_mth_order_components(hn)

    # Create perturbed version for comparison
    hn_perturbed = Hypernetwork()
    hn_perturbed.add_hyperedge(0, [1, 2])
    hn_perturbed.add_hyperedge(1, [2, 4])

    hocr = ho_metrics.compute_hocr_m(hn, hn_perturbed)
    lhc = ho_metrics.compute_lhc_m(hn, hn_perturbed)

    # TOPSIS ranking for targeted attacks
    hyperedge_ranker = HyperedgeTopsisRanker()
    ranked_hyperedges = hyperedge_ranker.rank_hyperedges(hn)
"""

from .average_hyperedge_cardinality import AverageHyperedgeCardinality
from .centrality_disruption import CentralityDisruptionIndex
from .comprehensive_resilience import ComprehensiveResilienceExperiment
from .connectivity import HypergraphConnectivity, HyperedgeConnectivity
from .distance import HypergraphDistance
from .entropy_loss import EntropyLoss
from .experiments import ResilienceExperiment
from .higher_order_cohesion import HigherOrderCohesionMetrics
from .hyperedge_fragmentation import HyperedgeFragmentationIndex
from .hyperedge_integrity import HyperedgeIntegrity
from .hyperedge_topsis import HyperedgeTopsisRanker
from .moora import MOORANodeRanker
from .redundancy import RedundancyCoefficient
from .swalk import SwalkEfficiency
from .topsis import TopsisNodeRanker
from .wsm import WSMNodeRanker

__all__ = [
    'HypergraphConnectivity',
    'HyperedgeConnectivity',
    'HypergraphDistance',
    'RedundancyCoefficient',
    'SwalkEfficiency',
    'TopsisNodeRanker',
    'WSMNodeRanker',
    'MOORANodeRanker',
    'ResilienceExperiment',
    'HyperedgeIntegrity',
    'CentralityDisruptionIndex',
    'HyperedgeFragmentationIndex',
    'EntropyLoss',
    'AverageHyperedgeCardinality',
    'HigherOrderCohesionMetrics',
    'HyperedgeTopsisRanker',
    'ComprehensiveResilienceExperiment'
]

__version__ = '1.0.0'
