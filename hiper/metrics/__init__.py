# -*- coding: utf-8 -*-
"""
metrics package

Provides resilience metrics for hypergraphs, including connectivity measures,
redundancy coefficients, efficiency metrics, and structural integrity measures.

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

These metrics enable comprehensive analysis of hypergraph resilience under
various attack scenarios including node and hyperedge removal.

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

Example usage:
    from hiper.core import Hypernetwork
    from hiper.metrics import (
        HypergraphConnectivity,
        RedundancyCoefficient,
        SwalkEfficiency,
        HyperedgeIntegrity,
        AverageHyperedgeCardinality,
        HyperedgeFragmentationIndex,
        CentralityDisruptionIndex,
        EntropyLoss
    )

    # Create hypergraph
    hn = Hypernetwork()
    hn.add_hyperedge(0, [1, 2, 3])
    hn.add_hyperedge(1, [2, 3, 4])

    # Compute metrics
    kappa = HypergraphConnectivity().compute(hn)
    rho = RedundancyCoefficient().compute(hn)
    efficiency = SwalkEfficiency(s=1).compute(hn)
    hi = HyperedgeIntegrity()
    ahc = AverageHyperedgeCardinality()
    hfi = HyperedgeFragmentationIndex()

    # Create a perturbed version for comparison
    hn_perturbed = Hypernetwork()
    hn_perturbed.add_hyperedge(0, [1, 2])  # Missing node 3
    hn_perturbed.add_hyperedge(1, [2, 4])  # Missing node 3

    # Compute integrity and fragmentation metrics
    integrity = hi.compute(hn_perturbed, hn)
    cardinality = ahc.compute(hn_perturbed)
    fragmentation = hfi.compute(hn_perturbed, hn)

    # Compute disruption metrics
    cdi = CentralityDisruptionIndex(centrality_type='degree')
    disruption = cdi.compute(hn, hn_perturbed)

    el = EntropyLoss(distribution_type='node_degree')
    entropy_loss = el.compute(hn, hn_perturbed)
"""

from .connectivity import HypergraphConnectivity, HyperedgeConnectivity
from .distance import HypergraphDistance
from .experiments import ResilienceExperiment
from .redundancy import RedundancyCoefficient
from .swalk import SwalkEfficiency
from .topsis import TopsisNodeRanker
from .hyperedge_integrity import HyperedgeIntegrity
from .average_hyperedge_cardinality import AverageHyperedgeCardinality
from .hyperedge_fragmentation import HyperedgeFragmentationIndex
from .centrality_disruption import CentralityDisruptionIndex
from .entropy_loss import EntropyLoss

__all__ = [
    'HypergraphConnectivity',
    'HyperedgeConnectivity',
    'RedundancyCoefficient',
    'SwalkEfficiency',
    'HypergraphDistance',
    'TopsisNodeRanker',
    'ResilienceExperiment',
    'HyperedgeIntegrity',
    'AverageHyperedgeCardinality',
    'HyperedgeFragmentationIndex',
    'CentralityDisruptionIndex',
    'EntropyLoss'
]

__version__ = '1.0.0'
