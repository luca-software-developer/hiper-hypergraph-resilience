# -*- coding: utf-8 -*-
"""
metrics package

Provides resilience metrics for hypergraphs, including connectivity measures,
redundancy coefficients, and efficiency metrics.

The metrics package implements:
- Hypergraph connectivity (κ(H))
- Hyperedge connectivity (λ(H))
- Redundancy coefficient (ρ(H))
- Average s-walk efficiency (Es(H))

These metrics enable comprehensive analysis of hypergraph resilience under
various attack scenarios including node and hyperedge removal.

Classes:
    HypergraphConnectivity: Computes minimum nodes to disconnect hypergraph
    HyperedgeConnectivity: Computes minimum hyperedges to disconnect graph
    RedundancyCoefficient: Measures normalized overlap between hyperedges
    SwalkEfficiency: Computes average efficiency of s-walk distances

Example usage:
    from hiper.core import Hypernetwork
    from hiper.metrics import (
        HypergraphConnectivity,
        RedundancyCoefficient,
        SwalkEfficiency
    )

    # Create hypergraph
    hn = Hypernetwork()
    hn.add_hyperedge(0, [1, 2, 3])
    hn.add_hyperedge(1, [2, 3, 4])

    # Compute metrics
    kappa = HypergraphConnectivity().compute(hn)
    rho = RedundancyCoefficient().compute(hn)
    efficiency = SwalkEfficiency(s=1).compute(hn)
"""

from .connectivity import HypergraphConnectivity, HyperedgeConnectivity
from .distance import HypergraphDistance
from .experiments import ResilienceExperiment
from .redundancy import RedundancyCoefficient
from .swalk import SwalkEfficiency
from .topsis import TopsisNodeRanker

__all__ = [
    'HypergraphConnectivity',
    'HyperedgeConnectivity',
    'RedundancyCoefficient',
    'SwalkEfficiency',
    'HypergraphDistance',
    'TopsisNodeRanker',
    'ResilienceExperiment'
]

__version__ = '1.0.0'
