# -*- coding: utf-8 -*-
"""
core package

Provides the fundamental data structures and algorithms for hypernetwork
representation and manipulation. This package contains the core implementation
of hypernetworks using an optimized dict-of-sets representation for high
performance operations.

The core package implements a dual mapping structure where nodes maintain
references to their incident hyperedges and hyperedges maintain references
to their constituent nodes. This design enables efficient O(1) amortized
operations for most common hypernetwork manipulations.

Classes:
    Hypernetwork: Main hypernetwork data structure with optimized operations
    Node: Represents individual vertices in the hypernetwork
    Hyperedge: Represents hyperedges connecting multiple nodes
    Incidence: Links nodes and hyperedges in doubly-linked list structures

The Hypernetwork class serves as the primary interface for creating and
manipulating hypernetworks, providing methods for adding and removing nodes
and hyperedges, computing network metrics, and analyzing structural properties.

Example usage:
    from hiper.core import Hypernetwork

    # Create a new hypernetwork
    hn = Hypernetwork()

    # Add hyperedges connecting multiple nodes
    hn.add_hyperedge(0, [1, 2, 3])
    hn.add_hyperedge(1, [2, 3, 4, 5])

    # Query network properties
    print(f'Network order: {hn.order()}')
    print(f'Network size: {hn.size()}')
    print(f'Average degree: {hn.avg_deg():.2f}')

    # Analyze node connectivity
    neighbors = hn.get_neighbors(2)
    hyperedges = hn.get_hyperedges(2)
"""

from .hyperedge import Hyperedge
from .hypernetwork import Hypernetwork
from .incidence import Incidence
from .node import Node

__all__ = [
    'Hypernetwork',
    'Node',
    'Hyperedge',
    'Incidence'
]

__version__ = '1.0.0'
