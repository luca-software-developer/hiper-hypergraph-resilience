# -*- coding: utf-8 -*-
"""
test_core

Unit tests for the core hypergraph data structures and fundamental algorithms.
This package validates the foundational components of the hiper library
including the Hypernetwork class, Node and Hyperedge representations, and core
operations that enable high-performance hypergraph manipulation and analysis.

The test suite covers essential functionality including node and hyperedge
management, incidence operations, query methods, and performance-critical
algorithms that form the foundation for all higher-level analysis capabilities.

Test Modules:
    test_hypernetwork: Comprehensive tests for the main Hypernetwork class
"""

from .test_hypernetwork import TestHypernetwork

__all__ = [
    'TestHypernetwork'
]

__version__ = '1.0.0'
__description__ = 'Core hypergraph data structure test suite'
