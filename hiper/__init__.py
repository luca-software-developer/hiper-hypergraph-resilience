# -*- coding: utf-8 -*-
"""
hiper

A high-performance Python library for hypernetwork analysis and simulation.
This package provides comprehensive tools for creating, manipulating, analyzing,
and simulating attacks on hypernetworks with optimized data structures and
algorithms designed for computational efficiency.

The hiper library implements hypernetworks using a dual dict-of-sets
representation that enables O(1) amortized performance for most operations while
maintaining full flexibility for complex network analysis tasks.
The library supports both programmatic construction of hypernetworks and loading
standard dataset formats used in research and industry applications.

Key Features:
- High-performance hypernetwork optimized for large-scale analysis
- Comprehensive attack simulation framework for security and resilience testing
- Standard dataset loading and management capabilities
- Extensive metrics and analysis functions for network characterization
- Flexible configuration system for different computational environments

Package Structure:
    core: Fundamental hypernetwork data structures and algorithms
    datasets: Dataset loading, parsing, and management functionality
    simulation: Attack simulation framework for security analysis
    config: Configuration management for flexible deployment

Example usage:
    import hiper

    # Create and populate a hypernetwork
    hn = hiper.Hypernetwork()
    hn.add_hyperedge(0, [1, 2, 3])
    hn.add_hyperedge(1, [2, 3, 4])

    # Load data from external sources
    config = hiper.load_config()
    datafile = hiper.DataFile('path/to/dataset.txt')
    dataset = hiper.Dataset('Research Data', datafile)

    # Perform attack simulations
    simulator = hiper.HypernetworkSimulator('security_analysis')
    simulator.set_hypernetwork(dataset.get_hypernetwork())

    # Create and execute attack scenarios
    attack = hiper.RemoveNodeAttack('remove_critical_node', 5)
    result = simulator.simulate_attack(attack)

The library is designed to support both research applications requiring detailed
analysis capabilities and production systems needing robust performance under
high computational loads.
"""

# Configuration management
from .config import load_config
# Core hypernetwork functionality
from .core.hypernetwork import Hypernetwork
# Dataset management
from .datasets.constants import DATASETS
from .datasets.datafile import DataFile
from .datasets.dataset import Dataset
# Attack simulation framework
from .simulation import (
    Attack,
    AddNodeAttack,
    RemoveNodeAttack,
    AddHyperedgeAttack,
    RemoveHyperedgeAttack,
    AttackSequence,
    HypernetworkSimulator
)

__all__ = [
    # Core classes
    'Hypernetwork',

    # Dataset management
    'DataFile',
    'Dataset',
    'DATASETS',

    # Attack simulation
    'Attack',
    'AddNodeAttack',
    'RemoveNodeAttack',
    'AddHyperedgeAttack',
    'RemoveHyperedgeAttack',
    'AttackSequence',
    'HypernetworkSimulator',

    # Configuration
    'load_config'
]

__version__ = '1.0.0'
__description__ = 'High-performance hypernetwork analysis & simulation library'
__url__ \
    = 'https://github.com/luca-software-developer/hiper-hypergraph-resilience'

# Package metadata for distribution
__license__ = 'MIT'

# Supported Python versions
__python_requires__ = '>=3.8'

# Key dependencies for reference
__dependencies__ = [
    'typing',
    'json',
    'pathlib',
    'time'
]
