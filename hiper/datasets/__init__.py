# -*- coding: utf-8 -*-
"""
datasets package

Provides functionality for loading, parsing, and managing hypernetwork datasets
from various file formats. This package handles the conversion of external data
sources into hypernetwork representations suitable for analysis and computation.

The datasets package supports standard hypernetwork data formats where each line
in a dataset file represents one hyperedge by listing the node IDs that comprise
it. The package automatically handles data validation, type conversion, and the
construction of optimized hypernetwork objects from the parsed data.

Classes:
    DataFile: Parses hypernetwork data files with robust error handling
    Dataset: Constructs Hypernetwork objects from parsed data files

Constants:
    DATASETS: List of standard benchmark datasets available for testing

The DataFile class provides low-level file parsing with support for comments,
empty lines, and flexible whitespace handling. The Dataset class builds upon
this foundation to create complete hypernetwork objects ready for analysis.

Example usage:
    from hiper.datasets import DataFile, Dataset, DATASETS

    # Load a dataset from file
    datafile = DataFile('path/to/dataset.txt')
    dataset = Dataset('MyDataset', datafile)

    # Access the constructed hypergraph
    hypernetwork = dataset.get_hypernetwork()
    hypernetwork.print_info()

    # Work with standard benchmark datasets
    for dataset_name in DATASETS:
        print(f'Available dataset: {dataset_name}')

The package integrates seamlessly with the configuration system to support
flexible dataset path management and standardized benchmarking workflows
across different computational environments.
"""

from .constants import DATASETS
from .datafile import DataFile
from .dataset import Dataset

__all__ = [
    'DataFile',
    'Dataset',
    'DATASETS'
]

__version__ = '1.0.0'
