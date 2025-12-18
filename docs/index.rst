HIPER: Hypergraph-based Investigation of Perturbation Effects and Resilience
=============================================================================

.. image:: https://img.shields.io/badge/version-1.0.0-blue
   :alt: Version 1.0.0

.. image:: https://img.shields.io/badge/License-MIT-blue
   :alt: MIT License

Welcome to HIPER's documentation! HIPER provides optimized data structures and
algorithms for hypernetwork analysis and attack simulation.

The library implements hypernetworks using efficient dict-of-sets representation
to achieve O(1) amortized performance for most operations.

Quick Start
-----------

.. code-block:: python

   from hiper import Hypernetwork

   # Create new hypernetwork instance
   hn = Hypernetwork()

   # Add hyperedges connecting multiple nodes
   hn.add_hyperedge(0, [1, 2, 3])
   hn.add_hyperedge(1, [2, 3, 4, 5])
   hn.add_hyperedge(2, [1, 4, 6])

   # Analyze network properties
   print(f"Network order: {hn.order()}")
   print(f"Network size: {hn.size()}")
   print(f"Average degree: {hn.avg_deg():.2f}")

Key Features
------------

* **Optimized Data Structures**: Efficient dict-of-sets representation for O(1) amortized operations
* **Attack Simulation**: Comprehensive security analysis through individual and coordinated attacks
* **Resilience Analysis**: TOPSIS-based node ranking and removal strategies
* **Modular Architecture**: Selective importing based on application requirements
* **Comprehensive Metrics**: Detailed network characterization and structural features

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   experiments
   examples

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/core
   api/datasets
   api/metrics
   api/simulation

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
