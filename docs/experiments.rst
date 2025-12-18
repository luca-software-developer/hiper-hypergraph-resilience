Running Experiments
===================

The library provides several experimental scripts for comprehensive hypernetwork
analysis. Each script performs specific experiments and saves results in
dedicated output directories.

Resilience Experiments
----------------------

Test hypernetwork resilience by removing nodes using different strategies.

**Command:**

.. code-block:: bash

   python examples/run_resilience_experiments.py [optional_dataset_path]

**What it does:**

* Removes nodes at percentages: 1%, 2%, 5%, 10%, 25%
* Uses three strategies: Random, TOPSIS Top (critical), TOPSIS Bottom (peripheral)
* Computes metrics: connectivity, redundancy coefficient, s-walk efficiency

**Results location:** ``resilience_results/``

* ``resilience_analysis.png`` - Metric degradation plots
* ``impact_comparison.png`` - Strategy comparison
* ``resilience_summary.csv`` - Detailed numeric results

Perturbation Analysis
---------------------

Analyze single and multiple perturbations with targeted attacks.

**Command:**

.. code-block:: bash

   python examples/run_perturbation_analysis.py

**What it does:**

* Single perturbations: 1%, 2%, 5%, 10% node removal
* Multiple perturbations: Attack sequences with k ∈ {2, 5, 10, 25, 50, 100}
* Compares Random vs TOPSIS targeting strategies
* Analyzes component fragmentation and largest component evolution

**Results location:** ``results/``

* ``{dataset}_single_comparison.png`` - Single perturbation plots
* ``{dataset}_multiple_timeline.png`` - Evolution over attack sequences
* ``{dataset}_component_analysis.png`` - Fragmentation analysis
* ``{dataset}_largest_components.png`` - Component metrics
* ``perturbation_results_{timestamp}.json`` - Complete experimental data
* ``analysis_summary.json`` - Executive summary

Comprehensive Node and Hyperedge Experiments
---------------------------------------------

Run experiments on both node and hyperedge removal.

**Command:**

.. code-block:: bash

   python examples/run_node_hyperedge_experiments.py

**What it does:**

* Tests removal of both nodes and hyperedges
* Computes traditional metrics (connectivity, redundancy)
* Computes higher-order cohesion metrics (HOCR_m, LHC_m)
* Removal percentages: 1%, 2%, 5%, 10%, 25%

**Results location:** ``resilience_results/plots/``

* ``node_removal_traditional_metrics.png`` - Node removal analysis
* ``hyperedge_removal_traditional_metrics.png`` - Hyperedge removal analysis
* ``higher_order_cohesion_comparison.png`` - Advanced metrics
* ``strategy_effectiveness_heatmap.png`` - Comparative effectiveness

MCDM Methods Comparison
-----------------------

Compare different Multi-Criteria Decision Making methods for node selection.

**Command:**

.. code-block:: bash

   python examples/compare_selection_methods.py [dataset_name]
   # Or combine all results:
   python examples/compare_selection_methods.py --combine

**What it does:**

* Compares three MCDM methods: TOPSIS, WSM (Weighted Sum Model), MOORA
* Tests targeted node removal using each method
* Removal percentages: 5%, 10%, 25%
* Analyzes whether simpler methods (WSM, MOORA) achieve comparable results to TOPSIS

**Results location:** ``comparison_results/``

* ``methods_comparison_all_datasets.png`` - Side-by-side comparison of all three methods
* ``results_<dataset>.json`` - Results for individual datasets
* ``comparison_results_all_datasets.json`` - Complete numerical results
* ``<dataset>_comparison.png`` - Individual dataset plots

Statistical Analysis
--------------------

Perform cross-domain statistical analysis on hypernetwork features.

**Command:**

.. code-block:: bash

   python examples/run_statistical_analysis.py

**What it does:**

* Computes structural features for all datasets in ``data/`` directory
* Performs ANOVA/Kruskal-Wallis tests across hypergraph families
* Correlation analysis between features and resilience metrics
* Normalized metrics for size-independent comparisons

**Results location:** ``statistical_analysis_results/``

* ``data_directory_features_by_family.png`` - Feature distributions by family
* ``structural_correlations_heatmap.png`` - Feature correlation matrix
* ``structural_features_scatter.png`` - Relationship visualizations
* ``significant_correlations.png`` - Feature-resilience correlations
* ``statistical_analysis_summary.csv`` - Complete statistical results
* ``data_directory_features.csv`` - Computed features for all datasets

Working with Custom Datasets
-----------------------------

To run experiments on your own hypernetwork data:

1. **Prepare your dataset**

   Create a text file where each line represents a hyperedge with
   space-separated node IDs:

   .. code-block:: text

      1 2 3
      2 3 4 5
      3 4 6

2. **Place in data directory**

   Save the file in the ``data/`` folder

3. **Run experiments**

   Execute any of the experiment scripts above

4. **View results**

   Check the corresponding results directories for plots and data files

Example Workflow
----------------

Complete analysis workflow for a new dataset:

.. code-block:: bash

   # 1. Run perturbation analysis
   python examples/run_perturbation_analysis.py

   # 2. Run resilience experiments
   python examples/run_resilience_experiments.py data/your_dataset.txt

   # 3. Run comprehensive analysis
   python examples/run_node_hyperedge_experiments.py

   # 4. Perform statistical analysis
   python examples/run_statistical_analysis.py

Results will be organized in:

* ``results/`` - Perturbation analysis outputs
* ``resilience_results/`` - Resilience experiment outputs
* ``statistical_analysis_results/`` - Statistical analysis outputs
