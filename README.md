# HIPER: Hypergraph-based Investigation of Perturbation Effects and Resilience

![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-blue)

## Overview

HIPER provides optimized data structures and algorithms for hypernetwork
analysis and attack simulation. The library implements hypernetworks using
efficient dict-of-sets representation to achieve O(1) amortized performance
for most operations.

## Installation

Install HIPER locally using pip:

```bash
pip install -e .
```

This installs the package in editable mode, which is ideal for development as
changes to the source code are immediately available without reinstalling.

For a standard installation:

```bash
pip install .
```

## Key Features

The library offers comprehensive hypernetwork manipulation capabilities with
optimized performance characteristics. The core implementation supports
efficient addition and removal of nodes and hyperedges while maintaining
complete structural integrity. Advanced attack simulation functionality
enables analysis through both individual attacks and coordinated
attack sequences.

The modular architecture facilitates selective importing of functionality
based on application requirements, reducing computational overhead for
specialized use cases. Comprehensive metrics and analysis functions support
detailed characterization of network properties and structural features.

## Quick Start

The following example demonstrates fundamental hypernetwork creation and
analysis operations.

```python
"""Basic hypernetwork creation and analysis operations."""
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

# Query node connectivity
neighbors = hn.get_neighbors(2)
hyperedges = hn.get_hyperedges(2)
```

## Dataset Loading

The library provides streamlined access to standard hypernetwork datasets
for research and benchmarking applications.

```python
"""Load standard datasets for hypernetwork analysis."""
from hiper import DataFile, Dataset, load_config

# Load configuration and dataset
config = load_config()
datafile = DataFile('data/Algebra.txt')
dataset = Dataset('Algebra', datafile)

# Access constructed hypernetwork
hypernetwork = dataset.get_hypernetwork()
hypernetwork.print_info()
```

## Attack Simulation

The simulation framework enables comprehensive analysis through individual
attacks and coordinated sequences.

```python
"""Execute attack simulations for analysis."""
from hiper import (
    HypernetworkSimulator, AddNodeAttack, RemoveNodeAttack,
    AttackSequence, Hypernetwork
)

# Create simulator and set target network
hn = Hypernetwork()
hn.add_hyperedge(0, [1, 2, 3])

simulator = HypernetworkSimulator('analysis')
simulator.set_hypernetwork(hn)

# Execute individual attack with restoration
attack = RemoveNodeAttack('remove_critical_node', 2)
result = simulator.simulate_attack(attack)

print(f"Attack success: {result['success']}")
print(f"Network impact: {result['changes']['order_change']} nodes")

# Create coordinated attack sequence
sequence = AttackSequence('coordinated_attack')
sequence.add_attack(AddNodeAttack('add_decoy', 10))
sequence.add_attack(RemoveNodeAttack('remove_target', 1))

# Execute sequence with detailed analysis
sequence_result = simulator.simulate_sequence(sequence)
execution_stats = sequence_result['execution_stats']
print(f"Success rate: {execution_stats['success_rate']:.2%}")
```

## Running Experiments

The library provides several example scripts in the `examples/` directory for
comprehensive hypernetwork analysis. Each script performs specific experiments
and saves results in dedicated output directories.

For detailed documentation of all examples, see [
`examples/README.md`](examples/README.md).

### Resilience Experiments

Test hypernetwork resilience by removing nodes using different strategies:

```bash
python examples/run_resilience_experiments.py [optional_dataset_path]
```

**What it does:**

- Removes nodes at percentages: 1%, 2%, 5%, 10%, 25%
- Uses three strategies: Random, TOPSIS Top (critical), TOPSIS Bottom (
  peripheral)
- Supports multiple MCDM methods: TOPSIS, WSM, MOORA
- Computes metrics: connectivity, redundancy coefficient, s-walk efficiency

**Results location:** `resilience_results/`

- `resilience_analysis.png` - Metric degradation plots
- `impact_comparison.png` - Strategy comparison
- `resilience_summary.csv` - Detailed numeric results

### Perturbation Analysis

Analyze single and multiple perturbations with targeted attacks:

```bash
python examples/run_perturbation_analysis.py
```

**What it does:**

- Single perturbations: 1%, 2%, 5%, 10% node removal
- Multiple perturbations: Attack sequences with k ∈ {2, 5, 10, 25, 50, 100}
- Compares Random vs TOPSIS targeting strategies
- Analyzes component fragmentation and largest component evolution

**Results location:** `results/`

- `{dataset}_single_comparison.png` - Single perturbation plots
- `{dataset}_multiple_timeline.png` - Evolution over attack sequences
- `{dataset}_component_analysis.png` - Fragmentation analysis
- `{dataset}_largest_components.png` - Component metrics
- `perturbation_results_{timestamp}.json` - Complete experimental data
- `analysis_summary.json` - Executive summary

### Comprehensive Node and Hyperedge Experiments

Run experiments on both node and hyperedge removal:

```bash
python examples/run_node_hyperedge_experiments.py
```

**What it does:**

- Tests removal of both nodes and hyperedges
- Computes traditional metrics (connectivity, redundancy)
- Computes higher-order cohesion metrics (HOCR_m, LHC_m)
- Removal percentages: 1%, 2%, 5%, 10%, 25%

**Results location:** `resilience_results/plots/`

- `node_removal_traditional_metrics.png` - Node removal analysis
- `hyperedge_removal_traditional_metrics.png` - Hyperedge removal analysis
- `higher_order_cohesion_comparison.png` - Advanced metrics
- `strategy_effectiveness_heatmap.png` - Comparative effectiveness

### MCDM Methods Comparison

Compare different Multi-Criteria Decision Making methods for node selection:

```bash
python examples/compare_selection_methods.py
```

**What it does:**

- Compares three MCDM methods: TOPSIS, WSM (Weighted Sum Model), MOORA
- Tests targeted node removal using each method
- Removal percentages: 5%, 10%, 25%
- Analyzes whether simpler methods (WSM, MOORA) achieve comparable results to
  TOPSIS

**Results location:** `comparison_results/`

- `methods_comparison.png` - Side-by-side comparison of all three methods
- `difference_from_topsis.png` - Percentage difference from TOPSIS baseline
- `comparison_results.json` - Complete numerical results

### Statistical Analysis

Perform cross-domain statistical analysis on hypernetwork features:

```bash
python examples/run_statistical_analysis.py
```

**What it does:**

- Computes structural features for all datasets in `data/` directory
- Performs ANOVA/Kruskal-Wallis tests across hypergraph families
- Correlation analysis between features and resilience metrics
- Normalized metrics for size-independent comparisons

**Results location:** `statistical_analysis_results/`

- `data_directory_features_by_family.png` - Feature distributions by family
- `structural_correlations_heatmap.png` - Feature correlation matrix
- `structural_features_scatter.png` - Relationship visualizations
- `significant_correlations.png` - Feature-resilience correlations
- `statistical_analysis_summary.csv` - Complete statistical results
- `data_directory_features.csv` - Computed features for all datasets

### Working with Custom Datasets

To run experiments on your own hypernetwork data:

1. **Prepare your dataset**: Create a text file where each line represents a
   hyperedge with space-separated node IDs:
   ```
   1 2 3
   2 3 4 5
   3 4 6
   ```

2. **Place in data directory**: Save the file in the `data/` folder

3. **Run experiments**: Execute any of the experiment scripts above

4. **View results**: Check the corresponding results directories for plots
   and data files

### Example Workflow

Complete analysis workflow for a new dataset:

```bash
# 1. Run perturbation analysis
python examples/run_perturbation_analysis.py

# 2. Run resilience experiments
python examples/run_resilience_experiments.py data/your_dataset.txt

# 3. Run comprehensive analysis
python examples/run_node_hyperedge_experiments.py

# 4. Perform statistical analysis
python examples/run_statistical_analysis.py
```

Results will be organized in:

- `results/` - Perturbation analysis outputs
- `resilience_results/` - Resilience experiment outputs
- `statistical_analysis_results/` - Statistical analysis outputs

## API Documentation

Complete API documentation is available and built with **Sphinx**. The
documentation
includes:

- Complete class and function references with type hints
- Method signatures and parameters
- Detailed docstrings from the source code
- Module hierarchies and dependencies
- Code examples and tutorials
- Searchable interface

### Viewing the Documentation

To view the pre-built documentation, open `docs/_build/html/index.html` in your
web browser:

```bash
# Windows
start docs/_build/html/index.html

# macOS
open docs/_build/html/index.html

# Linux
xdg-open docs/_build/html/index.html
```

### Building the Documentation

If you make changes to the code or documentation, rebuild it locally:

**Option 1: Using sphinx-build directly:**

```bash
cd docs
sphinx-build -b html . _build/html
```

**Option 2: Using make:**

```bash
cd docs
make html  # Unix/macOS
make.bat html  # Windows
```

### Documentation Structure

- **User Guide**: Getting started, experiments, and examples
- **API Reference**: Complete module, class, and function documentation
    - Core modules (Hypernetwork, Node, Hyperedge)
    - Dataset management (loading and configuration)
    - Metrics (experiments, TOPSIS, connectivity, distance, etc.)
    - Simulation framework (simulator, attacks, sequences)
- **License**: Project license information

## Configuration

The library supports flexible configuration through JSON-based configuration
files that specify dataset paths and simulation parameters.

```json
{
  "dataset_base_path": "data/",
  "dataset_name": "Algebra.txt"
}
```

## License

This project is licensed under the MIT License, enabling both academic and
commercial use while ensuring appropriate attribution to the development team.
See the [LICENSE](LICENSE) file for complete terms and conditions.
