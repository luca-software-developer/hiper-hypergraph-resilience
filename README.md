# HIPER: Hypergraph-based Investigation of Perturbation Effects and Resilience

## Overview

HIPER provides optimized data structures and algorithms for hypernetwork
analysis and attack simulation. The library implements hypernetworks using
efficient dict-of-sets representation to achieve O(1) amortized performance
for most operations.

## Key Features

The library offers comprehensive hypernetwork manipulation capabilities with
optimized performance characteristics. The core implementation supports
efficient addition and removal of nodes and hyperedges while maintaining
complete structural integrity. Advanced attack simulation functionality
enables security analysis through both individual attacks and coordinated
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

The simulation framework enables comprehensive security analysis through
individual attacks and coordinated sequences.

```python
"""Execute attack simulations for security analysis."""
from hiper import (
    HypernetworkSimulator, AddNodeAttack, RemoveNodeAttack,
    AttackSequence, Hypernetwork
)

# Create simulator and set target network
hn = Hypernetwork()
hn.add_hyperedge(0, [1, 2, 3])

simulator = HypernetworkSimulator('security_analysis')
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