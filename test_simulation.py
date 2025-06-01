# -*- coding: utf-8 -*-
"""
simulation_example.py

Complete example demonstrating the usage of the hiper.simulation package
for attack simulations on hypernetworks.
"""

import json

from hiper.core.hypernetwork import Hypernetwork
from hiper.simulation import (
    AddNodeAttack, RemoveNodeAttack,
    AddHyperedgeAttack, RemoveHyperedgeAttack,
    AttackSequence, HypernetworkSimulator
)


def create_sample_hypernetwork() -> Hypernetwork:
    """
    Create a sample hypernetwork for demonstration purposes.

    Returns:
        A hypernetwork with several nodes and hyperedges.
    """
    hn = Hypernetwork()

    # Add some hyperedges to create a sample network
    hn.add_hyperedge(0, [1, 2, 3])  # Triangle-like structure
    hn.add_hyperedge(1, [2, 3, 4])  # Overlapping with first edge
    hn.add_hyperedge(2, [4, 5, 6, 7])  # Larger hyperedge
    hn.add_hyperedge(3, [1, 5])  # Simple edge connecting distant nodes
    hn.add_hyperedge(4, [6, 7, 8, 9, 10])  # Another large hyperedge

    return hn


def demonstrate_single_attacks():
    """
    Demonstrate individual attack execution and analysis.
    """
    print("=== Single Attack Demonstrations ===\n")

    # Create hypernetwork and simulator
    hn = create_sample_hypernetwork()
    simulator = HypernetworkSimulator("single_attack_demo")
    simulator.set_hypernetwork(hn)

    print("Initial hypernetwork state:")
    hn.print_info()
    print()

    # Demonstrate node addition attack
    add_node_attack = AddNodeAttack("add_node_15", 15)
    result = simulator.simulate_attack(add_node_attack, restore_after=False)

    print(f"Attack: {add_node_attack.describe()}")
    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time_seconds']:.4f} seconds")
    print(f"Network order change: {result['changes']['order_change']}")
    print()

    # Demonstrate hyperedge addition attack
    add_edge_attack = AddHyperedgeAttack("add_edge_5", 5, [11, 12, 13, 14, 15])
    result = simulator.simulate_attack(add_edge_attack, restore_after=False)

    print(f"Attack: {add_edge_attack.describe()}")
    print(f"Success: {result['success']}")
    print(f"Network size change: {result['changes']['size_change']}")
    print(f"Network order change: {result['changes']['order_change']}")
    print()

    # Demonstrate node removal attack (this will affect multiple hyperedges)
    remove_node_attack = RemoveNodeAttack("remove_node_2", 2)
    result = simulator.simulate_attack(remove_node_attack, restore_after=False)

    print(f"Attack: {remove_node_attack.describe()}")
    print(f"Success: {result['success']}")
    print(f"Nodes removed: {result['attack_result']['nodes_removed']}")
    print(f"Edges affected: {result['attack_result']['edges_affected']}")
    print(f"Edges removed: {result['attack_result']['edges_removed']}")
    print()

    print("Final hypernetwork state:")
    hn.print_info()
    print("\n")


def demonstrate_attack_sequences():
    """
    Demonstrate attack sequence creation and execution.
    """
    print("=== Attack Sequence Demonstrations ===\n")

    # Create fresh hypernetwork and simulator
    hn = create_sample_hypernetwork()
    simulator = HypernetworkSimulator("sequence_demo")
    simulator.set_hypernetwork(hn)

    print("Initial hypernetwork state:")
    hn.print_info()
    print()

    # Create a complex attack sequence
    sequence = AttackSequence("complex_attack_scenario")

    # Phase 1: Add some new infrastructure
    sequence.add_attack(AddNodeAttack("add_node_20", 20))
    sequence.add_attack(AddNodeAttack("add_node_21", 21))
    sequence.add_attack(AddHyperedgeAttack("add_edge_10", 10, [20, 21, 1]))

    # Phase 2: Perform targeted removals
    sequence.add_attack(RemoveNodeAttack("remove_central_node",
                                         3))  # Node 3 is in multiple edges
    sequence.add_attack(RemoveHyperedgeAttack("remove_edge_2", 2))

    # Phase 3: Add recovery infrastructure
    sequence.add_attack(
        AddHyperedgeAttack("recovery_edge_11", 11, [1, 4, 20, 21]))
    sequence.add_attack(AddHyperedgeAttack("backup_edge_12", 12, [5, 6, 20]))

    print(f"Sequence description: {sequence.describe()}")
    print(f"Number of attacks in sequence: {sequence.size()}")
    print()

    # Execute the sequence
    result = simulator.simulate_sequence(sequence)

    print(f"Sequence execution success: {result['success']}")
    print(
        f"Total execution time: {result['execution_time_seconds']:.4f} seconds")
    print(f"Successful attacks: "
          f"{result['execution_stats']['successful_attacks']}")
    print(f"Failed attacks: {result['execution_stats']['failed_attacks']}")
    print(f"Success rate: {result['execution_stats']['success_rate']:.2%}")
    print()

    # Show detailed results for each attack
    print("Individual attack results:")
    for attack_result in result['sequence_results']:
        print(
            f"  - {attack_result['description']}: "
            f"{'SUCCESS' if attack_result['success'] else 'FAILED'}")
    print()

    # Network should be restored to original state
    print("Hypernetwork state after sequence (should be restored):")
    hn.print_info()
    print("\n")


def demonstrate_comprehensive_simulation():
    """
    Demonstrate comprehensive simulation with multiple scenarios and analysis.
    """
    print("=== Comprehensive Simulation Analysis ===\n")

    # Create hypernetwork and simulator
    hn = create_sample_hypernetwork()
    simulator = HypernetworkSimulator("comprehensive_analysis")
    simulator.set_hypernetwork(hn)

    # Scenario 1: Resilience test - Remove critical nodes
    resilience_sequence = AttackSequence("resilience_test")
    critical_nodes = [2, 3, 4]  # These nodes appear in multiple hyperedges
    for i, node_id in enumerate(critical_nodes):
        resilience_sequence.add_attack(
            RemoveNodeAttack(f"remove_critical_{i}", node_id))

    print("Scenario 1: Testing network resilience to critical node removal")
    result1 = simulator.simulate_sequence(resilience_sequence)
    print(f"Network survived: {result1['success']}")
    print(f"Order impact: {result1['changes']['order_change']} nodes")
    print(f"Size impact: {result1['changes']['size_change']} hyperedges")
    print()

    # Scenario 2: Growth simulation - Add infrastructure
    growth_sequence = AttackSequence("growth_simulation")
    new_nodes = list(range(50, 60))
    for node_id in new_nodes:
        growth_sequence.add_attack(
            AddNodeAttack(f"grow_node_{node_id}", node_id))

    # Add some new hyperedges connecting old and new infrastructure
    growth_sequence.add_attack(
        AddHyperedgeAttack("bridge_edge_1", 20, [1, 50, 51]))
    growth_sequence.add_attack(
        AddHyperedgeAttack("bridge_edge_2", 21, [5, 52, 53, 54]))
    growth_sequence.add_attack(
        AddHyperedgeAttack("new_cluster", 22, [55, 56, 57, 58, 59]))

    print("Scenario 2: Network growth simulation")
    result2 = simulator.simulate_sequence(growth_sequence)
    print(f"Growth successful: {result2['success']}")
    print(f"Order growth: {result2['changes']['order_change']} nodes")
    print(f"Size growth: {result2['changes']['size_change']} hyperedges")
    print()

    # Scenario 3: Mixed operations - Realistic network evolution
    evolution_sequence = AttackSequence("network_evolution")
    evolution_sequence.add_attack(AddNodeAttack("add_hub", 100))
    evolution_sequence.add_attack(
        AddHyperedgeAttack("central_hub", 30, [1, 2, 4, 100]))
    evolution_sequence.add_attack(RemoveHyperedgeAttack("obsolete_edge", 0))
    evolution_sequence.add_attack(
        AddHyperedgeAttack("replacement_edge", 31, [1, 3, 100]))

    print("Scenario 3: Network evolution with mixed operations")
    result3 = simulator.simulate_sequence(evolution_sequence)
    print(f"Evolution successful: {result3['success']}")
    print(
        f"Net structural change: {result3['changes']['order_change']} nodes, "
        f"{result3['changes']['size_change']} edges")
    print()

    # Generate comprehensive report
    print("=== Simulation Summary Report ===")
    report = simulator.generate_summary_report()

    print(f"Total simulations executed: {report['total_simulations']}")
    print(f"Overall success rate: {report['success_rate']:.2%}")
    print(
        f"Average execution time: "
        f"{report['timing_stats']['average_execution_time']:.4f} seconds")
    print()

    print("Attack type distribution:")
    for attack_type, count in report['attack_type_distribution'].items():
        print(f"  - {attack_type}: {count} attacks")
    print()


def save_simulation_results_to_file(simulator: HypernetworkSimulator,
                                    filename: str):
    """
    Save simulation results to a JSON file for later analysis.

    Args:
        simulator: The simulator containing results to save.
        filename: Name of the file to save results to.
    """
    report = simulator.generate_summary_report()
    history = simulator.get_simulation_history()

    data = {
        'summary_report': report,
        'simulation_history': history
    }

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Simulation results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """
    Main function demonstrating all simulation capabilities.
    """
    print("Hypernetwork Attack Simulation Demonstration")
    print("=" * 50)
    print()

    # Run demonstrations
    demonstrate_single_attacks()
    demonstrate_attack_sequences()
    demonstrate_comprehensive_simulation()


if __name__ == "__main__":
    main()
