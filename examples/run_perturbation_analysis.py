#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_perturbation_analysis.py

Perturbation analysis implementation including: single perturbations with both
random and TOPSIS targeting (1%, 2%, 5%, 10%), multiple perturbations with attack
sequences k ∈ {2,5,10,25,50,100}, plots for all metrics over time and on largest
components.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt

from hiper.core.hypernetwork import Hypernetwork
from hiper.simulation.attack import RemoveNodeAttack
from hiper.simulation.sequence import AttackSequence
from hiper.simulation.simulator import HypernetworkSimulator


class PerturbationAnalyzer:
    """Perturbation analyzer with both random and efficient TOPSIS targeting."""

    def __init__(self, output_dir: str = "results"):
        """Initialize the analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def load_hypergraph_from_file(filepath: str) -> Hypernetwork:
        """Load hypergraph from file."""
        hypernetwork = Hypernetwork()

        with open(filepath, 'r') as f:
            for edge_id, line in enumerate(f):
                nodes = [int(x) for x in line.strip().split()]
                if nodes:
                    hypernetwork.add_hyperedge(edge_id, nodes)

        return hypernetwork

    @staticmethod
    def compute_resilience_metrics(
            hypernetwork: Hypernetwork) -> Dict[str, float]:
        """Compute basic resilience metrics."""
        return {
            'order': float(hypernetwork.order()),
            'size': float(hypernetwork.size()),
            'avg_degree': hypernetwork.avg_deg() if hypernetwork.order() > 0 else 0.0,
            'avg_hyperedge_size': hypernetwork.avg_hyperedge_size() if hypernetwork.size() > 0 else 0.0
        }

    @staticmethod
    def get_random_targets(hypernetwork: Hypernetwork, percentage: float) -> \
            List[int]:
        """Get random nodes for targeting."""
        nodes = list(hypernetwork.nodes)
        n_targets = max(1, int(len(nodes) * percentage / 100.0))
        return random.sample(nodes, n_targets)

    @staticmethod
    def get_topsis_targets(hypernetwork: Hypernetwork, percentage: float) -> \
            List[int]:
        """Get top-ranked nodes using TOPSIS methodology."""
        from hiper.metrics.topsis import TopsisNodeRanker

        nodes = list(hypernetwork.nodes)
        n_targets = max(1, int(len(nodes) * percentage / 100.0))

        # Use full TOPSIS analysis without approximations
        ranker = TopsisNodeRanker()

        if len(nodes) > 1000:
            print(
                f"  Warning: Large hypergraph ({len(nodes)} nodes) - TOPSIS computation may be slow")

        # Apply TOPSIS to the complete hypergraph
        topsis_nodes = ranker.get_top_nodes(hypernetwork, percentage)

        return topsis_nodes[:n_targets]

    def single_perturbation_experiment(self, hypernetwork: Hypernetwork,
                                       dataset_name: str,
                                       percentages: Optional[List[int]] = None):
        """Run single perturbation experiments with both random and TOPSIS targeting."""
        if percentages is None:
            percentages = [1, 2, 5, 10]
        print(f"Running single perturbation experiments for {dataset_name}")

        results: Dict[str, Any] = {
            'dataset': dataset_name,
            'original_metrics': self.compute_resilience_metrics(hypernetwork),
            'random_targeting': {},
            'topsis_targeting': {}
        }

        simulator = HypernetworkSimulator(f"{dataset_name}_single")

        for percentage in percentages:
            print(f"  Processing {percentage}% attacks...")

            # Random targeting
            random_targets = self.get_random_targets(hypernetwork, percentage)
            results['random_targeting'][
                percentage] = self._execute_single_attacks(
                hypernetwork, random_targets,
                f"{dataset_name}_random_{percentage}", simulator
            )

            # TOPSIS targeting
            topsis_targets = self.get_topsis_targets(hypernetwork, percentage)
            results['topsis_targeting'][
                percentage] = self._execute_single_attacks(
                hypernetwork, topsis_targets,
                f"{dataset_name}_topsis_{percentage}", simulator
            )

        self.results[f"{dataset_name}_single"] = results
        return results

    def _execute_single_attacks(self, hypernetwork: Hypernetwork,
                                targets: List[int],
                                experiment_id: str,
                                simulator: HypernetworkSimulator) -> Dict[
        str, Any]:
        """Execute attacks on target nodes and measure resilience."""
        simulator.set_hypernetwork(hypernetwork, create_backup=True)

        # Create attack sequence for all targets
        sequence = AttackSequence(experiment_id)
        for i, target in enumerate(targets):
            attack = RemoveNodeAttack(f"{experiment_id}_attack_{i}", target)
            sequence.add_attack(attack)

        # Execute the sequence
        if sequence.size() > 0:
            simulator.simulate_sequence(sequence, restore_after=False)

        # Compute final metrics
        final_metrics = self.compute_resilience_metrics(
            simulator.current_hypernetwork)

        return {
            'targets': targets,
            'n_targets': len(targets),
            'final_metrics': final_metrics
        }

    def multiple_perturbation_experiment(self, hypernetwork: Hypernetwork,
                                         dataset_name: str,
                                         k_values: Optional[List[int]] = None):
        """Run multiple perturbation experiments with attack sequences."""
        if k_values is None:
            k_values = [2, 5, 10, 25, 50, 100]
        print(f"Running multiple perturbation experiments for {dataset_name}")

        results: Dict[str, Any] = {
            'dataset': dataset_name,
            'original_metrics': self.compute_resilience_metrics(hypernetwork),
            'sequences': {}
        }

        simulator = HypernetworkSimulator(f"{dataset_name}_multiple")

        for k in k_values:
            print(f"  Processing k={k} attack sequence...")

            # Determine step size for monitoring
            if k <= 10:
                step_size = 2
            elif k <= 50:
                step_size = 5
            else:
                step_size = 10

            # Generate attack sequence
            targets = self._generate_attack_sequence(hypernetwork, k)
            sequence_result = self._execute_monitored_sequence(
                hypernetwork, targets, k, step_size,
                f"{dataset_name}_seq_{k}", simulator
            )

            results['sequences'][k] = sequence_result

        self.results[f"{dataset_name}_multiple"] = results
        return results

    def _generate_attack_sequence(self, hypernetwork: Hypernetwork, k: int) -> \
            List[int]:
        """Generate attack sequence using TOPSIS for all k values."""
        nodes = list(hypernetwork.nodes)

        # Calculate the percentage needed to get k nodes
        percentage = min(100.0, (k * 100.0) / len(nodes))

        # Use TOPSIS to get the most critical nodes
        topsis_targets = self.get_topsis_targets(hypernetwork, percentage)

        # Return exactly k nodes
        return topsis_targets[:min(k, len(nodes))]

    def _execute_monitored_sequence(self, hypernetwork: Hypernetwork,
                                    targets: List[int], k: int, step_size: int,
                                    experiment_id: str,
                                    simulator: HypernetworkSimulator) -> Dict[
        str, Any]:
        """Execute attack sequence with periodic monitoring."""
        simulator.set_hypernetwork(hypernetwork, create_backup=True)

        # Track metrics over time
        metrics_timeline = []

        # Initial state
        initial_metrics = self.compute_resilience_metrics(
            simulator.current_hypernetwork)
        metrics_timeline.append({
            'step': 0,
            'attacks_executed': 0,
            'metrics': initial_metrics
        })

        # Execute attacks in batches
        attacks_executed = 0
        for i in range(0, len(targets), step_size):
            batch_targets = targets[i:i + step_size]

            # Create and execute batch attacks
            batch_sequence = AttackSequence(
                f"{experiment_id}_batch_{i // step_size}")
            for j, target in enumerate(batch_targets):
                # Check if target still exists
                if target in simulator.current_hypernetwork.nodes:
                    attack = RemoveNodeAttack(
                        f"{experiment_id}_attack_{attacks_executed + j}", target
                    )
                    batch_sequence.add_attack(attack)

            # Execute batch if not empty
            if batch_sequence.size() > 0:
                simulator.simulate_sequence(batch_sequence, restore_after=False)

            attacks_executed += len(batch_targets)

            # Record metrics
            current_metrics = self.compute_resilience_metrics(
                simulator.current_hypernetwork)
            metrics_timeline.append({
                'step': i // step_size + 1,
                'attacks_executed': attacks_executed,
                'metrics': current_metrics
            })

        return {
            'k': k,
            'step_size': step_size,
            'targets': targets,
            'metrics_timeline': metrics_timeline,
            'total_attacks': attacks_executed
        }

    @staticmethod
    def find_connected_components(hypernetwork: Hypernetwork) -> List[set]:
        """
        Find all connected components in the hypergraph.
        Two nodes are connected if there exists a path of hyperedges between them.
        """
        if hypernetwork.order() == 0:
            return []

        # Build node-to-node adjacency
        adjacency = {node: set() for node in hypernetwork.nodes}

        for edge_id in hypernetwork.edges:
            edge_nodes = list(hypernetwork.get_nodes(edge_id))
            # Connect all pairs of nodes in the same hyperedge
            for i, node1 in enumerate(edge_nodes):
                for node2 in edge_nodes[i + 1:]:
                    adjacency[node1].add(node2)
                    adjacency[node2].add(node1)

        # Find components using BFS
        visited = set()
        components = []

        for node in hypernetwork.nodes:
            if node not in visited:
                component = set()
                stack = [node]

                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue

                    visited.add(current)
                    component.add(current)

                    for neighbor in adjacency.get(current, set()):
                        if neighbor not in visited:
                            stack.append(neighbor)

                if component:
                    components.append(component)

        # Sort by size (largest first)
        components.sort(key=len, reverse=True)
        return components

    @staticmethod
    def get_component_subgraph(hypernetwork: Hypernetwork,
                               component_nodes: set) -> Hypernetwork:
        """Extract subgraph induced by component nodes with edge handling."""
        subgraph = Hypernetwork()

        for edge_id in hypernetwork.edges:
            edge_nodes = set(hypernetwork.get_nodes(edge_id))
            # Include hyperedges with ANY nodes in the component but filter
            # to only include nodes that are actually in the component
            intersection_nodes = edge_nodes.intersection(component_nodes)

            # Only add if at least 2 nodes from the component are in this hyperedge
            if len(intersection_nodes) >= 2:
                subgraph.add_hyperedge(edge_id, list(intersection_nodes))

        return subgraph

    def analyze_component_metrics(self, original: Hypernetwork,
                                  perturbed: Hypernetwork, m: int = 3) -> Dict[
        str, Any]:
        """Analyze connected components and compute fragmentation metrics."""
        orig_components = self.find_connected_components(original)
        pert_components = self.find_connected_components(perturbed)

        orig_largest_size = len(orig_components[0]) if orig_components else 0
        pert_largest_size = len(pert_components[0]) if pert_components else 0

        # Component metrics for top m components
        component_metrics = {}
        for i in range(min(m, len(pert_components))):
            component = pert_components[i]
            subgraph = self.get_component_subgraph(perturbed, component)

            component_metrics[f'component_{i + 1}'] = {
                'order': float(subgraph.order()),
                'size': float(subgraph.size()),
                'avg_degree': subgraph.avg_deg() if subgraph.order() > 0 else 0.0,
                'avg_hyperedge_size': subgraph.avg_hyperedge_size() if subgraph.size() > 0 else 0.0
            }

        return {
            'original_components': len(orig_components),
            'perturbed_components': len(pert_components),
            'original_largest_size': orig_largest_size,
            'perturbed_largest_size': pert_largest_size,
            'component_metrics': component_metrics,
            'largest_component_retention': (
                pert_largest_size / orig_largest_size if orig_largest_size > 0 else 0.0),
            'fragmentation_ratio': (
                len(pert_components) / len(orig_components) if len(
                    orig_components) > 0 else float('inf'))
        }

    def create_plots(self, dataset_name: str):
        """Create plots including component analysis."""
        print(f"Creating plots for {dataset_name}")

        # Single perturbation plots
        if f"{dataset_name}_single" in self.results:
            self._plot_single_perturbations(dataset_name)

        # Multiple perturbation plots  
        if f"{dataset_name}_multiple" in self.results:
            self._plot_multiple_perturbations(dataset_name)

        # Component analysis plots
        self._plot_component_analysis(dataset_name)

        # M-th largest component plots
        self._plot_largest_components(dataset_name)

    def _plot_single_perturbations(self, dataset_name: str):
        """Plot single perturbation results comparing random vs TOPSIS."""
        data = self.results[f"{dataset_name}_single"]
        percentages = sorted(data['random_targeting'].keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Single Perturbations: Random vs TOPSIS - {dataset_name}',
                     fontsize=16)

        metrics = ['order', 'size', 'avg_degree', 'avg_hyperedge_size']

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            # Plot both targeting strategies
            random_values = [
                data['random_targeting'][pct]['final_metrics'][metric] for pct
                in percentages]
            topsis_values = [
                data['topsis_targeting'][pct]['final_metrics'][metric] for pct
                in percentages]

            ax.plot(percentages, random_values, 'o-', label='Random Targeting',
                    linewidth=2, markersize=6)
            ax.plot(percentages, topsis_values, 's-', label='TOPSIS Targeting',
                    linewidth=2, markersize=6)
            ax.axhline(y=data['original_metrics'][metric], color='r',
                       linestyle='--', alpha=0.7, label='Original Value')

            ax.set_xlabel('Attack Percentage (%)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(
                f'{metric.replace("_", " ").title()} vs Attack Percentage')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_single_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_multiple_perturbations(self, dataset_name: str):
        """Plot multiple perturbation results over time."""
        data = self.results[f"{dataset_name}_multiple"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Multiple Perturbations Over Time - {dataset_name}',
                     fontsize=16)

        metrics = ['order', 'size', 'avg_degree', 'avg_hyperedge_size']

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            for k in sorted(data['sequences'].keys()):
                timeline = data['sequences'][k]['metrics_timeline']
                steps = [point['attacks_executed'] for point in timeline]
                values = [point['metrics'][metric] for point in timeline]

                ax.plot(steps, values, 'o-', label=f'k={k}', linewidth=2,
                        markersize=3)

            ax.axhline(y=data['original_metrics'][metric], color='r',
                       linestyle='--', alpha=0.7, label='Original')

            ax.set_xlabel('Number of Attacks Executed')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Degradation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_multiple_timeline.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_component_analysis(self, dataset_name: str):
        """Plot component analysis showing fragmentation and degradation metrics."""
        if f"{dataset_name}_single" not in self.results:
            return

        data = self.results[f"{dataset_name}_single"]
        percentages = sorted(data['random_targeting'].keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Component Fragmentation Analysis - {dataset_name}',
                     fontsize=16)

        # Show component count increase
        ax1 = axes[0, 0]
        ax1.set_title('Network Fragmentation')
        ax1.set_xlabel('Attack Percentage (%)')
        ax1.set_ylabel('Relative Metrics')

        # Show the degradation of the full network metrics
        full_order = [data['random_targeting'][pct]['final_metrics']['order']
                      for pct in percentages]
        full_size = [data['random_targeting'][pct]['final_metrics']['size'] for
                     pct in percentages]

        # Normalize to show relative degradation
        original_order = data['original_metrics']['order']
        original_size = data['original_metrics']['size']

        relative_order = [o / original_order for o in full_order]
        relative_size = [s / original_size for s in full_size]

        ax1.plot(percentages, relative_order, 'o-', label='Node Retention',
                 linewidth=2)
        ax1.plot(percentages, relative_size, 's-', label='Hyperedge Retention',
                 linewidth=2)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7,
                    label='Original')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Show comparison between random and TOPSIS
        ax2 = axes[0, 1]
        ax2.set_title('Attack Strategy Comparison')
        ax2.set_xlabel('Attack Percentage (%)')
        ax2.set_ylabel('Nodes Remaining')

        random_nodes = [data['random_targeting'][pct]['final_metrics']['order']
                        for pct in percentages]
        topsis_nodes = [data['topsis_targeting'][pct]['final_metrics']['order']
                        for pct in percentages]

        ax2.plot(percentages, random_nodes, 'o-', label='Random Attack',
                 linewidth=2)
        ax2.plot(percentages, topsis_nodes, 's-', label='TOPSIS Attack',
                 linewidth=2)
        ax2.axhline(y=original_order, color='r', linestyle='--', alpha=0.7,
                    label='Original')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Average degree degradation
        ax3 = axes[1, 0]
        ax3.set_title('Connectivity Degradation')
        ax3.set_xlabel('Attack Percentage (%)')
        ax3.set_ylabel('Average Degree')

        random_deg = [
            data['random_targeting'][pct]['final_metrics']['avg_degree'] for pct
            in percentages]
        topsis_deg = [
            data['topsis_targeting'][pct]['final_metrics']['avg_degree'] for pct
            in percentages]

        ax3.plot(percentages, random_deg, 'o-', label='Random Attack',
                 linewidth=2)
        ax3.plot(percentages, topsis_deg, 's-', label='TOPSIS Attack',
                 linewidth=2)
        ax3.axhline(y=data['original_metrics']['avg_degree'], color='r',
                    linestyle='--', alpha=0.7, label='Original')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Hyperedge size degradation  
        ax4 = axes[1, 1]
        ax4.set_title('Hyperedge Size Degradation')
        ax4.set_xlabel('Attack Percentage (%)')
        ax4.set_ylabel('Avg Hyperedge Size')

        random_hs = [
            data['random_targeting'][pct]['final_metrics']['avg_hyperedge_size']
            for pct in percentages]
        topsis_hs = [
            data['topsis_targeting'][pct]['final_metrics']['avg_hyperedge_size']
            for pct in percentages]

        ax4.plot(percentages, random_hs, 'o-', label='Random Attack',
                 linewidth=2)
        ax4.plot(percentages, topsis_hs, 's-', label='TOPSIS Attack',
                 linewidth=2)
        ax4.axhline(y=data['original_metrics']['avg_hyperedge_size'], color='r',
                    linestyle='--', alpha=0.7, label='Original')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_component_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_largest_components(self, dataset_name: str, m: int = 3):
        """Plot metrics evolution for the m-th largest connected components."""
        if f"{dataset_name}_single" not in self.results:
            return

        print(f"  Creating largest components plots (m={m})")

        # Load the original hypergraph to get component analysis
        original_path = f"data/{dataset_name}.txt"
        if not Path(original_path).exists():
            print(
                f"  Warning: Cannot find original dataset file {original_path}")
            return

        original_hypernetwork = self.load_hypergraph_from_file(original_path)
        data = self.results[f"{dataset_name}_single"]
        percentages = sorted(data['random_targeting'].keys())

        # Analyze components for each perturbation level
        component_data: Dict[str, Dict[int, Dict[str, Any]]] = {'random': {},
                                                                'topsis': {}}

        for pct in percentages:
            # Create simulator to reproduce the perturbed states
            simulator = HypernetworkSimulator(
                f"{dataset_name}_component_analysis")

            # Random targeting analysis
            simulator.set_hypernetwork(original_hypernetwork,
                                       create_backup=True)
            random_targets = data['random_targeting'][pct]['targets']

            # Execute the same attacks to get perturbed network
            sequence = AttackSequence(f"component_analysis_random_{pct}")
            for i, target in enumerate(random_targets):
                if target in simulator.current_hypernetwork.nodes:
                    attack = RemoveNodeAttack(f"comp_analysis_r_{pct}_{i}",
                                              target)
                    sequence.add_attack(attack)

            if sequence.size() > 0:
                simulator.simulate_sequence(sequence, restore_after=False)

            component_analysis = self.analyze_component_metrics(
                original_hypernetwork, simulator.current_hypernetwork, m
            )
            component_data['random'][pct] = component_analysis

            # TOPSIS targeting analysis
            simulator.set_hypernetwork(original_hypernetwork,
                                       create_backup=True)
            topsis_targets = data['topsis_targeting'][pct]['targets']

            sequence = AttackSequence(f"component_analysis_topsis_{pct}")
            for i, target in enumerate(topsis_targets):
                if target in simulator.current_hypernetwork.nodes:
                    attack = RemoveNodeAttack(f"comp_analysis_t_{pct}_{i}",
                                              target)
                    sequence.add_attack(attack)

            if sequence.size() > 0:
                simulator.simulate_sequence(sequence, restore_after=False)

            component_analysis = self.analyze_component_metrics(
                original_hypernetwork, simulator.current_hypernetwork, m
            )
            component_data['topsis'][pct] = component_analysis

        # Create plots for largest components
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f'M-th Largest Component Analysis (m=1 to {m}) - {dataset_name}',
            fontsize=16)

        metrics = ['order', 'size', 'avg_degree', 'avg_hyperedge_size']
        colors = ['blue', 'green', 'orange', 'purple', 'brown']

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            # Plot each of the top m components for both strategies
            for comp_idx in range(1, min(m + 1,
                                         4)):  # Limit to avoid too many lines
                # Random targeting
                random_values = []
                topsis_values = []

                for pct in percentages:
                    comp_key = f'component_{comp_idx}'

                    # Get component metric or 0 if component doesn't exist
                    if comp_key in component_data['random'][pct][
                        'component_metrics']:
                        random_val = \
                            component_data['random'][pct]['component_metrics'][
                                comp_key][metric]
                    else:
                        random_val = 0.0
                    random_values.append(random_val)

                    if comp_key in component_data['topsis'][pct][
                        'component_metrics']:
                        topsis_val = \
                            component_data['topsis'][pct]['component_metrics'][
                                comp_key][metric]
                    else:
                        topsis_val = 0.0
                    topsis_values.append(topsis_val)

                # Plot with different styles for random vs TOPSIS
                ax.plot(percentages, random_values, 'o-',
                        label=f'Random - Component {comp_idx}',
                        color=colors[comp_idx - 1], alpha=0.7, linewidth=2)
                ax.plot(percentages, topsis_values, 's--',
                        label=f'TOPSIS - Component {comp_idx}',
                        color=colors[comp_idx - 1], alpha=0.9, linewidth=2)

            ax.set_xlabel('Attack Percentage (%)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(
                f'{metric.replace("_", " ").title()} in Largest Components')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_largest_components.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"perturbation_results_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {output_path}")

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report."""
        report: Dict[str, Any] = {
            'analysis_summary': {
                'datasets_analyzed': len(
                    [k for k in self.results.keys() if '_single' in k]),
                'total_experiments': len(self.results),
                'analysis_timestamp': time.time()
            },
            'dataset_summaries': {}
        }

        for key in self.results.keys():
            if '_single' in key:
                dataset_name = key.replace('_single', '')
                data = self.results[key]

                report['dataset_summaries'][dataset_name] = {
                    'original_metrics': data['original_metrics'],
                    'max_degradation_random': self._calculate_max_degradation(
                        data['random_targeting']),
                    'max_degradation_topsis': self._calculate_max_degradation(
                        data['topsis_targeting'])
                }

        # Save summary
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Summary report saved to {summary_path}")
        return report

    @staticmethod
    def _calculate_max_degradation(targeting_data: Dict[int, Dict[str, Any]]) -> \
            Dict[str, Dict[str, Any]]:
        """Calculate maximum degradation for a targeting strategy."""
        max_degradations = {}
        for pct, data in targeting_data.items():
            for metric, value in data['final_metrics'].items():
                if metric not in max_degradations:
                    max_degradations[metric] = {'percentage': pct,
                                                'final_value': value}
                elif value < max_degradations[metric]['final_value']:
                    max_degradations[metric] = {'percentage': pct,
                                                'final_value': value}
        return max_degradations


def main():
    """Main function to run the perturbation analysis."""
    print("Starting Perturbation Analysis")
    print("=" * 60)

    analyzer = PerturbationAnalyzer()

    # List of datasets to analyze
    datasets = [
        "data/hypercl_20.txt",
        "data/hypercl_21.txt",
        "data/hypercl_23.txt",
        "data/hypercl_25.txt"
    ]

    for dataset_path in datasets:
        if Path(dataset_path).exists():
            dataset_name = Path(dataset_path).stem
            print(f"\n{'=' * 50}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'=' * 50}")

            # Load hypergraph
            hypernetwork = analyzer.load_hypergraph_from_file(dataset_path)
            print(
                f"Loaded hypergraph: {hypernetwork.order()} nodes, {hypernetwork.size()} hyperedges")

            # Run single perturbation experiments (both random and TOPSIS)
            analyzer.single_perturbation_experiment(hypernetwork, dataset_name,
                                                    [1, 2, 5, 10])

            # Run multiple perturbation experiments  
            analyzer.multiple_perturbation_experiment(hypernetwork,
                                                      dataset_name,
                                                      [2, 5, 10, 25, 50, 100])

            # Create plots
            analyzer.create_plots(dataset_name)

            print(f"Completed analysis for {dataset_name}")
        else:
            print(f"Dataset not found: {dataset_path}")

    # Save all results and generate summary
    analyzer.save_results()
    summary = analyzer.generate_summary_report()

    print(f"\n{'=' * 60}")
    print("PERTURBATION ANALYSIS FINISHED")
    print(f"{'=' * 60}")
    print(f"Results saved to: {analyzer.output_dir}")
    print(
        f"Datasets analyzed: {summary['analysis_summary']['datasets_analyzed']}")
    print(
        f"Total experiments: {summary['analysis_summary']['total_experiments']}")
    print("\nGenerated plots:")
    print("- Single perturbation comparisons (Random vs TOPSIS)")
    print("- Multiple perturbation timelines")
    print("- Component analysis plots")
    print("- M-th largest component analysis plots")


if __name__ == "__main__":
    main()
