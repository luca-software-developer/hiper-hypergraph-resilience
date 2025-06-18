# -*- coding: utf-8 -*-
"""
experiments.py

Implements resilience experiments for hypergraphs including node removal
simulations with random and TOPSIS-based selection strategies.
"""

import random
from typing import Dict, List, Any

import numpy as np

# Handle matplotlib import gracefully
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from hiper.core.hypernetwork import Hypernetwork
from .connectivity import HypergraphConnectivity, HyperedgeConnectivity
from .redundancy import RedundancyCoefficient
from .swalk import SwalkEfficiency
from .topsis import TopsisNodeRanker


class ResilienceExperiment:
    """
    Conducts resilience experiments on hypergraphs by simulating various
    attack scenarios and measuring their impact on structural metrics.

    This class provides comprehensive experimental capabilities for analyzing
    hypergraph resilience under different node removal strategies including
    random selection and TOPSIS-based targeted attacks.
    """

    def __init__(self, s: int = 1):
        """
        Initialize experiment framework.

        Args:
            s: Parameter for s-walk computations.
        """
        self.s = s
        self.metrics = {
            'hypergraph_connectivity': HypergraphConnectivity(s),
            'hyperedge_connectivity': HyperedgeConnectivity(s),
            'redundancy_coefficient': RedundancyCoefficient(),
            'swalk_efficiency': SwalkEfficiency(s)
        }
        self.topsis_ranker = TopsisNodeRanker()

    def run_node_removal_experiment(
            self,
            hypernetwork: Hypernetwork,
            removal_percentages: List[float],
            random_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Run comprehensive node removal experiment.

        Args:
            hypernetwork: Target hypergraph for analysis.
            removal_percentages: List of percentages to remove
                (e.g., [1, 2, 5, 10, 25]).
            random_trials: Number of trials for random removal strategy.

        Returns:
            Dictionary containing experimental results and analysis data.
        """
        # Initialize results structure
        results: Dict[str, Any] = {
            'original_metrics': {},
            'random_removal': {},
            'topsis_top_removal': {},
            'topsis_bottom_removal': {},
            'removal_percentages': list(removal_percentages)
        }

        print(f"Starting resilience experiment on hypergraph with "
              f"{hypernetwork.order()} nodes and {hypernetwork.size()} "
              f"hyperedges")

        # Compute original metrics
        original_metrics_raw = self._compute_all_metrics(hypernetwork)
        results['original_metrics'] = {
            k: float(v) for k, v in original_metrics_raw.items()
        }

        print("Original metrics:")
        for metric_name, value in results['original_metrics'].items():
            print(f"  {metric_name}: {value:.4f}")

        # Run experiments for each removal percentage
        for percentage in removal_percentages:
            print(f"\nTesting {percentage}% node removal...")

            # Random removal
            random_results_raw = self._test_random_removal(
                hypernetwork, percentage, random_trials)
            results['random_removal'][percentage] = {
                k: float(v) for k, v in random_results_raw.items()
            }

            # TOPSIS top removal
            top_results_raw = self._test_topsis_removal(
                hypernetwork, percentage, 'top')
            results['topsis_top_removal'][percentage] = {
                k: float(v) for k, v in top_results_raw.items()
            }

            # TOPSIS bottom removal
            bottom_results_raw = self._test_topsis_removal(
                hypernetwork, percentage, 'bottom')
            results['topsis_bottom_removal'][percentage] = {
                k: float(v) for k, v in bottom_results_raw.items()
            }

        # Generate plots with proper error handling
        if HAS_MATPLOTLIB:
            try:
                results['plots'] = self._generate_plots(results)
            except Exception as e:
                print(f"Warning: Plot generation failed: {e}")
                results['plots'] = {}
        else:
            print("Warning: matplotlib not available, plots not generated")
            results['plots'] = {}

        return results

    def _compute_all_metrics(self, hypernetwork: Hypernetwork) -> Dict[
        str, float]:
        """Compute all resilience metrics for a hypergraph."""
        metrics = {}

        for metric_name, metric_calculator in self.metrics.items():
            try:
                value = metric_calculator.compute(hypernetwork)
                # Ensure the value is always a float
                metrics[metric_name] = float(value)
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Warning: Could not compute {metric_name}: {e}")
                metrics[metric_name] = 0.0

        return metrics

    def _test_random_removal(self,
                             hypernetwork: Hypernetwork,
                             percentage: float,
                             trials: int) -> Dict[str, float]:
        """Test random node removal strategy."""
        all_metrics = []

        for trial in range(trials):
            test_hn = self._copy_hypernetwork(hypernetwork)
            all_nodes = list(test_hn.nodes)
            n_remove = max(1, int(len(all_nodes) * percentage / 100.0))
            nodes_to_remove = random.sample(all_nodes,
                                            min(n_remove, len(all_nodes)))

            for node_id in nodes_to_remove:
                test_hn.remove_node(node_id)

            trial_metrics = self._compute_all_metrics(test_hn)
            all_metrics.append(trial_metrics)

        # Calculate averages
        averaged_metrics = {}
        for metric_name in self.metrics.keys():
            try:
                values = [float(trial[metric_name]) for trial in all_metrics]
                averaged_metrics[metric_name] = float(np.mean(values))
                averaged_metrics[f"{metric_name}_std"] = float(np.std(values))
            except (KeyError, ValueError, TypeError):
                averaged_metrics[metric_name] = 0.0
                averaged_metrics[f"{metric_name}_std"] = 0.0

        return averaged_metrics

    def _test_topsis_removal(self,
                             hypernetwork: Hypernetwork,
                             percentage: float,
                             strategy: str) -> Dict[str, float]:
        """Test TOPSIS-based node removal strategy."""
        test_hn = self._copy_hypernetwork(hypernetwork)

        if strategy == 'top':
            nodes_to_remove = self.topsis_ranker.get_top_nodes(test_hn,
                                                               percentage)
        else:
            nodes_to_remove = self.topsis_ranker.get_bottom_nodes(test_hn,
                                                                  percentage)

        for node_id in nodes_to_remove:
            test_hn.remove_node(node_id)

        return self._compute_all_metrics(test_hn)

    @staticmethod
    def _copy_hypernetwork(hypernetwork: Hypernetwork) -> Hypernetwork:
        """
        Create a deep copy of a hypernetwork.

        Args:
            hypernetwork: Original hypernetwork to copy.

        Returns:
            New hypernetwork instance with identical structure.
        """
        new_hn = Hypernetwork()

        for edge_id in hypernetwork.edges:
            nodes = hypernetwork.get_nodes(edge_id)
            new_hn.add_hyperedge(edge_id, nodes)

        return new_hn

    def _generate_plots(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization plots for experiment results.

        Args:
            results: Dictionary containing experimental results data.

        Returns:
            Dictionary containing matplotlib figure objects.
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available for plotting")
            return {}

        percentages = results['removal_percentages']
        metric_names = list(self.metrics.keys())

        plots = {}

        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric_name in enumerate(metric_names):
            ax = axes[i]

            # Extract data for plotting
            random_values = [results['random_removal'][p][metric_name]
                             for p in percentages]
            random_stds = [results['random_removal'][p].get(
                f"{metric_name}_std", 0) for p in percentages]
            top_values = [results['topsis_top_removal'][p][metric_name]
                          for p in percentages]
            bottom_values = [results['topsis_bottom_removal'][p][metric_name]
                             for p in percentages]

            # Plot lines
            ax.errorbar(percentages, random_values, yerr=random_stds,
                        label='Random', marker='o', capsize=5)
            ax.plot(percentages, top_values, label='TOPSIS Top',
                    marker='s', linestyle='--')
            ax.plot(percentages, bottom_values, label='TOPSIS Bottom',
                    marker='^', linestyle=':')

            # Add original value as horizontal line
            original_value = results['original_metrics'][metric_name]
            ax.axhline(y=original_value, color='red', linestyle='-',
                       alpha=0.5, label='Original')

            ax.set_xlabel('Removal Percentage (%)')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} vs '
                         f'Node Removal')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plots['combined_metrics'] = fig

        # Summary statistics plot
        fig_summary, ax_summary = plt.subplots(figsize=(12, 8))

        # Compute relative impact for each strategy
        max_percentage = max(percentages)

        impact_data = {
            'Random': [],
            'TOPSIS Top': [],
            'TOPSIS Bottom': []
        }

        for metric_name in metric_names:
            original = results['original_metrics'][metric_name]

            # Impact at maximum removal percentage
            random_final = results['random_removal'][max_percentage][
                metric_name]
            top_final = results['topsis_top_removal'][max_percentage][
                metric_name]
            bottom_final = results['topsis_bottom_removal'][max_percentage][
                metric_name]

            # Relative change
            epsilon = 1e-10
            random_impact = abs(original - random_final) / (original + epsilon)
            top_impact = abs(original - top_final) / (original + epsilon)
            bottom_impact = abs(original - bottom_final) / (original + epsilon)

            impact_data['Random'].append(random_impact)
            impact_data['TOPSIS Top'].append(top_impact)
            impact_data['TOPSIS Bottom'].append(bottom_impact)

        # Create grouped bar chart
        x = np.arange(len(metric_names))
        width = 0.25

        ax_summary.bar(x - width, impact_data['Random'], width,
                       label='Random', alpha=0.8)
        ax_summary.bar(x, impact_data['TOPSIS Top'], width,
                       label='TOPSIS Top', alpha=0.8)
        ax_summary.bar(x + width, impact_data['TOPSIS Bottom'], width,
                       label='TOPSIS Bottom', alpha=0.8)

        ax_summary.set_xlabel('Metrics')
        ax_summary.set_ylabel('Relative Impact')
        ax_summary.set_title(f'Impact of {max_percentage}% Node Removal '
                             f'by Strategy')
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels([name.replace('_', ' ').title()
                                    for name in metric_names], rotation=45)
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3)

        plt.tight_layout()
        plots['impact_summary'] = fig_summary

        return plots

    @staticmethod
    def save_results(results: Dict[str, Any],
                     filename_prefix: str) -> None:
        """
        Save experiment results and plots to files.

        Args:
            results: Dictionary containing experimental results and plots.
            filename_prefix: Prefix for output filenames.
        """
        import json

        # Save numerical results
        results_to_save = {k: v for k, v in results.items() if k != 'plots'}

        results_filename = f"{filename_prefix}_results.json"
        with open(results_filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        # Save plots if available and matplotlib is present
        if 'plots' in results and HAS_MATPLOTLIB:
            plots = results['plots']
            if 'combined_metrics' in plots:
                plots['combined_metrics'].savefig(
                    f"{filename_prefix}_metrics.png",
                    dpi=300, bbox_inches='tight')
            if 'impact_summary' in plots:
                plots['impact_summary'].savefig(
                    f"{filename_prefix}_impact.png",
                    dpi=300, bbox_inches='tight')

        print(f"Results saved with prefix: {filename_prefix}")

    @staticmethod
    def print_summary(results: Dict[str, Any]) -> None:
        """
        Print a comprehensive summary of experiment results.

        Args:
            results: Dictionary containing experimental results data.
        """
        print("\n" + "=" * 60)
        print("RESILIENCE EXPERIMENT SUMMARY")
        print("=" * 60)

        original = results['original_metrics']
        percentages = results['removal_percentages']
        max_removal = max(percentages)

        print("\nOriginal Hypergraph Metrics:")
        for metric, value in original.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nImpact of {max_removal}% Node Removal:")
        print("-" * 40)

        for metric in original.keys():
            orig_val = original[metric]

            random_val = results['random_removal'][max_removal][metric]
            top_val = results['topsis_top_removal'][max_removal][metric]
            bottom_val = results['topsis_bottom_removal'][max_removal][metric]

            epsilon = 1e-10
            random_change = ((orig_val - random_val) /
                             (orig_val + epsilon)) * 100
            top_change = ((orig_val - top_val) /
                          (orig_val + epsilon)) * 100
            bottom_change = ((orig_val - bottom_val) /
                             (orig_val + epsilon)) * 100

            print(f"\n{metric}:")
            print(f"  Random: {random_change:+.1f}% change")
            print(f"  TOPSIS Top: {top_change:+.1f}% change")
            print(f"  TOPSIS Bottom: {bottom_change:+.1f}% change")
