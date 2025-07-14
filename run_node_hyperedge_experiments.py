#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_node_and_hyperedge_resilience_experiments.py

Script to perform comprehensive resilience experiments on hypergraphs by
removing both nodes and hyperedges using random and TOPSIS-based strategies.

The experiments test the impact of removing different percentages of
nodes and hyperedges (1%, 2%, 5%, 10%, 25%) using:
- Random selection
- TOPSIS method for most critical elements (top)
- TOPSIS method for least critical elements (bottom)

Analyzed metrics include traditional connectivity measures, redundancy
coefficients, s-walk efficiency, and new higher-order cohesion metrics
(HOCR_m and LHC_m).
"""

import time
from pathlib import Path
from typing import Dict, Any

# Plotting imports
import matplotlib.pyplot as plt
import numpy as np

# Import core classes
from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.comprehensive_resilience import (
    ComprehensiveResilienceExperiment
)


def create_sample_hypernetwork() -> Hypernetwork:
    """
    Create a sample hypernetwork for testing comprehensive resilience
    experiments.

    Returns:
        Hypernetwork: hypernetwork with a test structure designed to evaluate
                     both traditional and higher-order resilience metrics.
    """
    hn = Hypernetwork()

    # Add hyperedges to create an interesting structure with higher-order
    hyperedges = [
        (0, [1, 2, 3, 4]),  # Core component start
        (1, [2, 3, 4, 5]),  # Overlaps significantly with hyperedge 0
        (2, [3, 4, 5, 6]),  # Forms chain with hyperedge 1
        (3, [5, 6, 7, 8]),  # Continues the chain
        (4, [1, 9, 10, 11]),  # Secondary branch from core
        (5, [9, 10, 11, 12]),  # Extends secondary branch
        (6, [13, 14, 15]),  # Isolated component
        (7, [14, 15, 16]),  # Connected to isolated component
        (8, [17, 18]),  # Small component (previously isolated)
        (9, [1, 5, 9, 13]),  # Bridge hyperedge connecting components
        (10, [2, 6, 10, 14]),  # Another bridge hyperedge
        (11, [7, 17]),  # Bridge connecting [17,18] to main network
    ]

    for he_id, nodes in hyperedges:
        hn.add_hyperedge(he_id, nodes)

    # Verify connectivity for resilience experiments
    from hiper.metrics.distance import HypergraphDistance
    distance_calc = HypergraphDistance()
    is_connected = distance_calc.is_connected(hn)

    if not is_connected:
        print("WARNING: Hypernetwork is not fully connected!")
        print("This may cause issues in resilience experiments.")
    else:
        print(
            "Hypernetwork is fully connected - suitable for analysis")

    msg = (
        f"Sample hypernetwork created: {hn.order()} nodes, "
        f"{hn.size()} hyperedges, connected: {is_connected}"
    )
    print(msg)
    return hn


def setup_experiment_parameters() -> Dict[str, Any]:
    """
    Configure parameters for comprehensive resilience experiments.

    Returns:
        Dictionary with experiment parameters optimized for both node
        and hyperedge removal analysis
    """
    return {
        'removal_percentages': [1, 2, 5, 10, 25],
        'random_trials': 10,
        's_parameter': 1,
        'm_parameter': 2,
        'output_dir': Path('resilience_results'),
        'save_plots': True,
        'verbose': True,
    }


def run_comprehensive_analysis(
        hypernetwork: Hypernetwork,
        params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform full comprehensive resilience analysis on the hypernetwork.

    Args:
        hypernetwork: target hypernetwork for analysis
        params: experiment parameters including removal percentages and trials

    Returns:
        Comprehensive results dictionary containing both node and hyperedge
        removal experiment results with traditional and higher-order metrics
    """
    sep = "=" * 80
    print("\n" + sep)
    print("STARTING COMPREHENSIVE RESILIENCE EXPERIMENTS")
    print(sep)

    experiment = ComprehensiveResilienceExperiment(
        s=params['s_parameter'],
        m=params['m_parameter']
    )

    print("\nAnalyzing hypernetwork:")
    print(f"  Order (nodes): {hypernetwork.order()}")
    print(f"  Size (hyperedges): {hypernetwork.size()}")
    print(f"  Average degree: {hypernetwork.avg_deg():.2f}")

    print("\nComputing baseline metrics and starting experiments...")
    start_time = time.time()

    results = experiment.run_node_and_hyperedge_removal_experiments(
        hypernetwork=hypernetwork,
        removal_percentages=params['removal_percentages'],
        random_trials=params['random_trials'],
    )

    elapsed = time.time() - start_time
    results['execution_time'] = elapsed
    results['parameters'] = params.copy()

    print(f"\nExperiments completed in {elapsed:.2f} seconds")
    return results


def generate_comprehensive_plots(
        results: Dict[str, Any],
        output_dir: Path
) -> None:
    """
    Generate comprehensive plots for both node and hyperedge removal analysis.

    Args:
        results: comprehensive experimental results
        output_dir: directory for saving generated plots
    """
    print("\nGenerating comprehensive analysis plots...")

    # Setup plot style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': [12, 8],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9
    })

    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Traditional metrics comparison for node removal
    create_metrics_comparison_plot(
        results['node_removal_experiments'],
        results['removal_percentages'],
        results['original_metrics'],
        'Node Removal Analysis',
        plots_dir / 'node_removal_traditional_metrics.png'
    )

    # Plot 2: Traditional metrics comparison for hyperedge removal
    create_metrics_comparison_plot(
        results['hyperedge_removal_experiments'],
        results['removal_percentages'],
        results['original_metrics'],
        'Hyperedge Removal Analysis',
        plots_dir / 'hyperedge_removal_traditional_metrics.png'
    )

    # Plot 3: Higher-order cohesion metrics comparison
    create_higher_order_comparison_plot(
        results,
        plots_dir / 'higher_order_cohesion_comparison.png'
    )

    # Plot 4: Strategy effectiveness heatmap
    create_strategy_effectiveness_heatmap(
        results,
        plots_dir / 'strategy_effectiveness_heatmap.png'
    )

    print(f"Plots saved to: {plots_dir}")


def create_metrics_comparison_plot(
        experiment_results: Dict[str, Any],
        percentages: list,
        original_metrics: Dict[str, float],
        title: str,
        output_path: Path
) -> None:
    """
    Create comparison plot for traditional resilience metrics.

    Args:
        experiment_results: results from either node or hyperedge removal
                          experiments
        percentages: list of removal percentages tested
        original_metrics: baseline metric values
        title: plot title
        output_path: path for saving the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    traditional_metrics = [
        'hypergraph_connectivity',
        'hyperedge_connectivity',
        'redundancy_coefficient',
        'swalk_efficiency'
    ]
    strategies = [
        'random_removal',
        'topsis_top_removal',
        'topsis_bottom_removal'
    ]
    strategy_labels = ['Random', 'TOPSIS Top', 'TOPSIS Bottom']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, metric in enumerate(traditional_metrics):
        ax = axes[i // 2, i % 2]

        for strategy, label, color in zip(strategies, strategy_labels, colors):
            values = []
            for percentage in percentages:
                if percentage in experiment_results[strategy]:
                    metric_value = experiment_results[strategy][percentage].get(
                        metric, 0
                    )
                    values.append(metric_value)
                else:
                    values.append(0)

            # Normalize by original value for better comparison
            original_value = original_metrics.get(metric, 1)
            if original_value > 0:
                normalized_values = [v / original_value for v in values]
            else:
                normalized_values = values

            ax.plot(
                percentages, normalized_values, marker='o', label=label,
                color=color, linewidth=2, markersize=6
            )

        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Removal Percentage (%)')
        ax.set_ylabel('Normalized Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_higher_order_comparison_plot(
        results: Dict[str, Any],
        output_path: Path
) -> None:
    """
    Create comparison plot specifically for higher-order cohesion metrics.

    Args:
        results: comprehensive experimental results
        output_path: path for saving the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        'Higher-Order Cohesion Metrics Analysis',
        fontsize=16,
        fontweight='bold'
    )

    percentages = results['removal_percentages']
    experiments = [
        ('node_removal_experiments', 'Node Removal'),
        ('hyperedge_removal_experiments', 'Hyperedge Removal')
    ]

    # Find higher-order metrics
    sample_result = list(
        results['node_removal_experiments']['random_removal'].values()
    )[0]
    ho_metrics = [
        k for k in sample_result.keys()
        if k.startswith(('hocr_', 'lhc_'))
    ]

    if not ho_metrics:
        fig.text(
            0.5, 0.5, 'No higher-order cohesion metrics available',
            ha='center', va='center', fontsize=14
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    strategies = [
        'random_removal',
        'topsis_top_removal',
        'topsis_bottom_removal'
    ]
    strategy_labels = ['Random', 'TOPSIS Top', 'TOPSIS Bottom']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for exp_idx, (exp_type, exp_title) in enumerate(experiments):
        for metric_idx, metric in enumerate(ho_metrics[:2]):
            if exp_idx * 2 + metric_idx >= 4:
                break

            ax = axes[exp_idx, metric_idx]

            for strategy, label, color in zip(
                    strategies, strategy_labels, colors
            ):
                values = []
                for percentage in percentages:
                    if percentage in results[exp_type][strategy]:
                        metric_value = results[exp_type][strategy][
                            percentage
                        ].get(metric, 0)
                        values.append(metric_value)
                    else:
                        values.append(0)

                ax.plot(
                    percentages, values, marker='o', label=label,
                    color=color, linewidth=2, markersize=6
                )

            ax.set_title(f'{exp_title} - {metric.upper()}', fontweight='bold')
            ax.set_xlabel('Removal Percentage (%)')
            ax.set_ylabel('Metric Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(experiments) * 2, 4):
        axes.flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_strategy_effectiveness_heatmap(
        results: Dict[str, Any],
        output_path: Path
) -> None:
    """
    Create heatmap showing strategy effectiveness across different metrics and
    percentages.

    Args:
        results: comprehensive experimental results
        output_path: path for saving the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle(
        'Strategy Effectiveness Analysis',
        fontsize=16,
        fontweight='bold'
    )

    experiments = [
        ('node_removal_experiments', 'Node Removal'),
        ('hyperedge_removal_experiments', 'Hyperedge Removal')
    ]

    original_metrics = results['original_metrics']
    percentages = results['removal_percentages']
    max_percentage = max(percentages)

    colorbar_image = None

    for exp_idx, (exp_type, exp_title) in enumerate(experiments):
        strategies = [
            'random_removal',
            'topsis_top_removal',
            'topsis_bottom_removal'
        ]
        strategy_labels = ['Random', 'TOPSIS Top', 'TOPSIS Bottom']

        # Get available metrics
        sample_result = results[exp_type]['random_removal'][max_percentage]
        metrics = [k for k in sample_result.keys() if k in original_metrics]

        # Create effectiveness matrix
        effectiveness_matrix = []
        row_labels = []

        for strategy, strategy_label in zip(strategies, strategy_labels):
            if max_percentage in results[exp_type][strategy]:
                row_data = []
                for metric in metrics:
                    original_value = original_metrics[metric]
                    final_value = results[exp_type][strategy][
                        max_percentage
                    ].get(metric, 0)

                    # Calculate retention percentage
                    if original_value > 0:
                        retention = min((final_value / original_value) * 100,
                                        100.0)
                    else:
                        retention = 0

                    row_data.append(retention)

                effectiveness_matrix.append(row_data)
                row_labels.append(strategy_label)

        # Create heatmap
        if effectiveness_matrix:
            effectiveness_array = np.array(effectiveness_matrix)
            colorbar_image = axes[exp_idx].imshow(
                effectiveness_array, cmap='RdYlGn',
                vmin=0, vmax=100, aspect='auto'
            )

            # Configure labels
            axes[exp_idx].set_xticks(range(len(metrics)))
            axes[exp_idx].set_xticklabels(
                [m.replace('_', '\n') for m in metrics],
                rotation=45, ha='right'
            )
            axes[exp_idx].set_yticks(range(len(row_labels)))
            axes[exp_idx].set_yticklabels(row_labels)
            axes[exp_idx].set_title(
                f'{exp_title} (at {max_percentage}% removal)',
                fontweight='bold'
            )

            # Add percentage values to cells
            for i in range(len(row_labels)):
                for j in range(len(metrics)):
                    axes[exp_idx].text(
                        j, i, f'{effectiveness_array[i, j]:.0f}%',
                        ha="center", va="center", color="black",
                        fontweight='bold', fontsize=9
                    )

    # Add colorbar if we have any heatmap
    if colorbar_image is not None:
        fig.colorbar(
            colorbar_image, ax=axes, shrink=0.8,
            label='Metric Retention (%)'
        )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive summary of experimental results.

    Args:
        results: comprehensive experimental results
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESILIENCE EXPERIMENT SUMMARY")
    print("=" * 80)

    # General information
    print(f"\nExperiment Parameters:")
    print(f"  Removal percentages tested: {results['removal_percentages']}")
    print(f"  Random trials per percentage: "
          f"{results['parameters']['random_trials']}")
    print(f"  Higher-order parameter m: "
          f"{results['parameters']['m_parameter']}")

    # Higher-order analysis
    if ('higher_order_analysis' in results and
            'original' in results['higher_order_analysis']):
        ho_analysis = results['higher_order_analysis']['original']
        print(f"  Original higher-order components: "
              f"{ho_analysis['num_components']}")
        print(f"  Largest component size: "
              f"{ho_analysis['largest_component_size']}")

    # Strategy effectiveness analysis
    print(f"\nStrategy Effectiveness Analysis:")

    max_percentage = max(results['removal_percentages'])
    original_metrics = results['original_metrics']

    experiment_configs = [
        ('node_removal_experiments', 'Node Removal'),
        ('hyperedge_removal_experiments', 'Hyperedge Removal')
    ]

    for exp_type, exp_name in experiment_configs:
        print(f"\n{exp_name} (at {max_percentage}% removal):")

        strategies = {
            'random_removal': 'Random',
            'topsis_top_removal': 'TOPSIS Top',
            'topsis_bottom_removal': 'TOPSIS Bottom'
        }

        for strategy_key, strategy_name in strategies.items():
            if max_percentage in results[exp_type][strategy_key]:
                final_metrics = results[exp_type][strategy_key][max_percentage]

                print(f"  {strategy_name}:")

                # Calculate average retention across all metrics
                retentions = []
                for metric, original_value in original_metrics.items():
                    if metric in final_metrics and original_value > 0:
                        final_value = final_metrics[metric]
                        retention = min((final_value / original_value) * 100,
                                        100.0)
                        retentions.append(retention)
                        print(f"    {metric}: {retention:.1f}% retained")

                if retentions:
                    avg_retention = np.mean(retentions)
                    print(f"    Average retention: {avg_retention:.1f}%")


def main():
    """
    Main function for executing comprehensive resilience experiments.
    """
    print("Starting comprehensive node and hyperedge resilience experiments")
    print("=" * 70)

    # Create sample hypernetwork
    print("Creating sample hypernetwork for analysis...")
    hypernetwork = create_sample_hypernetwork()

    # Setup experiment parameters
    params = setup_experiment_parameters()
    output_dir = params['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment configuration:")
    print(f"  Removal percentages: {params['removal_percentages']}")
    print(f"  Random trials: {params['random_trials']}")
    print(f"  Higher-order parameter m: {params['m_parameter']}")
    print(f"  Output directory: {output_dir}")

    # Run comprehensive analysis
    print(f"\nExecuting comprehensive resilience analysis...")
    results = run_comprehensive_analysis(hypernetwork, params)

    # Generate comprehensive plots
    if params['save_plots']:
        try:
            generate_comprehensive_plots(results, output_dir)
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")

    # Print summary
    if params['verbose']:
        print_experiment_summary(results)

    print(f"\nComprehensive experiments completed successfully!")
    print(f"Results and plots available in: {output_dir}")


if __name__ == '__main__':
    main()
