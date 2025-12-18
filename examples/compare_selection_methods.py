#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_selection_methods.py

Script to compare different MCDM methods (TOPSIS, WSM, MOORA) for node
selection in hypergraph resilience experiments.

This script performs targeted node removal experiments using three different
multi-criteria decision making methods:
- TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
- WSM (Weighted Sum Model)
- MOORA (Multi-Objective Optimization on basis of Ratio Analysis)

The experiments help validate whether simpler methods (WSM, MOORA) can achieve
comparable results to TOPSIS for hypergraph resilience analysis.
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt

from hiper.core.hypernetwork import Hypernetwork
from hiper.datasets.datafile import DataFile
from hiper.datasets.dataset import Dataset
from hiper.metrics.comprehensive_resilience import (
    ComprehensiveResilienceExperiment
)


def load_dataset_hypernetwork(dataset_name: str,
                              data_dir: Path) -> Hypernetwork:
    """
    Load a hypernetwork from a dataset file.

    Args:
        dataset_name: Name of the dataset file (e.g., 'iAF1260b.txt')
        data_dir: Directory containing the dataset files

    Returns:
        Loaded Hypernetwork
    """
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"\nLoading dataset: {dataset_name}")
    datafile = DataFile(str(dataset_path))
    dataset = Dataset(dataset_name, datafile)
    hn = dataset.get_hypernetwork()

    msg = (
        f"  Loaded: {hn.order()} nodes, {hn.size()} hyperedges"
    )
    print(msg)
    return hn


def run_comparison_experiments(
        hypernetwork: Hypernetwork,
        removal_percentages: List[float],
        random_trials: int = 10
) -> Dict[str, Any]:
    """
    Run experiments comparing TOPSIS, WSM, and MOORA methods.

    Args:
        hypernetwork: Target hypergraph for analysis
        removal_percentages: List of percentages to test (e.g., [1, 2, 5, 10, 25])
        random_trials: Number of random trials to average

    Returns:
        Dictionary with results for all three methods
    """
    methods = ['topsis', 'wsm', 'moora']
    all_results = {}

    sep = "=" * 80
    print("\n" + sep)
    print("COMPARING MCDM METHODS FOR NODE SELECTION")
    print(sep)

    for method in methods:
        print(f"\n>>> Running experiments with {method.upper()} method...")
        start_time = time.time()

        # Create experiment with specific ranker
        experiment = ComprehensiveResilienceExperiment(
            s=1,
            m=2,
            node_ranker=method
        )

        # Run node removal experiments
        results = experiment.run_node_removal_experiments(
            hypernetwork=hypernetwork,
            removal_percentages=removal_percentages,
            random_trials=random_trials
        )

        elapsed = time.time() - start_time
        print(f"    {method.upper()} completed in {elapsed:.2f} seconds")

        all_results[method] = {
            'experiments': results,
            'execution_time': elapsed
        }

        gc.collect()

    # Add common metadata
    all_results['metadata'] = {
        'removal_percentages': list(removal_percentages),
        'random_trials': random_trials,
        'hypergraph_order': hypernetwork.order(),
        'hypergraph_size': hypernetwork.size(),
    }

    # Compute original metrics
    experiment_baseline = ComprehensiveResilienceExperiment(s=1, m=2)
    all_results['original_metrics'] = experiment_baseline.compute_all_metrics(
        hypernetwork
    )

    return all_results


def generate_comparison_plots(
        all_datasets_results: Dict[str, Dict[str, Any]],
        output_dir: Path
) -> None:
    """
    Generate combined plots comparing the three methods across all datasets.

    Args:
        all_datasets_results: Dictionary mapping dataset names to their results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)

    methods = ['topsis', 'wsm', 'moora']
    colors = {'topsis': '#1f77b4', 'wsm': '#ff7f0e', 'moora': '#2ca02c'}
    markers = {'topsis': 'o', 'wsm': 'o',
               'moora': 'o'}

    # Metrics to plot
    metrics_to_plot = [
        ('order', 'Order'),
        ('size', 'Size'),
        ('avg_degree', 'Avg Degree'),
        ('avg_hyperdegree', 'Avg Hyperedge Size'),
    ]

    dataset_names = list(all_datasets_results.keys())
    num_datasets = len(dataset_names)

    # Create figure: num_datasets cols x 4 metrics rows
    fig, axes = plt.subplots(4, num_datasets, figsize=(7 * num_datasets, 12))

    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        'Comparison of MCDM Methods (TOPSIS, WSM, MOORA)',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    for col_idx, dataset_name in enumerate(dataset_names):
        results = all_datasets_results[dataset_name]
        percentages = results['metadata']['removal_percentages']
        original = results['original_metrics']

        for row_idx, (metric_key, metric_title) in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]

            if metric_key not in original:
                continue

            orig_val = original[metric_key]

            # Plot each method
            for method in methods:
                method_results = results[method]['experiments']
                top_key = f'{method}_top_removal'

                if top_key in method_results:
                    values = [method_results[top_key][str(p)][metric_key]
                              for p in percentages]

                    # Normalize to percentage of original
                    if orig_val > 0:
                        values_norm = [(v / orig_val) * 100 for v in values]
                    else:
                        values_norm = values

                    ax.plot(
                        percentages, values_norm,
                        marker=markers[method],
                        linewidth=2.5,
                        markersize=8,
                        label=method.upper(),
                        color=colors[method],
                        alpha=0.9,
                        markeredgewidth=1.5,
                        markeredgecolor='white'
                    )

            # Add baseline
            ax.axhline(
                y=100, color='gray', linestyle='--', alpha=0.5,
                linewidth=1.5, label='Original', zorder=1
            )

            # Labels and styling
            if row_idx == 3:  # Bottom row
                ax.set_xlabel('Nodes Removed (%)', fontweight='bold',
                              fontsize=9)
            if col_idx == 0:  # Leftmost column
                ax.set_ylabel('% of original', fontweight='bold', fontsize=9)

            # Title: dataset name on top row, metric on left
            if row_idx == 0:
                ax.set_title(f"{dataset_name}\n{metric_title}",
                             fontweight='bold', fontsize=10, pad=8)
            else:
                ax.set_title(metric_title, fontweight='bold', fontsize=10,
                             pad=8)

            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5,
                    color='gray')
            ax.set_xlim(-0.5, max(percentages) + 0.5)
            ax.tick_params(labelsize=9)

            # Legend only on first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='best', framealpha=0.98, fontsize=9,
                          edgecolor='gray', fancybox=False)

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.99])

    # Save as PDF
    plot_path_pdf = output_dir / 'methods_comparison_all_datasets.pdf'
    plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\nCombined comparison plot saved: {plot_path_pdf}")

    # Save as PNG
    plot_path_png = output_dir / 'methods_comparison_all_datasets.png'
    plt.savefig(plot_path_png, format='png', bbox_inches='tight', dpi=300)
    print(f"Combined comparison plot saved: {plot_path_png}")

    plt.close(fig)


def generate_single_dataset_plot(
        dataset_name: str,
        results: Dict[str, Any],
        output_dir: Path
) -> None:
    """
    Generate a single plot for one dataset showing all metrics in a 2x2 grid.

    Args:
        dataset_name: Name of the dataset
        results: Results dictionary for this dataset
        output_dir: Directory to save plot
    """
    output_dir.mkdir(exist_ok=True)

    methods = ['topsis', 'wsm', 'moora']
    colors = {'topsis': '#1f77b4', 'wsm': '#2ca02c', 'moora': '#d62728'}
    markers = {'topsis': 'o', 'wsm': 's', 'moora': '^'}
    linestyles = {'topsis': '-', 'wsm': '-', 'moora': '-'}

    # Metrics to plot (4 metrics in 2x2 grid)
    metrics_to_plot = [
        ('order', 'Order (Number of Nodes)'),
        ('size', 'Size (Number of Hyperedges)'),
        ('avg_degree', 'Average Degree'),
        ('avg_hyperedge_size', 'Average Hyperedge Size'),
    ]

    percentages = results['metadata']['removal_percentages']
    original = results['original_metrics']

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fig.suptitle(
        f'Comparison of MCDM Methods for Targeted Node Removal\n{dataset_name}',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    for idx, (metric_key, metric_title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        if metric_key not in original:
            continue

        orig_val = original[metric_key]

        # Plot each method
        for method in methods:
            method_results = results[method]['experiments']
            top_key = f'{method}_top_removal'

            if top_key in method_results:
                # Handle both int and str keys (direct dict vs JSON-loaded)
                values = []
                for p in percentages:
                    # Try int key first, then str key
                    if p in method_results[top_key]:
                        values.append(method_results[top_key][p][metric_key])
                    elif str(p) in method_results[top_key]:
                        values.append(
                            method_results[top_key][str(p)][metric_key])
                    else:
                        continue

                # Normalize to percentage of original
                if orig_val > 0:
                    values_norm = [(v / orig_val) * 100 for v in values]
                else:
                    values_norm = values

                ax.plot(
                    percentages, values_norm,
                    marker=markers[method],
                    linestyle=linestyles[method],
                    linewidth=2.5,
                    markersize=8,
                    label=method.upper(),
                    color=colors[method],
                    alpha=0.9,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )

        # Add baseline
        ax.axhline(
            y=100, color='gray', linestyle='--', alpha=0.6,
            linewidth=2.0, label='Original', zorder=1
        )

        # Labels and styling
        ax.set_xlabel('Percentage of Nodes Removed (%)', fontweight='bold',
                      fontsize=11)
        ax.set_ylabel('Metric Value (% of original)', fontweight='bold',
                      fontsize=11)
        ax.set_title(metric_title, fontweight='bold', fontsize=12, pad=10)

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
        ax.set_xlim(-1, max(percentages) + 1)
        ax.tick_params(labelsize=10)

        # Legend on first subplot
        if idx == 0:
            ax.legend(loc='best', framealpha=0.98, fontsize=10,
                      edgecolor='gray', fancybox=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save as PDF
    plot_path_pdf = output_dir / f'{dataset_name}_comparison.pdf'
    plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Individual plot saved: {plot_path_pdf}")

    # Save as PNG
    plot_path_png = output_dir / f'{dataset_name}_comparison.png'
    plt.savefig(plot_path_png, format='png', bbox_inches='tight', dpi=300)
    print(f"Individual plot saved: {plot_path_png}")

    plt.close(fig)


def print_comparison_summary(
        all_datasets_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Print summary comparing the three methods across all datasets.

    Args:
        all_datasets_results: Dictionary mapping dataset names to their results
    """
    sep = "=" * 80
    print("\n" + sep)
    print("COMPARISON SUMMARY - ALL DATASETS")
    print(sep)

    methods = ['topsis', 'wsm', 'moora']

    for dataset_name, results in all_datasets_results.items():
        print(f"\n{'#' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'#' * 80}")

        percentages = results['metadata']['removal_percentages']
        original = results['original_metrics']

        print(f"\nNetwork: {results['metadata']['hypergraph_order']} nodes, "
              f"{results['metadata']['hypergraph_size']} hyperedges")
        print(f"Tested percentages: {percentages}")
        print(f"Random trials: {results['metadata']['random_trials']}")

        print("\nExecution times:")
        for method in methods:
            exec_time = results[method]['execution_time']
            print(f"  {method.upper()}: {exec_time:.2f} seconds")

        # Find a representative percentage to display
        display_pct = 10 if 10 in percentages else percentages[
            0] if percentages else None

        if display_pct is not None:
            print(
                f"\nImpact at {display_pct}% removal (% change from original):")
            print("-" * 50)

            for metric_key in original.keys():
                orig_val = original[metric_key]
                if not isinstance(orig_val, (int, float)) or orig_val == 0:
                    continue

                print(f"\n{metric_key.replace('_', ' ').title()}:")

                for method in methods:
                    method_results = results[method]['experiments']
                    top_key = f'{method}_top_removal'

                    if top_key in method_results and str(display_pct) in \
                            method_results[top_key]:
                        method_val = method_results[top_key][str(display_pct)][
                            metric_key]
                        change = ((orig_val - method_val) / orig_val) * 100
                        print(f"  {method.upper():8s}: {change:+7.2f}%")

    print(f"\n{sep}")


def save_comparison_results(
        all_datasets_results: Dict[str, Dict[str, Any]],
        output_dir: Path
) -> None:
    """
    Save comparison results to JSON file.

    Args:
        all_datasets_results: Dictionary mapping dataset names to their results
        output_dir: Output directory
    """
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / 'comparison_results_all_datasets.json'

    # Convert results to JSON-serializable format
    json_results = {}
    for dataset_name, results in all_datasets_results.items():
        dataset_json = {}
        for key, value in results.items():
            if isinstance(value, dict):
                dataset_json[key] = value
            else:
                dataset_json[key] = str(value)
        json_results[dataset_name] = dataset_json

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {json_path}")


def save_single_dataset_results(
        dataset_name: str,
        results: Dict[str, Any],
        output_dir: Path
) -> None:
    """
    Save results for a single dataset to JSON file.

    Args:
        dataset_name: Name of the dataset
        results: Results dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / f'results_{dataset_name}.json'

    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = value
        else:
            json_results[key] = str(value)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {json_path}")


def load_all_dataset_results(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all individual dataset results from JSON files.

    Args:
        output_dir: Directory containing result files

    Returns:
        Dictionary mapping dataset names to their results
    """
    all_results = {}

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return all_results

    for json_file in output_dir.glob('results_*.json'):
        dataset_name = json_file.stem.replace('results_', '')

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results[dataset_name] = results
                print(f"Loaded results for: {dataset_name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return all_results


def main():
    """Main function to run the comparison experiments."""
    sep = "=" * 80

    # Configuration
    removal_percentages = [5, 10, 20]
    random_trials = 3
    output_dir = Path('comparison_results')
    data_dir = Path('data')

    # Available datasets
    available_datasets = [
        'iAF1260b',
        'hypercl_20',
        'hypercl_23',
        'hypercl_25',
        'Music-Rev',
        'Restaurants-Rev'
    ]

    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Combine mode: load all results and generate plots
        if arg == '--combine':
            print(sep)
            print("COMBINING RESULTS AND GENERATING PLOTS")
            print(sep)

            all_results = load_all_dataset_results(output_dir)

            if all_results:
                print(f"\nLoaded {len(all_results)} dataset results")
                print_comparison_summary(all_results)
                save_comparison_results(all_results, output_dir)
                generate_comparison_plots(all_results, output_dir)

                # Generate individual plots for each dataset
                print(f"\n{sep}")
                print("GENERATING INDIVIDUAL PLOTS FOR EACH DATASET")
                print(f"{sep}")
                for dataset_name, results in all_results.items():
                    generate_single_dataset_plot(dataset_name, results,
                                                 output_dir)

                print(f"\n{sep}")
                print("ALL PLOTS GENERATED SUCCESSFULLY")
                print(f"Results saved in: {output_dir.absolute()}")
                print(f"{sep}")
            else:
                print(f"\n{sep}")
                print("ERROR: No dataset results found to combine")
                print(f"Run individual datasets first, then use --combine")
                print(f"{sep}")
            return

        # Single dataset mode
        dataset_name = arg.replace('.txt', '')

        if dataset_name not in available_datasets:
            print(f"\n{sep}")
            print(f"ERROR: Unknown dataset '{dataset_name}'")
            print(f"\nAvailable datasets:")
            for ds in available_datasets:
                print(f"  - {ds}")
            print(f"\nUsage:")
            print(f"  python {sys.argv[0]} <dataset_name>")
            print(f"  python {sys.argv[0]} --combine")
            print(f"{sep}")
            return

        dataset_file = f"{dataset_name}.txt"

        print(sep)
        print("MCDM METHODS COMPARISON FOR HYPERGRAPH RESILIENCE")
        print(f"Dataset: {dataset_name}")
        print(sep)

        try:
            # Load hypernetwork
            hypernetwork = load_dataset_hypernetwork(dataset_file, data_dir)

            # Run comparison experiments
            print(f"\n{'=' * 60}")
            print(f"Running experiments with TOPSIS, WSM, and MOORA...")
            print(f"{'=' * 60}")

            results = run_comparison_experiments(
                hypernetwork=hypernetwork,
                removal_percentages=removal_percentages,
                random_trials=random_trials
            )

            # Save individual results
            save_single_dataset_results(dataset_name, results, output_dir)

            # Generate individual plot for this dataset
            generate_single_dataset_plot(dataset_name, results, output_dir)

            print(f"\n{sep}")
            print(f"COMPLETED: {dataset_name}")
            print(f"Results saved in: {output_dir.absolute()}")
            print(f"\nTo combine all results and generate plots, run:")
            print(f"  python {sys.argv[0]} --combine")
            print(f"{sep}")

        except FileNotFoundError as e:
            print(f"\n{sep}")
            print(f"ERROR: {e}")
            print(f"{sep}")
        except Exception as e:
            print(f"\n{sep}")
            print(f"ERROR processing {dataset_name}: {e}")
            print(f"{sep}")

    else:
        # No arguments: show usage
        print(sep)
        print("MCDM METHODS COMPARISON FOR HYPERGRAPH RESILIENCE")
        print(sep)
        print("\nUsage:")
        print(f"  Run single dataset:")
        print(f"    python {sys.argv[0]} <dataset_name>")
        print(f"\n  Combine results and generate plots:")
        print(f"    python {sys.argv[0]} --combine")
        print(f"\nAvailable datasets:")
        for ds in available_datasets:
            print(f"  - {ds}")
        print(f"\nExample workflow:")
        print(f"  python {sys.argv[0]} hypercl_20")
        print(f"  python {sys.argv[0]} hypercl_23")
        print(f"  python {sys.argv[0]} iAF1260b")
        print(f"  python {sys.argv[0]} --combine")
        print(f"{sep}")


if __name__ == "__main__":
    main()
