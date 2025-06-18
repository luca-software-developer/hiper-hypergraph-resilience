#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_resilience_experiments.py

Script to perform resilience experiments on hypergraphs by removing
nodes using random and TOPSIS-based strategies.

The experiments test the impact of removing different percentages of
nodes (1%, 2%, 5%, 10%, 25%) using:
- Random selection
- TOPSIS method for most relevant nodes (top)
- TOPSIS method for least relevant nodes (bottom)

Analyzed metrics include connectivity, redundancy coefficient, and
s-walk efficiency.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Plotting imports
import matplotlib.pyplot as plt
import numpy as np

# Import core classes
from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.experiments import ResilienceExperiment


def create_sample_hypernetwork() -> Hypernetwork:
    """
    Create a sample hypernetwork for testing experiments.

    Returns:
        Hypernetwork: hypernetwork with a test structure
    """
    hn = Hypernetwork()

    # Add hyperedges to create an interesting structure
    hyperedges = [
        (0, [1, 2, 3]),
        (1, [2, 3, 4]),
        (2, [3, 4, 5, 6]),
        (3, [5, 6, 7]),
        (4, [1, 7, 8, 9]),
        (5, [8, 9, 10]),
        (6, [10, 11, 12, 13]),
        (7, [12, 13, 14]),
        (8, [1, 5, 9, 13]),
        (9, [2, 6, 10, 14]),
    ]

    for he_id, nodes in hyperedges:
        hn.add_hyperedge(he_id, nodes)

    msg = (
        f"Sample hypernetwork created: {hn.order()} nodes, "
        f"{hn.size()} hyperedges"
    )
    print(msg)
    return hn


def load_hypernetwork_from_file(
        filepath: str
) -> Optional[Hypernetwork]:
    """
    Load a hypernetwork from a file (placeholder).

    Args:
        filepath: path to hypernetwork data file

    Returns:
        Loaded Hypernetwork or None if error
    """
    print(
        f"Loading from {filepath} not implemented. "
        "Using sample hypernetwork..."
    )
    return None


def setup_experiment_parameters() -> Dict[str, Any]:
    """
    Configure parameters for resilience experiments.

    Returns:
        Dictionary with experiment parameters
    """
    return {
        'removal_percentages': [1, 2, 5, 10, 25],
        'random_trials': 10,
        's_parameter': 1,
        'topsis_criteria_weights': None,
        'output_dir': Path('resilience_results'),
        'save_plots': True,
        'verbose': True,
    }


def run_comprehensive_analysis(
        hypernetwork: Hypernetwork,
        params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform full resilience analysis on the hypernetwork.

    Args:
        hypernetwork: target hypernetwork
        params: experiment parameters

    Returns:
        Comprehensive results dict
    """
    sep = "=" * 80
    print("\n" + sep)
    print("STARTING RESILIENCE EXPERIMENTS")
    print(sep)

    experiment = ResilienceExperiment(s=params['s_parameter'])

    print("\nAnalyzing hypernetwork:")
    print(f"  Order (nodes): {hypernetwork.order()}")
    print(f"  Size (hyperedges): {hypernetwork.size()}")
    print(f"  Average degree: {hypernetwork.avg_deg():.2f}")

    print("\nComputing baseline metrics...")
    start_time = time.time()

    results = experiment.run_node_removal_experiment(
        hypernetwork=hypernetwork,
        removal_percentages=params['removal_percentages'],
        random_trials=params['random_trials'],
    )

    elapsed = time.time() - start_time
    results['execution_time'] = elapsed
    results['parameters'] = params.copy()

    print(f"\nExperiments completed in {elapsed:.2f} seconds")
    return results


def generate_detailed_plots(
        results: Dict[str, Any],
        output_dir: Path
) -> None:
    """
    Generate detailed plots for results analysis.

    Args:
        results: experiment results
        output_dir: directory to save plots
    """
    output_dir.mkdir(exist_ok=True)

    percentages = results['removal_percentages']
    original = results['original_metrics']

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        'Resilience Analysis: Impact of Node Removal',
        fontsize=16,
        fontweight='bold'
    )

    metrics_info = [
        ('hypergraph_connectivity', 'Hypergraph Connectivity', axes[0, 0]),
        ('hyperedge_connectivity', 'Hyperedge Connectivity', axes[0, 1]),
        ('redundancy_coefficient', 'Redundancy Coefficient', axes[1, 0]),
        ('swalk_efficiency', 'S-Walk Efficiency', axes[1, 1]),
    ]

    for key, title, ax in metrics_info:
        if key not in original:
            continue
        rnd = [results['random_removal'][p][key] for p in percentages]
        top = [results['topsis_top_removal'][p][key] for p in percentages]
        bot = [results['topsis_bottom_removal'][p][key] for p in percentages]

        orig_val = original[key]
        if orig_val > 0:
            rnd_norm = [(v / orig_val) * 100 for v in rnd]
            top_norm = [(v / orig_val) * 100 for v in top]
            bot_norm = [(v / orig_val) * 100 for v in bot]
        else:
            rnd_norm, top_norm, bot_norm = rnd, top, bot

        ax.plot(
            percentages, rnd_norm, 'o-', linewidth=2, markersize=6,
            label='Random Removal'
        )
        ax.plot(
            percentages, top_norm, 's-', linewidth=2, markersize=6,
            label='TOPSIS Top'
        )
        ax.plot(
            percentages, bot_norm, '^-', linewidth=2, markersize=6,
            label='TOPSIS Bottom'
        )
        ax.axhline(
            y=100, color='black', linestyle='--', alpha=0.5,
            linewidth=1
        )
        ax.set_xlabel('Percentage of Nodes Removed (%)',
                      fontweight='bold')
        ax.set_ylabel('Metric Value (% of original)',
                      fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, max(percentages) * 1.05)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plot_path = output_dir / 'resilience_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")

    # Impact comparison plot
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    max_p = max(percentages)
    impacts = {}
    for key in original.keys():
        orig_val = original[key]
        if orig_val > 0:
            rnd_imp = abs(orig_val -
                          results['random_removal'][max_p][
                              key]) / orig_val * 100
            top_imp = abs(orig_val -
                          results['topsis_top_removal'][max_p][
                              key]) / orig_val * 100
            bot_imp = abs(orig_val -
                          results['topsis_bottom_removal'][max_p][
                              key]) / orig_val * 100
            impacts[key] = {'random': rnd_imp, 'top': top_imp,
                            'bottom': bot_imp}

    if impacts:
        names = list(impacts.keys())
        x_pos = np.arange(len(names))
        width = 0.25
        rnd_vals = [impacts[m]['random'] for m in names]
        top_vals = [impacts[m]['top'] for m in names]
        bot_vals = [impacts[m]['bottom'] for m in names]
        ax2.bar(x_pos - width, rnd_vals, width, label='Random')
        ax2.bar(x_pos, top_vals, width, label='TOPSIS Top')
        ax2.bar(x_pos + width, bot_vals, width, label='TOPSIS Bottom')
        ax2.set_xlabel('Metrics', fontweight='bold')
        ax2.set_ylabel(
            f'Impact (% change with {max_p}% removal)',
            fontweight='bold'
        )
        ax2.set_title(
            'Impact Comparison by Removal Strategy', fontweight='bold'
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(
            [
                's-walk Efficiency' if m == 'swalk_efficiency'
                else m.replace('_', ' ').title()
                for m in names
            ],
            rotation=45
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        impact_path = output_dir / 'impact_comparison.png'
        plt.savefig(impact_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {impact_path}")
    plt.show()


def save_results_to_file(
        results: Dict[str, Any], output_dir: Path
) -> None:
    """
    Save results as image files for matplotlib Figure objects and CSV for
    numeric data.

    Args:
        results: experiment results
        output_dir: output directory
    """
    import csv
    from matplotlib.figure import Figure

    output_dir.mkdir(exist_ok=True)

    # Save Figure objects
    for key, val in results.items():
        if isinstance(val, Figure):
            fig_path = output_dir / f"{key}.png"
            try:
                val.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved: {fig_path}")
            except Exception as e:
                print(f"Failed to save figure {key}: {e}")

    # Save summary CSV
    if 'removal_percentages' in results and 'original_metrics' in results:
        csv_path = output_dir / 'resilience_summary.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                'Removal_Percentage', 'Strategy', 'Metric', 'Value',
                'Change_%'
            ]
            writer.writerow(header)
            for p in results.get('removal_percentages', []):
                strategies = {}
                if 'random_removal' in results and p in results[
                    'random_removal']:
                    strategies['Random'] = results['random_removal'][p]
                if ('topsis_top_removal' in results and
                        p in results['topsis_top_removal']):
                    strategies['TOPSIS_Top'] = results[
                        'topsis_top_removal'][p]
                if ('topsis_bottom_removal' in results and
                        p in results['topsis_bottom_removal']):
                    strategies['TOPSIS_Bottom'] = results[
                        'topsis_bottom_removal'][p]
                for strategy, data in strategies.items():
                    for metric, val in data.items():
                        orig_val = results['original_metrics'].get(metric)
                        if (isinstance(orig_val, (int, float)) and
                                orig_val > 0 and isinstance(val, (int, float))):
                            change = ((orig_val - val) / orig_val) * 100
                        else:
                            change = 0
                        writer.writerow([
                            p, strategy, metric,
                            f"{val:.4f}" if isinstance(val, (int, float))
                            else str(val),
                            f"{change:.2f}"
                        ])
        print(f"Summary CSV saved: {csv_path}")


def print_executive_summary(results: Dict[str, Any]) -> None:
    """
    Print an executive summary of key results.

    Args:
        results: experiment results
    """
    sep = "=" * 80
    print("\n" + sep)
    print("EXECUTIVE SUMMARY - RESILIENCE ANALYSIS")
    print(sep)

    max_p = max(results.get('removal_percentages', [0]))
    original = results.get('original_metrics', {})

    print(f"\nImpact of removing {max_p}% of nodes:")
    print("-" * 50)
    for metric, orig_val in original.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        rnd_val = results.get('random_removal', {})
        rnd_val = rnd_val.get(max_p, {}).get(metric)
        top_val = results.get('topsis_top_removal', {})
        top_val = top_val.get(max_p, {}).get(metric)
        bot_val = results.get('topsis_bottom_removal', {})
        bot_val = bot_val.get(max_p, {}).get(metric)
        if isinstance(orig_val, (int, float)) and orig_val > 0:
            if rnd_val is not None:
                change = ((orig_val - rnd_val) / orig_val) * 100
                print(f"  • Random removal: {change:+.1f}% change")
            if top_val is not None:
                change = ((orig_val - top_val) / orig_val) * 100
                print(f"  • TOPSIS top (critical nodes): {change:+.1f}% change")
            if bot_val is not None:
                change = ((orig_val - bot_val) / orig_val) * 100
                print(f"  • TOPSIS bottom (peripheral): {change:+.1f}% change")
        impacts = {}
        if rnd_val is not None:
            impacts['random'] = abs(orig_val - rnd_val)
        if top_val is not None:
            impacts['top'] = abs(orig_val - top_val)
        if bot_val is not None:
            impacts['bottom'] = abs(orig_val - bot_val)
        if impacts:
            worst = max(impacts.items(), key=lambda x: x[1])
            print(f"  → Most damaging: {worst[0]}")
    exec_time = results.get('execution_time')
    if isinstance(exec_time, (int, float)):
        print(f"\nTotal execution time: {exec_time:.2f} seconds")
    trials = results.get('parameters', {}).get('random_trials')
    if trials is not None:
        print(f"Number of random trials per percentage: {trials}")


def main():
    """Main function to run the experiments."""
    sep = "=" * 80
    print(sep)
    print("RESILIENCE EXPERIMENTS FOR HYPERGRAPHS")
    print("Node removal using random and TOPSIS strategies")
    print(sep)

    params = setup_experiment_parameters()
    hypernetwork = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        hypernetwork = load_hypernetwork_from_file(file_path)
    if hypernetwork is None:
        hypernetwork = create_sample_hypernetwork()
    results = run_comprehensive_analysis(hypernetwork, params)
    out_dir = params['output_dir']
    save_results_to_file(results, out_dir)
    if params['save_plots']:
        generate_detailed_plots(results, out_dir)
    print_executive_summary(results)
    print(f"\n{sep}")
    print("EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"Results saved in: {out_dir.absolute()}")
    print(f"{sep}")


if __name__ == "__main__":
    main()
