#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_statistical_analysis.py

Script to perform statistical analysis on hypergraph structural features:
- Structural feature comparison across different hypergraph families
- ANOVA/Kruskal-Wallis tests for cross-domain analysis
- Normalized metrics for size-independent comparisons

The analysis examines structural diversity across hypergraph families
from different domains.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, cast, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kruskal, f_oneway, shapiro


def load_hypergraph_from_file(file_path: Path) -> Tuple[
    Set[int], List[Set[int]]]:
    """
    Load hypergraph from a text file.

    Args:
        file_path: Path to the hypergraph file

    Returns:
        Tuple of (nodes, hyperedges) where nodes is a set of node IDs
        and hyperedges is a list of sets containing node IDs
    """
    nodes = set()
    hyperedges = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse node IDs from the line
                    node_ids = [int(x) for x in line.split()]
                    if node_ids:
                        hyperedge = set(node_ids)
                        hyperedges.append(hyperedge)
                        nodes.update(hyperedge)
    except Exception as e:
        print(f"Error loading hypergraph from {file_path}: {e}")
        return set(), []

    return nodes, hyperedges


def compute_hypergraph_metrics(nodes: Set[int], hyperedges: List[Set[int]]) -> \
        Dict[str, Any]:
    """
    Compute structural metrics for a hypergraph.

    Args:
        nodes: Set of node IDs
        hyperedges: List of hyperedges (each is a set of node IDs)

    Returns:
        Dictionary with computed metrics
    """
    if not nodes or not hyperedges:
        return {}

    order = len(nodes)
    size = len(hyperedges)

    # Compute node degrees
    node_degrees = {node: 0 for node in nodes}
    for hyperedge in hyperedges:
        for node in hyperedge:
            node_degrees[node] += 1

    # Compute hyperedge sizes
    hyperedge_sizes = [len(he) for he in hyperedges]

    # Average degree
    avg_degree = sum(node_degrees.values()) / order if order > 0 else 0

    # Average hyperedge size
    avg_hyperedge_size = sum(hyperedge_sizes) / size if size > 0 else 0

    # Density (hyperedges per node)
    density = avg_degree / order if order > 0 else 0

    # Hyperedge-node ratio
    hyperedge_node_ratio = size / order if order > 0 else 0

    # Normalized metrics
    # Normalized density: actual density / maximum possible density
    max_possible_edges = order * (order - 1) / 2 if order > 1 else 1
    total_connections = sum(len(he) * (len(he) - 1) / 2 for he in hyperedges)
    normalized_density = total_connections / max_possible_edges if max_possible_edges > 0 else 0

    # Clustering coefficient for hypergraphs (local density)
    # Measures how well-connected the neighborhood of each node is
    clustering_coeffs = []
    for node in nodes:
        # Find hyperedges containing this node
        node_hyperedges = [he for he in hyperedges if node in he]
        if len(node_hyperedges) < 2:
            clustering_coeffs.append(0)
            continue

        # Count triangular connections through hyperedges
        neighbors = set()
        for he in node_hyperedges:
            neighbors.update(he - {node})

        if len(neighbors) < 2:
            clustering_coeffs.append(0)
            continue

        # Actual connections between neighbors
        actual_connections = 0
        for he in hyperedges:
            neighbor_intersection = neighbors.intersection(he)
            if len(neighbor_intersection) >= 2:
                actual_connections += len(neighbor_intersection) * (
                        len(neighbor_intersection) - 1) / 2

        # Possible connections between neighbors
        possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
        clustering_coeffs.append(
            actual_connections / possible_connections if possible_connections > 0 else 0)

    avg_clustering = sum(clustering_coeffs) / len(
        clustering_coeffs) if clustering_coeffs else 0

    # Degree variance (measure of heterogeneity)
    degree_variance = np.var(list(node_degrees.values())) if node_degrees else 0

    # Hyperedge size variance (measure of hyperedge heterogeneity)
    hyperedge_size_variance = np.var(hyperedge_sizes) if hyperedge_sizes else 0

    # Effective size (entropy-based measure)
    # Higher entropy indicates more diverse hyperedge sizes
    if hyperedge_sizes:
        size_counts = {}
        for size_val in hyperedge_sizes:
            size_counts[size_val] = size_counts.get(size_val, 0) + 1

        entropy = 0
        total_hyperedges = len(hyperedge_sizes)
        for count in size_counts.values():
            prob = count / total_hyperedges
            if prob > 0:
                entropy -= prob * np.log2(prob)

        hyperedge_size_entropy = entropy
    else:
        hyperedge_size_entropy = 0

    return {
        'order': order,
        'size': size,
        'avg_degree': avg_degree,
        'avg_hyperedge_size': avg_hyperedge_size,
        'density': density,
        'hyperedge_node_ratio': hyperedge_node_ratio,
        # Normalized metrics
        'normalized_density': normalized_density,
        'avg_clustering': avg_clustering,
        'degree_variance': degree_variance,
        'hyperedge_size_variance': hyperedge_size_variance,
        'hyperedge_size_entropy': hyperedge_size_entropy
    }


def load_data_directory_hypergraphs(data_dir: Path) -> Dict[
    str, Dict[str, Any]]:
    """
    Load and analyze all hypergraphs from the data directory.

    Args:
        data_dir: Path to the data directory

    Returns:
        Dictionary mapping dataset names to their structural features
    """
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist")
        return {}

    features_dict = {}
    print(f"\nLoading hypergraphs from {data_dir}...")

    # Get all .txt files in the data directory
    hypergraph_files = list(data_dir.glob('*.txt'))

    for file_path in hypergraph_files:
        dataset_name = file_path.stem
        print(f"  Processing {dataset_name}...")

        # Load hypergraph
        nodes, hyperedges = load_hypergraph_from_file(file_path)

        if nodes and hyperedges:
            # Compute metrics
            metrics = compute_hypergraph_metrics(nodes, hyperedges)

            if metrics:
                # Add family information
                metrics['family'] = dataset_name.split('_')[
                    0] if '_' in dataset_name else dataset_name
                features_dict[dataset_name] = metrics

                print(
                    f"    Order: {metrics['order']}, Size: {metrics['size']}, "
                    f"Avg degree: {metrics['avg_degree']:.2f}")
        else:
            print(f"    Failed to load hypergraph from {file_path}")

    print(f"Loaded {len(features_dict)} hypergraphs from data directory")
    return features_dict


def load_analysis_results(results_dir: Path) -> Dict[str, Any]:
    """
    Load existing analysis results from JSON files.
    
    Args:
        results_dir: Directory containing analysis results
        
    Returns:
        Dictionary containing loaded analysis data
    """
    analysis_data = {}

    # Load analysis summary
    summary_file = results_dir / 'analysis_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            analysis_data['summary'] = json.load(f)
            print(f"Loaded analysis summary from {summary_file}")

    # Load perturbation results
    perturbation_files = list(results_dir.glob('perturbation_results_*.json'))
    if perturbation_files:
        latest_perturbation = max(perturbation_files,
                                  key=lambda x: x.stat().st_mtime)
        with open(latest_perturbation, 'r', encoding='utf-8') as f:
            analysis_data['perturbation'] = json.load(f)
            print(f"Loaded perturbation results from {latest_perturbation}")

    return analysis_data


def _extract_features_from_metrics(orig: Dict[str, Any], dataset_name: str) -> \
        Dict[str, Any]:
    """
    Extract structural features from original metrics.

    Args:
        orig: Original metrics dictionary
        dataset_name: Name of the dataset

    Returns:
        Dictionary with extracted structural features
    """
    features = {
        'order': orig['order'],
        'size': orig['size'],
        'avg_degree': orig['avg_degree'],
        'avg_hyperedge_size': orig['avg_hyperedge_size'],
        'density': orig['avg_degree'] / orig['order'] if orig[
                                                             'order'] > 0 else 0,
        'hyperedge_node_ratio': orig['size'] / orig['order'] if orig[
                                                                    'order'] > 0 else 0,
        'family': dataset_name.split('_')[
            0] if '_' in dataset_name else dataset_name
    }

    # Add normalized metrics if available
    normalized_metrics = ['normalized_density', 'avg_clustering',
                          'degree_variance',
                          'hyperedge_size_variance', 'hyperedge_size_entropy']
    for metric in normalized_metrics:
        if metric in orig:
            features[metric] = orig[metric]

    return features


def extract_structural_features(data: Dict[str, Any]) -> Dict[
    str, Dict[str, Any]]:
    """
    Extract structural features from hypergraph data.
    
    Args:
        data: Analysis data dictionary
        
    Returns:
        Dictionary with structural features for each dataset
    """
    features_dict = {}

    # Extract from summary data
    if 'summary' in data and 'dataset_summaries' in data['summary']:
        for dataset_name, dataset_info in data['summary'][
            'dataset_summaries'].items():
            if 'original_metrics' in dataset_info:
                orig = dataset_info['original_metrics']
                features_dict[dataset_name] = _extract_features_from_metrics(
                    orig, dataset_name)

    # Extract from perturbation data if summary not available
    if not features_dict and 'perturbation' in data:
        for key, dataset_info in data['perturbation'].items():
            if 'original_metrics' in dataset_info and 'dataset' in dataset_info:
                dataset_name = dataset_info['dataset']
                orig = dataset_info['original_metrics']
                features_dict[dataset_name] = _extract_features_from_metrics(
                    orig, dataset_name)

    return features_dict


def extract_resilience_metrics(data: Dict[str, Any]) -> Dict[
    str, List[Dict[str, Any]]]:
    """
    Extract resilience metrics from experimental data.
    
    Args:
        data: Analysis data dictionary
        
    Returns:
        Dictionary with resilience metrics for each dataset and strategy
    """
    resilience_dict = {'random': [], 'topsis': []}

    if 'summary' in data and 'dataset_summaries' in data['summary']:
        for dataset_name, dataset_info in data['summary'][
            'dataset_summaries'].items():
            original = dataset_info['original_metrics']
            family = dataset_name.split('_')[
                0] if '_' in dataset_name else dataset_name

            # Random targeting resilience
            if 'max_degradation_random' in dataset_info:
                random_deg = dataset_info['max_degradation_random']
                resilience_random = {'dataset': dataset_name, 'family': family}

                for metric, info in random_deg.items():
                    if metric in original and original[metric] > 0:
                        resilience_random[f'{metric}_resilience'] \
                            = (original[metric] - info['final_value']) / \
                              original[metric]

                resilience_dict['random'].append(resilience_random)

            # TOPSIS targeting resilience
            if 'max_degradation_topsis' in dataset_info:
                topsis_deg = dataset_info['max_degradation_topsis']
                resilience_topsis = {'dataset': dataset_name, 'family': family}

                for metric, info in topsis_deg.items():
                    if metric in original and original[metric] > 0:
                        resilience_topsis[f'{metric}_resilience'] \
                            = (original[metric] - info['final_value']) / \
                              original[metric]

                resilience_dict['topsis'].append(resilience_topsis)

    return resilience_dict


def perform_correlation_analysis(features_dict: Dict[str, Dict[str, Any]],
                                 resilience_dict: Dict[str,
                                 List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Perform correlation analysis between features and resilience metrics.
    
    Args:
        features_dict: Dictionary with structural features
        resilience_dict: Dictionary with resilience metrics
        
    Returns:
        Dictionary containing correlation results
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    feature_cols = ['order', 'size', 'avg_degree', 'avg_hyperedge_size',
                    'density', 'hyperedge_node_ratio', 'normalized_density',
                    'avg_clustering', 'degree_variance',
                    'hyperedge_size_variance',
                    'hyperedge_size_entropy']
    results = {'correlations': {}}

    for strategy, resilience_list in resilience_dict.items():
        if not resilience_list:
            continue

        print(f"\n--- {strategy.upper()} TARGETING STRATEGY ---")
        strategy_results = {}

        # Get all possible resilience metrics from first entry
        sample_entry = resilience_list[0]
        resilience_cols = [col for col in sample_entry.keys() if
                           col.endswith('_resilience')]

        for resilience_col in resilience_cols:
            resilience_metric = resilience_col.replace('_resilience', '')
            print(f"\nResilience metric: {resilience_metric}")
            print("-" * 40)

            feature_results = {}

            for feature_col in feature_cols:
                # Collect data points
                x_values = []
                y_values = []

                for resilience_entry in resilience_list:
                    dataset = resilience_entry['dataset']
                    if dataset in features_dict and resilience_col in resilience_entry:
                        x_val = features_dict[dataset].get(feature_col)
                        y_val = resilience_entry.get(resilience_col)

                        if x_val is not None and y_val is not None:
                            x_values.append(x_val)
                            y_values.append(y_val)

                if len(x_values) < 3:
                    print(
                        f"  {feature_col}: Insufficient data (n={len(x_values)})")
                    continue

                # Convert to numpy arrays for scipy functions
                x_vals = np.array(x_values)
                y_vals = np.array(y_values)

                # Check for constant arrays (zero variance)
                if np.var(x_vals) == 0 or np.var(y_vals) == 0:
                    print(
                        f"  {feature_col}: Skipped (constant values - no variance)")
                    continue

                # Pearson correlation
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*range zero.*")
                    pearson_r, pearson_p = pearsonr(cast(Any, x_vals),
                                                    cast(Any, y_vals))

                # Spearman correlation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*range zero.*")
                    spearman_rho, spearman_p = spearmanr(cast(Any, x_vals),
                                                         cast(Any, y_vals))

                feature_results[feature_col] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_rho': spearman_rho,
                    'spearman_p': spearman_p,
                    'n_samples': len(x_values)
                }

                # Print results
                print(f"  {feature_col}:")
                print(f"    Pearson r = {pearson_r:.4f}, p = {pearson_p:.4f}")
                print(
                    f"    Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.4f}")
                print(f"    n = {len(x_values)}")

            strategy_results[resilience_col] = feature_results

        results['correlations'][strategy] = strategy_results

    return results


def _perform_statistical_test(groups: List[List[float]],
                              family_names: List[str]) -> Dict[str, Any]:
    """
    Perform appropriate statistical test based on normality.
    
    Args:
        groups: List of data groups to compare
        family_names: Names of the families/groups
        
    Returns:
        Dictionary with test results
    """
    # Check normality assumption
    import warnings
    normality_ok = True
    for group in groups:
        if len(group) >= 3:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*range zero.*")
                _, p_norm = shapiro(group)
            if p_norm < 0.05:
                normality_ok = False
                break

    # Perform appropriate test
    if normality_ok and len(groups) > 1:
        # ANOVA (parametric)
        f_stat, p_value = f_oneway(*groups)
        test_used = 'ANOVA'
        statistic = f_stat
    else:
        # Kruskal-Wallis (non-parametric)
        h_stat, p_value = kruskal(*groups)
        test_used = 'Kruskal-Wallis'
        statistic = h_stat

    return {
        'test_used': test_used,
        'statistic': statistic,
        'p_value': p_value,
        'families': family_names,
        'group_sizes': [len(group) for group in groups]
    }


def _group_data_by_family(data_dict: Dict[str, Dict[str, Any]],
                          metric_key: str) -> Dict[str, List[float]]:
    """
    Group data by family for statistical comparison.
    
    Args:
        data_dict: Dictionary containing data to group
        metric_key: Key to extract values from data
        
    Returns:
        Dictionary mapping family names to lists of values
    """
    family_data = {}
    for item_key, item_data in data_dict.items():
        if isinstance(item_data, dict):
            family = item_data.get('family', item_key.split('_')[
                0] if '_' in item_key else item_key)
            if family not in family_data:
                family_data[family] = []
            if metric_key in item_data and item_data[metric_key] is not None:
                family_data[family].append(item_data[metric_key])
    return {k: v for k, v in family_data.items() if len(v) > 0}


def _group_resilience_by_family(resilience_list: List[Dict[str, Any]],
                                metric_key: str) -> Dict[str, List[float]]:
    """
    Group resilience data by family for statistical comparison.
    
    Args:
        resilience_list: List of resilience entries
        metric_key: Key to extract resilience values
        
    Returns:
        Dictionary mapping family names to lists of resilience values
    """
    family_data = {}
    for entry in resilience_list:
        family = entry.get('family', entry.get('dataset', '').split('_')[
            0] if '_' in entry.get('dataset', '') else entry.get('dataset', ''))
        if family not in family_data:
            family_data[family] = []
        if metric_key in entry and entry[metric_key] is not None:
            family_data[family].append(entry[metric_key])
    return {k: v for k, v in family_data.items() if len(v) > 0}


def perform_family_comparison(features_dict: Dict[str, Dict[str, Any]],
                              resilience_dict: Dict[
                                  str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Perform ANOVA/Kruskal-Wallis tests to compare metrics across hypergraphs.
    
    Args:
        features_dict: Dictionary with structural features
        resilience_dict: Dictionary with resilience metrics
        
    Returns:
        Dictionary containing comparison test results
    """
    print("\n" + "=" * 60)
    print("FAMILY COMPARISON ANALYSIS")
    print("=" * 60)

    results = {'structural_tests': {}, 'resilience_tests': {}}

    # Test structural features across families
    print("\n--- STRUCTURAL FEATURES COMPARISON ---")

    feature_cols = ['order', 'size', 'avg_degree', 'avg_hyperedge_size',
                    'density', 'hyperedge_node_ratio', 'normalized_density',
                    'avg_clustering', 'degree_variance',
                    'hyperedge_size_variance',
                    'hyperedge_size_entropy']

    for feature_col in feature_cols:
        print(f"\nFeature: {feature_col}")
        print("-" * 30)

        # Group data by family using helper function
        valid_families = _group_data_by_family(features_dict, feature_col)

        if len(valid_families) < 2:
            print(
                f"  Insufficient families for comparison (found {len(valid_families)} family/families)")
            continue

        groups = list(valid_families.values())
        family_names = list(valid_families.keys())

        # Perform statistical test using helper function
        test_results = _perform_statistical_test(groups, family_names)
        results['structural_tests'][feature_col] = test_results

        print(f"  Test: {test_results['test_used']}")
        print(f"  Statistic: {test_results['statistic']:.4f}")
        print(f"  p-value: {test_results['p_value']:.4f}")
        print(f"  Families: {test_results['families']}")

    # Test resilience metrics across families and strategies
    print("\n--- RESILIENCE METRICS COMPARISON ---")

    for strategy, resilience_list in resilience_dict.items():
        if not resilience_list:
            continue

        print(f"\nStrategy: {strategy.upper()}")
        print("-" * 30)

        # Get resilience columns from first entry
        sample_entry = resilience_list[0]
        resilience_cols = [col for col in sample_entry.keys() if
                           col.endswith('_resilience')]

        for resilience_col in resilience_cols:
            resilience_metric = resilience_col.replace('_resilience', '')
            print(f"\n  Resilience metric: {resilience_metric}")

            # Group data by family using helper function
            valid_families = _group_resilience_by_family(resilience_list,
                                                         resilience_col)

            if len(valid_families) < 2:
                print("    Insufficient families for comparison")
                continue

            groups = list(valid_families.values())
            family_names = list(valid_families.keys())

            # Perform statistical test using helper function
            test_results = _perform_statistical_test(groups, family_names)

            test_key = f"{strategy}_{resilience_col}"
            results['resilience_tests'][test_key] = test_results

            print(f"    Test: {test_results['test_used']}")
            print(f"    Statistic: {test_results['statistic']:.4f}")
            print(f"    p-value: {test_results['p_value']:.4f}")
            print(f"    Families: {test_results['families']}")

    return results


def _create_subplot_grid_with_boxplots(features_dict: Dict[str, Dict[str, Any]],
                                       feature_list: List[str], plot_title: str,
                                       subplot_count: int = 6) -> Tuple:
    """
    Helper function to create a subplot grid with box plots for given features.

    Args:
        features_dict: Dictionary with structural features
        feature_list: List of feature names to plot
        plot_title: Title for the entire plot
        subplot_count: Number of subplots to create (default 6 for 2x3 grid)

    Returns:
        Tuple of (figure, axes) objects
    """

    def _truncate_label(label: str, max_length: int = 12) -> str:
        """Truncate long labels for better display."""
        if len(label) <= max_length:
            return label
        else:
            return label[:max_length - 3] + '...'

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # Increased figure size
    axes = axes.flatten()

    for i, feature_col in enumerate(feature_list):
        if i >= subplot_count:
            break

        # Group data by family using helper function
        family_data = _group_data_by_family(features_dict, feature_col)

        # Create box plot
        if family_data:
            families = list(family_data.keys())
            data_lists = [family_data[family] for family in families]

            # Truncate long family names for display
            display_families = [_truncate_label(f) for f in families]

            axes[i].boxplot(data_lists, tick_labels=display_families)
            axes[i].set_title(feature_col.replace('_', ' ').title(),
                              fontsize=12, fontweight='bold')

            # Improve label formatting
            axes[i].tick_params(axis='x', rotation=45, labelsize=9)
            axes[i].tick_params(axis='y', labelsize=9)

            # Add grid for better readability
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(feature_list), subplot_count):
        if i < len(axes):
            axes[i].set_visible(False)

    plt.suptitle(plot_title, fontsize=16, fontweight='bold')
    plt.tight_layout(
        rect=(0, 0.03, 1, 0.95))  # Adjust layout to prevent overlap

    return fig, axes


def _create_features_boxplot(features_dict: Dict[str, Dict[str, Any]],
                             output_dir: Path,
                             title: str,
                             filename: str) -> None:
    """
    Create box plots for structural features by family.

    Args:
        features_dict: Dictionary with structural features
        output_dir: Directory to save the plot
        title: Title for the plot
        filename: Filename for saving the plot
    """
    # Basic features
    basic_features = ['order', 'size', 'avg_degree', 'avg_hyperedge_size',
                      'density', 'hyperedge_node_ratio']

    fig, _ = _create_subplot_grid_with_boxplots(features_dict, basic_features,
                                                title)
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Normalized features
    normalized_features = ['normalized_density', 'avg_clustering',
                           'degree_variance',
                           'hyperedge_size_variance', 'hyperedge_size_entropy']

    # Check if any dataset has normalized features
    has_normalized = any(
        any(feature in dataset for feature in normalized_features)
        for dataset in features_dict.values())

    if has_normalized:
        fig, _ = _create_subplot_grid_with_boxplots(
            features_dict,
            normalized_features,
            f"{title} - Normalized Metrics"
        )
        normalized_filename = filename.replace('.png', '_normalized.png')
        plt.savefig(output_dir / normalized_filename, dpi=300,
                    bbox_inches='tight')
        plt.close()


def _create_structural_correlations_plot(
        features_dict: Dict[str, Dict[str, Any]],
        output_dir: Path) -> None:
    """
    Create correlation plots between structural features.

    Args:
        features_dict: Dictionary with structural features
        output_dir: Directory to save the plot
    """
    # Convert to DataFrame for easier analysis
    data_list = []
    for dataset_name, features in features_dict.items():
        row = {'dataset': dataset_name,
               'family': features.get('family', 'unknown')}
        # Add all numerical features
        numerical_features = ['order', 'size', 'avg_degree',
                              'avg_hyperedge_size',
                              'density', 'hyperedge_node_ratio',
                              'normalized_density',
                              'avg_clustering', 'degree_variance',
                              'hyperedge_size_variance',
                              'hyperedge_size_entropy']

        for feature in numerical_features:
            if feature in features and features[feature] is not None:
                row[feature] = features[feature]

        data_list.append(row)

    df = pd.DataFrame(data_list)

    # Select only numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'dataset' in numerical_cols:
        numerical_cols.remove('dataset')

    if len(numerical_cols) < 2:
        print("Not enough numerical features for correlation analysis")
        return

    # Create correlation matrix
    corr_matrix = df[numerical_cols].corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Structural Features Correlation Matrix', fontsize=16,
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'structural_correlations_heatmap.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    # Create scatter plots for most interesting correlations
    _create_feature_scatter_plots(df, numerical_cols, output_dir)


def _create_feature_scatter_plots(df: pd.DataFrame,
                                  numerical_cols: List[str],
                                  output_dir: Path) -> None:
    """
    Create scatter plots for selected feature pairs.

    Args:
        df: DataFrame with features
        numerical_cols: List of numerical column names
        output_dir: Directory to save plots
    """
    # Select interesting feature pairs
    interesting_pairs = [
        ('order', 'size'),
        ('avg_degree', 'density'),
        ('order', 'avg_clustering'),
        ('hyperedge_node_ratio', 'normalized_density'),
        ('degree_variance', 'hyperedge_size_variance'),
        ('avg_hyperedge_size', 'hyperedge_size_entropy')
    ]

    # Filter pairs that exist in data
    available_pairs = [(x, y) for x, y in interesting_pairs
                       if x in numerical_cols and y in numerical_cols]

    if not available_pairs:
        print("No suitable feature pairs found for scatter plots")
        return

    # Create scatter plot grid
    n_pairs = len(available_pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Initialize families variable for later use
    families = df['family'].unique()

    for i, (x_feature, y_feature) in enumerate(available_pairs):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Create scatter plot colored by family
        colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(families)))

        for family, color in zip(families, colors):
            family_data = df[df['family'] == family]
            if len(family_data) > 0:
                ax.scatter(family_data[x_feature], family_data[y_feature],
                           c=[color], label=family, alpha=0.7, s=60)

        ax.set_xlabel(x_feature.replace('_', ' ').title())
        ax.set_ylabel(y_feature.replace('_', ' ').title())
        ax.set_title(
            f'{x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr_coef = df[x_feature].corr(df[y_feature])
        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Hide unused subplots
    for i in range(len(available_pairs), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    # Add legend
    if len(available_pairs) > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.02),
                   ncol=min(len(families), 6))

    plt.suptitle('Structural Features Relationships by Family', fontsize=16,
                 fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_dir / 'structural_features_scatter.png', dpi=300,
                bbox_inches='tight')
    plt.close()


def generate_statistical_plots(features_dict: Dict[str, Dict[str, Any]],
                               resilience_dict: Dict[str, List[Dict[str, Any]]],
                               correlation_results: Dict[str, Any],
                               output_dir: Path) -> None:
    """
    Generate plots for statistical analysis results.
    
    Args:
        features_dict: Dictionary with structural features
        resilience_dict: Dictionary with resilience metrics
        correlation_results: Results from correlation analysis
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)

    print("Generating statistical plots...")

    # Scatter plots for significant correlations
    significant_correlations = []
    for strategy, strategy_results in correlation_results.get('correlations',
                                                              {}).items():
        for resilience_col, feature_results in strategy_results.items():
            for feature_col, corr_data in feature_results.items():
                if corr_data['pearson_p'] < 0.05:
                    significant_correlations.append({
                        'strategy': strategy,
                        'resilience': resilience_col,
                        'feature': feature_col,
                        'pearson_r': corr_data['pearson_r'],
                        'pearson_p': corr_data['pearson_p']
                    })

    if significant_correlations:
        n_plots = min(len(significant_correlations),
                      6)  # Limit to 6 most significant
        significant_correlations.sort(key=lambda x: x['pearson_p'])

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, corr_info in enumerate(significant_correlations[:n_plots]):
            # Collect data points
            x_data = []
            y_data = []

            resilience_list = resilience_dict[corr_info['strategy']]
            for entry in resilience_list:
                dataset = entry['dataset']
                if dataset in features_dict and corr_info[
                    'resilience'] in entry:
                    x_val = features_dict[dataset].get(corr_info['feature'])
                    y_val = entry.get(corr_info['resilience'])

                    if x_val is not None and y_val is not None:
                        x_data.append(x_val)
                        y_data.append(y_val)

            if len(x_data) > 0:
                axes[i].scatter(x_data, y_data, alpha=0.7, s=80)

                # Add trend line
                if len(x_data) > 1:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_data), max(x_data), 100)
                    axes[i].plot(x_trend, p(x_trend), "r--", alpha=0.8)

                axes[i].set_xlabel(
                    corr_info['feature'].replace('_', ' ').title())
                axes[i].set_ylabel(
                    corr_info['resilience'].replace('_', ' ').title())
                axes[i].set_title(
                    f"{corr_info['strategy'].title()} - r={corr_info['pearson_r']:.3f}, p={corr_info['pearson_p']:.3f}")
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Significant Correlations: Features vs Resilience Metrics',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'significant_correlations.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Box plots for family comparisons
    _create_features_boxplot(features_dict, output_dir,
                             'Structural Features by Hypergraph Family',
                             'structural_features_by_family.png')

    print(f"Statistical plots saved in {output_dir}")


def save_statistical_results(correlation_results: Dict[str, Any],
                             comparison_results: Dict[str, Any],
                             output_dir: Path) -> None:
    """
    Save statistical analysis results to files.
    
    Args:
        correlation_results: Correlation analysis results
        comparison_results: Family comparison results
        output_dir: Directory to save results
    """
    output_dir.mkdir(exist_ok=True)

    # Save correlation results to JSON
    def clean_nan_values(obj):
        """Replace NaN values with None for JSON serialization."""
        if isinstance(obj, dict):
            return {k: clean_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan_values(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj

    cleaned_correlation_results = clean_nan_values(correlation_results)
    correlation_file = output_dir / 'correlation_analysis.json'
    with open(correlation_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_correlation_results, f, indent=2, default=str)
    print(f"Correlation results saved: {correlation_file}")

    # Save comparison results to JSON
    comparison_file = output_dir / 'family_comparison_tests.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"Comparison test results saved: {comparison_file}")

    # Save summary to CSV
    summary_data = []

    # Add correlation summary
    for strategy, strategy_results in correlation_results.get('correlations',
                                                              {}).items():
        for resilience_col, feature_results in strategy_results.items():
            for feature_col, corr_data in feature_results.items():
                summary_data.append([
                    'correlation', strategy, resilience_col, feature_col,
                    corr_data['pearson_r'], corr_data['pearson_p'],
                    'Pearson', corr_data['n_samples']
                ])
                summary_data.append([
                    'correlation', strategy, resilience_col, feature_col,
                    corr_data['spearman_rho'], corr_data['spearman_p'],
                    'Spearman', corr_data['n_samples']
                ])

    # Add comparison test summary
    for test_name, test_results in comparison_results.get('structural_tests',
                                                          {}).items():
        summary_data.append([
            'family_comparison', 'structural', test_name, test_name,
            test_results['statistic'], test_results['p_value'],
            test_results['test_used'], sum(test_results['group_sizes'])
        ])

    for test_name, test_results in comparison_results.get('resilience_tests',
                                                          {}).items():
        strategy, metric = test_name.split('_', 1)
        summary_data.append([
            'family_comparison', strategy, metric, 'family',
            test_results['statistic'], test_results['p_value'],
            test_results['test_used'], sum(test_results['group_sizes'])
        ])

    # Save summary CSV
    if summary_data:
        summary_file = output_dir / 'statistical_analysis_summary.csv'
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'analysis_type', 'strategy', 'resilience_metric',
                'structural_feature',
                'test_statistic', 'p_value', 'test_type', 'n_samples'
            ])
            writer.writerows(summary_data)
        print(f"Analysis summary saved: {summary_file}")


def print_executive_summary(correlation_results: Dict[str, Any],
                            comparison_results: Dict[str, Any]) -> None:
    """
    Print executive summary of statistical analysis results.
    
    Args:
        correlation_results: Correlation analysis results
        comparison_results: Family comparison results
    """
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY - STATISTICAL ANALYSIS")
    print("=" * 80)

    # Correlation analysis summary
    print("\n--- CORRELATION ANALYSIS SUMMARY ---")
    significant_correlations = 0
    total_correlations = 0

    for strategy, strategy_results in correlation_results.get('correlations',
                                                              {}).items():
        print(f"\n{strategy.upper()} targeting strategy:")
        strategy_significant = 0
        strategy_total = 0

        for resilience_col, feature_results in strategy_results.items():
            for feature_col, corr_data in feature_results.items():
                strategy_total += 1
                total_correlations += 1

                if corr_data['pearson_p'] < 0.05:
                    strategy_significant += 1
                    significant_correlations += 1
                    print(
                        f"  • {feature_col} <-> {resilience_col}: r={corr_data['pearson_r']:.3f}, p={corr_data['pearson_p']:.3f} *")

        if strategy_total > 0:
            print(
                f"  Significant correlations: {strategy_significant}/{strategy_total} ({strategy_significant / strategy_total * 100:.1f}%)")

    if total_correlations > 0:
        print(
            f"\nOverall significant correlations: {significant_correlations}/{total_correlations} ({significant_correlations / total_correlations * 100:.1f}%)")
    else:
        print("\nNo correlations analyzed")

    # Family comparison summary
    print("\n--- FAMILY COMPARISON SUMMARY ---")

    print("\nData Directory Structural Features:")
    structural_significant = 0
    for feature, test_result in comparison_results.get('structural_tests',
                                                       {}).items():
        if test_result['p_value'] < 0.05:
            structural_significant += 1
            print(
                f"  • {feature}: {test_result['test_used']} p={test_result['p_value']:.3f} *")
        else:
            print(
                f"  • {feature}: {test_result['test_used']} p={test_result['p_value']:.3f}")

    total_struct = len(comparison_results.get('structural_tests', {}))
    if total_struct > 0:
        print(
            f"  Significant differences: {structural_significant}/{total_struct} ({structural_significant / total_struct * 100:.1f}%)")

    print("\nResilience metrics:")
    print("  • Data directory lacks resilience measurements")
    print("  • Resilience analysis only available for Results directory")
    resilience_significant = 0
    for test_name, test_result in comparison_results.get('resilience_tests',
                                                         {}).items():
        strategy, metric = test_name.split('_', 1)
        if test_result['p_value'] < 0.05:
            resilience_significant += 1
            print(
                f"  • {strategy} {metric}: {test_result['test_used']} p={test_result['p_value']:.3f} *")
        else:
            print(
                f"  • {strategy} {metric}: {test_result['test_used']} p={test_result['p_value']:.3f}")

    total_resil = len(comparison_results.get('resilience_tests', {}))
    if total_resil > 0:
        print(
            f"  Significant differences: {resilience_significant}/{total_resil} ({resilience_significant / total_resil * 100:.1f}%)")

    print("\n* indicates statistical significance (p < 0.05)")


def perform_data_directory_analysis(
        data_features_dict: Dict[str, Dict[str, Any]],
        output_dir: Path) -> Dict[str, Any]:
    """
    Perform statistical analysis on hypergraphs from data directory.

    Args:
        data_features_dict: Features extracted from data directory hypergraphs
        output_dir: Directory to save results

    Returns:
        Dictionary containing analysis results
    """
    print("\n" + "=" * 60)
    print("DATA DIRECTORY HYPERGRAPH ANALYSIS")
    print("=" * 60)

    results = {'data_comparison_tests': {}}

    # Test structural features across families
    print("\n--- STRUCTURAL FEATURES COMPARISON (DATA DIRECTORY) ---")

    feature_cols = ['order', 'size', 'avg_degree', 'avg_hyperedge_size',
                    'density', 'hyperedge_node_ratio', 'normalized_density',
                    'avg_clustering', 'degree_variance',
                    'hyperedge_size_variance',
                    'hyperedge_size_entropy']

    for feature_col in feature_cols:
        print(f"\nFeature: {feature_col}")
        print("-" * 30)

        # Group data by family using helper function
        valid_families = _group_data_by_family(data_features_dict, feature_col)

        if len(valid_families) < 2:
            print(
                f"  Insufficient families for comparison (found {len(valid_families)} family/families)")
            continue

        groups = list(valid_families.values())
        family_names = list(valid_families.keys())

        # Perform statistical test using helper function
        test_results = _perform_statistical_test(groups, family_names)
        results['data_comparison_tests'][feature_col] = test_results

        print(f"  Test: {test_results['test_used']}")
        print(f"  Statistic: {test_results['statistic']:.4f}")
        print(f"  p-value: {test_results['p_value']:.4f}")
        print(f"  Families: {test_results['families']}")

    # Generate plots for data directory analysis
    print("\nGenerating data directory plots...")
    _create_features_boxplot(data_features_dict, output_dir,
                             'Data Directory: Structural Features by Hypergraph Family',
                             'data_directory_features_by_family.png')

    # Generate correlation and scatter plots
    print("Creating structural feature correlation plots...")
    _create_structural_correlations_plot(data_features_dict, output_dir)

    return results


def main():
    """Main function to run statistical analysis."""
    print("=" * 80)
    print("STATISTICAL ANALYSIS FOR HYPERGRAPH STRUCTURAL FEATURES")
    print("Cross-domain analysis of hypergraph datasets")
    print("=" * 80)

    # Setup paths
    results_dir = Path('results')
    data_dir = Path('data')
    output_dir = Path('statistical_analysis_results')

    # Load data from results directory (if available)
    results_features_dict = {}

    if results_dir.exists():
        print("Loading analysis data from results directory...")
        data = load_analysis_results(results_dir)

        if data:
            print("Extracting structural features from results...")
            results_features_dict = extract_structural_features(data)

            print("Extracting resilience metrics from results...")
            resilience_dict = extract_resilience_metrics(data)

            print(
                f"Loaded {len(results_features_dict)} datasets with structural features from results")
            total_resilience = sum(len(v) for v in resilience_dict.values())
            print(
                f"Loaded {total_resilience} resilience measurements from results")
        else:
            print("No analysis data found in results directory")
    else:
        print(
            f"Results directory '{results_dir}' not found, skipping results analysis")

    # Load data from data directory
    data_features_dict = {}
    if data_dir.exists():
        data_features_dict = load_data_directory_hypergraphs(data_dir)
    else:
        print(
            f"Data directory '{data_dir}' not found, skipping data directory analysis")

    # Check if we have any data to analyze
    if not results_features_dict and not data_features_dict:
        print("Error: No data found in either results or data directories.")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Perform analyses on data directory hypergraphs (if available)
    if data_features_dict:
        data_analysis_results = perform_data_directory_analysis(
            data_features_dict, output_dir)

        # Save data directory analysis results
        data_analysis_file = output_dir / 'data_directory_analysis.json'
        with open(data_analysis_file, 'w', encoding='utf-8') as f:
            json.dump(data_analysis_results, f, indent=2, default=str)
        print(f"Data directory analysis saved: {data_analysis_file}")

        # Save data directory features to CSV
        if data_features_dict:
            features_file = output_dir / 'data_directory_features.csv'
            with open(features_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header
                feature_cols = ['dataset', 'family', 'order', 'size',
                                'avg_degree', 'avg_hyperedge_size', 'density',
                                'hyperedge_node_ratio', 'normalized_density',
                                'avg_clustering', 'degree_variance',
                                'hyperedge_size_variance',
                                'hyperedge_size_entropy']
                writer.writerow(feature_cols)

                # Write data
                for dataset_name, features in data_features_dict.items():
                    row = [dataset_name] + [features.get(col, '') for col in
                                            feature_cols[1:]]
                    writer.writerow(row)
            print(f"Data directory features saved: {features_file}")

    # Combined analysis (if we have both data sources)
    if results_features_dict and data_features_dict:
        print("\n" + "=" * 60)
        print("FINAL STATISTICAL ANALYSIS - DATA DIRECTORY FOCUS")
        print("=" * 60)

        # Use only data directory for meaningful analysis
        final_features = data_features_dict
        print(
            f"Analysis based on {len(final_features)} hypergraph datasets")
        print(
            f"  - Data directory: {len(data_features_dict)} datasets")
        print(
            f"  - Results directory: {len(results_features_dict)} datasets")

        # Perform family comparison on data directory only
        final_comparison_results = perform_family_comparison(
            final_features, {})

        # Save final results
        final_file = output_dir / 'final_analysis.json'
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_comparison_results, f, indent=2, default=str)
        print(f"Final analysis results saved: {final_file}")

    print(f"\n{'=' * 80}")
    print("STRUCTURAL FEATURE ANALYSIS COMPLETED SUCCESSFULLY")
    print(
        f"Cross-domain hypergraph analysis results saved in: {output_dir.absolute()}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
