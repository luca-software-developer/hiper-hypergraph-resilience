# -*- coding: utf-8 -*-
"""
comprehensive_resilience.py

Comprehensive resilience experiments for hypergraphs including both node and
hyperedge removal simulations with random and TOPSIS-based selection
strategies.

This module provides enhanced experimental capabilities including TOPSIS-based
targeted attacks on both nodes and hyperedges, and evaluation using both
traditional metrics and new higher-order cohesion measures.
"""

import random
from typing import Dict, List, Any, Tuple

import numpy as np

# Handle matplotlib import gracefully
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.topsis import TopsisNodeRanker
from hiper.metrics.wsm import WSMNodeRanker
from hiper.metrics.moora import MOORANodeRanker
from hiper.metrics.hyperedge_topsis import HyperedgeTopsisRanker
from hiper.metrics.higher_order_cohesion import HigherOrderCohesionMetrics
from hiper.metrics.connectivity import (
    HypergraphConnectivity,
    HyperedgeConnectivity
)
from hiper.metrics.redundancy import RedundancyCoefficient
from hiper.metrics.swalk import SwalkEfficiency


class ComprehensiveResilienceExperiment:
    """
    Comprehensive resilience experiment framework supporting both node and
    hyperedge removal experiments with comprehensive metric evaluation.

    This class provides enhanced experimental capabilities including
    TOPSIS-based targeted attacks on both nodes and hyperedges, and evaluation
    using both traditional metrics and new higher-order cohesion measures.
    """

    def __init__(self, s: int = 1, m: int = 2, node_ranker: str = 'topsis'):
        """
        Initialize comprehensive experiment framework.

        Args:
            s: Parameter for s-walk computations.
            m: Parameter for m-th order component analysis.
            node_ranker: Node ranking method to use. Options: 'topsis', 'wsm',
                        'moora'. Default: 'topsis'.
        """
        self.s = s
        self.m = m
        self.node_ranker_name = node_ranker

        # Initialize traditional metrics
        self.traditional_metrics = {
            'hypergraph_connectivity': HypergraphConnectivity(s),
            'hyperedge_connectivity': HyperedgeConnectivity(s),
            'redundancy_coefficient': RedundancyCoefficient(),
            'swalk_efficiency': SwalkEfficiency(s)
        }

        # Initialize higher-order metrics
        self.higher_order_metrics = HigherOrderCohesionMetrics(m)

        # Initialize node ranker based on selection
        if node_ranker.lower() == 'wsm':
            self.node_ranker = WSMNodeRanker()
        elif node_ranker.lower() == 'moora':
            self.node_ranker = MOORANodeRanker()
        else:  # default to TOPSIS
            self.node_ranker = TopsisNodeRanker()

        # Initialize hyperedge ranker (always TOPSIS)
        self.hyperedge_topsis = HyperedgeTopsisRanker()

    def run_node_and_hyperedge_removal_experiments(
            self,
            hypernetwork: Hypernetwork,
            removal_percentages: List[float],
            random_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Run comprehensive resilience analysis including both node and
        hyperedge removal.

        Args:
            hypernetwork: Target hypergraph for analysis.
            removal_percentages: List of percentages to remove
                (e.g., [1, 2, 5, 10, 25]).
            random_trials: Number of trials for random removal strategies.

        Returns:
            Comprehensive results dictionary with all experiments and metrics.
        """
        results = {
            'original_metrics': {},
            'node_removal_experiments': {},
            'hyperedge_removal_experiments': {},
            'removal_percentages': list(removal_percentages),
            'higher_order_analysis': {}
        }

        print(f"Starting comprehensive resilience analysis on hypergraph "
              f"with {hypernetwork.order()} nodes and "
              f"{hypernetwork.size()} hyperedges")

        # Compute original metrics
        results['original_metrics'] = self._compute_all_metrics(hypernetwork)

        # Original higher-order analysis
        results['higher_order_analysis']['original'] = \
            self.higher_order_metrics.analyze_component_distribution(
                hypernetwork
            )

        print("Original metrics computed")
        for metric_name, value in results['original_metrics'].items():
            print(f"  {metric_name}: {value:.4f}")

        # Run node removal experiments
        print("\n=== NODE REMOVAL EXPERIMENTS ===")
        results['node_removal_experiments'] = \
            self._run_node_removal_experiments(
                hypernetwork, removal_percentages, random_trials
            )

        # Run hyperedge removal experiments
        print("\n=== HYPEREDGE REMOVAL EXPERIMENTS ===")
        results['hyperedge_removal_experiments'] = \
            self._run_hyperedge_removal_experiments(
                hypernetwork, removal_percentages, random_trials
            )

        return results

    def run_node_removal_experiments(
            self,
            hypernetwork: Hypernetwork,
            removal_percentages: List[float],
            random_trials: int
    ) -> Dict[str, Any]:
        """
        Run node removal experiments.

        Args:
            hypernetwork: Target hypergraph for analysis.
            removal_percentages: List of percentages to remove.
            random_trials: Number of trials for random removal strategies.

        Returns:
            Experiments results dictionary.
        """
        return self._run_node_removal_experiments(
            hypernetwork, removal_percentages, random_trials
        )

    def compute_all_metrics(self, hypernetwork: Hypernetwork) -> Dict[
        str, float
    ]:
        """
        Compute all traditional resilience metrics.

        Args:
            hypernetwork: Target hypergraph for analysis.

        Returns:
            Dictionary with all computed metrics.
        """
        return self._compute_all_metrics(hypernetwork)

    def _run_node_removal_experiments(
            self,
            hypernetwork: Hypernetwork,
            removal_percentages: List[float],
            random_trials: int
    ) -> Dict[str, Any]:
        """Run node removal experiments with random and ranking strategies."""
        # Use ranker name in experiment keys
        ranker_name = self.node_ranker_name
        experiments = {
            'random_removal': {},
            f'{ranker_name}_top_removal': {},
            f'{ranker_name}_bottom_removal': {}
        }

        for percentage in removal_percentages:
            print(f"\nTesting {percentage}% node removal...")

            # Random removal
            experiments['random_removal'][percentage] = \
                self._test_random_node_removal(
                    hypernetwork, percentage, random_trials
                )

            # Ranker-based top removal (remove most critical nodes)
            experiments[f'{ranker_name}_top_removal'][percentage] = \
                self._test_topsis_node_removal(hypernetwork, percentage, 'top')

            # Ranker-based bottom removal (remove the least critical nodes)
            experiments[f'{ranker_name}_bottom_removal'][percentage] = \
                self._test_topsis_node_removal(
                    hypernetwork, percentage, 'bottom'
                )

        return experiments

    def _run_hyperedge_removal_experiments(
            self,
            hypernetwork: Hypernetwork,
            removal_percentages: List[float],
            random_trials: int
    ) -> Dict[str, Any]:
        """
        Run hyperedge removal experiments with random and TOPSIS strategies.
        """
        experiments = {
            'random_removal': {},
            'topsis_top_removal': {},
            'topsis_bottom_removal': {}
        }

        for percentage in removal_percentages:
            print(f"\nTesting {percentage}% hyperedge removal...")

            # Random removal
            experiments['random_removal'][percentage] = \
                self._test_random_hyperedge_removal(
                    hypernetwork, percentage, random_trials
                )

            # TOPSIS top removal (remove most critical hyperedges)
            experiments['topsis_top_removal'][percentage] = \
                self._test_topsis_hyperedge_removal(
                    hypernetwork, percentage, 'top'
                )

            # TOPSIS bottom removal (remove least critical hyperedges)
            experiments['topsis_bottom_removal'][percentage] = \
                self._test_topsis_hyperedge_removal(
                    hypernetwork, percentage, 'bottom'
                )

        return experiments

    def _test_random_node_removal(
            self,
            hypernetwork: Hypernetwork,
            percentage: float,
            trials: int
    ) -> Dict[str, float]:
        """Test random node removal strategy."""
        results = []

        for trial in range(trials):
            # Create copy and remove random nodes
            hn_copy = hypernetwork.lightweight_copy()
            nodes_to_remove = self._select_random_nodes(hn_copy, percentage)

            for node_id in nodes_to_remove:
                hn_copy.remove_node(node_id)

            # Compute metrics
            trial_metrics = self._compute_all_metrics(hn_copy)

            # Add higher-order metrics
            ho_metrics = self.higher_order_metrics.compute_all_metrics(
                hypernetwork, hn_copy
            )
            trial_metrics.update(ho_metrics)

            results.append(trial_metrics)

        # Average results across trials
        return self._average_metrics(results)

    def _test_topsis_node_removal(
            self,
            hypernetwork: Hypernetwork,
            percentage: float,
            strategy: str
    ) -> Dict[str, float]:
        """Test ranking-based node removal strategy."""
        # Get node ranking using selected ranker
        ranked_nodes = self.node_ranker.rank_nodes(hypernetwork)

        # Select nodes based on strategy
        if strategy == 'top':
            # Remove most critical nodes (highest TOPSIS scores)
            nodes_to_remove = self._select_top_nodes(ranked_nodes, percentage)
        else:  # strategy == 'bottom'
            # Remove the least critical nodes (lowest TOPSIS scores)
            nodes_to_remove = self._select_bottom_nodes(
                ranked_nodes, percentage
            )

        # Create copy and remove selected nodes
        hn_copy = hypernetwork.lightweight_copy()
        for node_id in nodes_to_remove:
            hn_copy.remove_node(node_id)

        # Compute metrics
        metrics = self._compute_all_metrics(hn_copy)

        # Add higher-order metrics
        ho_metrics = self.higher_order_metrics.compute_all_metrics(
            hypernetwork, hn_copy
        )
        metrics.update(ho_metrics)

        return metrics

    def _test_random_hyperedge_removal(
            self,
            hypernetwork: Hypernetwork,
            percentage: float,
            trials: int
    ) -> Dict[str, float]:
        """Test random hyperedge removal strategy."""
        results = []

        for trial in range(trials):
            # Create copy and remove random hyperedges
            hn_copy = hypernetwork.lightweight_copy()
            hyperedges_to_remove = self._select_random_hyperedges(
                hn_copy, percentage
            )

            for he_id in hyperedges_to_remove:
                hn_copy.remove_hyperedge(he_id)

            # Compute metrics
            trial_metrics = self._compute_all_metrics(hn_copy)

            # Add higher-order metrics
            ho_metrics = self.higher_order_metrics.compute_all_metrics(
                hypernetwork, hn_copy
            )
            trial_metrics.update(ho_metrics)

            results.append(trial_metrics)

        # Average results across trials
        return self._average_metrics(results)

    def _test_topsis_hyperedge_removal(
            self,
            hypernetwork: Hypernetwork,
            percentage: float,
            strategy: str
    ) -> Dict[str, float]:
        """Test TOPSIS-based hyperedge removal strategy."""
        # Get TOPSIS ranking for hyperedges
        ranked_hyperedges = self.hyperedge_topsis.rank_hyperedges(
            hypernetwork
        )

        # Select hyperedges based on strategy
        if strategy == 'top':
            # Remove most critical hyperedges (highest TOPSIS scores)
            hyperedges_to_remove = self._select_top_hyperedges(
                ranked_hyperedges, percentage
            )
        else:  # strategy == 'bottom'
            # Remove least critical hyperedges (lowest TOPSIS scores)
            hyperedges_to_remove = self._select_bottom_hyperedges(
                ranked_hyperedges, percentage
            )

        # Create copy and remove selected hyperedges
        hn_copy = hypernetwork.lightweight_copy()
        for he_id in hyperedges_to_remove:
            hn_copy.remove_hyperedge(he_id)

        # Compute metrics
        metrics = self._compute_all_metrics(hn_copy)

        # Add higher-order metrics
        ho_metrics = self.higher_order_metrics.compute_all_metrics(
            hypernetwork, hn_copy
        )
        metrics.update(ho_metrics)

        return metrics

    @staticmethod
    def _compute_all_metrics(hypernetwork: Hypernetwork) -> Dict[
        str, float
    ]:
        """
        Compute structural metrics for the hypergraph.
        """
        metrics = {'order': float(hypernetwork.order()),
                   'size': float(hypernetwork.size())}

        if hypernetwork.order() > 0:
            metrics['avg_degree'] = float(hypernetwork.avg_deg())
            metrics['avg_hyperdegree'] = float(hypernetwork.avg_hyperdegree())
        else:
            metrics['avg_degree'] = 0.0
            metrics['avg_hyperdegree'] = 0.0

        if hypernetwork.size() > 0:
            metrics['avg_hyperedge_size'] = float(
                hypernetwork.avg_hyperedge_size()
            )
        else:
            metrics['avg_hyperedge_size'] = 0.0

        return metrics

    @staticmethod
    def _select_random_nodes(
            hypernetwork: Hypernetwork,
            percentage: float
    ) -> List[int]:
        """Select random nodes for removal."""
        all_nodes = list(hypernetwork.nodes.keys())
        num_to_remove = max(1, int(len(all_nodes) * percentage / 100))
        return random.sample(all_nodes, min(num_to_remove, len(all_nodes)))

    @staticmethod
    def _select_random_hyperedges(
            hypernetwork: Hypernetwork,
            percentage: float
    ) -> List[int]:
        """Select random hyperedges for removal."""
        all_hyperedges = list(hypernetwork.edges.keys())
        num_to_remove = max(1, int(len(all_hyperedges) * percentage / 100))
        return random.sample(
            all_hyperedges, min(num_to_remove, len(all_hyperedges))
        )

    @staticmethod
    def _select_top_nodes(
            ranked_nodes: List[Tuple[int, float]],
            percentage: float
    ) -> List[int]:
        """Select top-ranked nodes for removal."""
        num_to_remove = max(1, int(len(ranked_nodes) * percentage / 100))
        return [node_id for node_id, _ in ranked_nodes[:num_to_remove]]

    @staticmethod
    def _select_bottom_nodes(
            ranked_nodes: List[Tuple[int, float]],
            percentage: float
    ) -> List[int]:
        """Select bottom-ranked nodes for removal."""
        num_to_remove = max(1, int(len(ranked_nodes) * percentage / 100))
        return [node_id for node_id, _ in ranked_nodes[-num_to_remove:]]

    @staticmethod
    def _select_top_hyperedges(
            ranked_hyperedges: List[Tuple[int, float]],
            percentage: float
    ) -> List[int]:
        """Select top-ranked hyperedges for removal."""
        num_to_remove = max(1, int(len(ranked_hyperedges) * percentage / 100))
        return [he_id for he_id, _ in ranked_hyperedges[:num_to_remove]]

    @staticmethod
    def _select_bottom_hyperedges(
            ranked_hyperedges: List[Tuple[int, float]],
            percentage: float
    ) -> List[int]:
        """Select bottom-ranked hyperedges for removal."""
        num_to_remove = max(1, int(len(ranked_hyperedges) * percentage / 100))
        return [he_id for he_id, _ in ranked_hyperedges[-num_to_remove:]]

    @staticmethod
    def _average_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple trial results."""
        if not results:
            return {}

        averaged = {}
        for key in results[0].keys():
            values = [result[key] for result in results if key in result]
            averaged[key] = np.mean(values) if values else 0.0

        return averaged
