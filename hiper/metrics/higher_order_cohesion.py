# -*- coding: utf-8 -*-
"""
higher_order_cohesion.py

Implements Higher-Order Cohesion Resilience (HOCR) and Largest Higher-Order
Component (LHC) metrics for analyzing hypergraph resilience based on m-th
order components.

The m-th higher-order component is defined as a set of hyperedges where each
hyperedge shares at least m nodes with another hyperedge in the same component.
"""

from typing import List, Dict, Any

import numpy as np

from hiper.core.hypernetwork import Hypernetwork


class HigherOrderCohesionMetrics:
    """
    Computes Higher-Order Cohesion Resilience (HOCR) and Largest Higher-Order
    Component (LHC) metrics for hypergraph resilience analysis.

    These metrics evaluate how perturbations affect the higher-order structure
    of hypergraphs by analyzing components where hyperedges share at least m
    nodes.
    """

    def __init__(self, m: int = 2):
        """
        Initialize the higher-order cohesion metrics calculator.

        Args:
            m: Minimum number of shared nodes required between hyperedges
               in the same m-th order component. Must be >= 2.

        Raises:
            ValueError: If m < 2.
        """
        if m < 2:
            raise ValueError(
                "Parameter m must be >= 2 for meaningful analysis"
            )
        self.m = m

    def compute_mth_order_components(
            self, hypernetwork: Hypernetwork
    ) -> List[List[int]]:
        """
        Compute the m-th higher-order components of the hypernetwork.

        An m-th higher-order component is a set of hyperedges where each
        hyperedge shares at least m nodes with another hyperedge in the same
        component.

        Args:
            hypernetwork: The input hypergraph.

        Returns:
            List of components, where each component is a list of hyperedge
            indices.
        """
        hyperedges_list = list(hypernetwork.edges.values())
        hyperedge_ids = list(hypernetwork.edges.keys())
        num_edges = len(hyperedges_list)

        if num_edges == 0:
            return []

        # Create adjacency matrix for hyperedges based on m-th order
        # relationships
        edge_adjacency = np.zeros((num_edges, num_edges), dtype=int)

        # Compare each pair of hyperedges
        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                # Count shared nodes between hyperedges
                shared_nodes = len(
                    set(hyperedges_list[i]).intersection(hyperedges_list[j])
                )
                if shared_nodes >= self.m:
                    edge_adjacency[i, j] = 1
                    edge_adjacency[j, i] = 1

        def iterative_dfs(start_edge: int, visited_nodes: set) -> List[int]:
            """
            Perform iterative depth-first search to find connected components.

            Args:
                start_edge: Starting edge index for the search.
                visited_nodes: Set of already visited edge indices.

            Returns:
                List of edge IDs (not indices) in the connected component.
            """
            current_component = []
            stack = [start_edge]

            while stack:
                current_edge_idx = stack.pop()
                if current_edge_idx not in visited_nodes:
                    visited_nodes.add(current_edge_idx)
                    # Store actual edge ID, not index
                    current_component.append(hyperedge_ids[current_edge_idx])

                    # Add all unvisited neighbors to stack
                    for next_edge in range(num_edges):
                        is_connected = (
                                edge_adjacency[current_edge_idx, next_edge] == 1
                        )
                        if is_connected and next_edge not in visited_nodes:
                            stack.append(next_edge)

            return current_component

        # Find all components
        visited_edges = set()
        components = []

        for edge_index in range(num_edges):
            if edge_index not in visited_edges:
                # Check if this edge has any m-th order connections
                if any(edge_adjacency[edge_index]):
                    found_component = iterative_dfs(edge_index, visited_edges)
                    if found_component:  # Only add non-empty components
                        components.append(found_component)

        return components

    def compute_hocr_m(
            self,
            original_hypernetwork: Hypernetwork,
            perturbed_hypernetwork: Hypernetwork
    ) -> float:
        """
        Compute Higher-Order Cohesion Resilience (HOCR_m).

        HOCR_m measures the normalized preservation of higher-order structure
        after perturbation by comparing the total number of hyperedges in
        m-th order components before and after perturbation.

        Args:
            original_hypernetwork: The original hypergraph.
            perturbed_hypernetwork: The hypergraph after perturbation.

        Returns:
            HOCR_m value in [0, 1], where 1 indicates perfect preservation.
        """
        # Compute components for original hypergraph
        original_components = self.compute_mth_order_components(
            original_hypernetwork
        )
        original_total = sum(
            len(comp) for comp in original_components
        )

        # Compute components for perturbed hypergraph
        perturbed_components = self.compute_mth_order_components(
            perturbed_hypernetwork
        )
        perturbed_total = sum(
            len(comp) for comp in perturbed_components
        )

        # Compute HOCR_m with smoothing factor
        hocr_m = perturbed_total / (original_total + 1)

        return float(hocr_m)

    def compute_lhc_m(
            self,
            original_hypernetwork: Hypernetwork,
            perturbed_hypernetwork: Hypernetwork
    ) -> float:
        """
        Compute Largest Higher-Order Component (LHC_m) resilience.

        LHC_m measures how well the largest m-th order component is preserved
        after perturbation.

        Args:
            original_hypernetwork: The original hypergraph.
            perturbed_hypernetwork: The hypergraph after perturbation.

        Returns:
            LHC_m value in [0, 1], where 1 indicates perfect preservation.
        """
        # Compute components for original hypergraph
        original_components = self.compute_mth_order_components(
            original_hypernetwork
        )
        original_max = max(
            (len(comp) for comp in original_components), default=0
        )

        # Compute components for perturbed hypergraph
        perturbed_components = self.compute_mth_order_components(
            perturbed_hypernetwork
        )
        perturbed_max = max(
            (len(comp) for comp in perturbed_components), default=0
        )

        # Compute LHC_m with smoothing factor
        lhc_m = perturbed_max / (original_max + 1)

        return float(lhc_m)

    def compute_all_metrics(
            self,
            original_hypernetwork: Hypernetwork,
            perturbed_hypernetwork: Hypernetwork
    ) -> Dict[str, float]:
        """
        Compute both HOCR_m and LHC_m metrics.

        Args:
            original_hypernetwork: The original hypergraph.
            perturbed_hypernetwork: The hypergraph after perturbation.

        Returns:
            Dictionary containing both metrics.
        """
        return {
            f'hocr_{self.m}': self.compute_hocr_m(
                original_hypernetwork, perturbed_hypernetwork
            ),
            f'lhc_{self.m}': self.compute_lhc_m(
                original_hypernetwork, perturbed_hypernetwork
            )
        }

    def analyze_component_distribution(
            self, hypernetwork: Hypernetwork
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of m-th order components in the hypergraph.

        Args:
            hypernetwork: The input hypergraph.

        Returns:
            Dictionary with detailed component analysis.
        """
        components = self.compute_mth_order_components(hypernetwork)

        if not components:
            return {
                'num_components': 0,
                'total_hyperedges_in_components': 0,
                'largest_component_size': 0,
                'average_component_size': 0.0,
                'component_sizes': []
            }

        component_sizes = [len(comp) for comp in components]

        return {
            'num_components': len(components),
            'total_hyperedges_in_components': sum(component_sizes),
            'largest_component_size': max(component_sizes),
            'average_component_size': np.mean(component_sizes),
            'component_sizes': component_sizes
        }
