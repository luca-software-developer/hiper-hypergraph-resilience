# -*- coding: utf-8 -*-
"""
test_performance.py

Performance benchmark for insertion of nodes and hyperedges
comparing three implementations:
 - hiper.core.hypernetwork (Hiper)
 - hypernetx.Hypergraph (HyperNetX)
 - hypergraphx.Hypergraph (HypergraphX)
Datasets are defined in hiper.datasets.constants.DATASETS.
Measures latency and throughput for:
  - node addition
  - hyperedge addition
and provides overall average statistics by implementation.
"""

import statistics
import time
from typing import Any, Dict, List, Callable, Set

import hypergraphx as hgx
import hypernetx as hnx
from tabulate import tabulate

from hiper.config import load_config
from hiper.core.hypernetwork import Hypernetwork as HiperHN
from hiper.datasets.constants import DATASETS
from hiper.datasets.datafile import DataFile as NewDataFile
from hiper.datasets.dataset import Dataset as NewDataset


def measure(
        count: int,
        add_fn: Callable[[], None]
) -> Dict[str, float]:
    """
    Measure a batch of add operations using provided callback.

    Args:
        count: Number of operations to perform.
        add_fn: Function that performs the operations.

    Returns:
        Dictionary with:
          - Time: total time (seconds)
          - Count: number of operations performed
    """
    start = time.perf_counter()
    add_fn()
    elapsed = time.perf_counter() - start
    return {'Time': elapsed, 'Count': count}


def run_impl(label: str, nodes: List[int], edges: List[List[int]]) -> Dict[
    str, Dict[str, float]]:
    """
    Build and benchmark a hypernetwork implementation.

    Args:
        label: Implementation name ('Hiper', 'HyperNetX', 'HypergraphX').
        nodes: List of node IDs to add.
        edges: List of hyperedges, each as list of node IDs.

    Returns:
        Dict mapping metric type to measurement dict:
         - 'nodes': result of node additions
         - 'edges': result of hyperedge additions
    """
    if label == 'Hiper':
        def add_nodes():
            """
            Add each node ID to a fresh Hiper hypernetwork.
            """
            net = HiperHN()
            for nid in nodes:
                net.add_node(nid)

        def add_edges():
            """
            Add hyperedges to a fresh Hiper hypernetwork.
            Pre-add nodes to isolate edge insertion cost.
            """
            net = HiperHN()
            for nid in nodes:
                net.add_node(nid)
            for eid, members in enumerate(edges):
                net.add_hyperedge(eid, members)

    elif label == 'HyperNetX':
        def add_nodes():
            """
            Add each node ID to a fresh HyperNetX Hypergraph.
            """
            g = hnx.Hypergraph({})
            for nid in nodes:
                g.add_node(nid)

        def add_edges():
            """
            Add hyperedges to a fresh HyperNetX Hypergraph.
            Pre-add nodes to isolate edge insertion cost.
            """
            g = hnx.Hypergraph({})
            for nid in nodes:
                g.add_node(nid)
            for eid, members in enumerate(edges):
                g.add_edge(eid, set(members))

    else:  # HypergraphX
        def add_nodes():
            """
            Add each node ID to a fresh HypergraphX Hypergraph.
            """
            g = hgx.Hypergraph()
            for nid in nodes:
                g.add_node(nid)

        def add_edges():
            """
            Add hyperedges to a fresh HypergraphX Hypergraph.
            Pre-add nodes to isolate edge insertion cost.
            """
            g = hgx.Hypergraph()
            for nid in nodes:
                g.add_node(nid)
            for members in edges:
                # HypergraphX expects a tuple of node IDs to define an edge
                g.add_edge(tuple(members))

    nodes_metrics = measure(len(nodes), add_nodes)
    edges_metrics = measure(len(edges), add_edges)
    return {'nodes': nodes_metrics, 'edges': edges_metrics}


def main() -> None:
    """
    Execute performance comparison across datasets and implementations.

    Prints a table with latency (ms) and throughput (ops/ms) for:
      - node additions
      - hyperedge additions
    and overall averages by implementation.
    """
    config = load_config()
    base_path = config.get('dataset_base_path', '')

    headers = [
        'Dataset', 'Impl', 'Metric', 'Ops', 'Time (ms)', 'Rate (ops/ms)'
    ]
    rows: List[List[Any]] = []

    for ds_name in DATASETS:
        path = f"{base_path}/{ds_name}.txt"
        df = NewDataFile(path)
        ds_obj = NewDataset(ds_name, df)
        hn = ds_obj.get_hypernetwork()

        edges = [hn.get_nodes(e) for e in hn.edges.keys()]
        unique_nodes: Set[int] = set()
        for members in edges:
            unique_nodes.update(members)
        nodes = sorted(unique_nodes)

        for impl in ['Hiper', 'HyperNetX', 'HypergraphX']:
            metrics = run_impl(impl, nodes, edges)
            for metric_name, m in metrics.items():
                time_ms = m['Time'] * 1000
                count = m['Count']
                rate = count / time_ms if time_ms > 0 else float('inf')
                rows.append([
                    ds_name,
                    impl,
                    'node_add' if metric_name == 'nodes' else 'edge_add',
                    count,
                    f"{time_ms:.2f}",
                    f"{rate:.2f}"
                ])

    print('Comparison of Implementations:')
    print(tabulate(rows, headers=headers, tablefmt='grid'))

    summary_headers = [
        'Impl', 'Metric', 'AvgTime (ms)', 'AvgRate (ops/ms)'
    ]
    summary_rows: List[List[Any]] = []
    for impl in ['Hiper', 'HyperNetX', 'HypergraphX']:
        for metric_key, metric_label in [('nodes', 'node_add'),
                                         ('edges', 'edge_add')]:
            impl_rows = [r for r in rows if
                         r[1] == impl and r[2] == metric_label]
            times = [float(r[4]) for r in impl_rows]
            rates = [float(r[5]) for r in impl_rows]
            summary_rows.append([
                impl,
                metric_label,
                f"{statistics.mean(times):.2f}",
                f"{statistics.mean(rates):.2f}"
            ])

    print('\nOverall Averages by Implementation:')
    print(tabulate(summary_rows, headers=summary_headers, tablefmt='grid'))

    # Determine fastest implementation per metric
    fastest: Dict[str, str] = {}
    best_rates: Dict[str, float] = {'node_add': -1.0, 'edge_add': -1.0}
    for impl_name, metric_label, _, rate_str in summary_rows:
        rate = float(rate_str)
        if rate > best_rates[metric_label]:
            best_rates[metric_label] = rate
            fastest[metric_label] = impl_name

    print('\nFastest per metric:')
    print(f"  - node_add: {fastest['node_add']}")
    print(f"  - edge_add: {fastest['edge_add']}")


if __name__ == '__main__':
    main()
