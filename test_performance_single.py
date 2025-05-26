# -*- coding: utf-8 -*-
"""
test_performance_single.py

Automated performance test for insertion and deletion of nodes and hyperedges
on datasets defined in hiper.datasets.constants.DATASETS.
Measures time for add/remove operations in the new implementation.
Displays dataset stats and performance metrics (times and rates).
"""

import time
from typing import Any, Dict, List

from tabulate import tabulate

from hiper.config import load_config
from hiper.core.hypernetwork import Hypernetwork as TestHypernetwork
from hiper.datasets.constants import DATASETS
from hiper.datasets.datafile import DataFile as NewDataFile
from hiper.datasets.dataset import Dataset as NewDataset


def measure_operations(
        base_hn: TestHypernetwork
) -> Dict[str, float]:
    """
    Measure add/remove operations on a copy of the network structure.
    Returns timings in seconds and counts.
    """
    times: Dict[str, float] = {}
    edge_ids = list(base_hn.edges.keys())
    node_ids = list(base_hn.nodes.keys())

    # Time adding edges
    hn_tmp = TestHypernetwork()
    start = time.perf_counter()
    for eid in edge_ids:
        members = base_hn.get_nodes(eid)
        hn_tmp.add_hyperedge(eid, members)
    times['AddEdgeTime'] = time.perf_counter() - start

    # Time removing edges
    start = time.perf_counter()
    for eid in edge_ids:
        hn_tmp.remove_hyperedge(eid)
    times['RemoveEdgeTime'] = time.perf_counter() - start

    # Time adding nodes
    hn_tmp = TestHypernetwork()
    start = time.perf_counter()
    for nid in node_ids:
        hn_tmp.add_node(nid)
    times['AddNodeTime'] = time.perf_counter() - start

    # Time removing nodes
    start = time.perf_counter()
    for nid in node_ids:
        hn_tmp.remove_node(nid)
    times['RemoveNodeTime'] = time.perf_counter() - start

    # Counts
    times['EdgeCount'] = len(edge_ids)
    times['NodeCount'] = len(node_ids)
    return times


def gather_stats(
        base_hn: TestHypernetwork
) -> Dict[str, Any]:
    """
    Collect basic hypernetwork statistics.
    """
    return {
        'NodeCount': base_hn.order(),
        'EdgeCount': base_hn.size(),
        'AvgDegree': base_hn.avg_deg(),
        'AvgHyperdegree': base_hn.avg_hyperdegree(),
        'AvgEdgeSize': base_hn.avg_hyperedge_size(),
    }


def main():
    """Run tests and display results."""
    config = load_config()
    base_path = config.get("dataset_base_path", "")

    headers: List[str] = [
        "Dataset",
        "Nodes",
        "Edges",
        "Avg Degree",
        "Avg Hyperdegree",
        "Avg Edge Size",
        "Add Edge Time (ms)",
        "Add Rate (edges/ms)",
        "Remove Edge Time (ms)",
        "Remove Rate (edges/ms)",
        "Add Node Time (ms)",
        "Add Rate (nodes/ms)",
        "Remove Node Time (ms)",
        "Remove Rate (nodes/ms)"
    ]
    rows: List[List[Any]] = []

    for ds in DATASETS:
        path = f"{base_path}/{ds}.txt"
        df = NewDataFile(path)
        ds_obj = NewDataset(ds, df)
        hn = ds_obj.get_hypernetwork()

        stats = gather_stats(hn)
        perf = measure_operations(hn)

        # Convert seconds to milliseconds
        perf_ms = {
            k: v * 1000 for k, v in perf.items()
            if 'Time' in k
        }
        # Compute rates: count / ms
        rates = {
            'AddRate': perf['EdgeCount'] / perf_ms['AddEdgeTime']
            if perf_ms['AddEdgeTime'] > 0 else float('inf'),
            'RemRate': perf['EdgeCount'] / perf_ms['RemoveEdgeTime']
            if perf_ms['RemoveEdgeTime'] > 0 else float('inf'),
            'AddNodeRate': perf['NodeCount'] / perf_ms['AddNodeTime']
            if perf_ms['AddNodeTime'] > 0 else float('inf'),
            'RemNodeRate': perf['NodeCount'] / perf_ms['RemoveNodeTime']
            if perf_ms['RemoveNodeTime'] > 0 else float('inf'),
        }

        rows.append([
            ds,
            stats['NodeCount'],
            stats['EdgeCount'],
            f"{stats['AvgDegree']:.2f}",
            f"{stats['AvgHyperdegree']:.2f}",
            f"{stats['AvgEdgeSize']:.2f}",
            f"{perf_ms['AddEdgeTime']:.2f}",
            f"{rates['AddRate']:.2f}",
            f"{perf_ms['RemoveEdgeTime']:.2f}",
            f"{rates['RemRate']:.2f}",
            f"{perf_ms['AddNodeTime']:.2f}",
            f"{rates['AddNodeRate']:.2f}",
            f"{perf_ms['RemoveNodeTime']:.2f}",
            f"{rates['RemNodeRate']:.2f}"
        ])

    print("Performance Test Results: Structure, Latency (ms) and Throughput:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    main()
