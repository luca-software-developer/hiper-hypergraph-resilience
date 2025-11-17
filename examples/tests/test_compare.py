# -*- coding: utf-8 -*-
"""
test_compare.py

Automated test that compares statistics of the hypernetworks created by
the new (hiper) and old (old_source) implementations for all datasets.
"""

from hiper.config import load_config
from hiper.datasets.constants import DATASETS
from hiper.datasets.datafile import DataFile as NewDataFile
from hiper.datasets.dataset import Dataset as NewDataset
from old_source.datafile import DataFile as OldDataFile
from old_source.hypernetwork import Hypernetwork as OldHypernetwork


def compare_hypernetworks(new_hn, old_hn, dataset_name):
    """
    Compare main statistics and return True if they match, False otherwise.
    """
    attrs = [
        'order',
        'size',
        'avg_deg',
        'avg_hyperdegree',
        'avg_hyperedge_size'
    ]

    all_match = True
    for attr in attrs:
        new_val = getattr(new_hn, attr)()
        old_val = getattr(old_hn, attr)()
        if abs(new_val - old_val) > 1e-6:  # float tolerance
            print(
                f"[{dataset_name}] Difference in {attr}: "
                f"New={new_val} Old={old_val}"
            )
            all_match = False
    return all_match


def main():
    """Compare hypernetworks from new and old implementations."""
    config = load_config()
    base_path = config.get("dataset_base_path", "")

    all_passed = True
    for dataset_name in DATASETS:
        # Get the dataset path
        dataset_path = f"{base_path}/{dataset_name}.txt"

        # Create hypernetworks from both implementations
        new_datafile = NewDataFile(dataset_path)
        new_dataset = NewDataset(dataset_name, new_datafile)
        new_hn = new_dataset.get_hypernetwork()

        old_datafile = OldDataFile(dataset_path)
        order, size, nodes_map, hyperedges_list = old_datafile.get_data()
        old_hn = OldHypernetwork(order, size, nodes_map, hyperedges_list)

        print(f"\nDataset: {dataset_name}")
        new_hn.print_info()
        old_hn.print_info()

        if not compare_hypernetworks(new_hn, old_hn, dataset_name):
            print(f"*** Differences found for dataset {dataset_name} ***")
            all_passed = False
        else:
            print(f"All tests passed for dataset {dataset_name}")

    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Check the differences above.")


if __name__ == "__main__":
    main()
