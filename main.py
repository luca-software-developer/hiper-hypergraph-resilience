# -*- coding: utf-8 -*-
"""
main.py

Entry point for the project.
Loads a dataset, builds a hypernetwork, and prints basic statistics.
"""

import os

from hiper.config import load_config
from hiper.datasets.datafile import DataFile
from hiper.datasets.dataset import Dataset


def main() -> None:
    """Load a dataset, build a hypernetwork, and print statistics."""
    config = load_config()

    dataset_name = str(config["dataset_name"])
    dataset_base_path = str(config["dataset_base_path"])
    dataset_path = os.path.join(dataset_base_path, dataset_name)

    datafile = DataFile(dataset_path)
    dataset = Dataset(dataset_name, datafile)

    hypernetwork = dataset.get_hypernetwork()
    print(f"Dataset: {dataset.get_name()}")
    hypernetwork.print_info()


if __name__ == "__main__":
    main()
