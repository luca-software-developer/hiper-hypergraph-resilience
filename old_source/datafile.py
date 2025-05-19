from typing import Dict, Tuple, List
from hypernetwork import Hypernetwork

DATASETS = ['Restaurants-Rev', 'Music-Rev',
            'Geometry', 'NDC-classes-unique-hyperedges',
            'Algebra', 'Bars-Rev', 'iAF1260b', 'iJO1366']


class DataFile:
    def __init__(self, path: str):
        """Parses a file and creates a DataFile instance.
        The DataFile is used then to create a Hypernetwork instance via the get_data() method,
        e.g., h = Hypernetwork(**MyDataFile.get_data())"""
        self.hyperedges_list = []
        self.nodes_set = set()
        self.nodes_map = dict()

        with open(path) as hn_file:
            for line in hn_file:
                hyperedge = list(map(int, line.split(' ')))
                self.hyperedges_list.append(hyperedge)
                self.nodes_set = self.nodes_set.union(hyperedge)

        # this encodes nodes starting from 0
        for (i, node) in enumerate(self.nodes_set):
            self.nodes_map[node] = i

    def get_data(self) -> Tuple[int, int, Dict[int, int], List[List[int]]]:
        """Returns order, size, a map indicating node -> index, and the list of hyperedges"""
        return len(self.nodes_map), len(self.hyperedges_list), self.nodes_map, self.hyperedges_list


class Dataset:
    def __init__(self, name: str, datafile: DataFile):
        """Represents a dataset contained in a DataFile instance. Dataset
        also provides the hypernetwork represented by the dataset.

        :param name: name of the dataset.
        :param datafile: a DataFile instance.
        """
        self.name = name
        self.hypernetwork = Hypernetwork(*datafile.get_data())

    def get_name(self) -> str:
        """Returns the name of the dataset."""
        return self.name

    def get_hypernetwork(self) -> Hypernetwork:
        """Returns the hypernetwork represented by the dataset."""
        return self.hypernetwork
