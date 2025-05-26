# -*- coding: utf-8 -*-
"""
dataset.py

Wraps DataFile instances into Hypernetwork objects.
"""

from hiper.core.hypernetwork import Hypernetwork
from .datafile import DataFile


class Dataset:
    """
    Constructs a Hypernetwork from a DataFile.

    Attributes:
        name: Name of the dataset.
        hypernetwork: Hypernetwork built from the data.
    """

    def __init__(self, name: str, datafile: DataFile) -> None:
        self.name = name
        self.hypernetwork = Hypernetwork()
        for edge_id, members in enumerate(datafile.get_hyperedges()):
            self.hypernetwork.add_hyperedge(edge_id, members)

    def get_name(self) -> str:
        """Return the dataset name."""
        return self.name

    def get_hypernetwork(self) -> Hypernetwork:
        """Return the constructed Hypernetwork."""
        return self.hypernetwork
