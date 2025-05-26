# -*- coding: utf-8 -*-
"""
datafile.py

Parses a file containing hyperedges and stores them in memory.
"""

from typing import List


class DataFile:
    """
    Reads a file where each line lists node IDs for one hyperedge.
    Skips empty or comment lines (starting with '#').

    Attributes:
        hyperedges: List of hyperedges (each a list of ints).
    """

    def __init__(self, path: str) -> None:
        self.hyperedges: List[List[int]] = []
        with open(path, encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                members = list(map(int, line.split()))
                self.hyperedges.append(members)

    def get_hyperedges(self) -> List[List[int]]:
        """Return the list of parsed hyperedges."""
        return self.hyperedges
