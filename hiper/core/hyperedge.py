# -*- coding: utf-8 -*-
"""
hyperedge.py

Defines the Hyperedge class for hypernetworks.
"""

from typing import Optional

from .incidence import Incidence


class Hyperedge:
    """
    Represents a hyperedge, holding head pointer to incidence list.
    """
    __slots__ = ('edge_id', 'first_incidence')

    def __init__(self, edge_id: int) -> None:
        self.edge_id = edge_id
        self.first_incidence: Optional[Incidence] = None
