# -*- coding: utf-8 -*-
"""
node.py

Defines the Node class for hypernetworks.
"""

from typing import Optional

from .incidence import Incidence


class Node:
    """
    Represents a vertex, holding head pointer to incidence list.
    """
    __slots__ = ('node_id', 'first_incidence')

    def __init__(self, node_id: int) -> None:
        self.node_id = node_id
        self.first_incidence: Optional[Incidence] = None
