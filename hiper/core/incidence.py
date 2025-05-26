# -*- coding: utf-8 -*-
"""
incidence.py

Defines the Incidence class linking Node and Hyperedge.
"""

from typing import Optional


class Incidence:
    """
    Links a Node to a Hyperedge in two doubly-linked lists.
    """
    __slots__ = (
        'node', 'edge',
        'prev_in_node', 'next_in_node',
        'prev_in_edge', 'next_in_edge',
    )

    def __init__(self, node: 'Node', edge: 'Hyperedge') -> None:
        self.node = node
        self.edge = edge
        self.prev_in_node: Optional[Incidence] = None
        self.next_in_node: Optional[Incidence] = None
        self.prev_in_edge: Optional[Incidence] = None
        self.next_in_edge: Optional[Incidence] = None
