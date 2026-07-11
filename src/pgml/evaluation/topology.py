"""The networkx view of the grid topology, for evaluation plots.

The pure topology bookkeeping (slack anchor, drawable branch edges, electrical
distance from the slack) lives in the core, dependency-free
:mod:`pgml.topology`; this module re-exports it unchanged so existing importers
keep working, and adds the one helper that genuinely needs networkx —
:func:`grid_graph`, the ``nx.Graph`` used by graph-layout plots.
"""

from __future__ import annotations

import networkx as nx

from pgml.schemas.grid_schema import Grid, Line
from pgml.topology import (
    ProfileEdge,
    _branch_kind,
    _is_drawable,
    branch_edges,
    distance_from_slack,
    slack_node_id,
)

from ._util import to_float


def grid_graph(grid: Grid, *, weight: str = "km") -> nx.Graph:
    """Undirected branch graph; edge ``weight`` = line length in km (0 for non-lines).

    Switches / transformers / generic branches connect their endpoints with zero
    length. Out-of-service branches AND open switches are skipped (an open switch does
    not define a path, so it must not merge distances across it). Each edge carries a
    ``kind`` attribute (see :class:`ProfileEdge`).

    Parallel branches (two in-service branches sharing both endpoints — a ring
    tie, a doubled cable) collapse to ONE edge carrying the SHORTEST length, so
    the graph stays a simple ``nx.Graph`` while distances remain shortest-path
    correct; a longer parallel branch must never overwrite a shorter one.
    """
    g = nx.Graph()
    for node in grid.nodes:
        g.add_node(int(node.id))
    for b in grid.branches:
        if not _is_drawable(b, include_open_switches=False):
            continue
        length_km = to_float(b.length_m) / 1000.0 if isinstance(b, Line) else 0.0
        u, v = int(b.from_node), int(b.to_node)
        if g.has_edge(u, v) and g[u][v][weight] <= length_km:
            continue
        g.add_edge(
            u,
            v,
            **{weight: length_km, "branch_id": int(b.id), "kind": _branch_kind(b)},
        )
    return g


__all__ = [
    "ProfileEdge",
    "slack_node_id",
    "branch_edges",
    "grid_graph",
    "distance_from_slack",
]
