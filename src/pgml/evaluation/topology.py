"""Grid topology helpers for evaluation plots.

Two jobs:
- electrical DISTANCE from the slack bus (the x-axis of the profile plots), walked
  along the branch graph;
- the branch INTERCONNECTIONS used to draw lines on a profile — in power-grid plots
  the connecting lines are the actual branches (only adjacent nodes are joined), NOT
  the data-sorted sequence. Closed switches are drawable (dashed); OPEN switches are
  omitted entirely (they carry no current and don't define a path).

Pure topology bookkeeping (no differentiable quantities), so plain python/networkx
is fine here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from pgml.schemas.grid_schema import Grid, Line, Source, Switch

from ._util import to_float


@dataclass(frozen=True)
class ProfileEdge:
    """One drawable branch interconnection between two nodes.

    ``kind`` is ``"line"`` (solid), ``"switch"`` (a CLOSED switch — drawn dashed) or
    ``"other"`` (transformer / generic branch — solid). Open switches never produce a
    :class:`ProfileEdge`.
    """

    a: int
    b: int
    kind: str


def _branch_kind(b) -> str:
    if isinstance(b, Line):
        return "line"
    if isinstance(b, Switch):
        return "switch"
    return "other"


def _is_drawable(b, *, include_open_switches: bool) -> bool:
    """An in-service branch that connects two distinct nodes; open switches excluded."""
    if not getattr(b, "in_service", True):
        return False
    if isinstance(b, Switch) and not b.closed and not include_open_switches:
        return False
    return int(b.from_node) != int(b.to_node)


def slack_node_id(grid: Grid) -> int:
    """Node id of the first in-service :class:`Source` (the slack/reference bus)."""
    src = next(
        (
            a
            for a in grid.appliances
            if isinstance(a, Source) and getattr(a, "in_service", True)
        ),
        None,
    )
    if src is None:
        raise ValueError("Grid has no in-service Source to anchor distances to.")
    return int(src.node)


def branch_edges(
    grid: Grid, *, include_open_switches: bool = False
) -> list[ProfileEdge]:
    """Drawable branch interconnections ``[(a, b, kind)]`` for profile line-drawing.

    In-service branches connecting two distinct nodes; CLOSED switches are kept
    (``kind="switch"``) and OPEN switches dropped (unless ``include_open_switches``).
    """
    edges = []
    for b in grid.branches:
        if _is_drawable(b, include_open_switches=include_open_switches):
            edges.append(ProfileEdge(int(b.from_node), int(b.to_node), _branch_kind(b)))
    return edges


def grid_graph(grid: Grid, *, weight: str = "km") -> nx.Graph:
    """Undirected branch graph; edge ``weight`` = line length in km (0 for non-lines).

    Switches / transformers / generic branches connect their endpoints with zero
    length. Out-of-service branches AND open switches are skipped (an open switch does
    not define a path, so it must not merge distances across it). Each edge carries a
    ``kind`` attribute (see :class:`ProfileEdge`).
    """
    g = nx.Graph()
    for node in grid.nodes:
        g.add_node(int(node.id))
    for b in grid.branches:
        if not _is_drawable(b, include_open_switches=False):
            continue
        length_km = to_float(b.length_m) / 1000.0 if isinstance(b, Line) else 0.0
        g.add_edge(
            int(b.from_node),
            int(b.to_node),
            **{weight: length_km, "branch_id": int(b.id), "kind": _branch_kind(b)},
        )
    return g


def distance_from_slack(
    grid: Grid, slack: Optional[int] = None, *, weight: str = "km"
) -> dict[int, float]:
    """Map ``node_id -> shortest-path distance (km)`` from the slack bus.

    Disconnected nodes map to ``inf``. ``slack`` defaults to :func:`slack_node_id`.
    """
    if slack is None:
        slack = slack_node_id(grid)
    g = grid_graph(grid, weight=weight)
    lengths = nx.single_source_dijkstra_path_length(g, int(slack), weight=weight)
    return {
        int(node.id): lengths.get(int(node.id), float("inf")) for node in grid.nodes
    }


__all__ = [
    "ProfileEdge",
    "slack_node_id",
    "branch_edges",
    "grid_graph",
    "distance_from_slack",
]
