"""Pure grid-topology bookkeeping — slack anchor, branch edges, electrical distance.

Core, dependency-free helpers over the :class:`~pgml.schemas.grid_schema.Grid`
contract (plain python + stdlib, no torch / networkx / plotting):

- :func:`slack_node_id` — the reference bus (first in-service :class:`Source`).
- :func:`branch_edges` — the drawable branch interconnections (closed switches
  kept, open switches dropped — they carry no current and define no path).
- :func:`distance_from_slack` — shortest-path line distance from the slack along
  the branch graph (Dijkstra), the x-axis of the profile plots and a node
  feature of the ML layer.

The networkx graph view (:func:`pgml.evaluation.topology.grid_graph`) stays in
the evaluation package with the plotting stack; everything here is safe to
import from a lean training process.

No differentiable quantities pass through this module.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

from pgml.schemas.grid_schema import Grid, Line, Source, Switch


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


def _to_float(x) -> float:
    """A python float from a plain number OR a 0-d tensor (float/tensor duality).

    Topology is off the differentiable path, so collapsing a tensor length to a
    number here is correct and intended.
    """
    if hasattr(x, "detach"):
        return float(x.detach().cpu().reshape(()).item())
    return float(x)


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


def _line_adjacency(grid: Grid) -> dict[int, dict[int, float]]:
    """Undirected ``{u: {v: km}}`` over in-service branches; non-lines weigh 0 km.

    Parallel branches sharing both endpoints collapse to the SHORTEST length —
    only the minimum matters for a shortest-path distance.
    """
    adj: dict[int, dict[int, float]] = {int(n.id): {} for n in grid.nodes}
    for b in grid.branches:
        if not _is_drawable(b, include_open_switches=False):
            continue
        w = _to_float(b.length_m) / 1000.0 if isinstance(b, Line) else 0.0
        u, v = int(b.from_node), int(b.to_node)
        cur = adj[u].get(v)
        if cur is None or w < cur:
            adj[u][v] = w
            adj[v][u] = w
    return adj


def distance_from_slack(grid: Grid, slack: Optional[int] = None) -> dict[int, float]:
    """Map ``node_id -> shortest-path line distance (km)`` from the slack bus.

    Dijkstra over the in-service branch graph (open switches excluded; switches /
    transformers / generic branches contribute zero length). Disconnected nodes
    map to ``inf``. ``slack`` defaults to :func:`slack_node_id`.
    """
    if slack is None:
        slack = slack_node_id(grid)
    adj = _line_adjacency(grid)
    dist = {nid: float("inf") for nid in adj}
    if int(slack) not in dist:
        raise ValueError(f"slack node {slack} is not a node of the grid.")
    dist[int(slack)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(slack))]
    done: set[int] = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


__all__ = [
    "ProfileEdge",
    "slack_node_id",
    "branch_edges",
    "distance_from_slack",
]
