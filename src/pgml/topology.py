"""Pure grid-topology bookkeeping — slack anchor, branch edges, electrical distance.

Core, dependency-free helpers over the :class:`~pgml.schemas.grid_schema.Grid`
contract (plain python + stdlib, no torch / networkx / plotting):

- :func:`slack_node_id` — the reference bus (first in-service :class:`Source`).
- :func:`branch_edges` — the drawable branch interconnections (closed switches
  kept, open switches dropped — they carry no current and define no path).
- :func:`distance_from_slack` — shortest-path line distance from the slack along
  the branch graph (Dijkstra), the x-axis of the profile plots and a node
  feature of the ML layer.
- :func:`connectivity_report` / :func:`energized_subgrid` — the pre-solve
  connectivity check: which (node, phase) rows have a galvanic path to an
  in-service :class:`Source`, and the energized sub-grid for solving around
  deliberately disconnected areas.

The networkx graph view (:func:`pgml.evaluation.topology.grid_graph`) stays in
the evaluation package with the plotting stack; everything here is safe to
import from a lean training process.

No differentiable quantities pass through this module.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

from pgml.errors import ConnectivityError
from pgml.schemas.grid_schema import Grid, Line, Phase, ShuntReactor, Source, Switch


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


# ---------------------------------------------------------------------------
# connectivity (pre-solve energization check)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReconnectHint:
    """A currently open / out-of-service branch that would reconnect an island."""

    branch_id: int
    kind: str  # schema class name, lowercased (e.g. "switch", "line")
    from_node: int
    to_node: int
    reason: str  # "open" (a Switch with closed=False) or "not in service"


@dataclass(frozen=True)
class ConnectivityReport:
    """Which (node, phase) rows can reach an in-service :class:`Source`.

    Built by :func:`connectivity_report` from the CONDUCTING branch graph: every
    in-service branch (and CLOSED switch) merges all its terminal (node, phase)
    rows into one electrical component (branch primitives couple their terminal
    rows; the phase-exact sparsity of an uncoupled multi-phase branch is not
    resolved — a row is only reported unenergized when no conducting branch
    touches it at all, the practically relevant case).

    Attributes
    ----------
    connected:
        ``True`` iff every (node, phase) row reaches an in-service source.
    has_source:
        ``True`` iff the grid has at least one in-service :class:`Source`.
    unenergized:
        ``(node_id, phases)`` for every node with at least one unenergized row
        (``phases`` lists exactly the unenergized ones), in grid node order.
    islands:
        The unenergized components as tuples of node ids (a node with only some
        phases unenergized is included), largest first.
    reconnectable:
        :class:`ReconnectHint` entries for open switches / out-of-service branches
        whose terminals bridge an unenergized island to the energized grid.
    """

    connected: bool
    has_source: bool
    unenergized: tuple[tuple[int, tuple[Phase, ...]], ...] = ()
    islands: tuple[tuple[int, ...], ...] = ()
    reconnectable: tuple[ReconnectHint, ...] = ()

    def describe(self) -> str:
        """Human-readable multi-line summary (the :class:`ConnectivityError` body)."""
        if self.connected:
            return "All node phases are connected to an in-service source."
        if not self.has_source:
            return (
                "Grid has no in-service Source appliance: every node is unenergized. "
                "Add a Source (the slack / external grid) to the grid, or set an "
                "existing Source in_service=True."
            )
        parts = []
        by_node = ", ".join(
            f"node {n} (phases {', '.join(p.value.upper() for p in ph)})"
            for n, ph in self.unenergized[:10]
        )
        more = (
            ""
            if len(self.unenergized) <= 10
            else f", … (+{len(self.unenergized) - 10} more)"
        )
        parts.append(
            f"{len(self.unenergized)} node(s) have no galvanic path to an in-service "
            f"source: {by_node}{more}."
        )
        if self.islands:
            isl = "; ".join(
                "["
                + ", ".join(str(n) for n in island[:12])
                + ("" if len(island) <= 12 else ", …")
                + "]"
                for island in self.islands[:5]
            )
            parts.append(f"{len(self.islands)} disconnected island(s): {isl}.")
        if self.reconnectable:
            hints = "; ".join(
                (
                    f"close {h.kind} {h.branch_id} (nodes {h.from_node}-{h.to_node})"
                    if h.reason == "open"
                    else f"set {h.kind} {h.branch_id} (nodes {h.from_node}-{h.to_node}) in_service=True"
                )
                for h in self.reconnectable[:8]
            )
            parts.append(f"Reconnect options: {hints}.")
        parts.append(
            "Fix the grid (close a switch / set a branch in service / add a Source / "
            'remove the disconnected nodes), or pass on_disconnected="zero" to the '
            "solver to solve the energized part and report 0 V on the disconnected rows."
        )
        return " ".join(parts)


class _UnionFind:
    """Path-compressing union-find over hashable keys."""

    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, x):
        parent = self._parent
        root = parent.setdefault(x, x)
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _conducting(branch) -> bool:
    """An in-service branch that carries current (open switches do not)."""
    if not getattr(branch, "in_service", True):
        return False
    if isinstance(branch, Switch) and not branch.closed:
        return False
    return True


def _branch_rows(branch) -> list[tuple[int, Phase]]:
    """All (node, phase) rows a branch's primitive block touches."""
    rows = [(int(branch.from_node), ph) for ph in branch.from_phases]
    if not isinstance(branch, ShuntReactor):  # single-terminal: no TO side
        rows += [(int(branch.to_node), ph) for ph in branch.to_phases]
    return rows


def connectivity_report(grid: Grid) -> ConnectivityReport:
    """Pre-solve energization check: can every (node, phase) row reach a source?

    Merges the terminal rows of every CONDUCTING branch (in service; switches also
    closed) into electrical components (union-find), marks the components holding
    an in-service :class:`Source` terminal as energized, and reports the rest —
    including which currently open / out-of-service branches would reconnect them
    (the actionable fix). Pure bookkeeping (stdlib only), microseconds next to a
    solve; the solvers run it up front so a disconnected grid fails with an
    explanation instead of a numerical error (see
    :class:`~pgml.errors.ConnectivityError`).
    """
    uf = _UnionFind()
    for node in grid.nodes:  # register every row (isolated nodes must appear)
        for ph in node.phases:
            uf.find((int(node.id), ph))
    for b in grid.branches:
        if not _conducting(b):
            continue
        rows = _branch_rows(b)
        first = rows[0]
        for r in rows[1:]:
            uf.union(first, r)

    sources = [
        a
        for a in grid.appliances
        if isinstance(a, Source) and getattr(a, "in_service", True)
    ]
    energized_roots = {uf.find((int(s.node), ph)) for s in sources for ph in s.phases}

    unenergized: list[tuple[int, tuple[Phase, ...]]] = []
    dead_rows: list[tuple[int, Phase]] = []
    for node in grid.nodes:
        dead = tuple(
            ph
            for ph in node.phases
            if uf.find((int(node.id), ph)) not in energized_roots
        )
        if dead:
            unenergized.append((int(node.id), dead))
            dead_rows.extend((int(node.id), ph) for ph in dead)

    if not dead_rows:
        return ConnectivityReport(connected=True, has_source=bool(sources))

    # Group the dead rows into islands (by component root), node-level view.
    by_root: dict = {}
    for row in dead_rows:
        by_root.setdefault(uf.find(row), set()).add(row[0])
    islands = tuple(
        tuple(sorted(nodes))
        for nodes in sorted(by_root.values(), key=len, reverse=True)
    )

    # Actionable hints: non-conducting branches bridging a dead row to a live one.
    dead_set = set(dead_rows)
    hints: list[ReconnectHint] = []
    for b in grid.branches:
        if _conducting(b) or isinstance(b, ShuntReactor):
            continue
        rows = _branch_rows(b)
        touches_dead = any(r in dead_set for r in rows)
        touches_live = any(
            r not in dead_set and uf.find(r) in energized_roots for r in rows
        )
        if touches_dead and touches_live:
            reason = (
                "open"
                if isinstance(b, Switch) and getattr(b, "in_service", True)
                else "not in service"
            )
            hints.append(
                ReconnectHint(
                    branch_id=int(b.id),
                    kind=type(b).__name__.lower(),
                    from_node=int(b.from_node),
                    to_node=int(b.to_node),
                    reason=reason,
                )
            )

    return ConnectivityReport(
        connected=False,
        has_source=bool(sources),
        unenergized=tuple(unenergized),
        islands=islands,
        reconnectable=tuple(hints),
    )


def energized_subgrid(grid: Grid) -> tuple[Grid, tuple[int, ...]]:
    """The energized part of ``grid`` plus the ids of the dropped (dead) nodes.

    Drops every FULLY unenergized node together with the branches and appliances
    touching it; the result solves like any grid. This is the
    ``on_disconnected="zero"`` reduction: the solver runs on the sub-grid and
    scatters the solution back with 0 V on the dropped rows (a de-energized
    conductor carries no voltage — the power-grid-model "energized" convention).

    A node with only SOME phases unenergized cannot be split (a
    :class:`~pgml.schemas.grid_schema.Node`'s phase set is fixed), so partial-phase
    disconnection raises :class:`~pgml.errors.ConnectivityError` — fix the grid
    instead. Returns ``(grid, ())`` unchanged when everything is energized.
    """
    report = connectivity_report(grid)
    if report.connected:
        return grid, ()
    if not report.has_source:
        raise ConnectivityError(report.describe())
    node_phases = {int(nd.id): tuple(nd.phases) for nd in grid.nodes}
    partial = [
        (nid, dead) for nid, dead in report.unenergized if dead != node_phases[nid]
    ]
    if partial:
        nid, dead = partial[0]
        raise ConnectivityError(
            f"Node {nid} is only partially energized (phases "
            f"{', '.join(p.value.upper() for p in dead)} have no path to a source) — "
            'a node cannot be split per phase, so on_disconnected="zero" cannot '
            "reduce this grid. Fix the connectivity instead. " + report.describe(),
            unenergized_nodes=tuple(n for n, _ in report.unenergized),
            islands=report.islands,
            reconnectable=report.reconnectable,
        )
    dropped = {nid for nid, _ in report.unenergized}
    sub = grid.model_copy(
        update={
            "nodes": [nd for nd in grid.nodes if int(nd.id) not in dropped],
            "branches": [
                b
                for b in grid.branches
                if int(b.from_node) not in dropped
                and (isinstance(b, ShuntReactor) or int(b.to_node) not in dropped)
            ],
            "appliances": [a for a in grid.appliances if int(a.node) not in dropped],
        }
    )
    return sub, tuple(sorted(dropped))


__all__ = [
    "ProfileEdge",
    "slack_node_id",
    "branch_edges",
    "distance_from_slack",
    "ConnectivityReport",
    "ReconnectHint",
    "connectivity_report",
    "energized_subgrid",
]
