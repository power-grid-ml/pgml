"""Graph-structure plot: draw the grid topology, optionally coloring nodes by a value.

A spatial complement to the distance-based profiles: render the branch graph with
node color = a per-node quantity (e.g. voltage pu or harmonic magnitude), making the
spatial spread visible (useful for the "error spread across nodes" use case).
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pgml.schemas.grid_schema import Grid

from .data import node_numbering
from .topology import grid_graph, slack_node_id


def graph_layout(
    grid: Grid,
    *,
    layout: str = "spring",
    positions: Optional[dict] = None,
) -> dict:
    """Node positions ``{node_id: (x, y)}`` for drawing the grid graph.

    The single source of node placement for every grid-graph figure: pass the returned
    dict to :func:`plot_grid_graph` AND to any overlay drawn on top of it (sensor
    markers, annotations), so all layers share identical coordinates.

    ``positions`` overrides the computed layout with explicit (geographic) coordinates.
    When its keys (as integers) exactly cover the grid's node ids, they are taken as
    node ids directly — so an already-resolved layout passes through unchanged
    (idempotent). Otherwise each key resolves in order of precedence: an exact node
    NAME, else a zero-based node NUMBER (the node's position in ``grid.nodes`` — for a
    converted grid the source tool's bus index; see
    :func:`~pgml.evaluation.data.node_numbering`), else a node id. Values are ``(x, y)``
    pairs. Nodes missing from ``positions`` raise — a partial layout would silently
    misplace the rest of the graph. Without ``positions``, ``layout`` selects
    ``"spring"`` (deterministic, seed 0) or ``"kamada"``.
    """
    if positions is not None:
        by_name = {str(n.name): int(n.id) for n in grid.nodes}
        ids = {int(n.id) for n in grid.nodes}
        try:
            int_keys = {int(k) for k in positions}
        except (TypeError, ValueError):
            int_keys = None
        if int_keys is not None and int_keys == ids and len(int_keys) == len(positions):
            return {int(k): (float(xy[0]), float(xy[1])) for k, xy in positions.items()}
        resolved: dict[int, tuple[float, float]] = {}
        for key, xy in positions.items():
            if isinstance(key, str) and key in by_name:
                nid = by_name[key]
            else:
                try:
                    number = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"position key {key!r} is neither a node name nor an integer."
                    ) from exc
                if 0 <= number < len(grid.nodes):
                    nid = int(grid.nodes[number].id)  # zero-based node number
                elif number in ids:
                    nid = number
                else:
                    raise ValueError(
                        f"position key {key!r} matches no node name, number, or id."
                    )
            resolved[nid] = (float(xy[0]), float(xy[1]))
        missing = [int(n.id) for n in grid.nodes if int(n.id) not in resolved]
        if missing:
            raise ValueError(
                f"positions cover {len(resolved)} nodes but the grid has "
                f"{len(grid.nodes)}; missing node ids {missing[:8]}..."
                if len(missing) > 8
                else f"positions miss node ids {missing}."
            )
        return resolved
    g = grid_graph(grid)
    if layout == "kamada":
        return nx.kamada_kawai_layout(g)
    return nx.spring_layout(g, seed=0, weight=None)


def load_node_positions(path, grid: Grid) -> dict:
    """Read a node-position JSON -> ``{node_id: (x, y)}`` via :func:`graph_layout` key rules.

    The file maps node names, zero-based node numbers, or node ids to ``[x, y]`` pairs
    (e.g. ``examples/configs/cigre_lv_geo.json``, keyed by the CIGRE LV benchmark's
    zero-based bus numbers). Every grid node must be covered.
    """
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text())
    return graph_layout(grid, positions=raw)


def _node_value_array(grid: Grid, node_values, g: nx.Graph) -> Optional[np.ndarray]:
    """Resolve ``node_values`` (dict id->val OR array in grid.nodes order) to g order."""
    if node_values is None:
        return None
    if isinstance(node_values, dict):
        return np.array([node_values.get(int(nid), np.nan) for nid in g.nodes()])
    arr = np.asarray(node_values)
    by_id = {int(n.id): float(arr[i]) for i, n in enumerate(grid.nodes)}
    return np.array([by_id.get(int(nid), np.nan) for nid in g.nodes()])


def plot_grid_graph(
    grid: Grid,
    *,
    node_values: Optional[Union[dict, np.ndarray]] = None,
    value_label: str = "value",
    ax=None,
    layout: str = "spring",
    cmap: str = "viridis",
    node_size: int = 160,
    with_labels: bool = False,
    title: str = "Grid topology",
    mark_slack: bool = True,
    positions: Optional[dict] = None,
):
    """Draw the grid graph; color nodes by ``node_values`` if given. Returns ``(fig, ax)``.

    ``node_values`` is a ``{node_id: value}`` dict or an array aligned to
    ``grid.nodes``. ``layout`` is ``"spring"`` (default, deterministic) or ``"kamada"``;
    ``positions`` overrides it with explicit coordinates (resolved by
    :func:`graph_layout` — pass the SAME positions to anything drawn on top so overlays
    stay aligned). ``with_labels`` writes each node's zero-based display number
    (:func:`~pgml.evaluation.data.node_numbering` — the position in ``grid.nodes``, i.e.
    the source tool's bus numbering for a converted grid). The slack bus is outlined
    when ``mark_slack``.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    else:
        fig = ax.figure
    g = grid_graph(grid)
    pos = graph_layout(grid, layout=layout, positions=positions)

    # Lines solid, closed switches dashed (open switches are already absent from g).
    line_edges = [(u, v) for u, v, k in g.edges(data="kind") if k != "switch"]
    switch_edges = [(u, v) for u, v, k in g.edges(data="kind") if k == "switch"]
    nx.draw_networkx_edges(
        g, pos, ax=ax, edgelist=line_edges, edge_color="0.6", width=1.2
    )
    if switch_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edgelist=switch_edges,
            edge_color="0.4",
            width=1.2,
            style="dashed",
        )
    values = _node_value_array(grid, node_values, g)
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=(values if values is not None else "tab:blue"),
        cmap=cmap,
        node_size=node_size,
    )
    if values is not None:
        fig.colorbar(nodes, ax=ax, label=value_label, fraction=0.046, pad=0.04)
    if with_labels:
        numbering = node_numbering(grid)
        labels = {nid: str(numbering.get(int(nid), int(nid))) for nid in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    if mark_slack:
        sid = slack_node_id(grid)
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            nodelist=[sid],
            node_color="none",
            edgecolors="red",
            linewidths=2.5,
            node_size=node_size * 1.6,
        )
    ax.set_title(title)
    ax.axis("off")
    return fig, ax


__all__ = ["graph_layout", "load_node_positions", "plot_grid_graph"]
