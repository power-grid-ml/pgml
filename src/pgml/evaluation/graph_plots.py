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

from .topology import grid_graph, slack_node_id


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
):
    """Draw the grid graph; color nodes by ``node_values`` if given. Returns ``(fig, ax)``.

    ``node_values`` is a ``{node_id: value}`` dict or an array aligned to
    ``grid.nodes``. ``layout`` is ``"spring"`` (default, deterministic) or
    ``"kamada"``. The slack bus is outlined when ``mark_slack``.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    else:
        fig = ax.figure
    g = grid_graph(grid)
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.spring_layout(g, seed=0, weight=None)

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
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
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


__all__ = ["plot_grid_graph"]
