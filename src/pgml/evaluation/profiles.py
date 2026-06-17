"""Profile plots along the feeder: voltage drop and the harmonic magnitude/angle plot.

In power-grid profiles the connecting LINES are the actual branches: a marker is
placed per node at its (distance, value), and two markers are joined ONLY if a branch
connects them (so the plot reflects the real radial/meshed structure, not the
data-sorted order). Pass ``grid`` to enable this. CLOSED switches are drawn dashed;
OPEN switches are not drawn. Without ``grid`` the functions fall back to a
distance-sorted polyline (legacy).

- :func:`plot_voltage_profile` — REUSABLE voltage-drop diagram (distance vs pu); each
  implementation a different color; adjustable ``alpha`` so close lines stay visible.
- :func:`plot_harmonic_profile` — one harmonic order: distance (x) vs magnitude (y),
  LINE/MARKER COLOR = voltage angle, LINE STYLE = implementation.
- :func:`plot_profile_error` — companion bar chart of |Δ| between two voltage profiles.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .data import HarmonicProfile, VoltageProfile
from .style import ANGLE_CMAP, COMPARE_ALPHA, LINESTYLES
from .topology import branch_edges

# Branch kind -> matplotlib line style for the connecting lines.
_LINESTYLE_BY_KIND = {"line": "-", "switch": "--", "other": "-."}
_KIND_LABEL = {"line": "line", "switch": "closed switch", "other": "transformer/branch"}


def _node_xy(node_ids, xs, ys) -> dict[int, tuple[float, float]]:
    """Map ``node_id -> (x, y)`` from a profile's aligned arrays."""
    return {int(nid): (float(x), float(y)) for nid, x, y in zip(node_ids, xs, ys)}


def _circmean_deg(a: float, b: float) -> float:
    """Circular mean of two angles in degrees (handles the +-180 wrap)."""
    return float(
        np.rad2deg(np.angle(np.exp(1j * np.deg2rad(a)) + np.exp(1j * np.deg2rad(b))))
    )


def _branch_kind_legend(ax, kinds: set[str]) -> list[Line2D]:
    """Proxy handles describing the branch-type line styles actually drawn."""
    return [
        Line2D(
            [], [], color="0.35", linestyle=_LINESTYLE_BY_KIND[k], label=_KIND_LABEL[k]
        )
        for k in ("line", "switch", "other")
        if k in kinds
    ]


def plot_voltage_profile(
    profiles: Sequence[VoltageProfile],
    *,
    grid=None,
    ax=None,
    colors: Optional[Sequence] = None,
    alpha: float = COMPARE_ALPHA,
    markers: bool = True,
    title: str = "Voltage profile",
    xlabel: str = "Distance from slack [km]",
    ylabel: str = "Voltage [pu]",
    legend: bool = True,
):
    """Voltage magnitude vs distance from slack, one color per implementation.

    Reusable across evaluations (pass ``ax`` to embed). With ``grid`` the connecting
    lines follow the real branches (closed switches dashed, open switches omitted);
    without it, a distance-sorted polyline is drawn. ``alpha`` (every line) keeps
    overlapping implementations visible. Returns ``(fig, ax)``.
    """
    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    cycle = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    edges = branch_edges(grid) if grid is not None else None
    kinds: set[str] = set()

    for i, p in enumerate(profiles):
        color = cycle[i % len(cycle)]
        if markers:
            ax.plot(
                p.distances_km,
                p.v_pu,
                linestyle="none",
                marker="o",
                markersize=4,
                color=color,
                alpha=alpha,
                zorder=3,
            )
        if edges is not None and p.node_ids is not None:
            xy = _node_xy(p.node_ids, p.distances_km, p.v_pu)
            for kind in {e.kind for e in edges}:
                segs = [
                    [xy[e.a], xy[e.b]]
                    for e in edges
                    if e.kind == kind and e.a in xy and e.b in xy
                ]
                if not segs:
                    continue
                kinds.add(kind)
                ax.add_collection(
                    LineCollection(
                        segs,
                        colors=[color],
                        linestyles=_LINESTYLE_BY_KIND.get(kind, "-"),
                        linewidths=1.9,
                        alpha=alpha,
                        zorder=2,
                    )
                )
        else:  # legacy: connect in distance order
            ax.plot(p.distances_km, p.v_pu, linewidth=1.9, color=color, alpha=alpha)
        ax.plot([], [], color=color, marker="o", label=p.label)  # legend proxy (impl)

    ax.autoscale_view()
    ax.margins(x=0.02, y=0.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend and profiles:
        handles = ax.get_legend_handles_labels()[0] + _branch_kind_legend(ax, kinds)
        ax.legend(handles=handles, frameon=False)
    return fig, ax


def plot_harmonic_profile(
    profiles: Sequence[HarmonicProfile],
    *,
    grid=None,
    ax=None,
    linestyles: Optional[Sequence[str]] = None,
    cmap: str = ANGLE_CMAP,
    alpha: float = 1.0,
    angle_range: tuple[float, float] = (-180.0, 180.0),
    markers: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Distance from slack [km]",
    ylabel: Optional[str] = None,
    colorbar: bool = True,
    legend: bool = True,
):
    """Harmonic magnitude vs distance, colored by angle; implementations by line style.

    For a single harmonic order. x = distance, y = harmonic magnitude, COLOR = voltage
    angle (cyclic colormap), LINE STYLE = implementation (so a reference overlays the
    same color scale as ours). With ``grid`` the connecting lines follow the real
    branches (open switches omitted); without it, a distance-sorted polyline.
    Returns ``(fig, ax)``.
    """
    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    styles = linestyles or LINESTYLES
    norm = Normalize(*angle_range)
    edges = branch_edges(grid) if grid is not None else None
    mappable = None

    for i, p in enumerate(profiles):
        x, y, ang = p.distances_km, p.magnitude, p.angle_deg
        style = styles[i % len(styles)]
        if edges is not None and p.node_ids is not None:
            xy = _node_xy(p.node_ids, x, y)
            ang_by_node = {int(nid): float(a) for nid, a in zip(p.node_ids, ang)}
            segs, seg_ang = [], []
            for e in edges:
                if e.a in xy and e.b in xy:
                    segs.append([xy[e.a], xy[e.b]])
                    seg_ang.append(_circmean_deg(ang_by_node[e.a], ang_by_node[e.b]))
            if segs:
                lc = LineCollection(
                    segs,
                    cmap=cmap,
                    norm=norm,
                    linestyles=style,
                    linewidths=2.2,
                    alpha=alpha,
                )
                lc.set_array(np.asarray(seg_ang))
                ax.add_collection(lc)
                mappable = mappable or lc
        elif x.size >= 2:  # legacy: connect in distance order
            pts = np.column_stack([x, y]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(
                segs,
                cmap=cmap,
                norm=norm,
                linestyles=style,
                linewidths=2.2,
                alpha=alpha,
            )
            lc.set_array(
                np.rad2deg(
                    np.angle(
                        np.exp(1j * np.deg2rad(ang[:-1]))
                        + np.exp(1j * np.deg2rad(ang[1:]))
                    )
                )
            )
            ax.add_collection(lc)
            mappable = mappable or lc
        if markers:
            mappable = ax.scatter(
                x,
                y,
                c=ang,
                cmap=cmap,
                norm=norm,
                s=24,
                edgecolors="k",
                linewidths=0.3,
                zorder=3,
            )
        ax.plot([], [], linestyle=style, color="0.35", label=p.label)  # legend (impl)

    ax.autoscale()
    ax.margins(x=0.02, y=0.08)
    order = profiles[0].order if profiles else None
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or (f"|V| [{profiles[0].unit}]" if profiles else "|V|"))
    ax.set_title(
        title
        or (f"Harmonic voltage profile (h={order})" if order else "Harmonic profile")
    )
    ax.grid(True, alpha=0.3)
    if colorbar and mappable is not None:
        fig.colorbar(mappable, ax=ax, label="Voltage angle [deg]", pad=0.02)
    if legend and profiles:
        ax.legend(frameon=False, loc="best")
    return fig, ax


def plot_profile_error(
    reference: VoltageProfile,
    ours: VoltageProfile,
    *,
    ax=None,
    title: str = "Voltage error vs reference",
    ylabel: str = "|Δ| [pu]",
):
    """Bar chart of per-node ``|ours - reference|`` voltage error (aligned by node id)."""
    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ref_by_node = dict(zip([int(i) for i in reference.node_ids], reference.v_pu))
    nids = [int(i) for i in ours.node_ids]
    err = np.array([abs(v - ref_by_node[nid]) for nid, v in zip(nids, ours.v_pu)])
    ax.bar(range(len(err)), err, color="tab:red", alpha=0.8)
    ax.set_xticks(range(len(nids)))
    ax.set_xticklabels(nids, rotation=90, fontsize=6)
    ax.set_xlabel("Node id")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (max={err.max():.2e} pu)")
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


__all__ = ["plot_voltage_profile", "plot_harmonic_profile", "plot_profile_error"]
