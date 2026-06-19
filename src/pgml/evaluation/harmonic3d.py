"""Interactive 3D harmonic profile (plotly -> self-contained HTML).

Generalizes the 2D harmonic plot by making the angle its OWN axis, which frees the
color channel to distinguish harmonic ORDER — so several prominent harmonics (e.g.
3, 5, 7, 9) are shown at once:

    x = distance from slack [km],  y = magnitude,  z = angle [deg]

A marker is placed per node; markers are joined by lines along the real branches when
``grid`` is given (open switches omitted), else along the distance order. Color =
harmonic order; line DASH = implementation (solid = ours, dashed = reference). Output
is an interactive, self-contained HTML file (rotate/zoom/hover).
"""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.colors as pcolors
import plotly.graph_objects as go

from .data import HarmonicProfile
from .topology import branch_edges

_PALETTE = pcolors.qualitative.Plotly


def _order_colors(orders: Sequence[int]) -> dict[int, str]:
    return {o: _PALETTE[i % len(_PALETTE)] for i, o in enumerate(sorted(set(orders)))}


def _edge_polyline(p: HarmonicProfile, edges):
    """(x, y, z) arrays tracing each branch as a segment, with ``None`` breaks.

    Returns ``None`` if the profile has no node ids to map edges onto.
    """
    if p.node_ids is None:
        return None
    xyz = {
        int(nid): (float(x), float(y), float(z))
        for nid, x, y, z in zip(p.node_ids, p.distances_km, p.magnitude, p.angle_deg)
    }
    xs: list = []
    ys: list = []
    zs: list = []
    for e in edges:
        if e.a in xyz and e.b in xyz:
            (xa, ya, za), (xb, yb, zb) = xyz[e.a], xyz[e.b]
            xs += [xa, xb, None]
            ys += [ya, yb, None]
            zs += [za, zb, None]
    return xs, ys, zs


def plot_harmonic_profile_3d(
    profiles: Sequence[HarmonicProfile],
    *,
    grid=None,
    reference_labels: Sequence[str] = (),
    dash_map: Optional[dict] = None,
    title: str = "Harmonic voltage profiles (3D)",
    out_html: Optional[str] = None,
    x_title: str = "Distance from slack [km]",
    y_title: Optional[str] = None,
    z_title: str = "Angle [deg]",
    width: int = 900,
    height: int = 700,
) -> "go.Figure":
    """Interactive 3D plot of several harmonics; ours vs reference by line dash.

    Parameters
    ----------
    profiles:
        One :class:`HarmonicProfile` per (order, implementation). Use
        :func:`pgml.evaluation.data.harmonic_profiles` to build them per order.
    grid:
        If given, the connecting lines follow the real branches (open switches
        omitted); else the markers are joined in distance order.
    reference_labels:
        Labels (matching ``profile.label``) to render DASHED (the reference runs);
        all others are solid. Ignored when ``dash_map`` is given.
    dash_map:
        Optional ``{label: plotly_dash}`` (e.g. ``{"L1": "solid", "L2": "dash",
        "L3": "dot"}``) — line DASH per label, so a third dimension such as PHASE can be
        encoded by dash while COLOR stays the harmonic order. Overrides the binary
        ``reference_labels`` dashing.
    out_html:
        If given, write a self-contained interactive HTML file there.

    Returns the plotly ``Figure``.
    """
    profiles = list(profiles)
    colors = _order_colors([p.order for p in profiles])
    ref = set(reference_labels)
    unit = profiles[0].unit if profiles else "pu"
    edges = branch_edges(grid) if grid is not None else None

    fig = go.Figure()
    for p in profiles:
        is_ref = p.label in ref
        dash = (
            dash_map.get(p.label, "solid")
            if dash_map is not None
            else ("dash" if is_ref else "solid")
        )
        marker_symbol = (
            "circle" if dash_map is not None else ("diamond" if is_ref else "circle")
        )
        color = colors[p.order]
        group = f"h{p.order}"
        # Markers at each node (always in node order).
        fig.add_trace(
            go.Scatter3d(
                x=p.distances_km,
                y=p.magnitude,
                z=p.angle_deg,
                mode="markers",
                marker=dict(size=3, color=color, symbol=marker_symbol),
                name=f"{group} · {p.label}",
                legendgroup=group,
                showlegend=False,
                hovertemplate=(
                    f"{group} ({p.label})<br>"
                    "dist=%{x:.3f} km<br>|V|=%{y:.4g}<br>angle=%{z:.2f}°<extra></extra>"
                ),
            )
        )
        # Connecting lines: real branches if grid given, else distance order.
        poly = _edge_polyline(p, edges) if edges is not None else None
        lx, ly, lz = (
            poly if poly is not None else (p.distances_km, p.magnitude, p.angle_deg)
        )
        fig.add_trace(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                line=dict(color=color, width=5, dash=dash),
                name=f"{group} · {p.label}",
                legendgroup=group,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title or f"Magnitude [{unit}]",
            zaxis_title=z_title,
        ),
        legend=dict(itemsizing="constant"),
    )
    if out_html is not None:
        from pathlib import Path

        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    return fig


__all__ = ["plot_harmonic_profile_3d"]
