"""Interactive 2D harmonic profile (plotly -> self-contained HTML).

The 2D companion to :func:`pgml.evaluation.harmonic3d.plot_harmonic_profile_3d`. For a
fixed harmonic order it plots magnitude vs distance with ONE colour per implementation
(not per angle), so heavily overlapping models stay distinguishable AND each model can be
toggled on/off by clicking its legend entry (plotly hides the whole legend group). A
marker is placed per node; markers are joined along the real branches when ``grid`` is
given (open switches omitted), else along the distance order. Hover shows the model,
distance, magnitude and angle. Lines are drawn semi-transparent so overlaps read clearly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.colors as pcolors
import plotly.graph_objects as go

from .data import HarmonicProfile
from .topology import branch_edges

_PALETTE = pcolors.qualitative.Plotly


def _rgba(color: str, alpha: float) -> str:
    """Convert a plotly colour (``#hex`` or ``rgb(...)``) to an ``rgba(...)`` string."""
    r, g, b = (
        pcolors.hex_to_rgb(color)
        if color.startswith("#")
        else pcolors.unlabel_rgb(color)
    )
    return f"rgba({r:.0f}, {g:.0f}, {b:.0f}, {alpha})"


def _label_colors(labels: Sequence[str]) -> dict[str, str]:
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(seen)}


def _edge_polyline_2d(p: HarmonicProfile, edges):
    """``(x, y)`` arrays tracing each branch as a segment with ``None`` breaks (or None)."""
    if p.node_ids is None:
        return None
    xy = {
        int(nid): (float(x), float(y))
        for nid, x, y in zip(p.node_ids, p.distances_km, p.magnitude)
    }
    xs: list = []
    ys: list = []
    for e in edges:
        if e.a in xy and e.b in xy:
            (xa, ya), (xb, yb) = xy[e.a], xy[e.b]
            xs += [xa, xb, None]
            ys += [ya, yb, None]
    return xs, ys


def plot_harmonic_profile_interactive(
    profiles: Sequence[HarmonicProfile],
    *,
    grid=None,
    title: Optional[str] = None,
    out_html: Optional[str] = None,
    x_title: str = "Distance from slack [km]",
    y_title: Optional[str] = None,
    line_alpha: float = 0.6,
    width: int = 950,
    height: int = 560,
) -> "go.Figure":
    """Toggleable 2D harmonic magnitude-vs-distance plot, one colour per implementation.

    Parameters
    ----------
    profiles:
        One :class:`HarmonicProfile` per implementation (and optionally per order). Each
        becomes its own legend group ``"h{order} · {label}"`` that toggles on/off on a
        legend click (markers + connecting lines together).
    grid:
        If given, connecting lines follow the real branches (open switches omitted);
        else markers are joined in distance order.
    out_html:
        If given, write a self-contained interactive HTML file there.
    line_alpha:
        Opacity of the connecting lines (markers stay opaque) so overlaps read clearly.

    Returns the plotly ``Figure``.
    """
    profiles = list(profiles)
    colors = _label_colors([p.label for p in profiles])
    unit = profiles[0].unit if profiles else "pu"
    edges = branch_edges(grid) if grid is not None else None
    multi_order = len({p.order for p in profiles}) > 1

    fig = go.Figure()
    for p in profiles:
        color = colors[p.label]
        group = f"h{p.order} · {p.label}" if multi_order else p.label
        # Connecting lines: real branches if grid given, else distance order.
        poly = _edge_polyline_2d(p, edges) if edges is not None else None
        lx, ly = poly if poly is not None else (p.distances_km, p.magnitude)
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                line=dict(color=_rgba(color, line_alpha), width=3),
                name=group,
                legendgroup=group,
                hoverinfo="skip",
            )
        )
        # Markers at each node carry the angle for hover.
        fig.add_trace(
            go.Scatter(
                x=p.distances_km,
                y=p.magnitude,
                customdata=p.angle_deg,
                mode="markers",
                marker=dict(size=6, color=color, line=dict(width=0.5, color="#222")),
                name=group,
                legendgroup=group,
                showlegend=False,
                hovertemplate=(
                    f"{group}<br>dist=%{{x:.3f}} km<br>|V|=%{{y:.4g}} {unit}"
                    "<br>angle=%{customdata:.2f}°<extra></extra>"
                ),
            )
        )

    order = profiles[0].order if profiles and not multi_order else None
    fig.update_layout(
        title=title
        or (f"Harmonic voltage profile (h={order})" if order else "Harmonic profile"),
        width=width,
        height=height,
        xaxis_title=x_title,
        yaxis_title=y_title or f"Magnitude [{unit}]",
        legend=dict(groupclick="togglegroup", itemsizing="constant"),
        hovermode="closest",
    )
    if out_html is not None:
        from pathlib import Path

        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    return fig


__all__ = ["plot_harmonic_profile_interactive"]
