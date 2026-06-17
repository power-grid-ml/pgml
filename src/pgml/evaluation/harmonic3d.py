"""Interactive 3D harmonic profile (plotly -> self-contained HTML).

Generalizes the 2D harmonic plot by making the angle its OWN axis, which frees the
color channel to distinguish harmonic ORDER — so several prominent harmonics (e.g.
3, 5, 7, 9) are shown at once:

    x = distance from slack [km],  y = magnitude,  z = angle [deg]

Each (order, implementation) is a 3D polyline; color = harmonic order; line DASH =
implementation (solid = ours, dashed = reference). Output is an interactive,
self-contained HTML file (rotate/zoom/hover).
"""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.colors as pcolors
import plotly.graph_objects as go

from .data import HarmonicProfile

_PALETTE = pcolors.qualitative.Plotly


def _order_colors(orders: Sequence[int]) -> dict[int, str]:
    return {o: _PALETTE[i % len(_PALETTE)] for i, o in enumerate(sorted(set(orders)))}


def plot_harmonic_profile_3d(
    profiles: Sequence[HarmonicProfile],
    *,
    reference_labels: Sequence[str] = (),
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
    reference_labels:
        Labels (matching ``profile.label``) to render DASHED (the reference runs);
        all others are solid.
    out_html:
        If given, write a self-contained interactive HTML file there.

    Returns the plotly ``Figure``.
    """
    profiles = list(profiles)
    colors = _order_colors([p.order for p in profiles])
    ref = set(reference_labels)
    unit = profiles[0].unit if profiles else "pu"

    fig = go.Figure()
    for p in profiles:
        is_ref = p.label in ref
        color = colors[p.order]
        fig.add_trace(
            go.Scatter3d(
                x=p.distances_km,
                y=p.magnitude,
                z=p.angle_deg,
                mode="lines+markers",
                line=dict(color=color, width=5, dash="dash" if is_ref else "solid"),
                marker=dict(
                    size=3, color=color, symbol="diamond" if is_ref else "circle"
                ),
                name=f"h{p.order} · {p.label}",
                legendgroup=f"h{p.order}",
                hovertemplate=(
                    f"h{p.order} ({p.label})<br>"
                    "dist=%{x:.3f} km<br>|V|=%{y:.4g}<br>angle=%{z:.2f}°<extra></extra>"
                ),
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
