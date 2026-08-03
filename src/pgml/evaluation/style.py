"""Shared plot styling + a high-quality save helper.

Paper-ready defaults: figures save at >= 300 DPI for raster formats; vector formats
(svg/pdf) are resolution-independent. The format is chosen from the file extension.
"""

from __future__ import annotations

from pathlib import Path

# Paper-ready raster resolution (vector formats ignore it).
DPI: int = 300
# Default transparency for overlaid comparison lines (so two close lines stay visible).
COMPARE_ALPHA: float = 0.8
# Matplotlib line styles cycled across compared implementations.
LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")
# A cyclic colormap is the right choice for angles (wraps at +-180 deg).
ANGLE_CMAP: str = "twilight"


def save_figure(
    fig, path, *, dpi: int = DPI, transparent: bool = False, **kwargs
) -> str:
    """Save a matplotlib ``Figure`` paper-ready (>=300 DPI raster, or vector svg/pdf).

    Parameters
    ----------
    fig:
        A matplotlib ``Figure``.
    path:
        Output path; the extension (``.png``/``.svg``/``.pdf``) selects the format.
    dpi:
        Raster resolution (default 300). Ignored by vector formats.
    """
    import matplotlib as mpl

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Embed fonts as TrueType (fonttype 42), never Type 3: IEEE/publisher PDF
    # checks reject Type 3 fonts, and matplotlib's default writes them.
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent, **kwargs)
    return str(p)


def positive_log_norm(values):
    """A matplotlib ``LogNorm`` spanning the positive finite entries, or ``None``.

    The shared guard for every log-scaled heatmap: no positive finite data →
    ``None`` (the caller falls back to a linear scale); a degenerate
    ``vmax <= vmin`` widens to one decade so the scale stays valid. NaN/inf
    entries (e.g. empty statistic buckets) are ignored.
    """
    import numpy as np
    from matplotlib.colors import LogNorm

    values = np.asarray(values)
    pos = values[np.isfinite(values) & (values > 0)]
    if pos.size == 0:
        return None
    vmin = float(pos.min())
    vmax = float(pos.max())
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def save_html(fig, path) -> str:
    """Save a plotly ``Figure`` as a self-contained HTML file (plotly.js inlined).

    Creates the parent directory; the file opens offline in any browser. The
    plotly counterpart of :func:`save_figure`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(p, include_plotlyjs=True, full_html=True)
    return str(p)


__all__ = [
    "DPI",
    "COMPARE_ALPHA",
    "LINESTYLES",
    "ANGLE_CMAP",
    "positive_log_norm",
    "save_figure",
    "save_html",
]
