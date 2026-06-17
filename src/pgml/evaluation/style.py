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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent, **kwargs)
    return str(p)


__all__ = ["DPI", "COMPARE_ALPHA", "LINESTYLES", "ANGLE_CMAP", "save_figure"]
