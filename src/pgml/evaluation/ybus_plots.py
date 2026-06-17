"""Y-bus comparison heatmaps + a difference heatmap.

``plot_ybus_heatmaps`` lays out several Y-bus versions side by side (ours vs each
reference); ``plot_ybus_difference`` shows ``|Y_a - Y_b|`` for the two most-similar
versions. Y-bus entries span a huge dynamic range (line admittances ~1 S vs a source
Norton shunt ~1e6 S), so magnitude heatmaps use a logarithmic color scale.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

from .data import LabeledMatrix


def _component(m: np.ndarray, part: str) -> np.ndarray:
    if part == "abs":
        return np.abs(m)
    if part == "real":
        return np.real(m)
    if part == "imag":
        return np.imag(m)
    raise ValueError(f"part must be 'abs'/'real'/'imag', got {part!r}.")


def _make_norm(data: np.ndarray, part: str):
    """Log norm for magnitudes, symmetric-log norm for signed real/imag parts."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    if part == "abs":
        pos = finite[finite > 0]
        vmin = float(pos.min()) if pos.size else 1e-12
        vmax = float(finite.max()) if finite.size else 1.0
        return LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
    amax = float(np.abs(finite).max()) or 1.0
    linthresh = max(amax * 1e-6, 1e-12)
    return SymLogNorm(linthresh=linthresh, vmin=-amax, vmax=amax)


def _ticklabels(ax, labels: Optional[list[str]]) -> None:
    if labels is not None and len(labels) <= 16:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xlabel("column (node·phase)")
        ax.set_ylabel("row (node·phase)")


def plot_ybus_heatmaps(
    matrices: Sequence[LabeledMatrix],
    *,
    part: str = "abs",
    cmap: Optional[str] = None,
    share_scale: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    suptitle: Optional[str] = None,
):
    """Heatmaps of several Y-bus versions side by side.

    Parameters
    ----------
    matrices:
        The versions to compare (e.g. ours, pandapower, OpenDSS), aligned to the same
        node·phase row order.
    part:
        ``"abs"`` (log magnitude, default), ``"real"`` or ``"imag"`` (symmetric log).
    share_scale:
        If True (default) all panels share one color scale (so they are directly
        comparable) with a single shared colorbar.
    """
    matrices = list(matrices)
    if not matrices:
        raise ValueError("Need at least one matrix to plot.")
    comps = [_component(m.matrix, part) for m in matrices]
    cmap = cmap or ("viridis" if part == "abs" else "RdBu_r")
    shared_norm = (
        _make_norm(np.concatenate([c.ravel() for c in comps]), part)
        if share_scale
        else None
    )

    k = len(matrices)
    figsize = figsize or (4.3 * k + 1.4, 4.6)
    fig, axes = plt.subplots(
        1, k, figsize=figsize, constrained_layout=True, squeeze=False
    )
    axes = axes[0]
    im = None
    for ax, m, c in zip(axes, matrices, comps):
        norm = shared_norm if share_scale else _make_norm(c, part)
        im = ax.imshow(c, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        ax.set_title(f"{m.label}\n({part})", fontsize=10)
        _ticklabels(ax, m.row_labels)
        if not share_scale:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if share_scale and im is not None:
        label = "|Y| [S]" if part == "abs" else f"Y {part} [S]"
        fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.02, label=label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig


def plot_ybus_difference(
    a: LabeledMatrix,
    b: LabeledMatrix,
    *,
    cmap: str = "magma",
    figsize: tuple[float, float] = (5.6, 4.8),
    title: Optional[str] = None,
):
    """Heatmap of the entrywise complex difference magnitude ``|Y_a - Y_b|`` [S].

    Intended for the two MOST SIMILAR versions (e.g. our network Y vs a reference's
    network Y), where the residual should be near floating-point zero. Annotates the
    max entry error and the Frobenius norm of the difference.
    """
    if a.matrix.shape != b.matrix.shape:
        raise ValueError(f"Shape mismatch: {a.matrix.shape} vs {b.matrix.shape}.")
    diff = a.matrix - b.matrix
    mag = np.abs(diff)
    pos = mag[mag > 0]
    norm = LogNorm(vmin=float(pos.min()), vmax=float(mag.max())) if pos.size else None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mag, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|ΔY| [S]")
    _ticklabels(ax, a.row_labels)
    max_err = float(mag.max())
    fro = float(np.linalg.norm(diff))
    ax.set_title(
        title or f"|ΔY|: {a.label} − {b.label}\nmax={max_err:.2e} S, ‖Δ‖_F={fro:.2e} S",
        fontsize=11,
    )
    return fig


__all__ = ["plot_ybus_heatmaps", "plot_ybus_difference"]
