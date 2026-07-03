"""Framework-agnostic plot DATA objects + builders from solver results.

The plot functions never touch torch or a reference library; they consume these
plain-numpy containers. Builders here adapt our solver outputs
(:class:`~pgml.solver.PowerFlowResult`, :class:`~pgml.solver.HarmonicFlowResult`,
assembled Y-bus tensors) into them. Reference-library adapters that emit the SAME
containers live in :mod:`pgml.evaluation.oracles`, so "ours vs reference" is just
a list of these objects handed to one plot function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pgml.assembly._params import phase_voltage_magnitude
from pgml.schemas.grid_schema import Grid, Phase

from ._util import to_float, to_numpy
from .topology import distance_from_slack


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoltageProfile:
    """Per-node voltage magnitude vs distance from slack (one implementation/run).

    Sorted by ``distances_km`` ascending. ``v_pu`` is per-unit on the node's
    line-to-neutral base. ``label`` names the implementation (e.g. ``"pgml"``).
    """

    distances_km: np.ndarray
    v_pu: np.ndarray
    label: str
    node_ids: Optional[np.ndarray] = None


@dataclass(frozen=True)
class HarmonicProfile:
    """Per-node harmonic voltage (magnitude + angle) vs distance, at one order.

    ``magnitude`` is in ``unit`` (``"pu"`` of the L-N base, or ``"V"``);
    ``angle_deg`` is the voltage phasor angle in degrees. Sorted by distance.
    """

    distances_km: np.ndarray
    magnitude: np.ndarray
    angle_deg: np.ndarray
    order: int
    frequency_hz: float
    label: str
    node_ids: Optional[np.ndarray] = None
    unit: str = "pu"


@dataclass(frozen=True)
class LabeledMatrix:
    """A labeled complex matrix (e.g. a Y-bus version) for heatmap comparison."""

    matrix: np.ndarray  # complex [N, N]
    label: str
    row_labels: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def row_labels(index) -> list[str]:
    """``"<node>·<phase>"`` label per row of a :class:`NodePhaseIndex`."""
    phase_names = ("a", "b", "c", "n")
    nids = to_numpy(index.node_ids).astype(int)
    pcs = to_numpy(index.phase_codes).astype(int)
    return [f"{int(nid)}·{phase_names[int(pc)]}" for nid, pc in zip(nids, pcs)]


def _phase_base(node) -> float:
    """Line-to-neutral base voltage of a node (per-unit denominator)."""
    return phase_voltage_magnitude(to_float(node.u_rated_v), len(node.phases))


def _scenario_slice(
    v_np: np.ndarray, n_rows: int, scenario: int, extra_axes: int = 0
) -> np.ndarray:
    """Reduce leading batch dims of a voltage array to a single scenario.

    ``v_np`` has trailing shape ``(..., N)`` (power flow) or ``(..., H, N)``
    (harmonic, ``extra_axes=1``). Picks scenario ``scenario`` from the flattened
    leading batch and returns ``(N,)`` or ``(H, N)``.
    """
    keep = extra_axes + 1
    if v_np.ndim <= keep:
        return v_np
    tail = v_np.shape[-keep:]
    flat = v_np.reshape(-1, *tail)
    return flat[scenario]


# ---------------------------------------------------------------------------
# builders from our solver results
# ---------------------------------------------------------------------------
def voltage_profile(
    result,
    grid: Grid,
    *,
    label: str = "pgml",
    phase: Phase = Phase.A,
    slack: Optional[int] = None,
    scenario: int = 0,
) -> VoltageProfile:
    """Build a :class:`VoltageProfile` from a :class:`PowerFlowResult`.

    For each node carrying ``phase``, takes ``|V|`` at that phase, converts to pu on
    the node's L-N base, and pairs it with the node's distance from the slack. Batched
    results are reduced to ``scenario`` (default 0).
    """
    index = result.index
    n = index.size
    v = _scenario_slice(to_numpy(result.v), n, scenario)
    dist = distance_from_slack(grid, slack)
    ds, pus, nids = [], [], []
    for node in grid.nodes:
        if phase not in node.phases:
            continue
        vc = v[index.row(int(node.id), phase)]
        ds.append(dist[int(node.id)])
        pus.append(abs(vc) / _phase_base(node))
        nids.append(int(node.id))
    order = np.argsort(ds)
    return VoltageProfile(
        distances_km=np.asarray(ds)[order],
        v_pu=np.asarray(pus)[order],
        label=label,
        node_ids=np.asarray(nids)[order],
    )


def harmonic_profile(
    hresult,
    grid: Grid,
    order: int,
    *,
    label: str = "pgml",
    phase: Phase = Phase.A,
    slack: Optional[int] = None,
    scenario: int = 0,
    unit: str = "pu",
) -> HarmonicProfile:
    """Build a :class:`HarmonicProfile` (magnitude + angle vs distance) at ``order``.

    Selects the slice of ``hresult.v`` whose frequency matches ``order * f0``.
    ``unit="pu"`` divides by the node L-N base; ``unit="V"`` keeps volts.
    """
    index = hresult.index
    n = index.size
    freqs = to_numpy(hresult.frequencies_hz)
    f0 = float(grid.base_frequency_hz)
    k = int(np.argmin(np.abs(freqs - order * f0)))
    vmat = _scenario_slice(to_numpy(hresult.v), n, scenario, extra_axes=1)  # (H, N)
    vh = vmat[k]
    dist = distance_from_slack(grid, slack)
    ds, mags, angs, nids = [], [], [], []
    for node in grid.nodes:
        if phase not in node.phases:
            continue
        vc = vh[index.row(int(node.id), phase)]
        base = _phase_base(node) if unit == "pu" else 1.0
        ds.append(dist[int(node.id)])
        mags.append(abs(vc) / base)
        angs.append(np.degrees(np.angle(vc)))
        nids.append(int(node.id))
    o = np.argsort(ds)
    return HarmonicProfile(
        distances_km=np.asarray(ds)[o],
        magnitude=np.asarray(mags)[o],
        angle_deg=np.asarray(angs)[o],
        order=int(order),
        frequency_hz=float(freqs[k]),
        label=label,
        node_ids=np.asarray(nids)[o],
        unit=unit,
    )


def harmonic_profiles(
    hresult, grid: Grid, orders: Sequence[int], *, label: str = "pgml", **kw
) -> list[HarmonicProfile]:
    """Convenience: a :class:`HarmonicProfile` per order (e.g. for the 3D plot)."""
    return [harmonic_profile(hresult, grid, int(h), label=label, **kw) for h in orders]


def labeled_matrix(y, index, *, label: str, freq_index: int = 0) -> LabeledMatrix:
    """Wrap an assembled Y-bus tensor as a :class:`LabeledMatrix` for heatmaps.

    Accepts ``[N, N]``, ``[H, N, N]`` or ``[*batch, H, N, N]``; selects
    ``freq_index`` from the flattened leading axes.
    """
    arr = to_numpy(y)
    if arr.ndim > 2:
        nn = arr.shape[-1]
        arr = arr.reshape(-1, nn, nn)[freq_index]
    return LabeledMatrix(
        matrix=np.asarray(arr), label=label, row_labels=row_labels(index)
    )


__all__ = [
    "VoltageProfile",
    "HarmonicProfile",
    "LabeledMatrix",
    "row_labels",
    "voltage_profile",
    "harmonic_profile",
    "harmonic_profiles",
    "labeled_matrix",
]
