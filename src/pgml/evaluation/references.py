"""Adapters that turn reference results into evaluation DATA objects.

Heavy reference libraries (pandapower, OpenDSS) are imported INSIDE the functions so
that ``import pgml.evaluation`` stays light (matplotlib/plotly/networkx only). Each
adapter returns the same containers the plot functions consume
(:class:`LabeledMatrix`, :class:`VoltageProfile`, :class:`HarmonicProfile`), so a
comparison plot is just ``[ours, *references]``.

Also provides an INDEPENDENT numpy harmonic oracle (single-phase, R-const/X∝h model,
matching ``references/opendss/harmonics.md``) for the harmonic-profile plots — the
fundamental operating point is shared (validated separately against pandapower); the
oracle independently propagates the harmonics.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Sequence

import numpy as np

from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    Phase,
    Source,
    StaticSpectrum,
)

from ._util import to_float
from .data import HarmonicProfile, LabeledMatrix, VoltageProfile, row_labels
from .topology import distance_from_slack


def _numpy_shim() -> None:
    """numpy 2.x compatibility shim required by pandapower 2.14 (Inf/in1d)."""
    np.Inf = np.inf  # type: ignore[attr-defined]
    np.in1d = np.isin  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# pandapower
# ---------------------------------------------------------------------------
def pandapower_ybus(
    net, grid: Grid, id_map: dict, index, *, label: str = "pandapower"
) -> LabeledMatrix:
    """pandapower internal Ybus (pu -> SI siemens), aligned to our node·phase rows.

    This is the PURE NETWORK admittance (lines + explicit shunts, no const-Z load /
    source Norton shunts) — compare it to our ``assemble_network_ybus``.
    """
    _numpy_shim()
    y_pu = net._ppc["internal"]["Ybus"].toarray()
    base_mva = float(net._ppc["baseMVA"])
    base_kv = float(net._ppc["bus"][0, 9])
    y_base = base_mva / (base_kv**2)  # 1 / z_base
    y_si = y_pu * y_base
    bus_lookup = net._pd2ppc_lookups["bus"]

    n = index.size
    out = np.zeros((n, n), dtype=complex)
    for pp_i, node_i in id_map["bus"].items():
        ri = index.row(node_i, Phase.A)
        ppi = int(bus_lookup[pp_i])
        for pp_j, node_j in id_map["bus"].items():
            rj = index.row(node_j, Phase.A)
            ppj = int(bus_lookup[pp_j])
            out[ri, rj] = y_si[ppi, ppj]
    return LabeledMatrix(matrix=out, label=label, row_labels=row_labels(index))


def pandapower_voltage_profile(
    net,
    grid: Grid,
    id_map: dict,
    *,
    label: str = "pandapower",
    slack: Optional[int] = None,
) -> VoltageProfile:
    """Voltage profile from a solved pandapower net (``res_bus.vm_pu`` is already pu)."""
    dist = distance_from_slack(grid, slack)
    ds, pus, nids = [], [], []
    for pp_bus, node_id in id_map["bus"].items():
        ds.append(dist[int(node_id)])
        pus.append(float(net.res_bus.at[pp_bus, "vm_pu"]))
        nids.append(int(node_id))
    order = np.argsort(ds)
    return VoltageProfile(
        distances_km=np.asarray(ds)[order],
        v_pu=np.asarray(pus)[order],
        label=label,
        node_ids=np.asarray(nids)[order],
    )


# ---------------------------------------------------------------------------
# OpenDSS
# ---------------------------------------------------------------------------
def align_dss_systemy(
    y_dss: np.ndarray, node_order: Sequence[str], id_map: dict, index
) -> np.ndarray:
    """Reorder a single-phase OpenDSS ``SystemY`` to our node·phase row layout.

    Assumes DSS bus names ``BUS<n>`` and a single phase ``.1`` per bus (the IEEE-feeder
    positive-sequence circuits used here), mapped through ``id_map["bus"]``.
    """
    rowmap: dict[int, int] = {}
    for di, entry in enumerate(node_order):
        bus = entry.upper().split(".")[0]
        num = int(bus[3:])
        rowmap[di] = index.row(id_map["bus"][num], Phase.A)
    n = index.size
    out = np.zeros((n, n), dtype=complex)
    for di in range(len(node_order)):
        for dj in range(len(node_order)):
            out[rowmap[di], rowmap[dj]] = y_dss[di, dj]
    return out


def dss_systemy() -> tuple[np.ndarray, list[str]]:
    """Extract ``(SystemY [N,N] complex, YNodeOrder)`` from the active OpenDSS circuit."""
    import opendssdirect as dss

    node_order = list(dss.Circuit.YNodeOrder())
    n = len(node_order)
    flat = np.array(dss.Circuit.SystemY(), dtype=np.float64)
    y = (flat[0::2] + 1j * flat[1::2]).reshape(n, n)
    return y, node_order


def opendss_ybus(
    y_dss, node_order, id_map, index, *, label: str = "OpenDSS"
) -> LabeledMatrix:
    """Wrap an extracted OpenDSS SystemY (aligned to our rows) as a LabeledMatrix."""
    aligned = align_dss_systemy(y_dss, node_order, id_map, index)
    return LabeledMatrix(matrix=aligned, label=label, row_labels=row_labels(index))


# ---------------------------------------------------------------------------
# independent numpy harmonic oracle (single phase)
# ---------------------------------------------------------------------------
def _resolve_spectrum(app, harmonic_injection):
    if harmonic_injection is not None and app.id in harmonic_injection:
        return dict(harmonic_injection[app.id])
    spec = getattr(app, "spectrum", None)
    if isinstance(spec, StaticSpectrum):
        return {
            c.order: (to_float(c.magnitude_pu), to_float(c.phase_deg))
            for c in spec.spectrum.components
        }
    return None


def numpy_harmonic_profiles(
    grid: Grid,
    v1,
    index,
    orders: Sequence[int],
    *,
    label: str = "numpy oracle",
    slack: Optional[int] = None,
    unit: str = "pu",
    operating_point: Optional[dict] = None,
    harmonic_injection: Optional[dict] = None,
) -> list[HarmonicProfile]:
    """Independent numpy harmonic solve -> one :class:`HarmonicProfile` per order.

    Single-phase only. ``v1`` is the converged fundamental node-voltage vector (numpy
    or tensor, aligned to ``index`` rows) — the SHARED operating point. For each order
    ``h > 1`` this builds ``Y(h)`` (line series/shunt with R const & X∝h, plus the
    source Norton shunt), injects each device's harmonic current per the OpenDSS
    convention, and solves ``Y(h) V(h) = I(h)``. Order 1 returns ``v1``.
    """
    for node in grid.nodes:
        if len(node.phases) != 1:
            raise ValueError(
                "numpy_harmonic_profiles supports single-phase grids only."
            )
    v1 = np.asarray(v1.detach().cpu().numpy() if hasattr(v1, "detach") else v1).reshape(
        -1
    )
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    n = index.size

    def _build_y(h: int) -> np.ndarray:
        y = np.zeros((n, n), dtype=complex)
        for b in grid.branches:
            if not (isinstance(b, Line) and getattr(b, "in_service", True)):
                continue
            length = to_float(b.length_m)
            r = to_float(b.series_resistance_ohm_per_m[0][0]) * length
            ind = to_float(b.series_inductance_h_per_m[0][0]) * length
            c = (
                to_float(b.shunt_capacitance_f_per_m[0][0]) * length
                if b.shunt_capacitance_f_per_m
                else 0.0
            )
            ys = 1.0 / (r + 1j * h * w0 * ind)
            ysh = 1j * h * w0 * c
            fr = index.row(b.from_node, Phase.A)
            to = index.row(b.to_node, Phase.A)
            y[fr, fr] += ys + 0.5 * ysh
            y[to, to] += ys + 0.5 * ysh
            y[fr, to] -= ys
            y[to, fr] -= ys
        for a in grid.appliances:
            if isinstance(a, Source) and getattr(a, "in_service", True):
                z = to_float(a.resistance_ohm[0][0]) + 1j * h * w0 * to_float(
                    a.inductance_h[0][0]
                )
                r0 = index.row(a.node, Phase.A)
                y[r0, r0] += 1.0 / z
        return y

    # Per-device fundamental current I1 (load convention) for the injection scaling.
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, (Load, Generator)) and getattr(a, "in_service", True)):
            continue
        spec = _resolve_spectrum(a, harmonic_injection)
        if spec is None:
            continue
        row = index.row(a.node, Phase.A)
        sign = 1.0 if isinstance(a, Load) else -1.0
        p = to_float(a.p_nom_w)
        q = to_float(a.q_nom_var)
        if operating_point and a.id in operating_point:
            op = operating_point[a.id]
            if "p_w" in op:
                p = to_float(op["p_w"])
            if "q_var" in op:
                q = to_float(op["q_var"])
        s0 = complex(sign * p, sign * q)
        i1 = np.conj(s0) / np.conj(v1[row])
        devs.append((spec, row, i1))

    out: list[HarmonicProfile] = []
    dist = distance_from_slack(grid, slack)
    for h in orders:
        if h == 1:
            vh = v1
        else:
            y = _build_y(h)
            i = np.zeros(n, dtype=complex)
            for spec, row, i1 in devs:
                mag1, ang1 = spec.get(1, (1.0, 0.0))
                mag_h, ang_h = spec.get(h, (0.0, 0.0))
                i_drawn = (
                    (mag_h / mag1)
                    * abs(i1)
                    * cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (cmath.phase(i1) - math.radians(ang1))
                        )
                    )
                )
                i[row] += -i_drawn  # nodal injection
            vh = np.linalg.solve(y, i)
        ds, mags, angs, nids = [], [], [], []
        for node in grid.nodes:
            row = index.row(int(node.id), Phase.A)
            base = (to_float(node.u_rated_v)) if unit == "pu" else 1.0
            ds.append(dist[int(node.id)])
            mags.append(abs(vh[row]) / base)
            angs.append(math.degrees(cmath.phase(complex(vh[row]))))
            nids.append(int(node.id))
        o = np.argsort(ds)
        out.append(
            HarmonicProfile(
                distances_km=np.asarray(ds)[o],
                magnitude=np.asarray(mags)[o],
                angle_deg=np.asarray(angs)[o],
                order=int(h),
                frequency_hz=float(h * f0),
                label=label,
                node_ids=np.asarray(nids)[o],
                unit=unit,
            )
        )
    return out


__all__ = [
    "pandapower_ybus",
    "pandapower_voltage_profile",
    "dss_systemy",
    "align_dss_systemy",
    "opendss_ybus",
    "numpy_harmonic_profiles",
]
