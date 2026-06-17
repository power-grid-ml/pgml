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
# Feeder builders (pandapower -> pgml grid + synthesized Carson geometry + spectra)
# ---------------------------------------------------------------------------
# Typical 6-pulse converter line-current spectrum (fraction of fundamental).
CONVERTER_SPECTRUM = [
    (1, 1.0, 0.0),
    (5, 0.20, 0.0),
    (7, 0.14, 0.0),
    (11, 0.09, 0.0),
    (13, 0.07, 0.0),
]


def _attach_spectrum_farthest(grid, n_loads: int, spectrum) -> None:
    from pgml.schemas.grid_schema import (
        HarmonicComponent,
        Load,
        SpectrumPoint,
        StaticSpectrum,
    )

    from .topology import distance_from_slack

    dist = distance_from_slack(grid)
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    loads.sort(key=lambda a: dist.get(int(a.node), 0.0), reverse=True)
    comps = [
        HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a) for o, m, a in spectrum
    ]
    for ld in loads[:n_loads]:
        ld.spectrum = StaticSpectrum(spectrum=SpectrumPoint(components=comps))


def ieee33_geometry_grid(*, n_harmonic_loads: int = 3, spectrum=None):
    """IEEE-33 as a pgml grid with synthesized Carson geometry + converter spectra."""
    _numpy_shim()
    import pandapower as pp
    import pandapower.networks as pn

    from pgml.convert.pandapower import to_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    net = pn.case33bw()
    pp.runpp(net, numba=False)
    grid, id_map = to_grid(net)
    synthesize_grid_geometry(grid)
    _attach_spectrum_farthest(grid, n_harmonic_loads, spectrum or CONVERTER_SPECTRUM)
    return grid, id_map


def cigre_lv_geometry_grid(*, n_harmonic_loads: int = 3, spectrum=None):
    """CIGRE LV residential feeder (fed by a Thévenin source at its LV busbar) as a
    pgml grid with synthesized Carson geometry + converter spectra.

    The 20/0.4 kV transformer + MV grid are abstracted to a stiff 0.4 kV source so the
    comparison isolates the LV line (Carson) model.
    """
    _numpy_shim()
    import networkx as nx
    import pandapower as pp
    import pandapower.networks as pn
    import pandapower.topology as top

    from pgml.convert.pandapower import to_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    net = pn.create_cigre_network_lv()
    mg = top.create_nxgraph(net, include_trafos=False)
    comp = list(nx.node_connected_component(mg, 2))  # residential LV busbar = bus 2
    sub = pp.select_subnet(net, comp, include_results=False)
    pp.create_ext_grid(sub, bus=2, vm_pu=1.0)
    pp.runpp(sub, numba=False)
    grid, id_map = to_grid(sub)
    synthesize_grid_geometry(grid)
    _attach_spectrum_farthest(grid, n_harmonic_loads, spectrum or CONVERTER_SPECTRUM)
    return grid, id_map


# ---------------------------------------------------------------------------
# OpenDSS from a pgml conductor-geometry grid (the Carson harmonic comparison)
# ---------------------------------------------------------------------------
def build_opendss_geometry_circuit(grid, *, slack_node: Optional[int] = None) -> dict:
    """Build a PASSIVE single-phase OpenDSS circuit from a pgml geometry ``grid``.

    Emits the slack Vsource (Thévenin from the :class:`Source`) and one WireData +
    LineGeometry + Line per geometry line, using the SAME synthesized conductor data
    pgml uses — so OpenDSS's ``SystemY(h)`` equals pgml's harmonic ``Y(h)`` up to the
    Carson model (which is bit-exact). No loads (harmonic injection is applied
    externally). Returns ``{node_id: dss_bus_name}``. DERI earth model (OpenDSS default).
    """
    import opendssdirect as dss

    from pgml.schemas.grid_schema import Line as GridLine, Source as GridSource

    f0 = float(grid.base_frequency_hz)
    src = next(a for a in grid.appliances if isinstance(a, GridSource) and a.in_service)
    if slack_node is None:
        slack_node = int(src.node)
    node_by_id = {int(n.id): n for n in grid.nodes}
    busname = {int(n.id): f"bus{int(n.id)}" for n in grid.nodes}

    kv = float(node_by_id[slack_node].u_rated_v) / 1000.0
    u_ref = (
        float(src.u_ref_v[0])
        if isinstance(src.u_ref_v, (list, tuple))
        else float(src.u_ref_v)
    )
    ang = (
        float(src.u_angle_deg[0])
        if isinstance(src.u_angle_deg, (list, tuple))
        else float(src.u_angle_deg)
    )
    rs = float(src.resistance_ohm[0][0])
    xs = 2.0 * math.pi * f0 * float(src.inductance_h[0][0])

    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.pgml_geom basekv={kv} phases=1 bus1={busname[slack_node]}.1 "
        f"pu={u_ref / (kv * 1000.0):.10g} angle={ang} frequency={f0} r1={rs} x1={xs}"
    )
    dss.Text.Command("Set earthmodel=Deri")
    for ln in grid.branches:
        if not (
            isinstance(ln, GridLine)
            and ln.in_service
            and ln.conductor_geometry is not None
        ):
            continue
        c = ln.conductor_geometry.conductors[0]  # single-phase synthesized geometry
        rho = float(ln.conductor_geometry.earth_resistivity_ohm_m)
        wd, gn = f"wd{ln.id}", f"geo{ln.id}"
        dss.Text.Command(
            f"New WireData.{wd} Rdc={to_float(c.r_dc_ohm_per_m)} GMRac={to_float(c.gmr_m)} "
            f"radius={to_float(c.radius_m)} Runits=m GMRunits=m radunits=m"
        )
        dss.Text.Command(
            f"New LineGeometry.{gn} nconds=1 nphases=1 cond=1 wire={wd} "
            f"x={to_float(c.x_m)} h={to_float(c.y_m)} units=m"
        )
        dss.Text.Command(
            f"New Line.l{ln.id} phases=1 bus1={busname[ln.from_node]}.1 "
            f"bus2={busname[ln.to_node]}.1 geometry={gn} length={to_float(ln.length_m)} "
            f"units=m rho={rho}"
        )
    dss.Text.Command(f"Set voltagebases=[{kv}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    return busname


def _pgml_harmonic_y(grid, index, h: int):
    """pgml's harmonic admittance ``Y(h)`` = passive network + source Norton (numpy)."""
    import torch

    from pgml.assembly import assemble_network_ybus
    from pgml.assembly._stamps import _cdtype, _rdtype
    from pgml.assembly.ybus import _stamp_sources

    f0 = float(grid.base_frequency_hz)
    cdt = torch.complex128
    f = torch.tensor([h * f0], dtype=torch.float64)
    yb = assemble_network_ybus(grid, [h * f0], dtype=cdt).Y.clone()
    yb = _stamp_sources(
        grid, f, yb, index, _cdtype(cdt), _rdtype(cdt), torch.device("cpu"), None
    )
    return yb[0].detach().cpu().numpy()


def opendss_geometry_harmonic_profiles(
    grid,
    hres,
    orders,
    *,
    slack_node: Optional[int] = None,
    unit: str = "pu",
    label: str = "OpenDSS (Carson)",
) -> list[HarmonicProfile]:
    """OpenDSS line-model harmonic voltage profiles for the SAME geometry as pgml.

    Builds OpenDSS ``SystemY(h)`` from the grid's conductor geometry and solves each
    harmonic with the SAME nodal injection pgml converged to (``I(h)=Y_pgml(h)·V_pgml(h)``,
    a fixed physical current). The only difference vs pgml is the line admittance, so
    this isolates the Carson line model — paired with :func:`pgml.evaluation.data.harmonic_profiles`
    it is a true OpenDSS-vs-pgml harmonic comparison. Single-phase feeders only.
    """
    index = hres.index
    f0 = float(grid.base_frequency_hz)
    freqs = hres.frequencies_hz.detach().cpu().numpy()
    dssY = opendss_geometry_systemy(grid, index, orders, slack_node=slack_node)
    dist = distance_from_slack(grid, slack_node)
    v = hres.v.detach().cpu().numpy()  # [H, N]
    profs = []
    for h in orders:
        k = int(np.argmin(np.abs(freqs - h * f0)))
        vp = v[k]
        i_inj = _pgml_harmonic_y(grid, index, h) @ vp
        vd = np.linalg.solve(dssY[int(h)], i_inj)
        ds, mags, angs, nids = [], [], [], []
        for node in grid.nodes:
            row = index.row(int(node.id), Phase.A)
            base = to_float(node.u_rated_v) if unit == "pu" else 1.0
            ds.append(dist[int(node.id)])
            mags.append(abs(vd[row]) / base)
            angs.append(np.degrees(np.angle(vd[row])))
            nids.append(int(node.id))
        o = np.argsort(ds)
        profs.append(
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
    return profs


def _align_systemy_by_nodeid(y_dss, node_order, index) -> np.ndarray:
    """Align OpenDSS SystemY whose buses are named ``bus<node_id>.1`` to our rows."""
    rowmap = {}
    for di, entry in enumerate(node_order):
        nid = int(entry.upper().split(".")[0][3:])  # 'BUS<nid>'
        rowmap[di] = index.row(nid, Phase.A)
    n = index.size
    out = np.zeros((n, n), dtype=complex)
    for di in range(len(node_order)):
        for dj in range(len(node_order)):
            out[rowmap[di], rowmap[dj]] = y_dss[di, dj]
    return out


def opendss_geometry_systemy(
    grid, index, orders, *, slack_node: Optional[int] = None
) -> dict:
    """``{order: SystemY(order·f0) aligned to our rows}`` from a pgml geometry grid.

    Builds the passive OpenDSS circuit once (Carson/DERI) and reads ``SystemY`` at each
    harmonic order (forcing a Y rebuild at each frequency). This is OpenDSS's harmonic
    admittance for the SAME geometry pgml uses — the ground truth for the comparison.
    """
    import opendssdirect as dss

    build_opendss_geometry_circuit(grid, slack_node=slack_node)
    f0 = float(grid.base_frequency_hz)
    out = {}
    for h in orders:
        dss.Text.Command(f"set frequency={h * f0}")
        dss.Solution.BuildYMatrix(2, 1)
        y, node_order = dss_systemy()
        out[int(h)] = _align_systemy_by_nodeid(y, node_order, index)
    return out


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
    "ieee33_geometry_grid",
    "cigre_lv_geometry_grid",
    "pandapower_ybus",
    "pandapower_voltage_profile",
    "dss_systemy",
    "align_dss_systemy",
    "opendss_ybus",
    "build_opendss_geometry_circuit",
    "opendss_geometry_systemy",
    "opendss_geometry_harmonic_profiles",
    "numpy_harmonic_profiles",
]
