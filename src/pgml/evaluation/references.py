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

The full-network oracles extend this to the complete CIGRE LV grid including transformers
(all three 20/0.4 kV units) and switches, supporting both single-phase and three-phase
grids:

- :func:`numpy_harmonic_voltages` — pure-numpy oracle using pgml's EXACT Y-bus formulas
  (R const / X∝h for all elements).  Machine-precision parity (~1e-13 V absolute) vs
  :func:`pgml.solver.solve_harmonic_flow`.  Kept as the regression oracle.

- :func:`opendss_harmonic_voltages` — LIVE OpenDSS oracle.  For single-phase grids with
  synthesized conductor geometry, OpenDSS builds the geometry lines (Carson/Deri) and the
  resulting ``SystemY(h)`` is used directly for the line contributions, while source Norton,
  transformer stamps, and switch stamps are stamped using pgml's exact formulas.  For
  three-phase grids tagged with the ``sequence_aware`` harmonic model, a full OpenDSS
  circuit is built with R1/X1/R0/X0 lines and native Transformer/Reactor elements.
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
    Switch,
    Transformer,
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


def cigre_lv_full_grid(*, phase_mode=None, source_impedance_ohm=None):
    """The FULL CIGRE LV benchmark grid (all 3 feeders + MV source + 3 transformers).

    Unlike :func:`cigre_lv_geometry_grid` (one residential feeder with synthesized
    Carson geometry), this returns the WHOLE pandapower CIGRE LV network converted to a
    pgml :class:`~pgml.schemas.grid_schema.Grid` with standard R/X lines and the three
    20/0.4 kV transformers intact, fed by the single MV ext-grid source. The shared
    entry point for the full-grid examples + the OpenDSS oracle.

    The stock pandapower ext-grid converts to a near-ideal source (R~1e-6 Ohm) which
    short-circuits the bus at harmonics; a FINITE series impedance is applied so the
    source does not fully absorb injected harmonics (``source_impedance_ohm`` |Z| at the
    source's rated voltage, ``source.rx_ratio`` for the X/R split — config defaults under
    ``source.*``). Larger = weaker upstream grid = more cross-feeder coupling.

    Parameters
    ----------
    phase_mode:
        ``PhaseMode.SINGLE_PHASE_EQUIV`` (default) or ``PhaseMode.THREE_PHASE``.
    source_impedance_ohm:
        Source series-impedance magnitude [Ohm]; ``None`` -> config
        ``source.series_impedance_ohm``. Pass ``0`` to keep the converted (stiff) source.

    Returns
    -------
    (Grid, id_map)
    """
    _numpy_shim()
    import math

    import pandapower.networks as pn

    from pgml import config as _config
    from pgml.convert.pandapower import PhaseMode, to_grid
    from pgml.schemas.grid_schema import Source

    mode = phase_mode if phase_mode is not None else PhaseMode.SINGLE_PHASE_EQUIV
    grid, id_map = to_grid(pn.create_cigre_network_lv(), phase_mode=mode)

    z = (
        _config.get("source.series_impedance_ohm")
        if source_impedance_ohm is None
        else float(source_impedance_ohm)
    )
    if z > 0.0:
        rx = float(_config.get("source.rx_ratio"))
        r = z / math.sqrt(1.0 + rx * rx)
        ll = (rx * r) / (2.0 * math.pi * float(grid.base_frequency_hz))  # X = 2*pi*f0*L
        for a in grid.appliances:
            if isinstance(a, Source):
                p = len(a.phases)
                a.resistance_ohm = [
                    [r if i == j else 0.0 for j in range(p)] for i in range(p)
                ]
                a.inductance_h = [
                    [ll if i == j else 0.0 for j in range(p)] for i in range(p)
                ]
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


# ---------------------------------------------------------------------------
# Full-network harmonic oracle: CIGRE LV with transformers and switches
# ---------------------------------------------------------------------------


def _stamp_transformer_numpy(
    y: np.ndarray,
    b: Transformer,
    h: int,
    w0: float,
    fr_rows: list,
    to_rows: list,
    p: int,
) -> None:
    """Stamp a two-winding transformer into the numpy Y-bus (in-place).

    Mirrors :mod:`pgml.assembly._transformer` exactly:

    - **P == 1** (single-phase / positive-sequence equivalent): the vector group is
      folded into a complex line-to-line ratio ``t = (u_from/u_to) · tap_mag ·
      e^{j·shift_deg}`` and the textbook off-nominal-tap pi is applied::

          Y_ff = y_se / |t|² + y_m,   Y_ft = −y_se / conj(t)
          Y_tf = −y_se / t,            Y_tt = y_se

    - **P == 3** (phase-domain): the nodal block is built as ``Nᵀ·Y_winding·N``
      where ``N = blockdiag(N_hv, N_lv)`` (wye-grounded → ``I₃``, delta → ``M`` or
      ``Mᵀ`` per clock), and the coil turns ratio is::

          τ = (coil_from / coil_to) · tap_mag

      with delta coils rated at the line-to-line voltage and wye coils at
      ``u_rated / √3``.  The magnetizing shunt ``y_m = G_m + jB_m`` is added to the
      HV terminal phase diagonal outside the incidence transform — identical to pgml.

    ``y``, ``fr_rows``, ``to_rows``, and ``p`` are passed in to avoid re-computing
    them; ``b`` supplies all other transformer parameters.
    """
    from pgml.assembly._transformer import block_incidence, resolve_vector_group

    r_t = to_float(b.series_resistance_ohm)
    l_t = to_float(b.series_inductance_h)
    z_se = r_t + 1j * h * w0 * l_t
    y_se = 1.0 / z_se
    gm = to_float(b.magnetizing_conductance_s)
    lm = b.magnetizing_inductance_h
    bm = -1.0 / (h * w0 * to_float(lm)) if lm is not None else 0.0
    ym = complex(gm, bm)

    u_from = to_float(b.u_rated_from_v)
    u_to = to_float(b.u_rated_to_v)
    tap_mag = to_float(b.tap.ratio_magnitude)

    if p == 1:
        # Single-phase equivalent: nominal ratio = u_from/u_to (LL/LL), folded
        # together with the off-nominal tap and the clock phase shift.
        shift_rad = to_float(b.tap.shift_deg) * math.pi / 180.0
        t = (u_from / u_to) * tap_mag * cmath.exp(1j * shift_rad)
        abs_t2 = abs(t) ** 2
        y_ff = y_se / abs_t2 + ym
        y_ft = -y_se / complex(t).conjugate()
        y_tf = -y_se / t
        y_tt = y_se
        fr = fr_rows[0]
        to = to_rows[0]
        y[fr, fr] += y_ff
        y[fr, to] += y_ft
        y[to, fr] += y_tf
        y[to, to] += y_tt
    else:
        # Phase-domain: winding-incidence primitive Nᵀ Y_winding N.
        import torch as _torch

        vg = resolve_vector_group(b)
        # Coil turns ratio: delta → LL voltage, wye → LN voltage = u/√3.
        sqrt3 = math.sqrt(3.0)
        coil_from = u_from if vg.from_side.kind == "delta" else u_from / sqrt3
        coil_to = u_to if vg.to_side.kind == "delta" else u_to / sqrt3
        tau = (coil_from / coil_to) * tap_mag

        # Block incidence N [2P, 2P] (constant, real).
        rdt = _torch.float64
        n_blk = (
            block_incidence(vg, p, rdt, _torch.device("cpu")).numpy().astype(complex)
        )

        # 6×6 winding primitive Y_winding.
        eye_p = np.eye(p, dtype=complex)
        y_w = np.block(
            [
                [(y_se / tau**2) * eye_p, -(y_se / tau) * eye_p],
                [-(y_se / tau) * eye_p, y_se * eye_p],
            ]
        )
        # Nodal block: Nᵀ Y_winding N  [2P, 2P].
        y_node = n_blk.T @ y_w @ n_blk

        # Magnetizing shunt on the HV terminal diagonal (outside incidence).
        y_node[:p, :p] += ym * eye_p

        # Scatter into global Y.
        all_rows = list(fr_rows) + list(to_rows)
        for i in range(2 * p):
            for j in range(2 * p):
                y[all_rows[i], all_rows[j]] += y_node[i, j]


def _build_numpy_ybus(grid: Grid, h: int, index) -> np.ndarray:
    """Build the full per-harmonic Y-bus (numpy) mirroring pgml's assembly.

    Stamps Lines (series + shunt, R const / X∝h), Switches (series RL),
    Transformers (off-nominal complex-tap leakage-pi, magnetizing shunt on HV
    diagonal), and Source Norton shunts — exactly the same formulas as
    ``pgml.assembly.ybus._stamp_network`` + ``_stamp_sources``.

    Works for single-phase (P=1) and three-phase (P=3) grids: phase matrices
    are stamped into the compact ``NodePhaseIndex`` rows via ``index.rows()``.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    h:
        Harmonic order (integer >= 1).
    index:
        :class:`~pgml.assembly.NodePhaseIndex` for this grid.

    Returns
    -------
    numpy.ndarray
        Complex ``[N, N]`` admittance matrix.
    """
    n = index.size
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    y = np.zeros((n, n), dtype=complex)

    # Lines: series pi + shunt capacitance
    for b in grid.branches:
        if not (isinstance(b, Line) and getattr(b, "in_service", True)):
            continue
        if getattr(b, "conductor_geometry", None) is not None:
            # Carson geometry lines: not supported in this oracle (use
            # opendss_geometry_systemy for Carson validation)
            continue
        length = to_float(b.length_m)
        phases_b = b.from_phases
        p = len(phases_b)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        r_mat = (
            np.array(
                [
                    [to_float(b.series_resistance_ohm_per_m[i][j]) for j in range(p)]
                    for i in range(p)
                ]
            )
            * length
        )
        l_mat = (
            np.array(
                [
                    [to_float(b.series_inductance_h_per_m[i][j]) for j in range(p)]
                    for i in range(p)
                ]
            )
            * length
        )
        c_mat = (
            np.array(
                [
                    [to_float(b.shunt_capacitance_f_per_m[i][j]) for j in range(p)]
                    for i in range(p)
                ]
            )
            * length
            if b.shunt_capacitance_f_per_m is not None
            else np.zeros((p, p))
        )
        z_mat = r_mat + 1j * h * w0 * l_mat
        ys = np.linalg.inv(z_mat)
        ysh = 1j * h * w0 * c_mat
        half_ysh = 0.5 * ysh
        for i_ph, (fr, to) in enumerate(zip(fr_rows, to_rows)):
            for j_ph in range(p):
                fr2 = fr_rows[j_ph]
                to2 = to_rows[j_ph]
                y[fr_rows[i_ph], fr2] += ys[i_ph, j_ph] + (
                    half_ysh[i_ph, j_ph] if i_ph == j_ph else 0.0
                )
                y[to_rows[i_ph], to2] += ys[i_ph, j_ph] + (
                    half_ysh[i_ph, j_ph] if i_ph == j_ph else 0.0
                )
                y[fr_rows[i_ph], to2] -= ys[i_ph, j_ph]
                y[to_rows[i_ph], fr2] -= ys[i_ph, j_ph]

    # Switches: series RL only (no shunt)
    for b in grid.branches:
        if not (isinstance(b, Switch) and getattr(b, "in_service", True) and b.closed):
            continue
        phases_b = b.from_phases
        p = len(phases_b)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        r_sw = to_float(b.resistance_ohm)
        l_sw = to_float(b.inductance_h)
        z_sw = r_sw + 1j * h * w0 * l_sw
        ys_sw = 1.0 / z_sw
        # Diagonal per phase (pgml stamps each phase independently for switches)
        for i_ph in range(p):
            fr = fr_rows[i_ph]
            to = to_rows[i_ph]
            y[fr, fr] += ys_sw
            y[to, to] += ys_sw
            y[fr, to] -= ys_sw
            y[to, fr] -= ys_sw

    # Transformers: winding-incidence primitive (new vector-group model).
    # See pgml.assembly._transformer for the full derivation.
    for b in grid.branches:
        if not (isinstance(b, Transformer) and getattr(b, "in_service", True)):
            continue
        p = len(b.from_phases)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        _stamp_transformer_numpy(y, b, h, w0, fr_rows, to_rows, p)

    # Sources: Norton shunt Y_s = Z_s(h)^-1 (held at zero harmonic voltage)
    for a in grid.appliances:
        if not (isinstance(a, Source) and getattr(a, "in_service", True)):
            continue
        phases_a = a.phases
        p = len(phases_a)
        src_rows = index.rows(a.node)
        r_mat = np.array(
            [[to_float(a.resistance_ohm[i][j]) for j in range(p)] for i in range(p)]
        )
        l_mat = np.array(
            [[to_float(a.inductance_h[i][j]) for j in range(p)] for i in range(p)]
        )
        z_mat = r_mat + 1j * h * w0 * l_mat
        ys_mat = np.linalg.inv(z_mat)
        for i_ph in range(p):
            for j_ph in range(p):
                y[src_rows[i_ph], src_rows[j_ph]] += ys_mat[i_ph, j_ph]

    return y


def numpy_harmonic_voltages(
    grid: Grid,
    harmonic_injection: Optional[dict],
    orders: Sequence[int],
    *,
    slack: str = "norton",
    v1: Optional[np.ndarray] = None,
    operating_point: Optional[dict] = None,
    node_sources: Optional[Sequence] = None,
) -> np.ndarray:
    """Pure-numpy harmonic voltage oracle — exact pgml parity (R const / X∝h).

    Returns complex node voltages ``[len(orders), N]`` aligned to
    :func:`pgml.assembly.node_phase_index` rows, solving the same linear harmonic
    system as :func:`pgml.solver.solve_harmonic_flow`.

    This is the **regression oracle**: it reimplements pgml's EXACT Y-bus formulas
    (R const / X∝h for all elements including transformers and source Norton) in
    pure numpy, giving machine-precision parity (~1e-13 V absolute) vs
    ``solve_harmonic_flow``.  No live OpenDSS circuit is built; the
    ``import opendssdirect`` dependency is not required.

    **Model (exact pgml parity, R const / X∝h)**

    - Lines: series pi (``R`` fixed, ``X(h) = h·X(f0)``), shunt capacitance
      (``B(h) = h·B(f0)``).  Conductor-geometry lines are skipped (the Carson
      path lives in :func:`opendss_harmonic_voltages`).
    - Switches: series RL (identical scaling).
    - Transformers: per-phase diagonal off-nominal-tap leakage-pi stamp —
      ``Y_ff = y_se/|t|² + y_m``, ``Y_ft = −y_se/t*``, ``Y_tf = −y_se/t``,
      ``Y_tt = y_se`` — where ``y_se = (R + j·h·2πf₀·L)⁻¹`` and
      ``t = ratio_magnitude · exp(j·shift_deg)``.
    - Sources (Norton mode): shunt ``Y_s(h) = Z_s(h)⁻¹`` on the source diagonal.
    - Harmonic injection: OpenDSS convention (``references/opendss/harmonics.md``).

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.
    harmonic_injection:
        Per-device harmonic-current spec —
        ``{appliance_id: {order: (magnitude_pu, phase_deg)}}``.
    orders:
        Harmonic orders to solve (e.g. ``[1, 5, 11]``).
    slack:
        Only ``"norton"`` is implemented.
    v1:
        Optional pre-computed fundamental voltage vector ``[N]``.  When ``None``,
        a linear const-Z fundamental is solved internally.
    operating_point:
        Optional per-device operating-point override
        ``{appliance_id: {order: value}}`` — passed through to the linear
        fundamental solve when ``v1 is None``.  Currently unused when ``v1`` is
        provided (the passed ``v1`` already encodes the operating point).
    node_sources:
        Optional sequence of :class:`~pgml.solver.NodeHarmonicSource` — per-node
        harmonic disturbance sources applied ONLY at ``h > 1``.  Each source stamps
        its Norton current ``I_N`` (and, for ``kind="voltage"``, the shunt ``Y_s``)
        into the harmonic system using the same physics as
        :func:`pgml.solver.solve_harmonic_flow`.  When ``None`` (default), the oracle
        is byte-identical to the pre-node-source behaviour.

    Returns
    -------
    numpy.ndarray
        Complex ``[H, N]``.

    See Also
    --------
    opendss_harmonic_voltages : Live OpenDSS oracle (Carson lines).
    numpy_harmonic_profiles : Single-phase numpy oracle (lines + source only).
    """
    if slack != "norton":
        raise ValueError(
            f"numpy_harmonic_voltages supports only slack='norton'; got {slack!r}"
        )
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Generator as PgmlGen, Load as PgmlLoad

    orders_list = [int(h) for h in orders]
    index = node_phase_index(grid)
    n = index.size
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0

    # --- Fundamental voltage (operating point for injection scaling) ---
    if v1 is None:
        # Linear fundamental: network Y + const-Z load shunts + source Norton current
        y1 = _build_numpy_ybus(grid, 1, index)
        node_map_v = {nd.id: nd for nd in grid.nodes}
        for a in grid.appliances:
            if not (
                isinstance(a, (PgmlLoad, PgmlGen)) and getattr(a, "in_service", True)
            ):
                continue
            node_v = node_map_v[a.node]
            u_rated = to_float(node_v.u_rated_v)
            phases_a = a.phases
            n_ph = len(phases_a)
            src_rows_v = index.rows(a.node)
            sign = 1.0 if isinstance(a, PgmlLoad) else -1.0
            p_total = to_float(a.p_nom_w)
            q_total = to_float(a.q_nom_var)
            p_ph = p_total / n_ph
            q_ph = q_total / n_ph
            v0 = u_rated / math.sqrt(3.0) if n_ph >= 3 else u_rated
            y_elem = (sign * p_ph - 1j * sign * q_ph) / (v0**2)
            for r in src_rows_v:
                y1[r, r] += y_elem
        i1 = np.zeros(n, dtype=complex)
        for a in grid.appliances:
            if not (isinstance(a, Source) and getattr(a, "in_service", True)):
                continue
            phases_a = a.phases
            p = len(phases_a)
            src_rows_v = index.rows(a.node)
            r_mat = np.array(
                [
                    [to_float(a.resistance_ohm[i_][j_]) for j_ in range(p)]
                    for i_ in range(p)
                ]
            )
            l_mat = np.array(
                [
                    [to_float(a.inductance_h[i_][j_]) for j_ in range(p)]
                    for i_ in range(p)
                ]
            )
            z_mat = r_mat + 1j * w0 * l_mat
            ys_mat = np.linalg.inv(z_mat)
            u_ref = np.array([to_float(a.u_ref_v[k]) for k in range(p)])
            u_ang = np.array(
                [to_float(a.u_angle_deg[k]) * math.pi / 180.0 for k in range(p)]
            )
            v_th = u_ref * np.exp(1j * u_ang)
            i_s = ys_mat @ v_th
            for k in range(p):
                i1[src_rows_v[k]] += i_s[k]
        v1_eff = np.linalg.solve(y1, i1)
    else:
        v1_arr = np.asarray(v1).reshape(-1)
        if v1_arr.shape[0] != n:
            raise ValueError(
                f"v1 has {v1_arr.shape[0]} entries but grid has N={n} rows"
            )
        v1_eff = v1_arr.astype(complex)

    # --- Per-device fundamental current + spectrum (for h > 1 injection) ---
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, (PgmlLoad, PgmlGen)) and getattr(a, "in_service", True)):
            continue
        # Resolve spectrum from override or stored spectrum
        if harmonic_injection is not None and a.id in harmonic_injection:
            spec: dict = {
                int(o): (float(mag), float(ang))
                for o, (mag, ang) in harmonic_injection[a.id].items()
            }
        else:
            s = getattr(a, "spectrum", None)
            if not isinstance(s, StaticSpectrum):
                continue
            spec = {
                c.order: (to_float(c.magnitude_pu), to_float(c.phase_deg))
                for c in s.spectrum.components
            }
        if not spec:
            continue
        phases_a = a.phases
        n_ph = len(phases_a)
        src_rows = index.rows(a.node)
        sign = 1.0 if isinstance(a, PgmlLoad) else -1.0
        p_total = to_float(a.p_nom_w)
        q_total = to_float(a.q_nom_var)
        # Per-phase S0: honor an explicit per-phase nameplate (asymmetric loads), else
        # split the total equally — mirroring pgml's resolve_operating_power so the
        # per-phase fundamental current I1 (and thus the harmonic injection) matches.
        p_pp = (
            [to_float(x) for x in a.p_nom_per_phase_w]
            if getattr(a, "p_nom_per_phase_w", None) is not None
            else [p_total / n_ph] * n_ph
        )
        q_pp = (
            [to_float(x) for x in a.q_nom_per_phase_var]
            if getattr(a, "q_nom_per_phase_var", None) is not None
            else [q_total / n_ph] * n_ph
        )
        # Fundamental current per phase: I1 = conj(S0_ph) / conj(V_term)
        i1_list = []
        for k_ph, row in enumerate(src_rows):
            v_term = v1_eff[row]
            s0_ph = complex(sign * p_pp[k_ph], sign * q_pp[k_ph])
            if abs(v_term) < 1e-300:
                i1_list.append(0.0 + 0j)
            else:
                i1_list.append(np.conj(s0_ph) / np.conj(v_term))
        devs.append((spec, src_rows, i1_list))

    # --- Per-order solve ---
    result_slices: list[np.ndarray] = []
    for h in orders_list:
        if h == 1:
            result_slices.append(v1_eff)
            continue
        y_h = _build_numpy_ybus(grid, h, index)
        i_h = np.zeros(n, dtype=complex)
        for spec, src_rows, i1_list in devs:
            mag1, ang1 = spec.get(1, (1.0, 0.0))
            mag_h, ang_h = spec.get(h, (0.0, 0.0))
            if mag1 == 0.0:
                continue
            ratio = mag_h / mag1
            for i1_val, row in zip(i1_list, src_rows):
                i_drawn = (
                    ratio
                    * abs(i1_val)
                    * cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (cmath.phase(i1_val) - math.radians(ang1))
                        )
                    )
                )
                i_h[row] += -i_drawn  # nodal injection (drawn = negative source)

        # --- Per-node harmonic disturbance sources ---
        if node_sources:
            _apply_node_sources_numpy(grid, node_sources, v1_eff, index, h, y_h, i_h)

        result_slices.append(np.linalg.solve(y_h, i_h))

    return np.stack(result_slices, axis=0)  # [H, N]


def _apply_node_sources_numpy(
    grid: Grid,
    node_sources: Sequence,
    v1: np.ndarray,
    index,
    h: int,
    y_h: np.ndarray,
    i_h: np.ndarray,
) -> None:
    """Stamp per-node harmonic disturbance sources in-place for order ``h``.

    Implements the physics from ``references/error_injection.md``:

    - ``V_base`` = node ``u_rated_v`` (L-N for ≥3-phase, else L-L).
    - ``Y_s = S_sc / V_base^2`` (real, frequency-flat).
    - ``E_h = (mag_h/mag_1) * |V1_row| * exp(j*(ang_h + h*(arg(V1_row) - ang_1)))``.
    - ``I_N = E_h * Y_s``.
    - ``kind="voltage"``: add ``Y_s`` to ``y_h[row, row]``; add ``I_N`` to ``i_h[row]``.
    - ``kind="current"``: add ``I_N`` to ``i_h[row]`` only.

    Modifies ``y_h`` and ``i_h`` in-place (numpy arrays).
    """
    node_map = {int(nd.id): nd for nd in grid.nodes}
    from pgml.assembly._params import phase_voltage_magnitude

    for ns in node_sources:
        nid = int(ns.node_id)
        node = node_map[nid]
        n_phases_node = len(node.phases)
        v_base = phase_voltage_magnitude(float(node.u_rated_v), n_phases_node)
        y_s = float(ns.source_power_va) / (v_base * v_base)  # real admittance S

        spec = {
            int(o): (float(mag), float(ang)) for o, (mag, ang) in ns.spectrum.items()
        }
        mag1, ang1 = spec.get(1, (1.0, 0.0))
        ang1_rad = math.radians(ang1)
        if mag1 == 0.0:
            continue

        mag_h, ang_h = spec.get(h, (0.0, 0.0))
        if mag_h == 0.0:
            continue
        ang_h_rad = math.radians(ang_h)
        ratio = mag_h / mag1

        phases = ns.phases if ns.phases is not None else list(node.phases)
        for phase in phases:
            row = index.row(nid, phase)
            v1_row = v1[row]
            e_h = (
                ratio
                * abs(v1_row)
                * cmath.exp(1j * (ang_h_rad + h * (cmath.phase(v1_row) - ang1_rad)))
            )
            i_n = e_h * y_s

            i_h[row] += i_n
            if ns.kind == "voltage":
                y_h[row, row] += y_s


# ---------------------------------------------------------------------------
# Live OpenDSS harmonic oracle (Carson geometry lines + pgml-exact non-line stamps)
# ---------------------------------------------------------------------------
def _is_geometry_grid(grid: Grid) -> bool:
    """True if ALL in-service lines carry a conductor_geometry (full Carson path)."""
    lines = [b for b in grid.branches if isinstance(b, Line) and b.in_service]
    if not lines:
        return False
    return all(getattr(ln, "conductor_geometry", None) is not None for ln in lines)


def _is_sequence_aware_grid(grid: Grid) -> bool:
    """True if any in-service line is tagged ``harmonic_line_model=sequence_aware``."""
    return any(
        isinstance(b, Line)
        and b.in_service
        and (b.tags or {}).get("harmonic_line_model") == "sequence_aware"
        for b in grid.branches
    )


def _emit_geometry_line_to_dss(dss, ln: Line, busname: dict) -> None:
    """Emit WireData + LineGeometry + Line commands for one conductor-geometry line.

    Handles both single-conductor (1-phase) and multi-conductor (3-phase) geometries.
    Each conductor gets its own ``WireData`` element (indexed by ``<line_id>_c<k>``);
    the ``LineGeometry`` lists all conductors with their (x, h) positions.  The ``Line``
    element references the geometry and uses bus-phase suffixes matching the line's
    ``from_phases`` order (phase A → ``.1``, B → ``.2``, C → ``.3``).

    OpenDSS conductor order for a ``LineGeometry`` with ``nconds=N nphases=N`` maps
    ``cond=k`` to DSS phase ``.k`` — this matches pgml's ``_geom_conductor_arrays``
    which orders conductors as ``from_phases[0], from_phases[1], …``.
    """
    geo = ln.conductor_geometry
    n_cond = len(geo.conductors)
    rho = float(geo.earth_resistivity_ohm_m)

    # Order conductors to match pgml's _geom_conductor_arrays: phases first (by
    # from_phases order), then neutrals.  For the synthesized equilateral 3-conductor
    # geometry all conductors are non-neutral phases in (A, B, C) order.
    phase_list = list(ln.from_phases)
    _phase_to_dss_suffix = {Phase.A: 1, Phase.B: 2, Phase.C: 3, Phase.N: 4}
    phase_conds = []
    for ph in phase_list:
        c = next(
            (c for c in geo.conductors if not c.is_neutral and c.phase == ph), None
        )
        if c is not None:
            phase_conds.append(c)
    neutral_conds = [c for c in geo.conductors if c.is_neutral]
    ordered_conds = phase_conds + neutral_conds

    # Emit one WireData per conductor
    for k, c in enumerate(ordered_conds):
        wd = f"wd{ln.id}_c{k + 1}"
        dss.Text.Command(
            f"New WireData.{wd} Rdc={to_float(c.r_dc_ohm_per_m)} "
            f"GMRac={to_float(c.gmr_m)} radius={to_float(c.radius_m)} "
            "Runits=m GMRunits=m radunits=m"
        )

    # Build LineGeometry: each cond line provides wire name + (x, h) coordinates.
    # OpenDSS requires specifying cond parameters in sequential cond= blocks; the
    # first cond= block also sets nconds/nphases, subsequent blocks update the active
    # conductor.  All cond blocks must be on separate commands or semicolon-separated.
    n_phase = len(phase_list)  # number of non-neutral (phase) conductors
    gn = f"geo{ln.id}"
    geo_cmd = (
        f"New LineGeometry.{gn} nconds={n_cond} nphases={n_phase} "
        f"cond=1 wire=wd{ln.id}_c1 x={to_float(ordered_conds[0].x_m)} "
        f"h={to_float(ordered_conds[0].y_m)} units=m"
    )
    dss.Text.Command(geo_cmd)
    for k in range(1, n_cond):
        c = ordered_conds[k]
        dss.Text.Command(
            f"~ cond={k + 1} wire=wd{ln.id}_c{k + 1} "
            f"x={to_float(c.x_m)} h={to_float(c.y_m)} units=m"
        )

    # Emit Line element.  Bus suffixes follow the phase order in from_phases:
    # cond=1 → from_phases[0], cond=2 → from_phases[1], etc.
    ph_suffix = ".".join(str(_phase_to_dss_suffix.get(ph, 1)) for ph in phase_list)
    dss.Text.Command(
        f"New Line.l{ln.id} phases={n_phase} "
        f"bus1={busname[ln.from_node]}.{ph_suffix} "
        f"bus2={busname[ln.to_node]}.{ph_suffix} "
        f"geometry={gn} length={to_float(ln.length_m)} units=m rho={rho}"
    )


def _build_geometry_circuit_stub(grid: Grid, busname: dict) -> None:
    """Build an OpenDSS circuit with a stub Vsource + all conductor-geometry lines.

    The stub Vsource has near-zero impedance so its Carson-corrected Norton shunt can
    be subtracted at each harmonic and replaced with pgml's exact source Norton.
    Supports both single-phase (1-conductor) and three-phase (3-conductor) geometry
    lines via :func:`_emit_geometry_line_to_dss`.

    For a single-phase grid the Vsource has ``phases=1``; for a three-phase geometry
    grid (3-conductor lines) the Vsource has ``phases=3`` with balanced stub impedance
    so that the resulting ``YNodeOrder`` aligns to pgml rows at all three phases.
    """
    import opendssdirect as dss

    src = next(a for a in grid.appliances if isinstance(a, Source) and a.in_service)
    slack_node = int(src.node)
    node_by_id = {int(n.id): n for n in grid.nodes}
    kv = float(node_by_id[slack_node].u_rated_v) / 1000.0
    f0 = float(grid.base_frequency_hz)
    p_src = len(src.phases)
    ph_conn = ".".join(str(k + 1) for k in range(p_src))

    dss.Text.Command("Clear")
    if p_src == 1:
        dss.Text.Command(
            f"New Circuit.pgml_live basekv={kv} phases=1 "
            f"bus1={busname[slack_node]}.1 pu=1.0 angle=0 frequency={f0} "
            "r1=1e-6 x1=1e-6"
        )
    else:
        dss.Text.Command(
            f"New Circuit.pgml_live phases={p_src} basekv={kv} "
            f"bus1={busname[slack_node]}.{ph_conn} pu=1.0 angle=0.0 frequency={f0} "
            "r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6"
        )
    dss.Text.Command("Set earthmodel=Deri")

    for ln in grid.branches:
        if not (
            isinstance(ln, Line) and ln.in_service and ln.conductor_geometry is not None
        ):
            continue
        _emit_geometry_line_to_dss(dss, ln, busname)

    dss.Text.Command(f"Set voltagebases=[{kv}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")


def _get_stub_norton_from_dss() -> complex:
    """Read the actual stub Vsource Norton admittance from the active DSS circuit.

    Returns ``YPrim[0, 0]`` of ``Vsource.Source`` (the positive-terminal Norton
    contribution at the current frequency, including Carson corrections).
    """
    import opendssdirect as dss

    dss.Circuit.SetActiveElement("Vsource.Source")
    yp = np.array(dss.CktElement.YPrim())
    n = int(round((len(yp) / 2) ** 0.5))
    yy = (yp[0::2] + 1j * yp[1::2]).reshape(n, n)
    return complex(yy[0, 0])


def _build_seq_aware_circuit_stub(grid: Grid, busname: dict) -> None:
    """Build an OpenDSS stub circuit for the sequence-aware 3-phase path.

    Contains ONLY:
    - A near-zero-impedance stub Vsource (so its Carson-corrected Norton can be read
      and subtracted later, exactly as in the geometry path).
    - Lines defined via R1/X1/R0/X0 derived from the 3×3 phase matrices (the
      ``sequence_aware`` lines).  OpenDSS applies its own Carson/Deri corrections at
      harmonics, which is the expected discrepancy vs pgml's ``sequence_aware`` model.
    - Switches modelled as pure-R ``Line`` elements (negligible Carson effect).

    Transformers are intentionally omitted: OpenDSS ``Transformer`` elements create
    neutral/delta neutral buses (``.0`` / ``.123`` in ``YNodeOrder``) that have no
    pgml equivalent, making Y-bus alignment impossible.  Transformer contributions are
    stamped with pgml-exact formulas after reading the OpenDSS ``SystemY``.

    The Vsource bus phases connect to ``bus<n>.1.2.3`` so that OpenDSS sees a
    three-phase source at the slack node; the resulting ``YNodeOrder`` contains only
    ``.1``, ``.2``, ``.3`` suffixed entries that map cleanly to pgml rows.
    """
    import opendssdirect as dss

    src = next(a for a in grid.appliances if isinstance(a, Source) and a.in_service)
    slack_node = int(src.node)
    node_by_id = {int(nd.id): nd for nd in grid.nodes}
    kv_slack = float(node_by_id[slack_node].u_rated_v) / 1000.0
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    p_src = len(src.phases)

    dss.Text.Command("Clear")
    ph_conn = ".".join(str(k + 1) for k in range(p_src))
    dss.Text.Command(
        f"New Circuit.pgml_live phases={p_src} basekv={kv_slack} "
        f"bus1={busname[slack_node]}.{ph_conn} pu=1.0 angle=0.0 frequency={f0} "
        "r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6"
    )

    # Lines: derive R1/X1/R0/X0 per metre from the 3x3 phase matrix
    for ln in grid.branches:
        if not (
            isinstance(ln, Line) and ln.in_service and ln.conductor_geometry is None
        ):
            continue
        ph = ln.from_phases
        p = len(ph)
        length = to_float(ln.length_m)
        ph_suffix = ".".join(str(k + 1) for k in range(p))
        bus1 = f"{busname[ln.from_node]}.{ph_suffix}"
        bus2 = f"{busname[ln.to_node]}.{ph_suffix}"

        if p == 1:
            r1 = to_float(ln.series_resistance_ohm_per_m[0][0]) * length
            l1 = to_float(ln.series_inductance_h_per_m[0][0]) * length
            x1 = w0 * l1
            c1 = (
                to_float(ln.shunt_capacitance_f_per_m[0][0]) * length
                if ln.shunt_capacitance_f_per_m is not None
                else 0.0
            )
            dss.Text.Command(
                f"New Line.l{ln.id} phases=1 bus1={bus1} bus2={bus2} "
                f"r1={r1:.10g} x1={x1:.10g} c1={c1 * 1e9:.10g} "
                f"r0={r1:.10g} x0={x1:.10g} c0={c1 * 1e9:.10g} "
                "length=1 units=m"
            )
        elif p == 3:
            # Extract Z1/Z0 from the phase matrix via the balanced symmetry approximation
            r_mat = (
                np.array(
                    [
                        [
                            to_float(ln.series_resistance_ohm_per_m[i][j])
                            for j in range(p)
                        ]
                        for i in range(p)
                    ]
                )
                * length
            )
            l_mat = (
                np.array(
                    [
                        [to_float(ln.series_inductance_h_per_m[i][j]) for j in range(p)]
                        for i in range(p)
                    ]
                )
                * length
            )
            c_mat = (
                np.array(
                    [
                        [to_float(ln.shunt_capacitance_f_per_m[i][j]) for j in range(p)]
                        for i in range(p)
                    ]
                )
                * length
                if ln.shunt_capacitance_f_per_m is not None
                else np.zeros((3, 3))
            )
            # Z1 = zs - zm, Z0 = zs + 2*zm (symmetric matrix approximation)
            z_f0 = r_mat + 1j * w0 * l_mat
            diag = np.diag(z_f0)
            zs = diag.mean()
            zm = (z_f0.sum() - diag.sum()) / 6.0
            z1 = zs - zm
            z0 = zs + 2.0 * zm
            r1 = z1.real
            x1 = z1.imag
            r0 = z0.real
            x0 = z0.imag
            c1 = np.diag(c_mat).mean()
            c0 = c_mat.sum() / 3.0
            dss.Text.Command(
                f"New Line.l{ln.id} phases=3 bus1={bus1} bus2={bus2} "
                f"r1={r1:.10g} x1={x1:.10g} c1={c1 * 1e9:.10g} "
                f"r0={r0:.10g} x0={x0:.10g} c0={c0 * 1e9:.10g} "
                "length=1 units=m"
            )
        else:
            # For other phase counts fall back to Rmatrix/Xmatrix
            r_mat = (
                np.array(
                    [
                        [
                            to_float(ln.series_resistance_ohm_per_m[i][j])
                            for j in range(p)
                        ]
                        for i in range(p)
                    ]
                )
                * length
            )
            l_mat = (
                np.array(
                    [
                        [to_float(ln.series_inductance_h_per_m[i][j]) for j in range(p)]
                        for i in range(p)
                    ]
                )
                * length
            )
            r_str = " | ".join(
                " ".join(f"{r_mat[i][j]:.10g}" for j in range(i + 1)) for i in range(p)
            )
            x_str = " | ".join(
                " ".join(f"{w0 * l_mat[i][j]:.10g}" for j in range(i + 1))
                for i in range(p)
            )
            dss.Text.Command(
                f"New Line.l{ln.id} phases={p} bus1={bus1} bus2={bus2} "
                f"Rmatrix=[{r_str}] Xmatrix=[{x_str}] Cmatrix=[{'0 | '.join(['0'] * p)}] "
                "length=1 units=m"
            )

    # Switches: model as pure-R Lines (x1=0 => no Carson correction, safe to include)
    for b in grid.branches:
        if not (isinstance(b, Switch) and b.in_service and b.closed):
            continue
        p = len(b.from_phases)
        ph_suffix = ".".join(str(k + 1) for k in range(p))
        bus1 = f"{busname[b.from_node]}.{ph_suffix}"
        bus2 = f"{busname[b.to_node]}.{ph_suffix}"
        r_sw = to_float(b.resistance_ohm)
        dss.Text.Command(
            f"New Line.sw{b.id} phases={p} bus1={bus1} bus2={bus2} "
            f"r1={r_sw:.10g} x1=0.0 c1=0.0 r0={r_sw:.10g} x0=0.0 c0=0.0 "
            "length=1 units=m"
        )

    # Voltage bases: use only the slack node kV (LV nodes are isolated without transformers)
    dss.Text.Command(f"Set voltagebases=[{kv_slack:.6g}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")


def opendss_harmonic_voltages(
    grid: Grid,
    harmonic_injection: Optional[dict],
    orders: Sequence[int],
    *,
    slack: str = "norton",
    v1: Optional[np.ndarray] = None,
    operating_point: Optional[dict] = None,
    node_sources: Optional[Sequence] = None,
) -> np.ndarray:
    """Live OpenDSS harmonic oracle for full multi-voltage-level grids.

    Returns complex node voltages ``[len(orders), N]`` aligned to
    :func:`pgml.assembly.node_phase_index` rows.

    This is the **genuine live-OpenDSS comparison**: OpenDSS's own ``SystemY(h)``
    is read at each harmonic order, harmonic injection is applied using pgml's
    converged fundamental operating point, and the system is solved.

    **Single-phase geometry path** (all lines carry ``conductor_geometry``):

    A stub-source OpenDSS circuit is built with the same conductor-geometry data
    used by pgml (``WireData`` / ``LineGeometry`` / ``Line`` commands).  At each
    harmonic the stub Vsource Norton shunt (which carries OpenDSS's own Carson
    correction) is read from ``Vsource.Source.YPrim``, subtracted from the
    ``SystemY``, and replaced with pgml's exact ``Y_s = (R + j·h·2πf₀·L)⁻¹``.
    Switch and Transformer stamps are then added using pgml's exact formulas
    (``R const, X∝h, complex tap``).  The resulting Y-bus matches pgml's
    assembled Y at Carson-line precision (~1e-14 relative for lines).

    **Three-phase sequence-aware path** (lines tagged ``harmonic_line_model=sequence_aware``):

    A full OpenDSS circuit is built with the source, Lines defined via
    ``R1/X1/R0/X0`` derived from the 3×3 phase matrices, Switches as R-only
    Lines, and Transformers with ``XRConst=No`` (OpenDSS default: R const, X∝h,
    matching pgml's transformer model).  OpenDSS applies its own Carson/DERI
    correction to all ``R1/X1``-defined lines at harmonics; this correction differs
    from pgml's ``sequence_aware`` earth-return resistance term by the Carson model
    difference (~5–15 % at harmonics 5–11).  The residual is documented by the
    parity tests in ``tests/reference/test_cigre_lv_live_opendss.py``.

    **Harmonic injection convention** (identical in both paths):

    Per ``references/opendss/harmonics.md``:
    ``|I_h| = (mag_h / mag_1) · |I₁|``,
    ``arg(I_h) = ang_h + h · (arg(I₁) − ang_1)``
    where ``I₁ = conj(S₀) / conj(V₁)`` at the device terminal.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.  Must be one of:
        (a) a single-phase grid with ``conductor_geometry`` on all lines, or
        (b) a three-phase grid with ``harmonic_line_model=sequence_aware`` tags.
        Grids with neither path raise ``ValueError``.
    harmonic_injection:
        Per-device harmonic-current spec
        ``{appliance_id: {order: (magnitude_pu, phase_deg)}}``.
        ``None`` means no injection.
    orders:
        Harmonic orders to solve (e.g. ``[1, 5, 11]``).  Order 1 returns ``v1``
        directly when ``v1`` is given.
    slack:
        Only ``"norton"`` is implemented (source held at zero harmonic EMF).
    v1:
        Optional pre-computed fundamental voltage vector ``[N]`` (complex numpy).
        Pass ``hres.pf.v.detach().cpu().numpy()`` to share pgml's converged
        operating point.  When ``None``, a linear const-Z fundamental is solved
        internally.
    operating_point:
        Optional per-device operating-point override.  Currently used only when
        ``v1 is None`` (passed to the internal linear fundamental solve).
        Accepted for API compatibility with the scenario example scripts.

    node_sources:
        Optional sequence of :class:`~pgml.solver.NodeHarmonicSource` — per-node
        harmonic disturbance sources applied ONLY at ``h > 1``.  These are stamped
        directly into the Y-bus / injection vector (after the OpenDSS SystemY stub-
        subtract and pgml-exact element stamps) using the same physics as
        :func:`numpy_harmonic_voltages`.  No OpenDSS ``ISource`` / ``VSource``
        elements are added; the stamping is done in Python so that parity with pgml
        is exact (same formulas, same ``V1``).  When ``None`` (default), the oracle
        is backward-compatible with the existing signature.

    Returns
    -------
    numpy.ndarray
        Complex ``[H, N]`` — ``H = len(orders)``, ``N = index.size``.

    Raises
    ------
    ValueError
        If ``slack != "norton"`` or the grid type is not recognised.

    See Also
    --------
    numpy_harmonic_voltages : Pure-numpy regression oracle (machine-precision parity).
    opendss_geometry_harmonic_profiles : Carson-line profile oracle (passive feeder).
    numpy_harmonic_profiles : Single-phase numpy oracle (lines + source only).
    """
    import opendssdirect as dss

    if slack != "norton":
        raise ValueError(
            f"opendss_harmonic_voltages supports only slack='norton'; got {slack!r}"
        )

    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Generator as PgmlGen, Load as PgmlLoad

    orders_list = [int(h) for h in orders]
    index = node_phase_index(grid)
    n = index.size
    f0 = float(grid.base_frequency_hz)
    busname = {int(nd.id): f"bus{int(nd.id)}" for nd in grid.nodes}

    use_geometry = _is_geometry_grid(grid)
    use_seq_aware = _is_sequence_aware_grid(grid)

    if not use_geometry and not use_seq_aware:
        raise ValueError(
            "opendss_harmonic_voltages requires either (a) all lines to carry "
            "conductor_geometry (single-phase Carson path) or (b) lines tagged "
            "harmonic_line_model=sequence_aware (3-phase sequence-aware path). "
            "For plain R/X grids without these tags, use numpy_harmonic_voltages."
        )

    # --- Build OpenDSS stub circuit (lines + stub source; NO transformers) ---
    if use_geometry:
        _build_geometry_circuit_stub(grid, busname)
    else:
        _build_seq_aware_circuit_stub(grid, busname)

    node_order_dss = list(dss.Circuit.YNodeOrder())
    n_dss = len(node_order_dss)

    def _dss_node_id(entry: str) -> int:
        """Extract node id from OpenDSS node name ``BUS<id>.<phase>``."""
        return int(entry.upper().split(".")[0][3:])

    def _dss_phase_index(entry: str) -> int:
        """Extract 0-based phase index from DSS suffix (.1 -> 0, .2 -> 1, ...)."""
        parts = entry.split(".")
        if len(parts) < 2:
            return 0
        try:
            return int(parts[1]) - 1
        except ValueError:
            return 0

    # Build rowmap: DSS row -> pgml row
    phases_list = [Phase.A, Phase.B, Phase.C, Phase.N]
    rowmap_dss_to_pgml: dict[int, int] = {}
    for di, entry in enumerate(node_order_dss):
        nid = _dss_node_id(entry)
        ph_idx = _dss_phase_index(entry)
        phase = phases_list[ph_idx] if ph_idx < len(phases_list) else Phase.A
        try:
            pgml_row = index.row(nid, phase)
        except (KeyError, ValueError):
            pgml_row = index.row(nid, Phase.A)
        rowmap_dss_to_pgml[di] = pgml_row

    # Locate the stub source's DSS rows (one per phase) for the Norton subtraction
    src = next(a for a in grid.appliances if isinstance(a, Source) and a.in_service)
    slack_node = int(src.node)
    slack_dss_rows = [
        di for di, e in enumerate(node_order_dss) if _dss_node_id(e) == slack_node
    ]

    # --- Determine fundamental voltage ---
    if v1 is None:
        v1_eff = numpy_harmonic_voltages(
            grid,
            None,  # no injection at fundamental
            [1],
            slack="norton",
            v1=None,
            operating_point=operating_point,
        )[0]
    else:
        v1_arr = np.asarray(v1).reshape(-1)
        if v1_arr.shape[0] != n:
            raise ValueError(
                f"v1 has {v1_arr.shape[0]} entries but grid has N={n} rows"
            )
        v1_eff = v1_arr.astype(complex)

    # --- Per-device injection setup ---
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, (PgmlLoad, PgmlGen)) and getattr(a, "in_service", True)):
            continue
        if harmonic_injection is not None and a.id in harmonic_injection:
            spec: dict = {
                int(o): (float(mag), float(ang))
                for o, (mag, ang) in harmonic_injection[a.id].items()
            }
        else:
            s = getattr(a, "spectrum", None)
            if not isinstance(s, StaticSpectrum):
                continue
            spec = {
                c.order: (to_float(c.magnitude_pu), to_float(c.phase_deg))
                for c in s.spectrum.components
            }
        if not spec:
            continue
        n_ph = len(a.phases)
        src_rows = index.rows(a.node)
        sign = 1.0 if isinstance(a, PgmlLoad) else -1.0
        # Per-phase S0: honor an explicit per-phase nameplate (asymmetric loads), else
        # split the total equally — mirroring pgml's resolve_operating_power so the
        # per-phase fundamental current I1 (hence the harmonic injection) matches.
        p_total, q_total = to_float(a.p_nom_w), to_float(a.q_nom_var)
        p_pp = (
            [to_float(x) for x in a.p_nom_per_phase_w]
            if getattr(a, "p_nom_per_phase_w", None) is not None
            else [p_total / n_ph] * n_ph
        )
        q_pp = (
            [to_float(x) for x in a.q_nom_per_phase_var]
            if getattr(a, "q_nom_per_phase_var", None) is not None
            else [q_total / n_ph] * n_ph
        )
        i1_list = []
        for k_ph, row in enumerate(src_rows):
            v_term = v1_eff[row]
            s0_ph = complex(sign * p_pp[k_ph], sign * q_pp[k_ph])
            if abs(v_term) < 1e-300:
                i1_list.append(0.0 + 0j)
            else:
                i1_list.append(np.conj(s0_ph) / np.conj(v_term))
        devs.append((spec, src_rows, i1_list))

    def _build_injection(h: int) -> np.ndarray:
        i_h = np.zeros(n, dtype=complex)
        for spec, src_rows, i1_list in devs:
            mag1, ang1 = spec.get(1, (1.0, 0.0))
            mag_h, ang_h = spec.get(h, (0.0, 0.0))
            if mag1 == 0.0:
                continue
            ratio = mag_h / mag1
            for i1_val, row in zip(i1_list, src_rows):
                i_drawn = (
                    ratio
                    * abs(i1_val)
                    * cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (cmath.phase(i1_val) - math.radians(ang1))
                        )
                    )
                )
                i_h[row] += -i_drawn
        return i_h

    def _build_harmonic_ybus(h: int) -> np.ndarray:
        """Build full harmonic Y-bus for order h using OpenDSS lines + pgml-exact stamps.

        For both the geometry and sequence-aware paths:
        1. Set OpenDSS frequency, rebuild Y, read ``SystemY`` (contains stub source
           Norton + lines ± switches).
        2. Read actual stub Norton from ``Vsource.Source.YPrim`` and subtract it from
           the diagonal block at the slack node.
        3. Stamp all non-line elements (switches, transformers, source Norton) with
           pgml-exact formulas.

        Since switches are already in the OpenDSS SystemY (as pure-R Line elements with
        no Carson correction at x1=0) AND are also stamped by
        ``_stamp_non_line_elements_no_source``, we must avoid double-counting.
        The function uses the following invariant:
        - For the geometry path: the stub circuit has NO switches and NO transformers,
          so ``_stamp_non_line_elements_no_source`` + ``_stamp_source_nortons`` add
          exactly the missing elements.
        - For the sequence-aware path: the stub circuit includes switches, so we stamp
          ONLY transformers and source Norton (not switches again).
        """
        dss.Text.Command(f"set frequency={h * f0}")
        dss.Solution.BuildYMatrix(2, 1)

        # Read stub Norton from YPrim (full p×p block)
        dss.Circuit.SetActiveElement("Vsource.Source")
        yp_flat = np.array(dss.CktElement.YPrim())
        p_src = len(src.phases)
        yp = (yp_flat[0::2] + 1j * yp_flat[1::2]).reshape(2 * p_src, 2 * p_src)
        # top-left p×p block = Norton admittance stamped at from-bus
        y_stub_block = yp[:p_src, :p_src]

        y_flat = np.array(dss.Circuit.SystemY(), dtype=np.float64)
        y_dss = (y_flat[0::2] + 1j * y_flat[1::2]).reshape(n_dss, n_dss)
        y_out = np.zeros((n, n), dtype=complex)
        for di in range(n_dss):
            for dj in range(n_dss):
                y_out[rowmap_dss_to_pgml[di], rowmap_dss_to_pgml[dj]] = y_dss[di, dj]

        # Subtract stub Norton from the slack block
        for pi in range(p_src):
            for pj in range(p_src):
                ri = rowmap_dss_to_pgml[slack_dss_rows[pi]]
                rj = rowmap_dss_to_pgml[slack_dss_rows[pj]]
                y_out[ri, rj] -= y_stub_block[pi, pj]

        # Add pgml-exact non-line stamps:
        # - geometry stub: no switches, no transformers -> stamp all
        # - seq-aware stub: switches already in Y -> stamp transformers + source only
        if use_geometry:
            _stamp_non_line_elements_no_source(y_out, grid, h, index)
        else:
            _stamp_transformers_only(y_out, grid, h, index)
        _stamp_source_nortons(y_out, grid, h, index)

        return y_out

    # --- Per-order solve ---
    result_slices: list[np.ndarray] = []
    for h in orders_list:
        if h == 1:
            result_slices.append(v1_eff)
            continue

        y_h = _build_harmonic_ybus(h)
        i_h = _build_injection(h)

        # Stamp per-node harmonic disturbance sources (same physics as pgml solver).
        # These are added AFTER the OpenDSS SystemY stub-subtract and pgml-exact
        # element stamps, so the only difference vs the numpy oracle is the line model.
        if node_sources:
            _apply_node_sources_numpy(grid, node_sources, v1_eff, index, h, y_h, i_h)

        result_slices.append(np.linalg.solve(y_h, i_h))

    return np.stack(result_slices, axis=0)  # [H, N]


def _stamp_non_line_elements_no_source(
    y: np.ndarray,
    grid: Grid,
    h: int,
    index,
) -> np.ndarray:
    """Stamp Switch and Transformer elements only (no Source Norton).

    Transformer stamps use the winding-incidence model (see
    :func:`_stamp_transformer_numpy`); Switch stamps are diagonal per-phase
    series RL (pgml-exact, no Carson correction at ``x1=0``).
    """
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0

    for b in grid.branches:
        if isinstance(b, Switch) and getattr(b, "in_service", True) and b.closed:
            p = len(b.from_phases)
            fr_rows = index.rows(b.from_node)
            to_rows = index.rows(b.to_node)
            r_sw = to_float(b.resistance_ohm)
            l_sw = to_float(b.inductance_h)
            z_sw = r_sw + 1j * h * w0 * l_sw
            ys_sw = 1.0 / z_sw
            for i_ph in range(p):
                fr = fr_rows[i_ph]
                to = to_rows[i_ph]
                y[fr, fr] += ys_sw
                y[to, to] += ys_sw
                y[fr, to] -= ys_sw
                y[to, fr] -= ys_sw

        elif isinstance(b, Transformer) and getattr(b, "in_service", True):
            p = len(b.from_phases)
            fr_rows = index.rows(b.from_node)
            to_rows = index.rows(b.to_node)
            _stamp_transformer_numpy(y, b, h, w0, fr_rows, to_rows, p)

    return y


def _stamp_transformers_only(
    y: np.ndarray,
    grid: Grid,
    h: int,
    index,
) -> np.ndarray:
    """Stamp Transformer admittances only (winding-incidence vector-group model).

    Used by the sequence-aware path where switches are already in the OpenDSS SystemY.
    Uses :func:`_stamp_transformer_numpy` for the new winding-incidence primitive;
    for P == 1 the result is the classical off-nominal-tap pi (with the full
    nominal ratio from ``u_rated_from_v / u_rated_to_v``); for P == 3 it is the
    ``Nᵀ·Y_winding·N`` block that correctly blocks zero-sequence in a delta winding.
    """
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0

    for b in grid.branches:
        if not (isinstance(b, Transformer) and getattr(b, "in_service", True)):
            continue
        p = len(b.from_phases)
        fr_rows = index.rows(b.from_node)
        to_rows = index.rows(b.to_node)
        _stamp_transformer_numpy(y, b, h, w0, fr_rows, to_rows, p)

    return y


def _stamp_source_nortons(
    y: np.ndarray,
    grid: Grid,
    h: int,
    index,
) -> np.ndarray:
    """Stamp all Source Norton shunts (pgml-exact formulas)."""
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0

    for a in grid.appliances:
        if not (isinstance(a, Source) and getattr(a, "in_service", True)):
            continue
        phases_a = a.phases
        p = len(phases_a)
        src_rows = index.rows(a.node)
        r_mat = np.array(
            [[to_float(a.resistance_ohm[i][j]) for j in range(p)] for i in range(p)]
        )
        l_mat = np.array(
            [[to_float(a.inductance_h[i][j]) for j in range(p)] for i in range(p)]
        )
        z_mat = r_mat + 1j * h * w0 * l_mat
        ys_mat = np.linalg.inv(z_mat)
        for i_ph in range(p):
            for j_ph in range(p):
                y[src_rows[i_ph], src_rows[j_ph]] += ys_mat[i_ph, j_ph]

    return y


# ---------------------------------------------------------------------------
# True live-OpenDSS Dyn transformer oracle
# ---------------------------------------------------------------------------


def _build_circuit_with_real_transformer(grid: Grid, busname: dict) -> None:
    """Build a full OpenDSS circuit including REAL Transformer elements.

    Unlike :func:`_build_seq_aware_circuit_stub`, this circuit contains a
    genuine OpenDSS ``Transformer`` element for each :class:`~pgml.schemas.grid_schema.Transformer`
    in the grid, using the ``delta``/``wye`` connections and ``LeadLag`` setting
    derived from the schema.  OpenDSS's own transformer model is used; ``XRConst=No``
    (the OpenDSS default) matches pgml's ``R const, X∝h`` harmonic model.

    The LV winding is declared with ``Rneut=0 Xneut=0`` (solidly grounded neutral),
    so OpenDSS does NOT add a ``.0`` row to ``YNodeOrder``.  The resulting
    ``YNodeOrder`` contains only ``.1``, ``.2``, ``.3`` suffixed entries that map
    cleanly to pgml (node, phase) rows.

    The stub Vsource still has near-zero impedance (r1=1e-6, x1=1e-6) and is treated
    identically to the seq-aware path: its Carson-corrected Norton is read from
    ``Vsource.Source.YPrim`` and subtracted from ``SystemY`` before the pgml-exact
    source Norton is added back.

    Lines and switches are built the same way as in
    :func:`_build_seq_aware_circuit_stub` (R1/X1/R0/X0 from the phase matrices,
    pure-R for switches).

    Transformer impedance parameters derive from the stored per-unit SI values:
    ``kVA`` is set to 1000 kVA so that ``Z_base_LV = U_to² / kVA``; ``%R`` and
    ``XHL`` are back-calculated from ``series_resistance_ohm`` and
    ``series_inductance_h`` accordingly.  ``LeadLag=Lag`` maps to Dyn1 (LV lags HV
    by 30°, matching ``tap.shift_deg = 30``); ``LeadLag=Lead`` maps to Dyn11.

    Parameters
    ----------
    grid:
        Materialised three-phase :class:`~pgml.schemas.grid_schema.Grid` whose
        lines carry ``harmonic_line_model=sequence_aware`` tags.
    busname:
        ``{node_id: dss_bus_name}`` mapping built by the caller.
    """
    import opendssdirect as dss

    src = next(a for a in grid.appliances if isinstance(a, Source) and a.in_service)
    slack_node = int(src.node)
    node_by_id = {int(nd.id): nd for nd in grid.nodes}
    kv_slack = float(node_by_id[slack_node].u_rated_v) / 1000.0
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    p_src = len(src.phases)

    dss.Text.Command("Clear")
    ph_conn = ".".join(str(k + 1) for k in range(p_src))
    dss.Text.Command(
        f"New Circuit.pgml_dyn phases={p_src} basekv={kv_slack} "
        f"bus1={busname[slack_node]}.{ph_conn} pu=1.0 angle=0.0 frequency={f0} "
        "r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6"
    )

    # Lines: R1/X1/R0/X0 from the 3×3 phase matrices (same as seq-aware path).
    for ln in grid.branches:
        if not (
            isinstance(ln, Line) and ln.in_service and ln.conductor_geometry is None
        ):
            continue
        p = len(ln.from_phases)
        ph_suffix = ".".join(str(k + 1) for k in range(p))
        bus1 = f"{busname[ln.from_node]}.{ph_suffix}"
        bus2 = f"{busname[ln.to_node]}.{ph_suffix}"
        length = to_float(ln.length_m)

        if p == 3:
            r_mat = (
                np.array(
                    [
                        [
                            to_float(ln.series_resistance_ohm_per_m[i][j])
                            for j in range(p)
                        ]
                        for i in range(p)
                    ]
                )
                * length
            )
            l_mat = (
                np.array(
                    [
                        [to_float(ln.series_inductance_h_per_m[i][j]) for j in range(p)]
                        for i in range(p)
                    ]
                )
                * length
            )
            z_f0 = r_mat + 1j * w0 * l_mat
            diag = np.diag(z_f0)
            zs = diag.mean()
            zm = (z_f0.sum() - diag.sum()) / 6.0
            z1 = zs - zm
            z0 = zs + 2.0 * zm
            r1, x1 = z1.real, z1.imag
            r0, x0 = z0.real, z0.imag
            dss.Text.Command(
                f"New Line.l{ln.id} phases=3 bus1={bus1} bus2={bus2} "
                f"r1={r1:.10g} x1={x1:.10g} c1=0 "
                f"r0={r0:.10g} x0={x0:.10g} c0=0 "
                "length=1 units=m"
            )
        else:
            r1 = to_float(ln.series_resistance_ohm_per_m[0][0]) * length
            l1 = to_float(ln.series_inductance_h_per_m[0][0]) * length
            x1 = w0 * l1
            dss.Text.Command(
                f"New Line.l{ln.id} phases=1 bus1={bus1} bus2={bus2} "
                f"r1={r1:.10g} x1={x1:.10g} c1=0 "
                f"r0={r1:.10g} x0={x1:.10g} c0=0 "
                "length=1 units=m"
            )

    # Switches: pure-R Lines (no Carson correction at X=0).
    for b in grid.branches:
        if not (isinstance(b, Switch) and b.in_service and b.closed):
            continue
        p = len(b.from_phases)
        ph_suffix = ".".join(str(k + 1) for k in range(p))
        bus1 = f"{busname[b.from_node]}.{ph_suffix}"
        bus2 = f"{busname[b.to_node]}.{ph_suffix}"
        r_sw = to_float(b.resistance_ohm)
        dss.Text.Command(
            f"New Line.sw{b.id} phases={p} bus1={bus1} bus2={bus2} "
            f"r1={r_sw:.10g} x1=0.0 c1=0.0 r0={r_sw:.10g} x0=0.0 c0=0.0 "
            "length=1 units=m"
        )

    # Transformers: REAL OpenDSS Transformer elements.
    # Back-calculate %R per winding and XHL from the stored LV-referred R (Ω) and L (H).
    # Reference kVA = 1000 kVA chosen to keep %R and XHL in a numerically comfortable
    # range; any consistent kVA works because the per-unit values are kVA-independent.
    kva_ref = 1000.0
    lv_voltage_bases_kv: set = set()
    for t in grid.branches:
        if not (isinstance(t, Transformer) and t.in_service):
            continue
        p = len(t.from_phases)
        ph_str = ".".join(str(k + 1) for k in range(p))
        u_from_kv = float(t.u_rated_from_v) / 1000.0
        u_to_kv = float(t.u_rated_to_v) / 1000.0
        lv_voltage_bases_kv.add(round(u_to_kv, 6))

        r_t = to_float(t.series_resistance_ohm)
        l_t = to_float(t.series_inductance_h)
        x_t = w0 * l_t
        # Z_base_LV = U_to² / kVA (LV LL voltage, single-phase reference base).
        z_base_lv = (u_to_kv**2 * 1e6) / (kva_ref * 1e3)
        # Total leakage in % of base; split equally between the two windings.
        vkr_total = (r_t / z_base_lv) * 100.0
        vk_total = (abs(r_t + 1j * x_t) / z_base_lv) * 100.0
        xhl = math.sqrt(max(vk_total**2 - vkr_total**2, 0.0))
        pct_r_per_winding = vkr_total / 2.0

        # Connection strings and LeadLag from the schema.
        from_conn_str = "delta" if str(t.from_connection).endswith("DELTA") else "wye"
        to_conn_str = "delta" if str(t.to_connection).endswith("DELTA") else "wye"
        # shift_deg > 0 → LV lags HV (Dyn1) → LeadLag=Lag.
        # shift_deg < 0 or 330 → LV leads HV (Dyn11) → LeadLag=Lead.
        shift = float(t.tap.shift_deg)
        lead_lag = "Lag" if (0.0 < shift < 180.0) else "Lead"

        bus_from = f"{busname[t.from_node]}.{ph_str}"
        bus_to = f"{busname[t.to_node]}.{ph_str}"

        dss.Text.Command(f"New Transformer.T{t.id} windings=2")
        dss.Text.Command(
            f"~ wdg=1 bus={bus_from} conn={from_conn_str} kV={u_from_kv:.8g} "
            f"kVA={kva_ref:.6g} %R={pct_r_per_winding:.10g}"
        )
        dss.Text.Command(
            f"~ wdg=2 bus={bus_to} conn={to_conn_str} kV={u_to_kv:.8g} "
            f"kVA={kva_ref:.6g} %R={pct_r_per_winding:.10g} Rneut=0 Xneut=0"
        )
        # XRConst=No (default): R const, X∝h — matches pgml's harmonic model.
        dss.Text.Command(f"~ XHL={xhl:.10g} XRConst=No LeadLag={lead_lag}")

    # Voltage bases: slack (HV) + all LV buses.
    vbases_str = ", ".join(
        [f"{kv_slack:.6g}"] + [f"{kv:.6g}" for kv in sorted(lv_voltage_bases_kv)]
    )
    dss.Text.Command(f"Set voltagebases=[{vbases_str}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")


def opendss_dyn_transformer_harmonic_voltages(
    grid: Grid,
    harmonic_injection: Optional[dict],
    orders: Sequence[int],
    *,
    slack: str = "norton",
    v1: Optional[np.ndarray] = None,
    operating_point: Optional[dict] = None,
) -> np.ndarray:
    """Live OpenDSS harmonic oracle using REAL OpenDSS Transformer elements.

    This is the genuine vector-group validation oracle: OpenDSS's own
    ``Transformer`` element (with ``conn=delta``/``conn=wye``, ``LeadLag=Lag``
    for Dyn1, ``XRConst=No``) is included in the OpenDSS circuit alongside the
    lines and switches.  OpenDSS applies its own transformer model (R const,
    X∝h when ``XRConst=No``) plus the correct delta/wye incidence for zero-
    sequence blocking.

    Because OpenDSS's real transformer is in the circuit, the stub Norton
    subtract-and-replace technique from :func:`opendss_harmonic_voltages` is
    applied ONLY to the source Norton (no transformer formula correction is
    needed — OpenDSS models the transformer natively).  The transformer
    contribution is read directly from ``SystemY`` at each harmonic.

    **Key validation property:** a delta winding in OpenDSS (and in pgml's new
    assembly) blocks zero-sequence current, so triplen harmonic orders (h=3, 9,
    …) injected on the LV wye side do NOT propagate through the HV delta
    winding to the MV bus.  Both pgml and this oracle should agree tightly on
    these orders.

    **Residual discrepancy:** non-triplen orders (h=5, 11) will show the same
    ~1e-8 V discrepancy as the existing seq-aware path (OpenDSS applies its own
    Carson/Deri correction to the R1/X1 lines, which differs from pgml's
    ``sequence_aware`` earth-return model).  This is the expected and documented
    line-model gap.

    Parameters
    ----------
    grid:
        Materialised three-phase :class:`~pgml.schemas.grid_schema.Grid`.
        Lines must carry ``harmonic_line_model=sequence_aware`` tags (the same
        prerequisite as :func:`opendss_harmonic_voltages`).
    harmonic_injection:
        Per-device harmonic-current spec
        ``{appliance_id: {order: (magnitude_pu, phase_deg)}}``.
    orders:
        Harmonic orders to solve (e.g. ``[1, 3, 5, 9, 11]``).  Order 1 returns
        ``v1`` directly.
    slack:
        Only ``"norton"`` is implemented.
    v1:
        Optional pre-computed fundamental voltage vector ``[N]`` (complex numpy).
    operating_point:
        Optional per-device operating-point override; used only when ``v1 is None``.

    Returns
    -------
    numpy.ndarray
        Complex ``[H, N]`` — ``H = len(orders)``, ``N = index.size``.

    Raises
    ------
    ValueError
        If ``slack != "norton"`` or the grid is not three-phase seq-aware.

    See Also
    --------
    opendss_harmonic_voltages : Live oracle with pgml-stamped transformers.
    numpy_harmonic_voltages : Pure-numpy regression oracle (machine-precision parity).
    """
    import opendssdirect as dss

    if slack != "norton":
        raise ValueError(
            f"opendss_dyn_transformer_harmonic_voltages supports only slack='norton';"
            f" got {slack!r}"
        )
    if not _is_sequence_aware_grid(grid):
        raise ValueError(
            "opendss_dyn_transformer_harmonic_voltages requires lines tagged "
            "harmonic_line_model=sequence_aware (three-phase seq-aware path)."
        )

    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Generator as PgmlGen, Load as PgmlLoad

    orders_list = [int(h) for h in orders]
    index = node_phase_index(grid)
    n = index.size
    f0 = float(grid.base_frequency_hz)
    busname = {int(nd.id): f"bus{int(nd.id)}" for nd in grid.nodes}

    # --- Build OpenDSS circuit with real transformers ---
    _build_circuit_with_real_transformer(grid, busname)

    node_order_dss = list(dss.Circuit.YNodeOrder())
    n_dss = len(node_order_dss)

    def _dss_node_id(entry: str) -> int:
        return int(entry.upper().split(".")[0][3:])

    def _dss_phase_index(entry: str) -> int:
        parts = entry.split(".")
        if len(parts) < 2:
            return 0
        try:
            return int(parts[1]) - 1
        except ValueError:
            return 0

    phases_list = [Phase.A, Phase.B, Phase.C, Phase.N]
    rowmap_dss_to_pgml: dict[int, int] = {}
    for di, entry in enumerate(node_order_dss):
        nid = _dss_node_id(entry)
        ph_idx = _dss_phase_index(entry)
        phase = phases_list[ph_idx] if ph_idx < len(phases_list) else Phase.A
        try:
            pgml_row = index.row(nid, phase)
        except (KeyError, ValueError):
            pgml_row = index.row(nid, Phase.A)
        rowmap_dss_to_pgml[di] = pgml_row

    src = next(a for a in grid.appliances if isinstance(a, Source) and a.in_service)
    slack_node = int(src.node)
    slack_dss_rows = [
        di for di, e in enumerate(node_order_dss) if _dss_node_id(e) == slack_node
    ]
    p_src = len(src.phases)

    # --- Fundamental voltage ---
    if v1 is None:
        v1_eff = numpy_harmonic_voltages(
            grid,
            None,
            [1],
            slack="norton",
            v1=None,
            operating_point=operating_point,
        )[0]
    else:
        v1_arr = np.asarray(v1).reshape(-1)
        if v1_arr.shape[0] != n:
            raise ValueError(
                f"v1 has {v1_arr.shape[0]} entries but grid has N={n} rows"
            )
        v1_eff = v1_arr.astype(complex)

    # --- Per-device injection setup ---
    devs = []
    for a in grid.appliances:
        if not (isinstance(a, (PgmlLoad, PgmlGen)) and getattr(a, "in_service", True)):
            continue
        if harmonic_injection is not None and a.id in harmonic_injection:
            spec: dict = {
                int(o): (float(mag), float(ang))
                for o, (mag, ang) in harmonic_injection[a.id].items()
            }
        else:
            s = getattr(a, "spectrum", None)
            if not isinstance(s, StaticSpectrum):
                continue
            spec = {
                c.order: (to_float(c.magnitude_pu), to_float(c.phase_deg))
                for c in s.spectrum.components
            }
        if not spec:
            continue
        n_ph = len(a.phases)
        src_rows = index.rows(a.node)
        sign = 1.0 if isinstance(a, PgmlLoad) else -1.0
        p_total, q_total = to_float(a.p_nom_w), to_float(a.q_nom_var)
        p_pp = (
            [to_float(x) for x in a.p_nom_per_phase_w]
            if getattr(a, "p_nom_per_phase_w", None) is not None
            else [p_total / n_ph] * n_ph
        )
        q_pp = (
            [to_float(x) for x in a.q_nom_per_phase_var]
            if getattr(a, "q_nom_per_phase_var", None) is not None
            else [q_total / n_ph] * n_ph
        )
        i1_list = []
        for k_ph, row in enumerate(src_rows):
            v_term = v1_eff[row]
            s0_ph = complex(sign * p_pp[k_ph], sign * q_pp[k_ph])
            if abs(v_term) < 1e-300:
                i1_list.append(0.0 + 0j)
            else:
                i1_list.append(np.conj(s0_ph) / np.conj(v_term))
        devs.append((spec, src_rows, i1_list))

    def _build_injection(h: int) -> np.ndarray:
        i_h = np.zeros(n, dtype=complex)
        for spec, src_rows, i1_list in devs:
            mag1, ang1 = spec.get(1, (1.0, 0.0))
            mag_h, ang_h = spec.get(h, (0.0, 0.0))
            if mag1 == 0.0:
                continue
            ratio = mag_h / mag1
            for i1_val, row in zip(i1_list, src_rows):
                i_drawn = (
                    ratio
                    * abs(i1_val)
                    * cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (cmath.phase(i1_val) - math.radians(ang1))
                        )
                    )
                )
                i_h[row] += -i_drawn
        return i_h

    def _build_harmonic_ybus_with_real_trafo(h: int) -> np.ndarray:
        """Read OpenDSS SystemY (with real transformer) and replace stub Norton."""
        dss.Text.Command(f"set frequency={h * f0}")
        dss.Solution.BuildYMatrix(2, 1)

        # Read stub Norton from YPrim.
        dss.Circuit.SetActiveElement("Vsource.Source")
        yp_flat = np.array(dss.CktElement.YPrim())
        yp = (yp_flat[0::2] + 1j * yp_flat[1::2]).reshape(2 * p_src, 2 * p_src)
        y_stub_block = yp[:p_src, :p_src]

        y_flat = np.array(dss.Circuit.SystemY(), dtype=np.float64)
        y_dss = (y_flat[0::2] + 1j * y_flat[1::2]).reshape(n_dss, n_dss)
        y_out = np.zeros((n, n), dtype=complex)
        for di in range(n_dss):
            for dj in range(n_dss):
                y_out[rowmap_dss_to_pgml[di], rowmap_dss_to_pgml[dj]] = y_dss[di, dj]

        # Subtract stub Norton and add pgml-exact source Norton.
        for pi in range(p_src):
            for pj in range(p_src):
                ri = rowmap_dss_to_pgml[slack_dss_rows[pi]]
                rj = rowmap_dss_to_pgml[slack_dss_rows[pj]]
                y_out[ri, rj] -= y_stub_block[pi, pj]
        _stamp_source_nortons(y_out, grid, h, index)

        return y_out

    # --- Per-order solve ---
    result_slices: list[np.ndarray] = []
    for h in orders_list:
        if h == 1:
            result_slices.append(v1_eff)
            continue
        y_h = _build_harmonic_ybus_with_real_trafo(h)
        i_h = _build_injection(h)
        result_slices.append(np.linalg.solve(y_h, i_h))

    return np.stack(result_slices, axis=0)  # [H, N]


__all__ = [
    "ieee33_geometry_grid",
    "cigre_lv_geometry_grid",
    "cigre_lv_full_grid",
    "pandapower_ybus",
    "pandapower_voltage_profile",
    "dss_systemy",
    "align_dss_systemy",
    "opendss_ybus",
    "build_opendss_geometry_circuit",
    "opendss_geometry_systemy",
    "opendss_geometry_harmonic_profiles",
    "numpy_harmonic_profiles",
    "numpy_harmonic_voltages",
    "opendss_harmonic_voltages",
    "opendss_dyn_transformer_harmonic_voltages",
]
