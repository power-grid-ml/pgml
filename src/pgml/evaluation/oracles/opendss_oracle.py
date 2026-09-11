"""Live OpenDSS oracle adapters for harmonic admittance and voltage comparison.

This module provides reference implementations backed by a live OpenDSS process
(via ``opendssdirect``) and pure-numpy helpers that consume the OpenDSS-derived
admittance matrices.

**Geometry path** (single-phase or three-phase grids with ``conductor_geometry``):
Build an OpenDSS circuit from pgml's synthesized conductor data, read ``SystemY(h)``
at each harmonic, and solve with pgml-consistent injection.  Parity: ~1e-11 V.

**Sequence-aware path** (three-phase grids with ``harmonic_line_model='sequence_aware'``):
Build an OpenDSS circuit with R1/X1/R0/X0 lines (from the 3×3 phase matrices);
transformer contributions are overwritten with pgml's own formulas.  Parity: ~1e-8 V.

**Dynamic transformer path** (:func:`opendss_dyn_transformer_harmonic_voltages`):
Uses a real OpenDSS ``Transformer`` element (with ``conn=delta``/``conn=wye``,
``LeadLag``) to validate the vector-group model.  Parity vs pgml's transformer:
tight on triplen orders (zero-sequence blocked); ~1e-8 V residual on non-triplen
orders (line-model Carson gap).

``opendssdirect`` is imported lazily (inside functions) so that this module can be
imported without the package installed.  ``numpy`` is imported at module level.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Phase,
    Source,
    Switch,
    Transformer,
    WindingConnection,
)

from pgml.evaluation._util import to_float
from pgml.evaluation.data import HarmonicProfile, LabeledMatrix, row_labels
from pgml.evaluation.topology import distance_from_slack
from pgml.evaluation.oracles.numpy_oracle import (
    _apply_node_sources_numpy,
    _stamp_transformer_numpy,
    numpy_harmonic_voltages,
)


# ---------------------------------------------------------------------------
# Low-level SystemY extraction helpers
# ---------------------------------------------------------------------------


def dss_systemy() -> tuple[np.ndarray, list[str]]:
    """Extract ``(SystemY [N,N] complex, YNodeOrder)`` from the active OpenDSS circuit."""
    import opendssdirect as dss

    node_order = list(dss.Circuit.YNodeOrder())
    n = len(node_order)
    flat = np.array(dss.Circuit.SystemY(), dtype=np.float64)
    y = (flat[0::2] + 1j * flat[1::2]).reshape(n, n)
    return y, node_order


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


def opendss_ybus(
    y_dss, node_order, id_map, index, *, label: str = "OpenDSS"
) -> LabeledMatrix:
    """Wrap an extracted OpenDSS SystemY (aligned to our rows) as a LabeledMatrix."""
    aligned = align_dss_systemy(y_dss, node_order, id_map, index)
    return LabeledMatrix(matrix=aligned, label=label, row_labels=row_labels(index))


# ---------------------------------------------------------------------------
# Geometry circuit builder (passive single/multi-phase Carson feeder)
# ---------------------------------------------------------------------------


def build_opendss_geometry_circuit(grid, *, slack_node: Optional[int] = None) -> dict:
    """Build a PASSIVE single-phase OpenDSS circuit from a pgml geometry ``grid``.

    Emits the slack Vsource (Thévenin from the :class:`Source`) and one WireData +
    LineGeometry + Line per geometry line, using the SAME synthesized conductor data
    pgml uses — so OpenDSS's ``SystemY(h)`` equals pgml's harmonic ``Y(h)`` up to the
    Carson model (which agrees to ~5e-8 relative, the SI-vs-truncated ``mu0`` constant).
    No loads (harmonic injection is applied
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
# Private stub-circuit builders for the live harmonic oracle
# ---------------------------------------------------------------------------


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


def _dss_earth_params(
    line: Line, f0: float, length_m: float
) -> tuple[float, float, float]:
    """OpenDSS ``(Rg, Xg, rho)`` reproducing ONE pgml line's earth-return model.

    OpenDSS frequency-corrects a sequence-defined line as ``R += Rg*(h-1)`` and
    ``X = h*(X - 0.5*KXg*ln(h))`` per matrix entry, with
    ``KXg = Xg/ln(658.5*sqrt(rho/f0))`` — i.e. exactly pgml's lumped zero-sequence
    model with ``Rg = resistance_coeff*f0`` and ``KXg = reactance_coeff*f0``. Its own
    defaults (``Rg=0.01805``, ``Xg=0.155081``) are the physical Carson values at 60 Hz
    in ohms per 1000 ft and are reinterpreted in the line's ``units``, so they are
    passed explicitly here.

    The stub writes the TOTAL impedance onto a ``length=1 units=m`` line, so the earth
    terms are likewise per whole line. ``Xg=0`` is emitted when the pgml line scales
    ``X0`` linearly (``x0_frequency='linear'``), which is OpenDSS's way of switching the
    earth-return reactance correction off.
    """
    from pgml import defaults as _d

    er = getattr(line, "earth_return", None)
    rc = getattr(er, "resistance_coeff_ohm_per_m_per_hz", None)
    if rc is None:
        rc = _d.get("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
    kx = getattr(er, "reactance_coeff_ohm_per_m_per_hz", None)
    if kx is None:
        kx = _d.get("line.earth_return.reactance_coeff_ohm_per_m_per_hz")
    law = getattr(er, "x0_frequency", None) or _d.get("line.earth_return.x0_frequency")
    rho = float(_d.get("line.earth_return.resistivity_ohm_m"))
    rg = to_float(rc) * f0 * length_m
    if law == "carson_sublinear":
        xg = to_float(kx) * f0 * math.log(658.5 * math.sqrt(rho / f0)) * length_m
    else:
        xg = 0.0
    return rg, xg, rho


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
    stamped with pgml-exact formulas after reading the OpenDSS ``SystemY`` — so on this
    path the transformer is NOT validated against OpenDSS (only the lines are).  The
    genuine OpenDSS transformer oracle is
    :func:`opendss_dyn_transformer_harmonic_voltages`.

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
    # Every element's BASE frequency comes from DefaultBaseFrequency (60 Hz out of the
    # box), and OpenDSS scales a sequence-defined line's reactance by f/basefreq. On a
    # 50 Hz grid an unset base frequency therefore reports 5/6 of the line reactance the
    # grid actually stores, at every order including the fundamental.
    dss.Text.Command(f"Set DefaultBaseFrequency={f0:.10g}")
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
            rg, xg, rho = _dss_earth_params(ln, f0, length)
            dss.Text.Command(
                f"New Line.l{ln.id} phases=3 bus1={bus1} bus2={bus2} "
                f"r1={r1:.10g} x1={x1:.10g} c1={c1 * 1e9:.10g} "
                f"r0={r0:.10g} x0={x0:.10g} c0={c0 * 1e9:.10g} "
                f"rg={rg:.10g} xg={xg:.10g} rho={rho:.10g} "
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
        # rg/xg = 0: a switch is a lumped contact resistance with no earth-return path,
        # so OpenDSS must not frequency-correct it (its defaults would add ~0.018 Ohm
        # per order on a `units=m length=1` element).
        dss.Text.Command(
            f"New Line.sw{b.id} phases={p} bus1={bus1} bus2={bus2} "
            f"r1={r_sw:.10g} x1=0.0 c1=0.0 r0={r_sw:.10g} x0=0.0 c0=0.0 "
            "rg=0 xg=0 length=1 units=m"
        )

    # Voltage bases: use only the slack node kV (LV nodes are isolated without transformers)
    dss.Text.Command(f"Set voltagebases=[{kv_slack:.6g}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")


# ---------------------------------------------------------------------------
# Private numpy stamps (used by the live oracle to overwrite DSS elements)
# ---------------------------------------------------------------------------


def _stamp_non_line_elements_no_source(
    y: np.ndarray,
    grid: Grid,
    h: int,
    index,
) -> np.ndarray:
    """Stamp Switch and Transformer elements only (no Source Norton).

    Transformer stamps use the winding-incidence model (see
    :func:`~pgml.evaluation.oracles.numpy_oracle._stamp_transformer_numpy`);
    Switch stamps are diagonal per-phase series RL (pgml-exact, no Carson correction
    at ``x1=0``).
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
    Uses :func:`~pgml.evaluation.oracles.numpy_oracle._stamp_transformer_numpy` for
    the winding-incidence primitive; for P == 1 the result is the classical
    off-nominal-tap pi (with the full nominal ratio from ``u_rated_from_v /
    u_rated_to_v``); for P == 3 it is the ``Nᵀ·Y_winding·N`` block that correctly
    blocks zero-sequence in a delta winding.
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
# Path-detection helpers
# ---------------------------------------------------------------------------


def _is_geometry_grid(grid: Grid) -> bool:
    """True if ALL in-service lines carry a conductor_geometry (full Carson path)."""
    lines = [b for b in grid.branches if isinstance(b, Line) and b.in_service]
    if not lines:
        return False
    return all(getattr(ln, "conductor_geometry", None) is not None for ln in lines)


def _is_sequence_aware_grid(grid: Grid) -> bool:
    """True if any in-service line selected ``harmonic_line_model='sequence_aware'``."""
    return any(
        isinstance(b, Line)
        and b.in_service
        and b.harmonic_line_model == "sequence_aware"
        for b in grid.branches
    )


# ---------------------------------------------------------------------------
# Main live OpenDSS harmonic oracle
# ---------------------------------------------------------------------------


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

    **Three-phase sequence-aware path** (lines with ``harmonic_line_model='sequence_aware'``):

    In this path OpenDSS supplies ONLY the LINE admittance.  A full OpenDSS
    circuit is built whose Lines are defined via ``R1/X1/R0/X0`` derived from the
    3×3 phase matrices; OpenDSS applies its own Carson/DERI correction to these
    ``R1/X1``-defined lines at harmonics.  The source Norton is read from
    ``SystemY`` and the transformer and switch stamps are then OVERWRITTEN with
    pgml's OWN formulas — the transformer with the SAME winding-incidence
    vector-group primitive the solver uses (``_stamp_transformer_numpy``:
    ``Nᵀ·Y_winding·N``, so a delta winding blocks the zero sequence), the switch
    with ``R const``.  Because the transformer is stamped identically on both
    sides, it cancels from the pgml-vs-OpenDSS comparison and is therefore NOT
    independently validated against OpenDSS on this path — only the lines are.
    For a genuine OpenDSS transformer oracle (real OpenDSS ``Transformer`` element
    with the correct delta/wye vector group and zero-sequence blocking), use
    :func:`opendss_dyn_transformer_harmonic_voltages`.

    The line-model residual — OpenDSS's Carson correction differs from pgml's
    ``sequence_aware`` earth-return resistance term by the Carson model difference
    (~5–15 % at harmonics 5–11) — is documented by the parity tests in
    ``tests/reference/test_cigre_lv_live_opendss.py``.

    **Harmonic injection convention** (identical in both paths):

    Per ``docs/pgml/modeling/references/opendss/harmonics.md``:
    ``|I_h| = (mag_h / mag_1) · |I₁|``,
    ``arg(I_h) = ang_h + h · (arg(I₁) − ang_1)``
    where ``I₁ = conj(S₀) / conj(V₁)`` at the device terminal.

    Parameters
    ----------
    grid:
        Materialised :class:`~pgml.schemas.grid_schema.Grid`.  Must be one of:
        (a) a single-phase grid with ``conductor_geometry`` on all lines, or
        (b) a three-phase grid with ``harmonic_line_model='sequence_aware'`` lines.
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
        :func:`~pgml.evaluation.oracles.numpy_oracle.numpy_harmonic_voltages`.
        No OpenDSS ``ISource`` / ``VSource`` elements are added; the stamping is
        done in Python so that parity with pgml is exact (same formulas, same ``V1``).
        When ``None`` (default), the oracle is backward-compatible with the existing
        signature.

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
    opendss_dyn_transformer_harmonic_voltages : Genuine OpenDSS transformer oracle
        (real OpenDSS ``Transformer`` element / vector group).  Use this when the
        transformer itself must be validated against OpenDSS — the seq-aware path
        here stamps the transformer with pgml's own formula instead.
    numpy_harmonic_voltages : Pure-numpy regression oracle (machine-precision parity).
    opendss_geometry_harmonic_profiles : Carson-line profile oracle (passive feeder).
    numpy_harmonic_profiles : Single-phase numpy oracle (lines + source only).
    """
    import opendssdirect as dss

    from pgml.schemas.grid_schema import Generator as PgmlGen, Load as PgmlLoad
    from pgml.schemas.grid_schema import StaticSpectrum

    if slack != "norton":
        raise ValueError(
            f"opendss_harmonic_voltages supports only slack='norton'; got {slack!r}"
        )

    from pgml.assembly import node_phase_index

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
            "harmonic_line_model='sequence_aware' (3-phase sequence-aware path). "
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
        import cmath as _cmath

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
                    * _cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (_cmath.phase(i1_val) - math.radians(ang1))
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


# ---------------------------------------------------------------------------
# True live-OpenDSS Dyn transformer oracle
# ---------------------------------------------------------------------------


def _dss_rotated_phase_suffix(p: int, r: int) -> str:
    """DSS bus-conductor suffix ``"k1.k2..."`` for a cyclic rotation ``r`` of phases.

    ``r=0`` gives the identity ``"1.2.3"``; ``r=1`` gives ``"2.3.1"``; ``r=2``
    gives ``"3.1.2"`` -- i.e. winding terminal ``k`` (0-based) connects to bus
    conductor ``((k + r) % p) + 1``. This is the exact bus string pgml's own
    :func:`~pgml.convert.opendss.converter._parse_transformer_winding_bus`
    parses back into a rotation of the same sign (round-trip verified by
    ``tests/reference/test_opendss_transformer.py``).
    """
    return ".".join(str(((k + r) % p) + 1) for k in range(p))


def _dss_leadlag_and_rotation(clock: int, shifting_pairing: bool) -> tuple[str, int]:
    """``(LeadLag, r_to)`` reproducing ``clock`` via a LeadLag + TO-side rotation.

    OpenDSS's ``Transformer`` element has no explicit clock parameter. The
    only two mechanisms available are the binary ``LeadLag`` toggle (``Lag``
    -> clock 1 baseline, ``Lead`` -> clock 11 baseline -- meaningful only for
    a Dy/Yd pairing; a matching Yy/Dd pairing baselines at clock 0
    regardless) and a cyclic rotation of one winding's bus-conductor order
    (+-4 clock steps per step, verified against a live OpenDSS solve -- see
    :mod:`pgml.convert.opendss.converter`'s ``_cyclic_rotation_steps`` and its
    CONTEXT.md). This function always rotates the TO/LV side only
    (``r_from=0``) and searches the reachable baseline(s) for one whose
    residual to ``clock`` is a multiple of 4 clock steps -- true for every
    clock of the pairing's correct parity except {2, 6, 10} (the
    polarity-flip clocks, which need a genuinely reversed winding
    construction no bus wiring can express).

    Raises
    ------
    NotImplementedError
        If ``clock`` needs a reversed winding polarity (clocks 2, 6, 10) --
        not expressible by any OpenDSS ``Transformer`` element.
    """
    clock = clock % 12
    bases = (1, 11) if shifting_pairing else (0,)
    leadlags = ("Lag", "Lead") if shifting_pairing else ("Lag",)
    for base, leadlag in zip(bases, leadlags):
        residual = (base - clock) % 12
        if residual % 4 == 0:
            return leadlag, (residual // 4) % 3
    raise NotImplementedError(
        f"OpenDSS cannot express transformer clock {clock}: it has no "
        "explicit clock/polarity parameter, only LeadLag (+-30 deg, Dy/Yd "
        "only) and a bus-connection rotation (+-120 deg per step); clocks "
        "{2, 6, 10} need a reversed winding polarity, a construction "
        "parameter no OpenDSS Transformer element wiring can express."
    )


def _build_circuit_with_real_transformer(grid: Grid, busname: dict) -> None:
    """Build a full OpenDSS circuit including REAL Transformer elements.

    Unlike :func:`_build_seq_aware_circuit_stub`, this circuit contains a
    genuine OpenDSS ``Transformer`` element for each
    :class:`~pgml.schemas.grid_schema.Transformer` in the grid, using the
    ``delta``/``wye`` connections, ``LeadLag`` setting, and (for clocks beyond
    {0, 1, 11}) a rotated TO-side bus connection derived from the schema.
    OpenDSS's own transformer model is used; ``XRConst=No`` (the OpenDSS default)
    matches pgml's ``R const, X∝h`` harmonic model.

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
    ``kVA`` is set to 1000 kVA so that ``Z_base_LV = U_to² / kVA``. pgml stores
    the leakage referred to the ACTUAL TO-side coil, which is 3x the standard
    line-to-line base for a delta LV winding (see
    ``pgml.convert.opendss.converter`` and
    ``docs/pgml/modeling/references/opendss/index.md``); the back-calculation
    divides by that same factor before recovering ``%R``/``XHL`` on OpenDSS's
    standard (base-invariant) percent convention. The clock is realised via
    :func:`_dss_leadlag_and_rotation` -- ``LeadLag`` alone for clocks 0, 1, 11
    and a rotated TO-side bus connection (e.g. ``bus=lv.3.1.2.0``) for every
    other clock of the pairing's correct parity; clocks 2, 6, 10 raise
    ``NotImplementedError`` (OpenDSS cannot express them).

    Zigzag windings raise ``NotImplementedError``: OpenDSS's ``Transformer``
    element has no zigzag connection.

    Parameters
    ----------
    grid:
        Materialised three-phase :class:`~pgml.schemas.grid_schema.Grid` whose
        lines carry ``harmonic_line_model='sequence_aware'``.
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
        # rg/xg = 0: a switch is a lumped contact resistance with no earth-return path,
        # so OpenDSS must not frequency-correct it (its defaults would add ~0.018 Ohm
        # per order on a `units=m length=1` element).
        dss.Text.Command(
            f"New Line.sw{b.id} phases={p} bus1={bus1} bus2={bus2} "
            f"r1={r_sw:.10g} x1=0.0 c1=0.0 r0={r_sw:.10g} x0=0.0 c0=0.0 "
            "rg=0 xg=0 length=1 units=m"
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
        if t.from_connection in (
            WindingConnection.ZIGZAG,
            WindingConnection.ZIGZAG_GROUNDED,
        ) or t.to_connection in (
            WindingConnection.ZIGZAG,
            WindingConnection.ZIGZAG_GROUNDED,
        ):
            raise NotImplementedError(
                f"transformer {t.id}: OpenDSS's Transformer element has no "
                "zigzag winding connection; a zigzag winding cannot be "
                "expressed in a live OpenDSS oracle circuit."
            )
        p = len(t.from_phases)
        if p != 3:
            raise NotImplementedError(
                f"transformer {t.id}: _build_circuit_with_real_transformer only "
                f"supports 3-phase windings (got {p}); the clock-realising "
                "bus rotation is specific to the 3-phase A/B/C cyclic group."
            )
        u_from_kv = float(t.u_rated_from_v) / 1000.0
        u_to_kv = float(t.u_rated_to_v) / 1000.0
        lv_voltage_bases_kv.add(round(u_to_kv, 6))

        is_delta_from = t.from_connection == WindingConnection.DELTA
        is_delta_to = t.to_connection == WindingConnection.DELTA

        r_t = to_float(t.series_resistance_ohm)
        l_t = to_float(t.series_inductance_h)
        x_t = w0 * l_t
        # pgml stores the leakage referred to the ACTUAL TO-side coil, 3x the
        # standard line-to-line base for a delta LV winding (see the module
        # CONTEXT.md and pgml.convert.opendss.converter); undo that factor
        # before recovering OpenDSS's %R/XHL, which are on the standard base.
        _lv_coil_factor = 3.0 if is_delta_to else 1.0
        r_ll = r_t / _lv_coil_factor
        x_ll = x_t / _lv_coil_factor
        # Z_base_LV = U_to² / kVA (LV LL voltage, single-phase reference base).
        z_base_lv = (u_to_kv**2 * 1e6) / (kva_ref * 1e3)
        # Total leakage in % of base; split equally between the two windings.
        vkr_total = (r_ll / z_base_lv) * 100.0
        vk_total = (abs(r_ll + 1j * x_ll) / z_base_lv) * 100.0
        xhl = math.sqrt(max(vk_total**2 - vkr_total**2, 0.0))
        pct_r_per_winding = vkr_total / 2.0

        from_conn_str = "delta" if is_delta_from else "wye"
        to_conn_str = "delta" if is_delta_to else "wye"

        # Clock -> (LeadLag, TO-side rotation).
        shift = float(t.tap.shift_deg)
        clock = int(round(shift / 30.0)) % 12
        shifting_pairing = is_delta_from != is_delta_to
        lead_lag, r_to = _dss_leadlag_and_rotation(clock, shifting_pairing)

        bus_from = f"{busname[t.from_node]}.{_dss_rotated_phase_suffix(p, 0)}"
        bus_to = f"{busname[t.to_node]}.{_dss_rotated_phase_suffix(p, r_to)}"

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
        Lines must carry ``harmonic_line_model='sequence_aware'`` (the same
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
    import cmath as _cmath

    import opendssdirect as dss

    from pgml.schemas.grid_schema import Generator as PgmlGen, Load as PgmlLoad
    from pgml.schemas.grid_schema import StaticSpectrum

    if slack != "norton":
        raise ValueError(
            f"opendss_dyn_transformer_harmonic_voltages supports only slack='norton';"
            f" got {slack!r}"
        )
    if not _is_sequence_aware_grid(grid):
        raise ValueError(
            "opendss_dyn_transformer_harmonic_voltages requires lines tagged "
            "harmonic_line_model='sequence_aware' (three-phase seq-aware path)."
        )

    from pgml.assembly import node_phase_index

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
                    * _cmath.exp(
                        1j
                        * (
                            math.radians(ang_h)
                            + h * (_cmath.phase(i1_val) - math.radians(ang1))
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
    "dss_systemy",
    "align_dss_systemy",
    "opendss_ybus",
    "build_opendss_geometry_circuit",
    "opendss_geometry_systemy",
    "opendss_geometry_harmonic_profiles",
    "opendss_harmonic_voltages",
    "opendss_dyn_transformer_harmonic_voltages",
]
