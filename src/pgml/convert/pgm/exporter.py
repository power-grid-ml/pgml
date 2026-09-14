"""Export a :class:`pgml.schemas.Grid` to power-grid-model at the fundamental.

The mapping is the inverse of :func:`pgml.convert.pgm.to_grid` over the
documented shared scope, so the
comparison exercises the same conventions the library's own converter was
pinned against: coil-referred leakage, ``clock`` as the vector-group shift,
the tap-side ratio definition, and the no-load test quantities.

Exporting from the ``Grid`` rather than routing through
``power-grid-model-io``'s pandapower converter keeps one code path for every
grid in the study, including the ones that have no pandapower original, and
keeps a second third-party converter out of the measured deviation.

Deliberate reductions, each flagged on the returned object:

* power-grid-model is a balanced positive-sequence engine for symmetric
  calculations, so a three-phase ``Grid`` exports its positive-sequence equivalent.
* power-grid-model has no ZIP load. ZIP and other model-changing reductions
  raise unless ``allow_approximation=True``; enabled reductions are recorded.
* A ``Source`` with explicit series impedance preserves its positive-sequence
  Thevenin equivalent. An ideal ``Source`` exports with a large finite
  short-circuit power because power-grid-model has no ideal voltage boundary;
  that approximation is recorded on the returned object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pgml.convert._export import (
    UnsupportedGridError,
    detached,
    harmonic_fields,
    has_unbalanced_power,
    phase_totals,
    positive_sequence,
    scalar,
    shunt_is_unbalanced,
    shunt_positive_sequence,
    validate_balanced_grid_phases,
)

from pgml.schemas.grid_schema import (
    GenericBranch,
    Generator,
    Grid,
    Line,
    Load,
    LoadModel,
    Phase,
    ShuntAppliance,
    ShuntReactor,
    Source,
    Storage,
    Switch,
    Transformer,
    WindingConnection,
)

#: short-circuit power standing in for an ideal slack (VA)
IDEAL_SLACK_SK_VA = 1e14

_WINDING = {
    WindingConnection.WYE: 0,
    WindingConnection.WYE_GROUNDED: 1,
    WindingConnection.DELTA: 2,
    WindingConnection.ZIGZAG: 3,
    WindingConnection.ZIGZAG_GROUNDED: 4,
}

_LOAD_TYPE = {
    LoadModel.CONST_POWER: 0,
    LoadModel.CONST_IMPEDANCE: 1,
    LoadModel.CONST_CURRENT: 2,
}


@dataclass
class PgmExport:
    """A power-grid-model ``input_data`` dict plus the id maps for comparison."""

    input_data: dict[str, Any]
    pgm_of_node: dict[int, int]
    pgm_of_branch: dict[int, int]
    pgm_of_appliance: dict[int, int]
    reductions: list[str] = field(default_factory=list)


def from_grid(grid: Grid, *, allow_approximation: bool = False) -> PgmExport:
    """Build balanced power-grid-model input for a fundamental power flow.

    The finite source impedance and ideal-switch impedance are unavoidable
    power-grid-model representations and are always recorded. Model-changing
    reductions such as ZIP-to-constant-power require ``allow_approximation=True``.

    Parameters
    ----------
    grid:
        Source grid. It is read without mutation.
    allow_approximation:
        Permit documented reductions that change the fundamental model. The default
        raises instead.

    Returns
    -------
    PgmExport
        Structured input data, complete id maps, and every applied or out-of-scope
        reduction.

    Raises
    ------
    UnsupportedGridError
        If an element has no PGM representation, or an approximation is required but
        was not enabled.
    """
    from power_grid_model import initialize_array

    validate_balanced_grid_phases(grid)
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    out = PgmExport(
        input_data={}, pgm_of_node={}, pgm_of_branch={}, pgm_of_appliance={}
    )
    next_id = iter(range(1, 10_000_000))

    # -- nodes ------------------------------------------------------------- #
    nodes = initialize_array("input", "node", len(grid.nodes))
    for k, node in enumerate(grid.nodes):
        pid = next(next_id)
        nodes["id"][k] = pid
        nodes["u_rated"][k] = scalar(node.u_rated_v)
        out.pgm_of_node[node.id] = pid
    out.input_data["node"] = nodes

    lines: list[dict] = []
    trafos: list[dict] = []
    shunts: list[dict] = []
    for br in grid.branches:
        _record_branch_scope(out, br, allow_approximation)
        if isinstance(br, Line):
            lines.append(
                _line_row(
                    br,
                    out,
                    w0,
                    next(next_id),
                    allow_approximation=allow_approximation,
                )
            )
        elif isinstance(br, Switch):
            lines.append(_switch_row(br, out, w0, next(next_id)))
        elif isinstance(br, Transformer):
            trafos.append(
                _trafo_row(
                    br,
                    out,
                    w0,
                    next(next_id),
                    allow_approximation=allow_approximation,
                )
            )
        elif isinstance(br, ShuntReactor):
            shunt_id = next(next_id)
            out.pgm_of_branch[br.id] = shunt_id
            capacitance = positive_sequence(br.capacitance_f)
            inductance = (
                None if br.inductance_h is None else positive_sequence(br.inductance_h)
            )
            susceptance = w0 * capacitance
            if inductance is not None and inductance > 0.0:
                susceptance -= 1.0 / (w0 * inductance)
            shunts.append(
                {
                    "id": shunt_id,
                    "node": out.pgm_of_node[br.from_node],
                    "status": int(bool(br.in_service)),
                    "g1": positive_sequence(br.conductance_s),
                    "b1": susceptance,
                }
            )
        elif isinstance(br, GenericBranch):
            raise UnsupportedGridError(
                "generic_branch has no power-grid-model equivalent"
            )
        else:
            raise UnsupportedGridError(f"branch kind {type(br).__name__}")

    sym_loads: list[dict] = []
    sym_gens: list[dict] = []
    sources: list[dict] = []
    for ap in grid.appliances:
        if isinstance(ap, Source):
            _record_appliance_scope(out, ap, allow_approximation)
            sources.append(_source_row(ap, out, grid, next(next_id)))
        elif isinstance(ap, Load):
            _record_appliance_scope(out, ap, allow_approximation)
            sym_loads.append(
                _injection_row(
                    ap,
                    out,
                    next(next_id),
                    sign=+1.0,
                    allow_approximation=allow_approximation,
                )
            )
        elif isinstance(ap, (Generator, Storage)):
            _record_appliance_scope(out, ap, allow_approximation)
            if getattr(ap, "voltage_regulation", None) is not None:
                raise UnsupportedGridError(
                    f"generator {ap.id} regulates its terminal voltage; "
                    "power-grid-model has no PV bus"
                )
            sym_gens.append(
                _injection_row(
                    ap,
                    out,
                    next(next_id),
                    sign=+1.0,
                    allow_approximation=allow_approximation,
                )
            )
        elif isinstance(ap, ShuntAppliance):
            _record_appliance_scope(out, ap, allow_approximation)
            shunt_id = next(next_id)
            out.pgm_of_appliance[ap.id] = shunt_id
            conductance, susceptance = shunt_positive_sequence(ap, w0)
            shunts.append(
                {
                    "id": shunt_id,
                    "node": out.pgm_of_node[ap.node],
                    "status": int(bool(ap.in_service)),
                    "g1": conductance,
                    "b1": susceptance,
                }
            )
        else:
            raise UnsupportedGridError(f"appliance kind {type(ap).__name__}")
    if not sources:
        raise UnsupportedGridError(
            "grid has no Source -- no slack for power-grid-model"
        )

    for key, rows in (
        ("line", lines),
        ("transformer", trafos),
        ("shunt", shunts),
        ("sym_load", sym_loads),
        ("sym_gen", sym_gens),
        ("source", sources),
    ):
        if rows:
            out.input_data[key] = _pack(key, rows)
    return out


def _record_branch_scope(out: PgmExport, branch, allow_approximation: bool) -> None:
    """Record harmonic-only options and reject dropped switch shunts."""
    harmonic = []
    for name in (
        "harmonic_line_model",
        "harmonic_skin_effect",
        "earth_return",
        "frequency_impedance",
        "harmonic_xr_constant",
        "resistance_frequency",
    ):
        value = getattr(branch, name, None)
        if name in getattr(branch, "model_fields_set", set()) and value is not None:
            harmonic.append(name)
    if harmonic:
        out.reductions.append(
            f"branch {branch.id}: harmonic-only fields ignored by fundamental export "
            f"({', '.join(harmonic)})"
        )
    if isinstance(branch, Switch):
        dropped = []
        if scalar(branch.shunt_conductance_s) != 0.0:
            dropped.append("shunt conductance")
        if scalar(branch.shunt_capacitance_f) != 0.0:
            dropped.append("per-end shunt capacitance")
        if dropped and not allow_approximation:
            raise UnsupportedGridError(
                f"switch {branch.id}: power-grid-model link/line export cannot preserve "
                f"{', '.join(dropped)}; pass allow_approximation=True to drop and "
                "record these terms"
            )
        if dropped:
            out.reductions.append(f"switch {branch.id}: dropped {', '.join(dropped)}")


def _record_appliance_scope(
    out: PgmExport, appliance, allow_approximation: bool
) -> None:
    """Validate model-changing reductions and record harmonic-only fields."""
    ignored = harmonic_fields(appliance)
    if ignored:
        out.reductions.append(
            f"appliance {appliance.id}: harmonic-only fields ignored by fundamental "
            f"export ({', '.join(ignored)})"
        )

    approximations = []
    if has_unbalanced_power(appliance):
        approximations.append("unbalanced per-phase P/Q folded to a balanced total")
    if isinstance(appliance, ShuntAppliance) and shunt_is_unbalanced(appliance):
        approximations.append("unbalanced shunt elements averaged to positive sequence")
    if getattr(appliance, "control", None) is not None:
        approximations.append("inverter control replaced by nameplate P/Q")
    if isinstance(appliance, Source):
        u_ref = np.asarray(detached(appliance.u_ref_v), dtype=float).reshape(-1)
        angles = np.asarray(detached(appliance.u_angle_deg), dtype=float).reshape(-1)
        phase_offset = {Phase.A: 0.0, Phase.B: -120.0, Phase.C: 120.0}
        expected = np.asarray(
            [phase_offset.get(phase, 0.0) for phase in appliance.phases], dtype=float
        )
        relative_angle = (angles - angles[0]) - (expected - expected[0])
        if (u_ref.size > 1 and not np.allclose(u_ref, u_ref[0])) or (
            angles.size > 1 and not np.allclose((relative_angle + 180.0) % 360.0, 180.0)
        ):
            approximations.append("unbalanced source voltage reduced to phase A")
    if approximations and not allow_approximation:
        raise UnsupportedGridError(
            f"appliance {appliance.id}: {'; '.join(approximations)}; pass "
            "allow_approximation=True to enable and record this reduction"
        )
    out.reductions.extend(
        f"appliance {appliance.id}: {description}" for description in approximations
    )


def _pack(component: str, rows: list[dict]):
    from power_grid_model import initialize_array

    arr = initialize_array("input", component, len(rows))
    for k, row in enumerate(rows):
        for name, value in row.items():
            arr[name][k] = value
    return arr


def _line_row(
    br: Line,
    out: PgmExport,
    w0: float,
    pid: int,
    *,
    allow_approximation: bool,
) -> dict:
    if br.conductor_geometry is not None:
        raise UnsupportedGridError(
            "line with conductor_geometry (Carson/Deri): no power-grid-model equivalent"
        )
    if br.type_ref is not None and br.series_resistance_ohm_per_m is None:
        raise UnsupportedGridError(f"line {br.id} carries an unresolved type_ref")
    length_m = scalar(br.length_m)
    r1 = positive_sequence(br.series_resistance_ohm_per_m) * length_m
    x1 = w0 * positive_sequence(br.series_inductance_h_per_m) * length_m
    c1 = positive_sequence(br.shunt_capacitance_f_per_m) * length_m
    g1 = positive_sequence(br.shunt_conductance_s_per_m) * length_m
    if g1 != 0.0 and c1 <= 0.0:
        if not allow_approximation:
            raise UnsupportedGridError(
                f"line {br.id}: power-grid-model cannot represent shunt conductance "
                "without capacitance; pass allow_approximation=True to drop it"
            )
        out.reductions.append(
            f"line {br.id}: shunt conductance without capacitance dropped "
            "(power-grid-model stores a loss tangent)"
        )
        tan1 = 0.0
    else:
        tan1 = (g1 / (w0 * c1)) if c1 > 0.0 else 0.0
    status = int(bool(br.in_service))
    out.pgm_of_branch[br.id] = pid
    return {
        "id": pid,
        "from_node": out.pgm_of_node[br.from_node],
        "to_node": out.pgm_of_node[br.to_node],
        "from_status": status,
        "to_status": status,
        "r1": r1,
        "x1": x1,
        "c1": c1,
        "tan1": tan1,
        "i_n": 1e5,
    }


def _switch_row(br: Switch, out: PgmExport, w0: float, pid: int) -> dict:
    """A closed switch exports as a very short line carrying its contact impedance."""
    status = int(bool(br.closed) and bool(br.in_service))
    out.pgm_of_branch[br.id] = pid
    r = scalar(br.resistance_ohm)
    x = w0 * scalar(br.inductance_h)
    if r == 0.0 and x == 0.0:
        # power-grid-model rejects a zero-impedance line; an ideal switch is a
        # numerically negligible series resistance instead.
        r = 1e-9
        out.reductions.append(
            f"switch {br.id}: ideal (zero-impedance) switch exported as 1 nOhm"
        )
    return {
        "id": pid,
        "from_node": out.pgm_of_node[br.from_node],
        "to_node": out.pgm_of_node[br.to_node],
        "from_status": status,
        "to_status": status,
        "r1": r,
        "x1": x,
        "c1": scalar(br.shunt_capacitance_f),
        "tan1": 0.0,
        "i_n": 1e5,
    }


def _trafo_row(
    br: Transformer,
    out: PgmExport,
    w0: float,
    pid: int,
    *,
    allow_approximation: bool,
) -> dict:
    if br.type_ref is not None and br.series_resistance_ohm is None:
        raise UnsupportedGridError(
            f"transformer {br.id} carries an unresolved type_ref"
        )
    from_conn = br.from_connection or WindingConnection.WYE_GROUNDED
    to_conn = br.to_connection or WindingConnection.WYE_GROUNDED
    u1 = scalar(br.u_rated_from_v)
    u2 = scalar(br.u_rated_to_v)
    sn = scalar(br.s_rated_va)
    coil_factor = 3.0 if to_conn == WindingConnection.DELTA else 1.0
    r_ll = scalar(br.series_resistance_ohm) / coil_factor
    x_ll = w0 * scalar(br.series_inductance_h) / coil_factor
    z_base = u2**2 / sn
    uk = math.hypot(r_ll, x_ll) / z_base
    pk = r_ll * sn**2 / u2**2

    # no-load test quantities: pgml stores the magnetizing branch referred to
    # the FROM terminal, power-grid-model expects the to-side values.
    g_from = scalar(br.magnetizing_conductance_s)
    l_m = br.magnetizing_inductance_h
    b_from = 0.0 if l_m is None or scalar(l_m) <= 0.0 else 1.0 / (w0 * scalar(l_m))
    p0 = g_from * u1**2
    i0 = math.hypot(g_from, b_from) * u1**2 / sn
    if i0 > 0.9:
        if not allow_approximation:
            raise UnsupportedGridError(
                f"transformer {br.id}: no-load current {i0:g} pu exceeds the "
                "power-grid-model limit 0.9; pass allow_approximation=True to clip it"
            )
        out.reductions.append(
            f"transformer {br.id}: no-load current clipped from {i0:g} to 0.9 pu"
        )
    if uk < 1e-9:
        out.reductions.append(
            f"transformer {br.id}: ideal leakage represented by uk=1e-9 pu"
        )

    ratio = scalar(br.tap.ratio_magnitude)
    delta_u = (ratio - 1.0) * u1
    clock = int(round(float(br.tap.shift_deg) / 30.0)) % 12
    status = int(bool(br.in_service))
    out.pgm_of_branch[br.id] = pid
    return {
        "id": pid,
        "from_node": out.pgm_of_node[br.from_node],
        "to_node": out.pgm_of_node[br.to_node],
        "from_status": status,
        "to_status": status,
        "u1": u1,
        "u2": u2,
        "sn": sn,
        "uk": max(uk, 1e-9),
        "pk": pk,
        "i0": min(max(i0, 0.0), 0.9),
        "p0": p0,
        "winding_from": _WINDING[from_conn],
        "winding_to": _WINDING[to_conn],
        "clock": clock,
        # tap on the FROM side: ratio_magnitude = (u1 + delta_u) / u1
        "tap_side": 0,
        "tap_pos": 1 if delta_u >= 0.0 else -1,
        "tap_min": -10,
        "tap_max": 10,
        "tap_nom": 0,
        "tap_size": abs(delta_u),
    }


def _source_row(ap: Source, out: PgmExport, grid: Grid, pid: int) -> dict:
    u_ref = np.asarray(detached(ap.u_ref_v), dtype=float).reshape(-1)
    ang = np.asarray(detached(ap.u_angle_deg), dtype=float).reshape(-1)
    node = next(n for n in grid.nodes if n.id == ap.node)
    u_rated_ll = scalar(node.u_rated_v)
    n_phase = len([p for p in ap.phases if p != Phase.N])
    scale = math.sqrt(3.0) if n_phase >= 3 else 1.0
    resistance = positive_sequence(ap.resistance_ohm)
    reactance = (
        float(grid.base_frequency_hz)
        * 2.0
        * math.pi
        * positive_sequence(ap.inductance_h)
    )
    if resistance < 0.0 or reactance < 0.0:
        raise UnsupportedGridError(
            f"source {ap.id}: power-grid-model requires nonnegative positive-sequence "
            "source resistance and reactance"
        )
    impedance = math.hypot(resistance, reactance)
    if impedance == 0.0:
        short_circuit_power = IDEAL_SLACK_SK_VA
        rx_ratio = 0.1
        out.reductions.append(
            f"source {ap.id}: ideal voltage boundary represented by finite "
            f"short-circuit power {IDEAL_SLACK_SK_VA:g} VA"
        )
    else:
        if reactance <= 0.0:
            raise UnsupportedGridError(
                f"source {ap.id}: a purely resistive source cannot be represented by "
                "power-grid-model's positive R/X source parameter"
            )
        short_circuit_power = u_rated_ll**2 / impedance
        rx_ratio = resistance / reactance
    out.pgm_of_appliance[ap.id] = pid
    return {
        "id": pid,
        "node": out.pgm_of_node[ap.node],
        "status": int(bool(ap.in_service)),
        "u_ref": float(u_ref[0]) * scale / u_rated_ll,
        "u_ref_angle": math.radians(float(ang[0])),
        "sk": short_circuit_power,
        "rx_ratio": rx_ratio,
        "z01_ratio": 1.0,
    }


def _injection_row(
    ap,
    out: PgmExport,
    pid: int,
    *,
    sign: float,
    allow_approximation: bool,
) -> dict:
    p, q = phase_totals(ap)
    model = getattr(ap, "load_model", LoadModel.CONST_POWER)
    if model == LoadModel.ZIP:
        if not allow_approximation:
            raise UnsupportedGridError(
                f"appliance {ap.id}: ZIP has no power-grid-model equivalent; pass "
                "allow_approximation=True to export it as constant power"
            )
        out.reductions.append(
            f"appliance {ap.id}: ZIP load reduced to constant power "
            "(power-grid-model has no ZIP model)"
        )
        load_type = 0
    else:
        load_type = _LOAD_TYPE[model]
    out.pgm_of_appliance[ap.id] = pid
    return {
        "id": pid,
        "node": out.pgm_of_node[ap.node],
        "status": int(bool(ap.in_service)),
        "type": load_type,
        "p_specified": sign * p,
        "q_specified": sign * q,
    }


__all__ = ["IDEAL_SLACK_SK_VA", "PgmExport", "UnsupportedGridError", "from_grid"]
