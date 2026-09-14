"""Export a :class:`pgml.schemas.Grid` to pandapower at the fundamental.

This module supports the element kinds a distribution grid commonly uses: nodes, lines with
R/L/C, two-winding transformers, bus-bus switches, shunts, and the
injection appliances (source, load, generator, storage).

The mapping is the inverse of :func:`pgml.convert.pandapower.to_grid` over the
documented shared scope.

Scope and deliberate reductions
-------------------------------
* pandapower's ``runpp`` is a balanced positive-sequence solver, so a
  three-phase ``Grid`` exports its POSITIVE-SEQUENCE equivalent:
  ``Z1 = Z_self - Z_mutual`` on every per-phase matrix (the same reduction
  ``pgml.convert.opendss`` applies for ``SINGLE_PHASE_EQUIV``) and the total
  P/Q of every appliance.  For a grid with diagonal line matrices and
  balanced injections the reduction is exact; for a grid with conductor
  coupling or per-phase data it is a modelling reduction and the export is
  flagged.
* Conductor-geometry lines, unresolved ``type_ref`` references, zigzag
  windings and impedance-grounded neutrals raise :class:`UnsupportedGridError`.
* Unbalanced injections and active inverter controls change the fundamental
  model when reduced. They raise by default and require
  ``allow_approximation=True``. Every enabled reduction is named in the return value.
* Harmonic-only fields are outside this fundamental export and are named in the
  reduction ledger when present.

Units: pgml stores SI with L/C (never X/B); pandapower stores per-km
reactances and nF.  Every conversion factor is spelled out at its use site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pgml.convert._export import (
    UnsupportedGridError,
    check_matrix_shape,
    detached,
    harmonic_fields,
    has_unbalanced_power,
    is_coupled,
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


@dataclass
class PandapowerExport:
    """A pandapower net plus the maps needed to compare results element-wise."""

    net: Any
    bus_of_node: dict[int, int]
    node_of_bus: dict[int, int]
    line_of_branch: dict[int, int]
    trafo_of_branch: dict[int, int]
    switch_of_branch: dict[int, int]
    shunt_of_component: dict[int, int]
    load_of_appliance: dict[int, int]
    sgen_of_appliance: dict[int, int]
    ext_grid_of_appliance: dict[int, int]
    #: notes naming every modelling reduction the export had to apply
    reductions: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
def from_grid(
    grid: Grid,
    *,
    name: str = "pgml_export",
    allow_approximation: bool = False,
) -> PandapowerExport:
    """Build a balanced fundamental-frequency pandapower representation.

    Parameters
    ----------
    grid:
        Source grid. It is read without mutation.
    name:
        Name assigned to the new pandapower network.
    allow_approximation:
        Permit documented reductions that change the fundamental model, including
        unbalanced P/Q, controls, source impedance, and unsupported switch terms.
        The default raises instead.

    Returns
    -------
    PandapowerExport
        A new network, complete id maps, and every applied or out-of-scope reduction.

    Raises
    ------
    UnsupportedGridError
        If an element has no pandapower representation, or an approximation is required
        but was not enabled.
    """
    import pandapower as pp

    validate_balanced_grid_phases(grid)
    f0 = float(grid.base_frequency_hz)
    w0 = 2.0 * math.pi * f0
    net = pp.create_empty_network(name=name, f_hz=f0, sn_mva=1.0)
    out = PandapowerExport(
        net=net,
        bus_of_node={},
        node_of_bus={},
        line_of_branch={},
        trafo_of_branch={},
        switch_of_branch={},
        shunt_of_component={},
        load_of_appliance={},
        sgen_of_appliance={},
        ext_grid_of_appliance={},
    )

    # -- nodes ------------------------------------------------------------- #
    for node in grid.nodes:
        bus = pp.create_bus(
            net,
            vn_kv=scalar(node.u_rated_v) / 1e3,
            name=node.name or f"node{node.id}",
        )
        out.bus_of_node[node.id] = int(bus)
        out.node_of_bus[int(bus)] = node.id

    # -- branches ---------------------------------------------------------- #
    for br in grid.branches:
        _record_branch_scope(out, br, allow_approximation)
        if isinstance(br, Line):
            _export_line(net, out, br, w0)
        elif isinstance(br, Transformer):
            _export_transformer(net, out, br, w0)
        elif isinstance(br, Switch):
            _export_switch(net, out, br)
        elif isinstance(br, ShuntReactor):
            _export_shunt_reactor(net, out, br, w0)
        elif isinstance(br, GenericBranch):
            raise UnsupportedGridError("generic_branch has no pandapower equivalent")
        else:
            raise UnsupportedGridError(f"branch kind {type(br).__name__}")

    # -- appliances -------------------------------------------------------- #
    n_sources = 0
    for ap in grid.appliances:
        if isinstance(ap, Source):
            _record_appliance_scope(out, ap, allow_approximation)
            _export_source(net, out, ap, grid)
            n_sources += 1
        elif isinstance(ap, Load):
            _record_appliance_scope(out, ap, allow_approximation)
            _export_load(net, out, ap)
        elif isinstance(ap, (Generator, Storage)):
            _record_appliance_scope(out, ap, allow_approximation)
            _export_generator(net, out, ap)
        elif isinstance(ap, ShuntAppliance):
            _record_appliance_scope(out, ap, allow_approximation)
            _export_shunt_appliance(net, out, ap, w0)
        else:
            raise UnsupportedGridError(f"appliance kind {type(ap).__name__}")
    if n_sources == 0:
        raise UnsupportedGridError("grid has no Source -- no slack for pandapower")
    return out


def _record_appliance_scope(
    out: PandapowerExport, appliance, allow_approximation: bool
) -> None:
    """Validate fundamental reductions and record ignored harmonic data."""
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
    if getattr(appliance, "voltage_regulation", None) is not None:
        approximations.append("voltage regulation replaced by nameplate P/Q")
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
            approximations.append("unbalanced source reduced to phase A")
        source_impedance = abs(positive_sequence(appliance.resistance_ohm)) + abs(
            positive_sequence(appliance.inductance_h)
        )
        if source_impedance != 0.0:
            approximations.append("source impedance omitted from the ideal ext_grid")
    if approximations and not allow_approximation:
        raise UnsupportedGridError(
            f"appliance {appliance.id}: {'; '.join(approximations)}; pass "
            "allow_approximation=True to enable and record this reduction"
        )
    out.reductions.extend(
        f"appliance {appliance.id}: {description}" for description in approximations
    )


def _record_branch_scope(
    out: PandapowerExport, branch, allow_approximation: bool
) -> None:
    """Record harmonic-only branch options and reject dropped switch terms."""
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
        if scalar(branch.inductance_h) != 0.0:
            dropped.append("series inductance")
        if scalar(branch.shunt_conductance_s) != 0.0:
            dropped.append("shunt conductance")
        if scalar(branch.shunt_capacitance_f) != 0.0:
            dropped.append("shunt capacitance")
        if scalar(branch.resistance_ohm) != 0.0:
            dropped.append(
                "pure-resistance semantics (pandapower uses switch_rx_ratio)"
            )
        if dropped and not allow_approximation:
            raise UnsupportedGridError(
                f"switch {branch.id}: pandapower bus switches cannot represent "
                f"{', '.join(dropped)}; pass allow_approximation=True to drop and "
                "record these terms"
            )
        if dropped:
            out.reductions.append(f"switch {branch.id}: dropped {', '.join(dropped)}")


def _export_line(net, out: PandapowerExport, br: Line, w0: float) -> None:
    import pandapower as pp

    if br.conductor_geometry is not None:
        raise UnsupportedGridError(
            "line with conductor_geometry (Carson/Deri): the frequency-dependent "
            "earth-return model has no pandapower equivalent"
        )
    if br.type_ref is not None and br.series_resistance_ohm_per_m is None:
        raise UnsupportedGridError(f"line {br.id} carries an unresolved type_ref")
    for field_name in ("series_resistance_ohm_per_m", "series_inductance_h_per_m"):
        if is_coupled(getattr(br, field_name)):
            out.reductions.append(
                f"line {br.id}: mutual coupling reduced to positive sequence"
            )
            break
    for name in (
        "series_resistance_ohm_per_m",
        "series_inductance_h_per_m",
        "shunt_capacitance_f_per_m",
        "shunt_conductance_s_per_m",
    ):
        check_matrix_shape(getattr(br, name), f"line {br.id} {name}")
    r_per_m = positive_sequence(br.series_resistance_ohm_per_m)
    l_per_m = positive_sequence(br.series_inductance_h_per_m)
    c_per_m = positive_sequence(br.shunt_capacitance_f_per_m)
    g_per_m = positive_sequence(br.shunt_conductance_s_per_m)
    length_km = scalar(br.length_m) / 1e3
    idx = pp.create_line_from_parameters(
        net,
        from_bus=out.bus_of_node[br.from_node],
        to_bus=out.bus_of_node[br.to_node],
        length_km=length_km,
        r_ohm_per_km=r_per_m * 1e3,
        x_ohm_per_km=w0 * l_per_m * 1e3,
        # F/m -> nF/km: x 1e3 m/km x 1e9 nF/F
        c_nf_per_km=c_per_m * 1e12,
        # S/m -> uS/km: x 1e3 m/km x 1e6 uS/S
        g_us_per_km=g_per_m * 1e9,
        max_i_ka=10.0,
        name=br.name or f"line{br.id}",
        in_service=bool(br.in_service),
    )
    out.line_of_branch[br.id] = int(idx)


def _export_transformer(net, out: PandapowerExport, br: Transformer, w0: float) -> None:
    import pandapower as pp

    if br.type_ref is not None and br.series_resistance_ohm is None:
        raise UnsupportedGridError(
            f"transformer {br.id} carries an unresolved type_ref"
        )
    if br.zero_sequence is not None:
        out.reductions.append(
            f"transformer {br.id}: zero-sequence override dropped "
            "(positive-sequence export)"
        )
    if br.from_grounding is not None or br.to_grounding is not None:
        raise UnsupportedGridError(
            f"transformer {br.id}: impedance-grounded neutral is not representable"
        )
    from_conn = br.from_connection or WindingConnection.WYE_GROUNDED
    to_conn = br.to_connection or WindingConnection.WYE_GROUNDED
    if WindingConnection.ZIGZAG in (from_conn, to_conn) or (
        WindingConnection.ZIGZAG_GROUNDED in (from_conn, to_conn)
    ):
        raise UnsupportedGridError(f"transformer {br.id}: zigzag winding")

    s_va = scalar(br.s_rated_va)
    u_hv = scalar(br.u_rated_from_v)
    u_lv = scalar(br.u_rated_to_v)
    # Inverse of the converter's coil referral: the schema stores the TO-coil
    # value, 3x the terminal (line-to-line) quantity for a delta secondary.
    coil_factor = 3.0 if to_conn == WindingConnection.DELTA else 1.0
    r_ll = scalar(br.series_resistance_ohm) / coil_factor
    x_ll = w0 * scalar(br.series_inductance_h) / coil_factor
    z_base_lv = u_lv**2 / s_va
    vkr_percent = r_ll / z_base_lv * 100.0
    vk_percent = math.hypot(r_ll, x_ll) / z_base_lv * 100.0

    # Magnetizing branch, HV-referred: g_m = pfe_w/u_hv^2 and
    # b_m = 1/(w0 L_m); i0 is the no-load APPARENT current in % of rating.
    g_m = scalar(br.magnetizing_conductance_s)
    pfe_w = g_m * u_hv**2
    l_m = br.magnetizing_inductance_h
    if l_m is None or scalar(l_m) <= 0.0:
        q_nl = 0.0
    else:
        q_nl = u_hv**2 / (w0 * scalar(l_m))
    s_nl = math.hypot(pfe_w, q_nl)
    i0_percent = s_nl / s_va * 100.0

    ratio = scalar(br.tap.ratio_magnitude)
    shift_deg = float(br.tap.shift_deg)
    tap_kwargs: dict[str, Any] = {}
    if abs(ratio - 1.0) > 1e-14:
        # hv-side tap: ratio_magnitude = 1 + delta (converter's own convention).
        # pandapower 3 applies the tap only when tap_changer_type names one, and
        # silently ignores the tap columns without it.
        tap_kwargs = {
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_pos": 1,
            "tap_min": -1,
            "tap_max": 1,
            "tap_step_percent": (ratio - 1.0) * 100.0,
            "tap_changer_type": "Ratio",
        }
    # A BARE vector-group string (no clock digits) is what pandapower's own
    # three-phase model requires, and what the pgml converter accepts without
    # cross-checking against shift_degree; the clock travels in shift_degree.
    vector_group = _BARE_GROUP.get((from_conn, to_conn))

    idx = pp.create_transformer_from_parameters(
        net,
        hv_bus=out.bus_of_node[br.from_node],
        lv_bus=out.bus_of_node[br.to_node],
        sn_mva=s_va / 1e6,
        vn_hv_kv=u_hv / 1e3,
        vn_lv_kv=u_lv / 1e3,
        vkr_percent=vkr_percent,
        vk_percent=vk_percent,
        pfe_kw=pfe_w / 1e3,
        i0_percent=i0_percent,
        shift_degree=shift_deg,
        name=br.name or f"trafo{br.id}",
        in_service=bool(br.in_service),
        **tap_kwargs,
    )
    if vector_group is not None:
        net.trafo.at[idx, "vector_group"] = vector_group
    out.trafo_of_branch[br.id] = int(idx)


_BARE_GROUP = {
    (WindingConnection.DELTA, WindingConnection.WYE_GROUNDED): "Dyn",
    (WindingConnection.DELTA, WindingConnection.WYE): "Dy",
    (WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED): "YNyn",
    (WindingConnection.WYE_GROUNDED, WindingConnection.DELTA): "YNd",
    (WindingConnection.WYE, WindingConnection.WYE): "Yy",
    (WindingConnection.WYE, WindingConnection.DELTA): "Yd",
    (WindingConnection.WYE_GROUNDED, WindingConnection.WYE): "YNy",
    (WindingConnection.WYE, WindingConnection.WYE_GROUNDED): "Yyn",
    (WindingConnection.DELTA, WindingConnection.DELTA): "Dd",
}


def _export_switch(net, out: PandapowerExport, br: Switch) -> None:
    import pandapower as pp

    z_ohm = scalar(br.resistance_ohm)
    idx = pp.create_switch(
        net,
        bus=out.bus_of_node[br.from_node],
        element=out.bus_of_node[br.to_node],
        et="b",
        closed=bool(br.closed) and bool(br.in_service),
        z_ohm=z_ohm,
        name=br.name or f"switch{br.id}",
    )
    out.switch_of_branch[br.id] = int(idx)
    if z_ohm > 0.0:
        out.reductions.append(
            f"switch {br.id}: z_ohm={z_ohm:g} -- pandapower splits it across R "
            "and X at switch_rx_ratio, pgml keeps it purely resistive"
        )


def _export_shunt_reactor(net, out: PandapowerExport, br: ShuntReactor, w0) -> None:
    import pandapower as pp

    g = positive_sequence(br.conductance_s)
    c = positive_sequence(br.capacitance_f)
    inductance = None if br.inductance_h is None else positive_sequence(br.inductance_h)
    susceptance = w0 * c
    if inductance is not None and inductance > 0.0:
        susceptance -= 1.0 / (w0 * inductance)
    bus = out.bus_of_node[br.from_node]
    u_kv = float(net.bus.at[bus, "vn_kv"])
    # pandapower's shunt is specified as the power it draws at its own rated
    # voltage: P = G U^2, Q = -B U^2 (generation-negative for a capacitor).
    u_v = u_kv * 1e3
    idx = pp.create_shunt(
        net,
        bus=bus,
        q_mvar=-susceptance * u_v**2 / 1e6,
        p_mw=g * u_v**2 / 1e6,
        vn_kv=u_kv,
        name=br.name or f"shunt{br.id}",
        in_service=bool(br.in_service),
    )
    out.shunt_of_component[br.id] = int(idx)


def _export_shunt_appliance(net, out, ap: ShuntAppliance, w0) -> None:
    import pandapower as pp

    g, susceptance = shunt_positive_sequence(ap, w0)
    bus = out.bus_of_node[ap.node]
    u_v = float(net.bus.at[bus, "vn_kv"]) * 1e3
    idx = pp.create_shunt(
        net,
        bus=bus,
        q_mvar=-susceptance * u_v**2 / 1e6,
        p_mw=g * u_v**2 / 1e6,
        vn_kv=u_v / 1e3,
        name=ap.name or f"shunt{ap.id}",
        in_service=bool(ap.in_service),
    )
    out.shunt_of_component[ap.id] = int(idx)


def _export_source(net, out: PandapowerExport, ap: Source, grid: Grid) -> None:
    import pandapower as pp

    u_ref = np.asarray(detached(ap.u_ref_v), dtype=float).reshape(-1)
    ang = np.asarray(detached(ap.u_angle_deg), dtype=float).reshape(-1)
    bus = out.bus_of_node[ap.node]
    u_rated_ll = float(net.bus.at[bus, "vn_kv"]) * 1e3
    n_phase = len([p for p in ap.phases if p != Phase.N])
    # Source u_ref_v is line-to-neutral for a 3-phase appliance; pandapower's
    # vm_pu is on the bus's line-to-line base.
    scale = math.sqrt(3.0) if n_phase >= 3 else 1.0
    vm_pu = float(u_ref[0]) * scale / u_rated_ll
    idx = pp.create_ext_grid(
        net,
        bus=bus,
        vm_pu=vm_pu,
        va_degree=float(ang[0]),
        name=ap.name or f"source{ap.id}",
        in_service=bool(ap.in_service),
    )
    out.ext_grid_of_appliance[ap.id] = int(idx)


def _zip_columns(ap) -> dict[str, float]:
    """pandapower's four per-load ZIP percentages from the schema's fractions."""
    zc = getattr(ap, "zip_coefficients", None)
    model = getattr(ap, "load_model", LoadModel.CONST_POWER)
    if model == LoadModel.CONST_POWER:
        return {}
    if model == LoadModel.CONST_IMPEDANCE:
        return {
            "const_z_p_percent": 100.0,
            "const_i_p_percent": 0.0,
            "const_z_q_percent": 100.0,
            "const_i_q_percent": 0.0,
        }
    if model == LoadModel.CONST_CURRENT:
        return {
            "const_z_p_percent": 0.0,
            "const_i_p_percent": 100.0,
            "const_z_q_percent": 0.0,
            "const_i_q_percent": 100.0,
        }
    if zc is None:
        return {}
    return {
        "const_z_p_percent": float(zc.z_p) * 100.0,
        "const_i_p_percent": float(zc.i_p) * 100.0,
        "const_z_q_percent": float(zc.z_q) * 100.0,
        "const_i_q_percent": float(zc.i_q) * 100.0,
    }


def _export_load(net, out: PandapowerExport, ap: Load) -> None:
    import pandapower as pp

    p, q = phase_totals(ap)
    idx = pp.create_load(
        net,
        bus=out.bus_of_node[ap.node],
        p_mw=p / 1e6,
        q_mvar=q / 1e6,
        name=ap.name or f"load{ap.id}",
        in_service=bool(ap.in_service),
        **_zip_columns(ap),
    )
    out.load_of_appliance[ap.id] = int(idx)
    if ap.connection == WindingConnection.DELTA:
        out.reductions.append(f"load {ap.id}: delta connection folded to a bus total")


def _export_generator(net, out: PandapowerExport, ap) -> None:
    import pandapower as pp

    p, q = phase_totals(ap)
    idx = pp.create_sgen(
        net,
        bus=out.bus_of_node[ap.node],
        p_mw=p / 1e6,
        q_mvar=q / 1e6,
        name=ap.name or f"gen{ap.id}",
        in_service=bool(ap.in_service),
    )
    out.sgen_of_appliance[ap.id] = int(idx)


__all__ = [
    "PandapowerExport",
    "UnsupportedGridError",
    "from_grid",
]
