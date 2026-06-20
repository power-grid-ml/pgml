"""Pure conversion function: pandapower net -> (Grid, id_map).

Conventions applied
-------------------
Unit conversion (engineering -> SI):
  - vn_kv [kV]  -> u_rated_v [V]  : multiply by 1000
  - length_km   -> length_m        : multiply by 1000
  - r_ohm_per_km -> series_resistance_ohm_per_m  : divide by 1000
  - x_ohm_per_km -> series_inductance_h_per_m    : x/(2*pi*f0) / 1000
  - c_nf_per_km  -> shunt_capacitance_f_per_m    : multiply by 1e-9, divide by 1000 = 1e-12
  - g_us_per_km  -> shunt_conductance_s_per_m    : multiply by 1e-6, divide by 1000 = 1e-9
  - p_mw         -> p_nom_w [W]   : multiply by 1e6
  - q_mvar       -> q_nom_var [VAr]: multiply by 1e6

Phase mode
----------
``phase_mode=PhaseMode.SINGLE_PHASE_EQUIV`` (default) keeps today's positive-sequence
single-phase equivalent: every node/branch is ``phases=(Phase.A,)`` and lines carry
1x1 matrices. ``u_rated_v = vn_kv * 1000`` (line-to-line magnitude, retained as-is for
the 1-phase node because ``phase_voltage_magnitude`` returns ``u_rated_v`` unchanged
for nodes with fewer than 3 phases); this matches pandapower's const-Z reference
``y_const = conj(S_total)/(V_LL)^2``.

``phase_mode=PhaseMode.THREE_PHASE`` produces a genuine abc grid: nodes/branches become
``(A, B, C)``; lines are expanded from sequence quantities via the symmetric-component
identity (zero-sequence from ``net.line`` ``r0/x0/c0`` columns when present, else from
``pgml.config`` defaults); the slack becomes a balanced 3-phase Thevenin (angles
``0 / -120 / +120``); a non-empty ``net.asymmetric_load`` table is captured with its
WYE/DELTA connection and per-phase P/Q. The shared scaffold in
:mod:`pgml.convert._common` is the single place the phase decision lives.

ext_grid -> Source
------------------
The slack (ext_grid) is converted to a ``Source`` with a very small Thevenin
impedance (1e-6 Ohm, 1e-12 H) so the Norton stamp is near-zero.  In the oracle
test we use **ideal-slack mode** (``fixed_rows`` / ``v_fixed``) which makes the
Thevenin impedance irrelevant; the Source is still required by the schema so the
slack bus has an appliance. Its zero-sequence source impedance equals the
positive-sequence impedance (no short-circuit data is read here).

Slack voltage phasor stored in id_map
--------------------------------------
``id_map["slack_v_complex"]`` holds the complex slack phasor (in V, LL) as a
Python complex number so the test can pass it directly to ``solve_harmonic``.

Only in-service elements are converted.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pgml.convert._common import (
    IdCounter,
    PhaseMode,
    build_line_from_sequence,
    build_load,
    build_node,
    build_source,
    make_metadata,
    phases_for,
)
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Provenance,
    SourceConvention,
    Switch,
    Transformer,
    WindingConnection,
)

_logger = logging.getLogger("pgml")

_TINY_R = 1.0e-6  # Ohm — near-ideal Thevenin for ext_grid in Norton stamp
_TINY_L = 1.0e-12  # H   — near-ideal Thevenin for ext_grid in Norton stamp
_SWITCH_R = 1.0e-4  # Ohm — near-ideal resistance for closed bus-bus switches
_PROVENANCE = Provenance(
    source_convention=SourceConvention.SEQUENCE,
    notes=(
        "Converted from pandapower positive-sequence network. "
        "Engineering units converted to SI."
    ),
)


def to_grid(
    net: Any, *, phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV
) -> tuple[Grid, dict[str, Any]]:
    """Convert a pandapower network to a :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    net:
        A pandapower network object (the result of e.g.
        ``pandapower.networks.case33bw()``). The network must already have
        basic DataFrames (``bus``, ``line``, ``load``, ``ext_grid``).
        Unmaterialised std_type references in lines are accepted as long as
        explicit per-km parameters are present.
    phase_mode:
        :class:`~pgml.convert._common.PhaseMode`. ``SINGLE_PHASE_EQUIV`` (default)
        reproduces the positive-sequence single-phase-equivalent output exactly;
        ``THREE_PHASE`` expands to a genuine abc grid (sequence->phase line
        matrices, balanced 3-phase source, asymmetric-load capture). Caveat:
        under ``THREE_PHASE`` transformers use a per-phase diagonal stamp with no
        vector-group phase coupling or zero-sequence path, so results are
        approximate for non-Dyn vector groups (e.g. Yyn/YNyn).

    Returns
    -------
    tuple[Grid, dict]
        A ``(Grid, id_map)`` pair.  ``Grid`` is the materialised schema object
        (no ``type_ref``).  ``id_map`` maps source element tables to our ids:
        ``"bus"`` -> ``{pp_bus_idx: Node.id}``,
        ``"line"`` -> ``{pp_line_idx: Line.id}``,
        ``"load"`` -> ``{pp_load_idx: Load.id}``,
        ``"asymmetric_load"`` -> ``{pp_asym_idx: Load.id}`` (THREE_PHASE only),
        ``"ext_grid"`` -> ``{pp_eg_idx: Source.id}``,
        ``"slack_v_complex"`` -> complex slack voltage phasor (V, LL) for
        ideal-slack mode.
    """
    f0_hz: float = float(getattr(net, "f_hz", 50.0))
    two_pi_f0 = 2.0 * math.pi * f0_hz

    _id = IdCounter()

    id_map: dict[str, Any] = {
        "bus": {},
        "line": {},
        "trafo": {},
        "switch": {},
        "load": {},
        "asymmetric_load": {},
        "ext_grid": {},
        "slack_v_complex": None,
    }

    # ------------------------------------------------------------------ #
    # 1. Nodes (buses)                                                     #
    # ------------------------------------------------------------------ #
    nodes: list = []
    for pp_idx, row in net.bus.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        node_id = _id.next()
        id_map["bus"][pp_idx] = node_id
        nodes.append(
            build_node(
                id=node_id,
                name=str(row.get("name", f"bus_{pp_idx}") or f"bus_{pp_idx}"),
                u_rated_v=float(row["vn_kv"]) * 1_000.0,
                mode=phase_mode,
            )
        )

    # ------------------------------------------------------------------ #
    # 2. Lines                                                             #
    # ------------------------------------------------------------------ #
    branches: list = []
    for pp_idx, row in net.line.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        # Skip lines whose buses were not converted (e.g. out-of-service buses)
        if from_bus not in id_map["bus"] or to_bus not in id_map["bus"]:
            continue

        line_id = _id.next()
        id_map["line"][pp_idx] = line_id

        length_m = float(row["length_km"]) * 1_000.0

        # Per-length positive-sequence SI parameters (1/m)
        r1 = float(row["r_ohm_per_km"]) / 1_000.0  # Ohm/m
        x1 = float(row["x_ohm_per_km"]) / 1_000.0  # Ohm/m (=2*pi*f0*L per m)
        c1 = float(row.get("c_nf_per_km", 0.0) or 0.0) * 1.0e-9 / 1_000.0  # F/m
        g1 = float(row.get("g_us_per_km", 0.0) or 0.0) * 1.0e-6 / 1_000.0  # S/m

        # Native zero-sequence columns (THREE_PHASE only; else config defaults).
        r0 = _opt_per_km(row, "r0_ohm_per_km", 1_000.0)
        x0 = _opt_per_km(row, "x0_ohm_per_km", 1_000.0)
        c0_nf = _opt_per_km(row, "c0_nf_per_km", None)
        c0 = c0_nf * 1.0e-12 if c0_nf is not None else None

        branches.append(
            build_line_from_sequence(
                id=line_id,
                name=str(row.get("name", f"line_{pp_idx}") or f"line_{pp_idx}"),
                from_node=id_map["bus"][from_bus],
                to_node=id_map["bus"][to_bus],
                mode=phase_mode,
                length_m=length_m,
                r1=r1,
                x1=x1,
                c1=c1,
                two_pi_f0=two_pi_f0,
                r0=r0,
                x0=x0,
                c0=c0,
                g1=g1,
                provenance=_PROVENANCE,
            )
        )

    # ------------------------------------------------------------------ #
    # 3. Transformers (two-winding, vector-group aware)                    #
    # ------------------------------------------------------------------ #
    # The nominal turns ratio and the vector-group phase shift come from the   #
    # rated voltages (`u_rated_from/to_v`) plus the winding connections, so    #
    # `tap` carries the OFF-NOMINAL ratio only (1.0 here — pandapower tap-     #
    # changer positions are not yet read). Assembly builds the winding-        #
    # incidence primitive Y = N^T Y_winding N: a delta winding blocks the      #
    # zero sequence (traps triplen harmonics) and supplies the √3 ratio + 30°  #
    # clock shift. The single-phase-equivalent mode collapses this to the      #
    # positive-sequence off-nominal-tap pi. Leakage is referred to the LV side #
    # (Z_sc_LV = vk·Z_base_LV). pandapower's CIGRE LV trafos are Dyn1          #
    # (shift_degree=30 -> clock 1).                                            #
    # ------------------------------------------------------------------ #
    if hasattr(net, "trafo") and len(net.trafo):
        for pp_idx, row in net.trafo.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            hv_bus = int(row["hv_bus"])
            lv_bus = int(row["lv_bus"])
            if hv_bus not in id_map["bus"] or lv_bus not in id_map["bus"]:
                continue

            trafo_id = _id.next()
            id_map["trafo"][pp_idx] = trafo_id

            sn_va = float(row["sn_mva"]) * 1.0e6  # VA
            vn_hv_v = float(row["vn_hv_kv"]) * 1.0e3  # V
            vn_lv_v = float(row["vn_lv_kv"]) * 1.0e3  # V
            vk_pct = float(row["vk_percent"])
            vkr_pct = float(row["vkr_percent"])
            pfe_w = float(row.get("pfe_kw", 0.0) or 0.0) * 1.0e3  # W
            i0_pct = float(row.get("i0_percent", 0.0) or 0.0)
            shift_deg = float(row.get("shift_degree", 0.0) or 0.0)

            # Leakage impedance referred to LV side (required by our stamp convention)
            z_base_lv = vn_lv_v**2 / sn_va
            z_sc_lv = vk_pct / 100.0 * z_base_lv
            r_sc_lv = vkr_pct / 100.0 * z_base_lv
            x_sc_sq = z_sc_lv**2 - r_sc_lv**2
            x_sc_lv = math.sqrt(max(x_sc_sq, 0.0))
            l_sc_lv = x_sc_lv / two_pi_f0

            # Magnetizing branch (referred to HV side; added to the HV diagonal)
            if pfe_w > 0.0:
                g_m = pfe_w / (vn_hv_v**2)
            else:
                g_m = 0.0

            l_m = None
            if i0_pct > 0.0:
                i0_amp = i0_pct / 100.0 * sn_va / vn_hv_v
                s_nl = vn_hv_v * i0_amp  # VA
                q_nl_sq = s_nl**2 - pfe_w**2
                if q_nl_sq > 0.0:
                    b_m = math.sqrt(q_nl_sq) / (vn_hv_v**2)
                    if b_m > 0.0:
                        l_m = 1.0 / (two_pi_f0 * b_m)

            tx_phases = phases_for(phase_mode)
            branches.append(
                Transformer(
                    id=trafo_id,
                    name=str(row.get("name", f"trafo_{pp_idx}") or f"trafo_{pp_idx}"),
                    from_node=id_map["bus"][hv_bus],  # from = HV side
                    to_node=id_map["bus"][lv_bus],  # to   = LV side
                    from_phases=tx_phases,
                    to_phases=tx_phases,
                    s_rated_va=sn_va,
                    u_rated_from_v=vn_hv_v,
                    u_rated_to_v=vn_lv_v,
                    from_connection=WindingConnection.DELTA,  # HV of Dyn
                    to_connection=WindingConnection.WYE_GROUNDED,  # LV of Dyn
                    series_resistance_ohm=r_sc_lv,
                    series_inductance_h=l_sc_lv,
                    magnetizing_conductance_s=g_m,
                    magnetizing_inductance_h=l_m,
                    # Nominal ratio comes from u_rated + connections; `tap` is the
                    # off-nominal ratio (1.0) plus the vector-group clock angle.
                    tap=ComplexTap(ratio_magnitude=1.0, shift_deg=shift_deg),
                    provenance=_PROVENANCE,
                )
            )

    # ------------------------------------------------------------------ #
    # 4. Bus-bus switches (et='b', closed=True -> near-ideal Switch)      #
    # ------------------------------------------------------------------ #
    if hasattr(net, "switch") and len(net.switch):
        for pp_idx, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b":
                continue  # only bus-bus switches
            if not bool(row.get("closed", True)):
                continue  # open switch: no branch
            bus_from = int(row["bus"])
            bus_to = int(row["element"])
            if bus_from not in id_map["bus"] or bus_to not in id_map["bus"]:
                continue

            sw_id = _id.next()
            id_map["switch"][pp_idx] = sw_id
            z_ohm = float(row.get("z_ohm", 0.0) or 0.0)
            r_sw = z_ohm if z_ohm > 0.0 else _SWITCH_R
            sw_phases = phases_for(phase_mode)
            branches.append(
                Switch(
                    id=sw_id,
                    name=str(row.get("name", f"switch_{pp_idx}") or f"switch_{pp_idx}"),
                    from_node=id_map["bus"][bus_from],
                    to_node=id_map["bus"][bus_to],
                    from_phases=sw_phases,
                    to_phases=sw_phases,
                    closed=True,
                    resistance_ohm=r_sw,
                    inductance_h=0.0,
                    provenance=_PROVENANCE,
                )
            )

    # ------------------------------------------------------------------ #
    # 5. ext_grid -> Source (Thevenin with near-zero Z; ideal-slack mode   #
    #    overrides this in the solver)                                     #
    # ------------------------------------------------------------------ #
    appliances: list = []
    for pp_idx, row in net.ext_grid.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_pp = int(row["bus"])
        if bus_pp not in id_map["bus"]:
            continue

        src_id = _id.next()
        id_map["ext_grid"][pp_idx] = src_id

        vm_pu = float(row.get("vm_pu", 1.0))
        va_deg = float(row.get("va_degree", 0.0))
        u_rated_v = float(net.bus.at[bus_pp, "vn_kv"]) * 1_000.0
        u_ref_v = vm_pu * u_rated_v  # magnitude of the slack phasor (LL)

        # First slack wins: a network may have several ext_grids, but the ideal
        # slack solve takes a single fixed phasor. Guard so the last ext_grid does
        # not silently overwrite it (matches the pgm converter).
        if id_map["slack_v_complex"] is None:
            id_map["slack_v_complex"] = u_ref_v * complex(
                math.cos(math.radians(va_deg)), math.sin(math.radians(va_deg))
            )

        appliances.append(
            build_source(
                id=src_id,
                name=str(row.get("name", f"ext_grid_{pp_idx}") or f"ext_grid_{pp_idx}"),
                node=id_map["bus"][bus_pp],
                mode=phase_mode,
                u_ref_v=u_ref_v,
                u_angle_deg=va_deg,
                r_ohm=_TINY_R,
                l_h=_TINY_L,
            )
        )

    # ------------------------------------------------------------------ #
    # 6. Loads (balanced net.load)                                         #
    # ------------------------------------------------------------------ #
    for pp_idx, row in net.load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_pp = int(row["bus"])
        if bus_pp not in id_map["bus"]:
            continue

        load_id = _id.next()
        id_map["load"][pp_idx] = load_id

        p_w = float(row["p_mw"]) * 1.0e6
        q_var = float(row["q_mvar"]) * 1.0e6

        # Balanced total: connection=None resolves to WYE from config; under
        # THREE_PHASE the symmetric/auto calc splits the total equally.
        appliances.append(
            build_load(
                id=load_id,
                name=str(row.get("name", f"load_{pp_idx}") or f"load_{pp_idx}"),
                node=id_map["bus"][bus_pp],
                mode=phase_mode,
                p_total_w=p_w,
                q_total_var=q_var,
            )
        )

    # ------------------------------------------------------------------ #
    # 7. Asymmetric loads (net.asymmetric_load) — genuine per-phase split  #
    # ------------------------------------------------------------------ #
    # Under THREE_PHASE each asymmetric_load becomes an abc Load carrying    #
    # its WYE/DELTA connection and the per-phase P/Q. Under SINGLE_PHASE_    #
    # EQUIV the per-phase split cannot be represented, so the three phases   #
    # are summed into a balanced 1-phase total (logged at INFO).             #
    # ------------------------------------------------------------------ #
    asym = getattr(net, "asymmetric_load", None)
    if asym is not None and len(asym):
        for pp_idx, row in asym.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            bus_pp = int(row["bus"])
            if bus_pp not in id_map["bus"]:
                continue

            p_a = float(row.get("p_a_mw", 0.0) or 0.0) * 1.0e6
            p_b = float(row.get("p_b_mw", 0.0) or 0.0) * 1.0e6
            p_c = float(row.get("p_c_mw", 0.0) or 0.0) * 1.0e6
            q_a = float(row.get("q_a_mvar", 0.0) or 0.0) * 1.0e6
            q_b = float(row.get("q_b_mvar", 0.0) or 0.0) * 1.0e6
            q_c = float(row.get("q_c_mvar", 0.0) or 0.0) * 1.0e6
            p_total = p_a + p_b + p_c
            q_total = q_a + q_b + q_c
            conn = (
                WindingConnection.DELTA
                if str(row.get("type", "wye")).lower() == "delta"
                else WindingConnection.WYE
            )

            load_id = _id.next()
            id_map["asymmetric_load"][pp_idx] = load_id
            name = str(row.get("name", f"asym_load_{pp_idx}") or f"asym_load_{pp_idx}")

            if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV:
                _logger.info(
                    "pandapower asymmetric_load %s collapsed to a balanced "
                    "single-phase total under SINGLE_PHASE_EQUIV "
                    "(per-phase split discarded); use THREE_PHASE to keep it.",
                    pp_idx,
                )
                appliances.append(
                    build_load(
                        id=load_id,
                        name=name,
                        node=id_map["bus"][bus_pp],
                        mode=phase_mode,
                        p_total_w=p_total,
                        q_total_var=q_total,
                    )
                )
            else:
                appliances.append(
                    build_load(
                        id=load_id,
                        name=name,
                        node=id_map["bus"][bus_pp],
                        mode=phase_mode,
                        p_total_w=p_total,
                        q_total_var=q_total,
                        connection=conn,
                        p_per_phase_w=(p_a, p_b, p_c),
                        q_per_phase_var=(q_a, q_b, q_c),
                    )
                )

    description = f"Imported from pandapower (f0={f0_hz} Hz). " + (
        "Single-phase positive-sequence equivalent."
        if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV
        else "Three-phase (abc) expansion from sequence quantities."
    )
    grid = Grid(
        base_frequency_hz=f0_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=make_metadata(
            name=str(getattr(net, "name", "") or "pandapower_import"),
            description=description,
        ),
    )
    return grid, id_map


def _opt_per_km(row: Any, column: str, divisor: float | None) -> float | None:
    """Read an optional per-km column from a pandapower line row, SI-scaled.

    Returns ``None`` when the column is absent or NaN (so the line falls back to
    config zero-sequence defaults). ``divisor`` converts per-km -> per-m when given
    (e.g. ``r0_ohm_per_km`` / 1000); pass ``None`` to leave the raw value (the
    caller scales it, e.g. nF -> F).
    """
    if not hasattr(row, "get"):
        return None
    val = row.get(column, None)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f / divisor if divisor is not None else f


__all__ = ["to_grid"]
