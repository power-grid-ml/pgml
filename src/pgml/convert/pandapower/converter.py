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

Single-phase positive-sequence equivalent
-----------------------------------------
Every node is modelled as a single-phase node with ``phases=(Phase.A,)`` and
``u_rated_v = vn_kv * 1000`` (line-to-line magnitude, retained as-is for the
1-phase node because ``phase_voltage_magnitude`` returns ``u_rated_v`` unchanged
for nodes with fewer than 3 phases).  This matches pandapower's const-Z
reference: y_const = conj(S_total)/(V_LL)^2 = (P - jQ)/(V_LL)^2.

ext_grid -> Source
------------------
The slack (ext_grid) is converted to a ``Source`` with a very small Thevenin
impedance (1e-6 Ohm, 1e-12 H) so the Norton stamp is near-zero.  In the oracle
test we use **ideal-slack mode** (``fixed_rows`` / ``v_fixed``) which makes the
Thevenin impedance irrelevant; the Source is still required by the schema so the
slack bus has an appliance.

Slack voltage phasor stored in id_map
--------------------------------------
``id_map["slack_v_complex"]`` holds the complex slack phasor (in V, LL) as a
Python complex number so the test can pass it directly to ``solve_harmonic``.

Only in-service elements are converted.
"""

from __future__ import annotations

import math
from typing import Any

from pgml.schemas.grid_schema import (
    ComplexTap,
    ConstantParam,
    Grid,
    GridMetadata,
    Line,
    Load,
    Node,
    Phase,
    Provenance,
    ResistanceFrequencyModel,
    Source,
    SourceConvention,
    Switch,
    Transformer,
    WindingConnection,
)

_PHASE_A = (Phase.A,)
_TINY_R = 1.0e-6    # Ohm — near-ideal Thevenin for ext_grid in Norton stamp
_TINY_L = 1.0e-12   # H   — near-ideal Thevenin for ext_grid in Norton stamp
_SWITCH_R = 1.0e-4  # Ohm — near-ideal resistance for closed bus-bus switches
_PROVENANCE = Provenance(
    source_convention=SourceConvention.SEQUENCE,
    notes=(
        "Converted from pandapower positive-sequence network. "
        "Single-phase-equivalent: phases=(A,), u_rated_v = vn_kv*1000 (line-to-line). "
        "Engineering units converted to SI."
    ),
)


def to_grid(net: Any) -> tuple[Grid, dict[str, Any]]:
    """Convert a pandapower network to a :class:`~pgml.schemas.grid_schema.Grid`.

    Parameters
    ----------
    net:
        A pandapower network object (the result of e.g.
        ``pandapower.networks.case33bw()``). The network must already have
        basic DataFrames (``bus``, ``line``, ``load``, ``ext_grid``).
        Unmaterialised std_type references in lines are accepted as long as
        explicit per-km parameters are present.

    Returns
    -------
    (Grid, id_map)
        ``Grid`` — materialised schema object (no ``type_ref``).
        ``id_map`` — ``dict`` mapping source element tables to our ids:
            - ``"bus"``      : ``{pp_bus_idx: Node.id}``
            - ``"line"``     : ``{pp_line_idx: Line.id}``
            - ``"load"``     : ``{pp_load_idx: Load.id}``
            - ``"ext_grid"`` : ``{pp_eg_idx: Source.id}``
            - ``"slack_v_complex"`` : complex slack voltage phasor (V, LL) for ideal-slack mode
    """
    f0_hz: float = float(getattr(net, "f_hz", 50.0))
    two_pi_f0 = 2.0 * math.pi * f0_hz

    # ------------------------------------------------------------------ #
    # ID counter: allocate monotonically increasing integer ids for all   #
    # elements (nodes then branches then appliances) to avoid collisions.  #
    # ------------------------------------------------------------------ #
    _id = _IdCounter()

    id_map: dict[str, Any] = {
        "bus": {},
        "line": {},
        "trafo": {},
        "switch": {},
        "load": {},
        "ext_grid": {},
    }

    # ------------------------------------------------------------------ #
    # 1. Nodes (buses)                                                     #
    # ------------------------------------------------------------------ #
    nodes: list[Node] = []
    for pp_idx, row in net.bus.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        node_id = _id.next()
        id_map["bus"][pp_idx] = node_id
        nodes.append(
            Node(
                id=node_id,
                name=str(row.get("name", f"bus_{pp_idx}") or f"bus_{pp_idx}"),
                u_rated_v=float(row["vn_kv"]) * 1_000.0,
                phases=_PHASE_A,
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

        # Per-length SI parameters (1/m)
        r_per_m = float(row["r_ohm_per_km"]) / 1_000.0       # Ohm/m
        x_per_m = float(row["x_ohm_per_km"]) / 1_000.0       # Ohm/m (=2*pi*f0*L per m)
        l_per_m = x_per_m / two_pi_f0                         # H/m

        c_nf_km = float(row.get("c_nf_per_km", 0.0) or 0.0)
        c_per_m = c_nf_km * 1.0e-9 / 1_000.0                 # F/m  (nF/km -> F/m)

        g_us_km = float(row.get("g_us_per_km", 0.0) or 0.0)
        g_per_m = g_us_km * 1.0e-6 / 1_000.0                 # S/m  (µS/km -> S/m)

        # 1x1 matrices for single-phase equivalent
        r_mat = [[r_per_m]]
        l_mat = [[l_per_m]]
        c_mat = [[c_per_m]]
        g_mat = [[g_per_m]] if g_per_m != 0.0 else None

        branches.append(
            Line(
                id=line_id,
                name=str(row.get("name", f"line_{pp_idx}") or f"line_{pp_idx}"),
                from_node=id_map["bus"][from_bus],
                to_node=id_map["bus"][to_bus],
                from_phases=_PHASE_A,
                to_phases=_PHASE_A,
                length_m=length_m,
                series_resistance_ohm_per_m=r_mat,
                series_inductance_h_per_m=l_mat,
                shunt_capacitance_f_per_m=c_mat,
                shunt_conductance_s_per_m=g_mat,
                resistance_frequency=ResistanceFrequencyModel(
                    multiplier=ConstantParam(value=1.0)
                ),
                provenance=_PROVENANCE,
            )
        )

    # ------------------------------------------------------------------ #
    # 3. Transformers (two-winding, positive-sequence equivalent)          #
    # ------------------------------------------------------------------ #
    # Conversion formulas for our assembly's transformer stamp convention: #
    #                                                                      #
    # Our assembly uses the MATPOWER off-nominal-tap PI model:             #
    #   Y_ff = y_se/|t|^2,  Y_ft = -y_se/conj(t)                         #
    #   Y_tf = -y_se/t,      Y_tt = y_se                                  #
    # where t = tap.ratio_magnitude * exp(j*tap.shift_deg).               #
    #                                                                      #
    # The model is correct in PER-UNIT where `y_se` is on the LV base and #
    # `t` is the off-nominal ratio (≈1).  In SI we use the full turns      #
    # ratio t = n = vn_hv/vn_lv as `tap.ratio_magnitude`, so `y_se` must  #
    # be referred to the LV side: Z_sc_LV = Z_sc_HV / n^2.  Then:         #
    #   Y_ff = y_se_LV/n^2 = y_se_HV  (HV SI)    ✓                        #
    #   Y_tt = y_se_LV = n^2*y_se_HV  (LV SI)    ✓                        #
    #                                                                      #
    # All quantities:                                                       #
    #   Z_base_LV = V_n_LV^2 / S_n  (LV SI ohm base)                      #
    #   Z_sc_LV = (vk_percent/100) * Z_base_LV                            #
    #   R_sc_LV = (vkr_percent/100) * Z_base_LV                           #
    #   X_sc_LV = sqrt(Z_sc_LV^2 - R_sc_LV^2);  L_sc_LV = X/(2*pi*f0)   #
    #   tap_ratio = vn_hv_kv / vn_lv_kv  (full turns ratio, SI)           #
    #   shift_deg = shift_degree from pandapower nameplate                 #
    # Magnetizing branch (referred to HV side for the shunt):              #
    #   G_m = P_fe / V_n_HV^2;  B_m = sqrt(...) / V_n_HV^2               #
    #   L_m = 1 / (2*pi*f0 * B_m) when B_m > 0, else None                #
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

            sn_va = float(row["sn_mva"]) * 1.0e6         # VA
            vn_hv_v = float(row["vn_hv_kv"]) * 1.0e3    # V
            vn_lv_v = float(row["vn_lv_kv"]) * 1.0e3    # V
            vk_pct = float(row["vk_percent"])
            vkr_pct = float(row["vkr_percent"])
            pfe_w = float(row.get("pfe_kw", 0.0) or 0.0) * 1.0e3  # W
            i0_pct = float(row.get("i0_percent", 0.0) or 0.0)
            shift_deg = float(row.get("shift_degree", 0.0) or 0.0)

            # Leakage impedance referred to LV side (required by our stamp convention)
            # Z_base_LV = Vn_LV^2 / Sn;  Z_sc_LV = vk%/100 * Z_base_LV
            z_base_lv = vn_lv_v ** 2 / sn_va
            z_sc_lv = vk_pct / 100.0 * z_base_lv
            r_sc_lv = vkr_pct / 100.0 * z_base_lv
            x_sc_sq = z_sc_lv ** 2 - r_sc_lv ** 2
            x_sc_lv = math.sqrt(max(x_sc_sq, 0.0))
            l_sc_lv = x_sc_lv / two_pi_f0

            # Magnetizing branch (referred to HV side; added to the HV diagonal)
            if pfe_w > 0.0:
                g_m = pfe_w / (vn_hv_v ** 2)
            else:
                g_m = 0.0

            l_m = None
            if i0_pct > 0.0:
                # no-load current amplitude as a fraction of rated current
                # I0 = i0_pct/100 * Sn / Vn_HV  (line-to-line, positive-seq)
                i0_amp = i0_pct / 100.0 * sn_va / vn_hv_v
                s_nl = vn_hv_v * i0_amp  # VA
                q_nl_sq = s_nl ** 2 - pfe_w ** 2
                if q_nl_sq > 0.0:
                    b_m = math.sqrt(q_nl_sq) / (vn_hv_v ** 2)
                    if b_m > 0.0:
                        l_m = 1.0 / (two_pi_f0 * b_m)

            # Off-nominal tap: full turns ratio = HV_rated / LV_rated (SI), plus
            # the nameplate phase shift from pandapower.
            tap_ratio = vn_hv_v / vn_lv_v

            branches.append(
                Transformer(
                    id=trafo_id,
                    name=str(row.get("name", f"trafo_{pp_idx}") or f"trafo_{pp_idx}"),
                    from_node=id_map["bus"][hv_bus],    # from = HV side
                    to_node=id_map["bus"][lv_bus],      # to   = LV side
                    from_phases=_PHASE_A,
                    to_phases=_PHASE_A,
                    s_rated_va=sn_va,
                    u_rated_from_v=vn_hv_v,
                    u_rated_to_v=vn_lv_v,
                    from_connection=WindingConnection.DELTA,   # HV of Dyn
                    to_connection=WindingConnection.WYE_GROUNDED,  # LV of Dyn
                    series_resistance_ohm=r_sc_lv,
                    series_inductance_h=l_sc_lv,
                    magnetizing_conductance_s=g_m,
                    magnetizing_inductance_h=l_m,
                    tap=ComplexTap(ratio_magnitude=tap_ratio, shift_deg=shift_deg),
                    provenance=_PROVENANCE,
                )
            )

    # ------------------------------------------------------------------ #
    # 4. Bus-bus switches (et='b', closed=True -> near-ideal Switch)      #
    # ------------------------------------------------------------------ #
    # pandapower's bus-bus switches (et='b') represent closed busbars or  #
    # coupling breakers that merge buses internally.  We convert each     #
    # closed bus-bus switch to a Switch with a small series resistance so  #
    # the series admittance is large but finite (no division-by-zero).    #
    # Open bus-bus switches are skipped (no branch stamped).              #
    # ------------------------------------------------------------------ #
    if hasattr(net, "switch") and len(net.switch):
        for pp_idx, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b":
                continue                             # only bus-bus switches
            if not bool(row.get("closed", True)):
                continue                             # open switch: no branch
            bus_from = int(row["bus"])
            bus_to = int(row["element"])
            if bus_from not in id_map["bus"] or bus_to not in id_map["bus"]:
                continue

            sw_id = _id.next()
            id_map["switch"][pp_idx] = sw_id
            z_ohm = float(row.get("z_ohm", 0.0) or 0.0)
            # Use z_ohm if provided; fall back to _SWITCH_R for numerical stability.
            r_sw = z_ohm if z_ohm > 0.0 else _SWITCH_R
            branches.append(
                Switch(
                    id=sw_id,
                    name=str(row.get("name", f"switch_{pp_idx}") or f"switch_{pp_idx}"),
                    from_node=id_map["bus"][bus_from],
                    to_node=id_map["bus"][bus_to],
                    from_phases=_PHASE_A,
                    to_phases=_PHASE_A,
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
        # u_rated_v of the slack bus (LL for 1-phase node)
        u_rated_v = float(net.bus.at[bus_pp, "vn_kv"]) * 1_000.0
        u_ref_v = vm_pu * u_rated_v  # magnitude of the slack phasor (LL)

        # Store complex phasor for ideal-slack use
        id_map["slack_v_complex"] = u_ref_v * complex(
            math.cos(math.radians(va_deg)), math.sin(math.radians(va_deg))
        )

        appliances.append(
            Source(
                id=src_id,
                name=str(row.get("name", f"ext_grid_{pp_idx}") or f"ext_grid_{pp_idx}"),
                node=id_map["bus"][bus_pp],
                phases=_PHASE_A,
                u_ref_v=(u_ref_v,),
                u_angle_deg=(va_deg,),
                resistance_ohm=[[_TINY_R]],
                inductance_h=[[_TINY_L]],
            )
        )

    # ------------------------------------------------------------------ #
    # 5. Loads                                                             #
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

        appliances.append(
            Load(
                id=load_id,
                name=str(row.get("name", f"load_{pp_idx}") or f"load_{pp_idx}"),
                node=id_map["bus"][bus_pp],
                phases=_PHASE_A,
                p_nom_w=p_w,
                q_nom_var=q_var,
            )
        )

    grid = Grid(
        base_frequency_hz=f0_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
        metadata=GridMetadata(
            name=str(getattr(net, "name", "") or "pandapower_import"),
            description=(
                f"Imported from pandapower (f0={f0_hz} Hz). "
                "Single-phase positive-sequence equivalent."
            ),
        ),
    )
    return grid, id_map


class _IdCounter:
    """Monotonically increasing integer id generator."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


__all__ = ["to_grid"]
