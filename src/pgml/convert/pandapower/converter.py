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
)

_PHASE_A = (Phase.A,)
_TINY_R = 1.0e-6   # Ohm — near-ideal Thevenin for ext_grid in Norton stamp
_TINY_L = 1.0e-12  # H   — near-ideal Thevenin for ext_grid in Norton stamp
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

    id_map: dict[str, Any] = {"bus": {}, "line": {}, "load": {}, "ext_grid": {}}

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
    # 3. ext_grid -> Source (Thevenin with near-zero Z; ideal-slack mode   #
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
    # 4. Loads                                                             #
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
