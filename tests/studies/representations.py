"""One MV/LV network in every grid representation, for the system-power study tests.

A 20 kV source feeds a Dyn5 250 kVA transformer and a 400 V feeder of two cables
(270 A) with loads at both LV buses and a generator at the far end. The network is
written four ways:

- ``"per_phase"``: the pandapower converter's per-phase equivalent (one phase,
  ``U_LL/sqrt(3)``, a third of every power; tag ``"single_phase_equiv"``);
- ``"legacy"``: the same one-phase grid with line-to-line voltages and three-phase totals,
  as converters wrote it before the per-phase equivalent (tag ``"legacy_ll"``);
- ``"three_phase"``: the converter's ``(A, B, C)`` build (tag ``"three_phase"``);
- ``"four_wire"``: a hand-built phase-domain grid whose LV part is four-wire (an
  ``(A, B, C, N)`` row at every LV bus, uncoupled 4x4 cable matrices with the
  positive-sequence impedance per conductor, the neutral earthed at the LV busbar),
  with the source and the transformer of the three-phase build (tag ``"as_built"``).

The operating point is balanced, so the neutral carries no current and every
representation describes the same physical network. The cables have ``R0 = R1`` and
``X0 = X1``, so the three-phase build's cable matrices are diagonal too.
"""

from __future__ import annotations

import pandapower as pp

from pgml.convert import PhaseMode
from pgml.convert._export import system_view
from pgml.convert.pandapower import to_grid
from pgml.representation import PHASE_MODE_TAG
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    LineType,
    Load,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    Transformer,
    TypeLibrary,
)

ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
#: Ids of the LV buses (the candidates of every study).
LV_NODES = (2, 3, 4)
#: Representation names, in the order of :func:`representations`.
NAMES = ("per_phase", "legacy", "three_phase", "four_wire")

_R_OHM_PER_KM = 0.206
_X_OHM_PER_KM = 0.08
_MAX_I_KA = 0.27


def mv_lv_net():
    """The pandapower source network (balanced, ideal external grid)."""
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    b3 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_transformer_from_parameters(
        net,
        b0,
        b1,
        sn_mva=0.25,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=1.2,
        vk_percent=4.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=150.0,
        vector_group="Dyn5",
    )
    for a, b, length_km in ((b1, b2, 0.25), (b2, b3, 0.3)):
        pp.create_line_from_parameters(
            net,
            a,
            b,
            length_km=length_km,
            r_ohm_per_km=_R_OHM_PER_KM,
            x_ohm_per_km=_X_OHM_PER_KM,
            c_nf_per_km=0.0,
            r0_ohm_per_km=_R_OHM_PER_KM,
            x0_ohm_per_km=_X_OHM_PER_KM,
            c0_nf_per_km=0.0,
            max_i_ka=_MAX_I_KA,
        )
    pp.create_load(net, b2, p_mw=0.06, q_mvar=0.015)
    pp.create_load(net, b3, p_mw=0.05, q_mvar=0.012)
    pp.create_sgen(net, b3, p_mw=0.03, q_mvar=0.0)
    return net


def _diag4(x: float) -> list[list[float]]:
    return [[x if i == j else 0.0 for j in range(4)] for i in range(4)]


def _four_wire(three: Grid) -> Grid:
    """The three-phase build with a four-wire LV part (module documentation)."""
    line_a = next(b for b in three.branches if isinstance(b, Line))
    r = float(line_a.series_resistance_ohm_per_m[0][0])
    l_h = float(line_a.series_inductance_h_per_m[0][0])
    nodes = [
        n if n.id not in LV_NODES else Node(id=n.id, u_rated_v=n.u_rated_v, phases=ABCN)
        for n in three.nodes
    ]
    branches = []
    for b in three.branches:
        if isinstance(b, Line):
            b = Line(
                id=b.id,
                from_node=b.from_node,
                to_node=b.to_node,
                from_phases=ABCN,
                to_phases=ABCN,
                length_m=b.length_m,
                type_ref="cable4",
                series_resistance_ohm_per_m=_diag4(r),
                series_inductance_h_per_m=_diag4(l_h),
                shunt_capacitance_f_per_m=_diag4(0.0),
                harmonic_line_model="naive",
            )
        branches.append(b)
    appliances = []
    for a in three.appliances:
        if isinstance(a, (Load, Generator)):
            a = a.model_copy(update={"return_path": "neutral"})
        appliances.append(a)
    first_id = max(int(a.id) for a in appliances) + 1
    appliances.append(
        ShuntAppliance(
            id=first_id,
            node=LV_NODES[0],
            phases=(Phase.N,),
            conductance_s=(100.0,),
            capacitance_f=(0.0,),
        )
    )
    types = TypeLibrary(
        lines={
            "cable4": LineType(
                id="cable4",
                n_phases=4,
                series_resistance_ohm_per_m=_diag4(r),
                series_inductance_h_per_m=_diag4(l_h),
                shunt_capacitance_f_per_m=_diag4(0.0),
                max_i_a=_MAX_I_KA * 1.0e3,
            )
        },
        transformers=three.types.transformers,
    )
    metadata = three.metadata.model_copy(
        update={"tags": {**three.metadata.tags, PHASE_MODE_TAG: "as_built"}}
    )
    assert all(isinstance(a, Source) for a in three.appliances if a.node == 1)
    assert sum(isinstance(b, Transformer) for b in branches) == 1
    return three.model_copy(
        update={
            "nodes": nodes,
            "branches": branches,
            "appliances": appliances,
            "types": types,
            "metadata": metadata,
        }
    )


def representations() -> dict[str, Grid]:
    """``{name: grid}`` of the network in every representation (module documentation)."""
    per_phase, _ = to_grid(mv_lv_net(), phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    legacy = system_view(per_phase)
    tags = {**legacy.metadata.tags, PHASE_MODE_TAG: "legacy_ll"}
    legacy = legacy.model_copy(
        update={"metadata": legacy.metadata.model_copy(update={"tags": tags})}
    )
    three, _ = to_grid(mv_lv_net(), phase_mode=PhaseMode.THREE_PHASE)
    return {
        "per_phase": per_phase,
        "legacy": legacy,
        "three_phase": three,
        "four_wire": _four_wire(three),
    }
