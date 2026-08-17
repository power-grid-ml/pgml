"""Hand-built tiny grids for forward-correctness / differentiability / gpu tests.

Each builder returns a materialised :class:`~pgml.schemas.grid_schema.Grid` (no
``type_ref``) whose Y-bus and solved voltages are simple enough to recompute
independently in plain numpy inside the tests.

Conventions used by the const-Z / source stamps (mirrored by the numpy oracle in
the tests):
- ``Source``: Thevenin ``u_ref ∠ u_angle`` behind ``Z_s = R + j*2*pi*f*L``;
  Norton shunt ``Y_s = Z_s^-1`` on the source-node diagonal, current
  ``I_s = Y_s @ V_th`` at the source rows.
- ``Line``: series ``Ys = (R*len + j*2*pi*f*L*len)^-1`` primitive
  ``[[Ys,-Ys],[-Ys,Ys]]``; shunt ``Y_sh = (G + j*2*pi*f*C)*len`` split half to
  each end diagonal.
- ``ShuntAppliance``: ``Y = G + j*2*pi*f*C`` on the node diagonal.
- const-Z ``Load``: ``y = conj(P + jQ)/|U_ln|^2`` per phase on the node diagonal
  (``U_ln = u_rated_v`` for 1-phase nodes, ``u_rated_v/sqrt(3)`` for 3-phase).
"""

from __future__ import annotations

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    Node,
    Phase,
    ShuntAppliance,
    Source,
)


def single_phase_chain() -> Grid:
    """Source -> n1 --line1--> n2 --line2--> n3, with a shunt load at n3.

    Single phase ``(A,)`` so every per-phase matrix is 1x1. Chosen with clean SI
    numbers; the test recomputes Y and V in numpy.
    """
    nodes = [
        Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
        Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        Node(id=3, u_rated_v=230.0, phases=(Phase.A,)),
    ]
    source = Source(
        id=10,
        node=1,
        phases=(Phase.A,),
        u_ref_v=(230.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[0.1]],
        inductance_h=[[1.0e-3]],
    )
    line1 = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=100.0,
        series_resistance_ohm_per_m=[[1.0e-3]],
        series_inductance_h_per_m=[[1.0e-6]],
        shunt_capacitance_f_per_m=[[1.0e-9]],
    )
    line2 = Line(
        id=21,
        from_node=2,
        to_node=3,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=50.0,
        series_resistance_ohm_per_m=[[2.0e-3]],
        series_inductance_h_per_m=[[1.5e-6]],
        shunt_capacitance_f_per_m=[[2.0e-9]],
    )
    load = Load(
        id=30,
        node=3,
        phases=(Phase.A,),
        p_nom_w=2000.0,
        q_nom_var=500.0,
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[line1, line2],
        appliances=[source, load],
    )


def single_phase_shunt_only() -> Grid:
    """Source -> n1 --line--> n2 with a pure ShuntAppliance at n2 (no const-Z load).

    Used by the injection / Norton-vs-ideal-slack tests where only passive shunts
    and the source appear.
    """
    nodes = [
        Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
        Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
    ]
    source = Source(
        id=10,
        node=1,
        phases=(Phase.A,),
        u_ref_v=(230.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[0.2]],
        inductance_h=[[2.0e-3]],
    )
    line = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=200.0,
        series_resistance_ohm_per_m=[[1.0e-3]],
        series_inductance_h_per_m=[[1.0e-6]],
        shunt_capacitance_f_per_m=[[1.0e-9]],
    )
    shunt = ShuntAppliance(
        id=40,
        node=2,
        phases=(Phase.A,),
        conductance_s=(1.0e-3,),
        capacitance_f=(5.0e-7,),
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[line],
        appliances=[source, shunt],
    )


def three_phase_two_bus() -> Grid:
    """Three-phase source -> n1 --line--> n2 with a 3-phase const-Z load at n2.

    Per-phase matrices are 3x3 with a small symmetric mutual term, exercising the
    matrix-inverse series stamp and the 3-phase const-Z load (U_ln = u/sqrt(3)).
    """
    nodes = [
        Node(id=1, u_rated_v=400.0, phases=(Phase.A, Phase.B, Phase.C)),
        Node(id=2, u_rated_v=400.0, phases=(Phase.A, Phase.B, Phase.C)),
    ]

    def sym(diag: float, off: float) -> list[list[float]]:
        return [[diag if i == j else off for j in range(3)] for i in range(3)]

    source = Source(
        id=10,
        node=1,
        phases=(Phase.A, Phase.B, Phase.C),
        u_ref_v=(400.0 / 3**0.5,) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=sym(0.1, 0.01),
        inductance_h=sym(1.0e-3, 1.0e-4),
    )
    line = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(Phase.A, Phase.B, Phase.C),
        to_phases=(Phase.A, Phase.B, Phase.C),
        length_m=100.0,
        series_resistance_ohm_per_m=sym(1.0e-3, 1.0e-4),
        series_inductance_h_per_m=sym(1.0e-6, 1.0e-7),
        shunt_capacitance_f_per_m=sym(1.0e-9, 1.0e-10),
    )
    load = Load(
        id=30,
        node=2,
        phases=(Phase.A, Phase.B, Phase.C),
        p_nom_w=9000.0,
        q_nom_var=3000.0,
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[line],
        appliances=[source, load],
    )


__all__ = [
    "single_phase_chain",
    "single_phase_shunt_only",
    "three_phase_two_bus",
]
