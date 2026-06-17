"""Shared fixture: a small single-phase grid with two distinct loads."""

from __future__ import annotations

import math

import pytest

from pgml.schemas.grid_schema import Grid, Line, Load, LoadModel, Node, Phase, Source

W = 2.0 * math.pi * 50.0


def _line(bid, a, b):
    return Line(
        id=bid,
        from_node=a,
        to_node=b,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[0.5]],
        series_inductance_h_per_m=[[0.5 / W]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


@pytest.fixture
def grid3() -> Grid:
    """3-bus radial: source@1, household load@2 (id 10), EV load@3 (id 11)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=3, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[_line(1, 1, 2), _line(2, 2, 3)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=500.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="household",
            ),
            Load(
                id=11,
                node=3,
                phases=(Phase.A,),
                p_nom_w=3000.0,
                q_nom_var=800.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="ev_charging",
            ),
        ],
    )
