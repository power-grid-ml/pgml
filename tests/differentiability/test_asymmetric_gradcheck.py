"""Differentiability gate for the connection-aware load models.

float64 gradcheck of the solved power-flow voltage ``v`` w.r.t. a load's per-phase
P/Q through BOTH a DELTA load and a WYE-with-neutral load. Per-phase P/Q leaves are
injected via the ``param_overrides`` hook
(``("load", id, "p_nom_per_phase_w"|"q_nom_per_phase_var")``), which
:func:`solve_power_flow` threads into ``device_current_injections`` and collects as
autograd leaves.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
    WindingConnection,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128
ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
torch.manual_seed(0)


def _source(node_phases=ABC):
    n = len(node_phases)
    angles = [0.0, -120.0, 120.0, 0.0][:n]
    # Neutral conductor referenced to (near) ground at the slack.
    mags = [231.0, 231.0, 231.0, 0.0][:n]
    r = [[0.05 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[1e-4 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Source(
        id=10,
        node=1,
        phases=node_phases,
        u_ref_v=tuple(mags),
        u_angle_deg=tuple(angles),
        resistance_ohm=r,
        inductance_h=ll,
    )


def _line(node_phases=ABC):
    n = len(node_phases)
    r = [[2e-3 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[2e-6 if i == j else 0.0 for j in range(n)] for i in range(n)]
    c = [[0.0] * n for _ in range(n)]
    return Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=node_phases,
        to_phases=node_phases,
        length_m=50.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=ll,
        shunt_capacitance_f_per_m=c,
    )


def _grid(load, node_phases=ABC):
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=node_phases),
            Node(id=2, u_rated_v=400.0, phases=node_phases),
        ],
        branches=[_line(node_phases)],
        appliances=[_source(node_phases), load],
    )


def test_gradcheck_delta_load_per_phase_pq():
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        connection=WindingConnection.DELTA,
    )
    grid = _grid(load, ABC)

    p = torch.tensor([2000.0, 1500.0, 1000.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([400.0, 300.0, 200.0], dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        ov = {
            ("load", 30, "p_nom_per_phase_w"): p,
            ("load", 30, "q_nom_per_phase_var"): q,
        }
        return solve_power_flow(
            grid, slack="ideal", dtype=CDT, param_overrides=ov, symmetry="asymmetric"
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_gradcheck_wye_neutral_load_per_phase_pq():
    # WYE load on a node WITH Phase.N -> the 4-wire incidence [I | -1].
    load = Load(id=30, node=2, phases=ABC, p_nom_w=4500.0, q_nom_var=900.0)
    grid = _grid(load, ABCN)

    p = torch.tensor([2000.0, 1500.0, 1000.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([400.0, 300.0, 200.0], dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        ov = {
            ("load", 30, "p_nom_per_phase_w"): p,
            ("load", 30, "q_nom_per_phase_var"): q,
        }
        return solve_power_flow(
            grid, slack="ideal", dtype=CDT, param_overrides=ov, symmetry="asymmetric"
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)
