"""Differentiability gate for inverter control + storage on the IFT path.

The control law enters the power-flow residual ``F(V) = Y_eff V + I_device(V) - I_slack``
through ``I_device``, so the implicit-function-theorem backward (which autograd-
differentiates one residual evaluation at ``V*``) must produce correct gradients of the
solved voltage w.r.t. the control curve, the inverter rating, and a storage setpoint —
with no new adjoint. float64 gradcheck (Newton forward, smoothed clamp so the map is
C\\ :sup:`1`). See ``references/der_pv_storage_modeling.md`` §4.2-4.3.
"""

from __future__ import annotations


import torch

from pgml.schemas.grid_schema import (
    Characteristic,
    Generator,
    Grid,
    Line,
    Node,
    Phase,
    Source,
    Storage,
    VoltVarControl,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128
RDT = torch.float64
ABC = (Phase.A, Phase.B, Phase.C)


def _src():
    return Source(
        id=0,
        node=0,
        phases=ABC,
        u_ref_v=[230.94] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[2e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )


def _line():
    return Line(
        id=0,
        from_node=0,
        to_node=1,
        from_phases=ABC,
        to_phases=ABC,
        length_m=300.0,
        series_resistance_ohm_per_m=[
            [3e-4 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        series_inductance_h_per_m=[
            [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
    )


def _nodes():
    return [
        Node(id=0, u_rated_v=400.0, phases=ABC),
        Node(id=1, u_rated_v=400.0, phases=ABC),
    ]


def test_gradcheck_volt_var_curve():
    """dV*/d(Q(V) curve): the recoverable-slope claim — gradients flow into the curve.

    A single-segment curve over [0.9, 1.1] keeps the operating voltage strictly interior
    (no breakpoint crossing under perturbation), so the map is smooth in the curve values.
    """
    y = torch.tensor([1.0, -1.0], dtype=RDT, requires_grad=True)

    def fn(y):
        ctrl = VoltVarControl(
            s_rated_va=20000.0,
            smoothing=0.05,
            characteristic=Characteristic(x_values=[0.9, 1.1], y_values=y),
        )
        gen = Generator(id=2, node=1, phases=ABC, p_nom_w=6000.0, control=ctrl)
        grid = Grid(nodes=_nodes(), branches=[_line()], appliances=[_src(), gen])
        return solve_power_flow(grid, method="newton", dtype=CDT).v

    assert torch.autograd.gradcheck(fn, (y,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_inverter_rating():
    """dV*/d(s_rated): rating enters the available-VAr base of the Volt-VAr control."""
    s_rated = torch.tensor(20000.0, dtype=RDT, requires_grad=True)

    def fn(s_rated):
        ctrl = VoltVarControl(
            s_rated_va=s_rated,
            q_reference="available",
            smoothing=0.05,
            characteristic=Characteristic(x_values=[0.9, 1.1], y_values=[1.0, -1.0]),
        )
        gen = Generator(id=2, node=1, phases=ABC, p_nom_w=6000.0, control=ctrl)
        grid = Grid(nodes=_nodes(), branches=[_line()], appliances=[_src(), gen])
        return solve_power_flow(grid, method="newton", dtype=CDT).v

    assert torch.autograd.gradcheck(fn, (s_rated,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_storage_setpoint():
    """dV*/d(storage p_nom_w): a Storage signed setpoint is on the differentiable path."""
    p = torch.tensor(12000.0, dtype=RDT, requires_grad=True)

    def fn(p):
        st = Storage(id=2, node=1, phases=ABC, p_nom_w=p, energy_capacity_wh=1e5)
        grid = Grid(nodes=_nodes(), branches=[_line()], appliances=[_src(), st])
        return solve_power_flow(grid, dtype=CDT).v

    assert torch.autograd.gradcheck(fn, (p,), eps=1e-3, atol=1e-5, rtol=1e-3)


def test_gradcheck_line_param_with_control():
    """Grid-parameter gradients still flow with a control law in the residual."""
    r = torch.tensor(
        [[3e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
        dtype=RDT,
        requires_grad=True,
    )

    def fn(r):
        ctrl = VoltVarControl(
            s_rated_va=20000.0,
            smoothing=0.05,
            characteristic=Characteristic(x_values=[0.9, 1.1], y_values=[1.0, -1.0]),
        )
        gen = Generator(id=2, node=1, phases=ABC, p_nom_w=6000.0, control=ctrl)
        grid = Grid(nodes=_nodes(), branches=[_line()], appliances=[_src(), gen])
        overrides = {("line", 0, "series_resistance_ohm_per_m"): r}
        return solve_power_flow(
            grid, method="newton", dtype=CDT, param_overrides=overrides
        ).v

    assert torch.autograd.gradcheck(fn, (r,), eps=1e-6, atol=1e-5, rtol=1e-3)
