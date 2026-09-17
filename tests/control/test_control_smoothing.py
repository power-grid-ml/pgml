"""The capability clamp's ``smoothing`` is a fraction of the inverter rating.

Single-phase devices throughout, so element and device quantities coincide.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.assembly._control import resolve_injection_power
from pgml.schemas.grid_schema import (
    ConstantReactivePowerControl,
    Generator,
    Grid,
    Line,
    Node,
    Phase,
    Source,
)
from pgml.solver import solve_power_flow

RDT = torch.float64
CDT = torch.complex128
A = (Phase.A,)
S_RATED = 20_000.0
P_GEN = 12_000.0
Q_MAX = math.sqrt(S_RATED**2 - P_GEN**2)  # 16 kvar
SMOOTHING = 0.05


def _q(q_request, *, s_rated=S_RATED, p=P_GEN, smoothing=SMOOTHING):
    control = ConstantReactivePowerControl(
        s_rated_va=s_rated, smoothing=smoothing, q_var=q_request
    )
    _, q = resolve_injection_power(
        control,
        torch.as_tensor([p], dtype=RDT),
        torch.ones((1, 1), dtype=RDT),
        rdt=RDT,
        device=None,
    )
    return q.reshape(())


def test_transition_half_width_is_a_fraction_of_the_rating():
    """At the corner a softplus clamp of half-width ``w`` falls short by ``w ln 2``."""
    width = SMOOTHING * S_RATED  # 1 kVA
    # (The active-power clamp, eight widths away, moves the limit by under 1 var.)
    assert float(Q_MAX - _q(Q_MAX)) == pytest.approx(width * math.log(2.0), rel=1e-3)
    # Several widths inside the limit the clamp is inactive, several outside it holds.
    assert float(_q(Q_MAX - 10 * width)) == pytest.approx(Q_MAX - 10 * width, abs=1.0)
    assert float(_q(Q_MAX + 10 * width)) == pytest.approx(Q_MAX, abs=1.0)
    # Inside the transition the slope is strictly between 0 and 1.
    q_req = torch.tensor(Q_MAX, dtype=RDT, requires_grad=True)
    (slope,) = torch.autograd.grad(_q(q_req), q_req)
    assert float(slope) == pytest.approx(0.5, abs=1e-3)


def test_soft_clamp_is_invariant_to_the_power_scale():
    """The same device ten times larger saturates the same way in per unit."""
    small = _q(0.98 * Q_MAX) / S_RATED
    large = _q(9.8 * Q_MAX, s_rated=10 * S_RATED, p=10 * P_GEN) / (10 * S_RATED)
    assert float(small) == pytest.approx(float(large), rel=1e-12)


def test_zero_smoothing_is_the_hard_clamp():
    assert float(_q(2 * Q_MAX, smoothing=0.0)) == pytest.approx(Q_MAX, rel=1e-12)
    assert float(_q(0.5 * Q_MAX, smoothing=0.0)) == pytest.approx(0.5 * Q_MAX)


def _grid(q_request, s_rated):
    line = Line(
        id=0,
        from_node=0,
        to_node=1,
        from_phases=A,
        to_phases=A,
        length_m=100.0,
        series_resistance_ohm_per_m=[[3e-4]],
        series_inductance_h_per_m=[[1e-6]],
        shunt_capacitance_f_per_m=[[0.0]],
    )
    source = Source(
        id=0,
        node=0,
        phases=A,
        u_ref_v=(230.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[0.05]],
        inductance_h=[[2e-4]],
    )
    control = ConstantReactivePowerControl(
        s_rated_va=s_rated, smoothing=SMOOTHING, q_var=q_request
    )
    gen = Generator(id=2, node=1, phases=A, p_nom_w=P_GEN, control=control)
    return Grid(
        nodes=[Node(id=i, u_rated_v=230.0, phases=A) for i in range(2)],
        branches=[line],
        appliances=[source, gen],
    )


def test_gradcheck_through_a_solve_inside_the_clamp_transition():
    """``dV*/d(q_request, s_rated)`` with the request on the limit itself."""
    q_request = torch.tensor(Q_MAX, dtype=RDT, requires_grad=True)
    s_rated = torch.tensor(S_RATED, dtype=RDT, requires_grad=True)

    def fn(q_request, s_rated):
        result = solve_power_flow(
            _grid(q_request, s_rated),
            method="newton",
            dtype=CDT,
            tol=1e-12,
            tol_update_pu=1e-12,
        )
        return torch.view_as_real(result.v)

    assert torch.autograd.gradcheck(
        fn, (q_request, s_rated), eps=1e-2, atol=1e-8, rtol=1e-5
    )

    # On the limit the soft clamp passes half of the request's sensitivity.
    def sensitivity(q_value):
        q_leaf = torch.tensor(q_value, dtype=RDT, requires_grad=True)
        (grad,) = torch.autograd.grad(fn(q_leaf, s_rated.detach())[1, 1], q_leaf)
        return float(grad)

    ratio = sensitivity(Q_MAX) / sensitivity(Q_MAX - 10 * SMOOTHING * S_RATED)
    assert ratio == pytest.approx(0.5, abs=0.05)
