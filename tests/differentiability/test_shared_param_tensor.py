"""Regression: the IFT backward supports a shared/derived parameter tensor.

The float/tensor duality lets a single leaf drive several Grid fields. When one
leaf ``p`` feeds a load's active power AND — via a derived expression
``q = p * k`` — its reactive power, the captured "parameter tensor" for ``q`` is
a NON-leaf node whose history is shared with the outer autograd tape. The IFT
parameter-vjp must keep that shared graph alive (``retain_graph=True``); freeing
it broke the FIRST ``.backward()`` with "backward through the graph a second
time". These checks pin the guarantee: a shared-history parameter tensor
backpropagates on the first pass, reaches the true leaf, and matches a finite
difference / ``gradcheck``.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_power_flow

CDT = torch.complex128
_Q_OVER_P = 0.3  # power factor coupling: q_nom_var = _Q_OVER_P * p_nom_w
torch.manual_seed(0)


def _two_bus(p, q, *, r=None, ind=None) -> Grid:
    """Single-phase two-bus grid; load P and Q may be (shared) tensors."""
    r = [[1e-3]] if r is None else r
    ind = [[1e-6]] if ind is None else ind
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=100.0,
                series_resistance_ohm_per_m=r,
                series_inductance_h_per_m=ind,
                shunt_capacitance_f_per_m=[[1e-9]],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1e-3]],
            ),
            Load(id=30, node=2, phases=(Phase.A,), p_nom_w=p, q_nom_var=q),
        ],
    )


def test_shared_leaf_backward_first_pass_succeeds():
    """One leaf feeds p_nom_w AND (via q = p * k) q_nom_var: the first backward
    must succeed and deliver a finite, nonzero gradient to the leaf."""
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = p * _Q_OVER_P  # derived, non-leaf; shares history with the outer tape

    res = solve_power_flow(_two_bus(p, q), slack="ideal", dtype=CDT)
    loss = res.v.abs().sum()
    loss.backward()  # would raise "backward through the graph a second time" pre-fix

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().item() > 0.0


def test_shared_leaf_grad_matches_finite_difference():
    """The IFT gradient of the shared leaf equals a central difference that lets
    BOTH the load's P and Q track the leaf (total derivative)."""

    def vsum(p_val: float) -> float:
        g = _two_bus(p_val, _Q_OVER_P * p_val)
        return float(solve_power_flow(g, slack="ideal", dtype=CDT).v.real.sum())

    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = p * _Q_OVER_P
    out = solve_power_flow(_two_bus(p, q), slack="ideal", dtype=CDT).v.real.sum()
    out.backward()
    analytic = float(p.grad)

    h = 1e-1
    fd = (vsum(2000.0 + h) - vsum(2000.0 - h)) / (2 * h)
    assert analytic == analytic  # not NaN
    assert abs(analytic - fd) <= 1e-3 * max(1.0, abs(fd))


def test_shared_leaf_gradcheck():
    """float64 gradcheck with the reactive power derived from the same leaf."""
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)

    def fn(p):
        q = p * _Q_OVER_P
        return solve_power_flow(_two_bus(p, q), slack="ideal", dtype=CDT).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p,), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_shared_leaf_across_network_and_device():
    """A leaf shared across a network param (line R) and a device param (load Q)
    still backpropagates cleanly on the first pass."""
    alpha = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    r = alpha * torch.tensor([[1e-3]], dtype=torch.float64)  # derived network param
    q = alpha * torch.tensor(600.0, dtype=torch.float64)  # derived device param
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)

    res = solve_power_flow(_two_bus(p, q, r=r), slack="ideal", dtype=CDT)
    res.v.abs().sum().backward()

    for leaf in (alpha, p):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert alpha.grad.abs().item() > 0.0
