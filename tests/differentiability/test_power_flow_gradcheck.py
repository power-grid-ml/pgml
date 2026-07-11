"""Differentiability gate for the nonlinear power flow (IMPLICIT FUNCTION THEOREM).

``solve_power_flow`` runs a current-injection fixed point under ``no_grad`` and
attaches the analytic gradient via a real-coordinate IFT adjoint. These checks
gradcheck ``v`` (float64) through that path w.r.t.

- a line R/L tensor (network params),
- a load's P/Q tensor stored DIRECTLY in the Grid (float/tensor duality),
- the slack voltage (u_ref),

for both slack modes, plus a finite-difference spot check and a batched
(scenario) gradcheck. No ``param_overrides`` needed: the tensors live in the Grid.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_power_flow

CDT = torch.complex128
torch.manual_seed(0)


def _two_bus(r, ind, p, q, *, u_ref=(230.0,)) -> Grid:
    """Single-phase two-bus grid; line R/L, load P/Q and slack u_ref may be tensors."""
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
                u_ref_v=u_ref,
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1e-3]],
            ),
            Load(id=30, node=2, phases=(Phase.A,), p_nom_w=p, q_nom_var=q),
        ],
    )


def test_gradcheck_line_rl_ideal():
    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-6]], dtype=torch.float64, requires_grad=True)

    def fn(r, ind):
        return solve_power_flow(
            _two_bus(r, ind, 2000.0, 500.0), slack="ideal", dtype=CDT
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r, ind), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_line_rl_norton():
    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-6]], dtype=torch.float64, requires_grad=True)

    def fn(r, ind):
        return solve_power_flow(
            _two_bus(r, ind, 2000.0, 500.0), slack="norton", dtype=CDT
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r, ind), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_load_pq_ideal_tensor_duality():
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        return solve_power_flow(
            _two_bus([[1e-3]], [[1e-6]], p, q), slack="ideal", dtype=CDT
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_gradcheck_load_pq_norton_tensor_duality():
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        return solve_power_flow(
            _two_bus([[1e-3]], [[1e-6]], p, q), slack="norton", dtype=CDT
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_gradcheck_line_rl_and_load_pq_together():
    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-6]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)

    def fn(r, ind, p, q):
        return solve_power_flow(
            _two_bus(r, ind, p, q), slack="ideal", dtype=CDT
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r, ind, p, q), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_slack_voltage():
    u_ref = torch.tensor([230.0], dtype=torch.float64, requires_grad=True)

    def fn(u_ref):
        return solve_power_flow(
            _two_bus([[1e-3]], [[1e-6]], 2000.0, 500.0, u_ref=u_ref),
            slack="ideal",
            dtype=CDT,
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (u_ref,), eps=1e-4, atol=1e-4, rtol=1e-3)


def test_gradcheck_batched_scenarios():
    """A scenario batch of P/Q solves and gradchecks in one call; v is [S, N]."""
    p = torch.tensor([1500.0, 2500.0, 3500.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([300.0, 600.0, 900.0], dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        return solve_power_flow(
            _two_bus([[1e-3]], [[1e-6]], p, q), slack="ideal", dtype=CDT
        ).v

    out = fn(p, q)
    assert out.shape == (3, 2)
    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_backward_jvp_fallback_matches_dense(monkeypatch):
    """The O(B) column-by-column JVP state Jacobian (used past the memory threshold)
    yields the SAME gradient as the dense [B,2N,B,2N] path — incl. batched device
    params, the batch source that off-diagonal-free per-element blocks get wrong."""
    import pgml.solver.power_flow as pf_mod

    p = torch.tensor([1500.0, 2500.0, 3500.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([300.0, 600.0, 900.0], dtype=torch.float64, requires_grad=True)

    def grad_pq():
        p.grad = q.grad = None
        solve_power_flow(
            _two_bus([[1e-3]], [[1e-6]], p, q), slack="ideal", dtype=CDT
        ).v.abs().sum().backward()
        return p.grad.clone(), q.grad.clone()

    monkeypatch.setattr(pf_mod, "_IFT_DENSE_JAC_MAX_ELEMS", 10**9)  # dense path
    dp_dense, dq_dense = grad_pq()
    monkeypatch.setattr(pf_mod, "_IFT_DENSE_JAC_MAX_ELEMS", 1)  # force JVP path
    dp_jvp, dq_jvp = grad_pq()
    assert torch.allclose(dp_dense, dp_jvp, rtol=1e-10, atol=1e-12)
    assert torch.allclose(dq_dense, dq_jvp, rtol=1e-10, atol=1e-12)


def test_finite_difference_spot_check_load_p():
    """Back up the IFT gradient with a central difference on the load active power."""

    def vsum(p_val):
        g = _two_bus([[1e-3]], [[1e-6]], p_val, 500.0)
        return float(solve_power_flow(g, slack="ideal", dtype=CDT).v.real.sum())

    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    out = solve_power_flow(
        _two_bus([[1e-3]], [[1e-6]], p, 500.0), slack="ideal", dtype=CDT
    ).v.real.sum()
    out.backward()
    analytic = float(p.grad)

    h = 1e-1
    fd = (vsum(2000.0 + h) - vsum(2000.0 - h)) / (2 * h)
    assert abs(analytic - fd) < 1e-5 * max(1.0, abs(fd))


def test_backward_reaches_all_leaves():
    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-6]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)
    res = solve_power_flow(_two_bus(r, ind, p, q), slack="ideal", dtype=CDT)
    res.v.abs().sum().backward()
    for leaf in (r, ind, p, q):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert r.grad.abs().sum() > 0
    assert p.grad.abs().sum() > 0


def test_gradcheck_operating_point():
    """A differentiable ``operating_point`` (per-appliance P/Q overrides) receives
    gradients through the IFT backward — including DERIVED expressions, so a
    neural network's output can drive the load powers directly."""
    from torch.autograd import gradcheck

    def f(p, q):
        g = _two_bus([[1e-3]], [[1e-6]], 2000.0, 500.0)
        r = solve_power_flow(
            g, slack="ideal", dtype=CDT, operating_point={30: {"p_w": p, "q_var": q}}
        )
        return r.v.real.sum() + r.v.imag.sum()

    p = torch.tensor(2100.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(400.0, dtype=torch.float64, requires_grad=True)
    assert gradcheck(f, (p, q), eps=1e-4, atol=1e-6, rtol=1e-4)


def test_backward_operating_point_derived_batched():
    """Batched operating point derived from a shared leaf (theta -> p, q): the
    gradient reaches theta once, without double counting."""
    theta = torch.full((3,), 1.1, dtype=torch.float64, requires_grad=True)
    g = _two_bus([[1e-3]], [[1e-6]], 2000.0, 500.0)
    op = {30: {"p_w": 2000.0 * theta, "q_var": 500.0 * theta}}
    res = solve_power_flow(g, slack="ideal", dtype=CDT, operating_point=op)
    assert res.v.shape[0] == 3
    res.v.abs().sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert theta.grad.abs().min() > 0


def test_backward_operating_point_neural_network():
    """Deep-graph leaf resolution: an nn.Sequential's output drives the load
    powers; gradients must reach EVERY network parameter (regression test for
    grad_fn-wrapper id reuse silently truncating the leaf walk)."""
    mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).double()
    theta = 1.0 + 0.1 * torch.tanh(mlp(torch.rand(4, dtype=torch.float64)))
    g = _two_bus([[1e-3]], [[1e-6]], 2000.0, 500.0)
    op = {30: {"p_w": 2000.0 * theta[0], "q_var": 500.0 * theta[1]}}
    res = solve_power_flow(g, slack="ideal", dtype=CDT, operating_point=op)
    res.v.abs().sum().backward()
    for name, p in mlp.named_parameters():
        assert p.grad is not None, f"no grad reached {name}"
        assert torch.isfinite(p.grad).all()
    assert sum(float(p.grad.abs().sum()) for p in mlp.parameters()) > 0
