"""Differentiability gate for the MIXED-precision solve path.

``precision="mixed"`` factors the system at complex64 and refines the solution against
residuals formed at complex128. Two things must hold for gradients:

- the LINEAR solve (:func:`pgml.solver.solve_harmonic` and the factored solve behind it)
  carries the EXACT linear-solve adjoint, not a derivative of the refinement iteration:
  differentiating the single-precision steps would push the gradient through their
  rounding, and a float64 ``gradcheck`` measures exactly that;
- the NONLINEAR solve is unaffected by construction — its forward runs under
  ``no_grad`` and the implicit-function-theorem backward works at the working precision —
  so a mixed-precision power flow must produce the same gradients as a full-precision one.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_harmonic, solve_harmonic_flow, solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored

CDT = torch.complex128
torch.manual_seed(0)


def _well_conditioned_y(n: int) -> torch.Tensor:
    a = torch.randn(n, n, dtype=CDT)
    return a + (n * 1.0) * torch.eye(n, dtype=CDT)


def _two_bus(r, ind, p, q) -> Grid:
    """Single-phase two-bus grid; the line R/L and the load P/Q may be tensors."""
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


# --------------------------------------------------------------------------- #
# the linear solve
# --------------------------------------------------------------------------- #
def test_gradcheck_mixed_norton_wrt_y_and_i():
    n = 4
    y = _well_conditioned_y(n).requires_grad_(True)
    i = torch.randn(n, dtype=CDT, requires_grad=True)

    def fn(y, i):
        return solve_harmonic(y, i, precision="mixed")

    assert torch.autograd.gradcheck(fn, (y, i), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_gradcheck_mixed_ideal_slack():
    n = 5
    y = _well_conditioned_y(n).requires_grad_(True)
    i = torch.randn(n, dtype=CDT, requires_grad=True)
    fixed_rows = torch.tensor([0, 2], dtype=torch.int64)
    v_fixed = torch.randn(2, dtype=CDT, requires_grad=True)

    def fn(y, i, v_fixed):
        return solve_harmonic(
            y, i, fixed_rows=fixed_rows, v_fixed=v_fixed, precision="mixed"
        )

    assert torch.autograd.gradcheck(fn, (y, i, v_fixed), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_gradcheck_mixed_sparse_backend():
    """The refined solve is backend-agnostic: SuperLU factors, complex128 residuals."""
    n = 6
    y = _well_conditioned_y(n).requires_grad_(True)
    i = torch.randn(n, dtype=CDT, requires_grad=True)

    def fn(y, i):
        return solve_factored(
            lu_factor_system(y, backend="sparse", precision="mixed"), i
        )

    assert torch.autograd.gradcheck(fn, (y, i), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_mixed_gradient_matches_full_precision():
    """The analytic adjoint is the same operator, so the gradients agree tightly."""
    n = 5
    y0 = _well_conditioned_y(n)
    i0 = torch.randn(n, dtype=CDT)
    grads = []
    for precision in ("full", "mixed"):
        y = y0.clone().requires_grad_(True)
        i = i0.clone().requires_grad_(True)
        solve_harmonic(y, i, precision=precision).abs().sum().backward()
        grads.append((y.grad.clone(), i.grad.clone()))
    for g_full, g_mixed in zip(*grads):
        assert torch.allclose(g_full, g_mixed, rtol=1e-7, atol=1e-10)


# --------------------------------------------------------------------------- #
# the nonlinear solve (implicit-function-theorem backward)
# --------------------------------------------------------------------------- #
def test_gradcheck_mixed_power_flow_line_rl():
    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-6]], dtype=torch.float64, requires_grad=True)

    def fn(r, ind):
        return solve_power_flow(
            _two_bus(r, ind, 2000.0, 500.0),
            slack="ideal",
            dtype=CDT,
            precision="mixed",
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r, ind), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_mixed_power_flow_load_pq():
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        return solve_power_flow(
            _two_bus([[1e-3]], [[1e-6]], p, q),
            slack="ideal",
            dtype=CDT,
            precision="mixed",
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_mixed_power_flow_gradients_match_full_precision():
    """Mixed precision changes the forward's arithmetic, not the gradient it carries."""
    grads = []
    for precision in ("full", "mixed"):
        r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
        res = solve_power_flow(
            _two_bus(r, [[1e-6]], 2000.0, 500.0),
            slack="ideal",
            dtype=CDT,
            precision=precision,
        )
        res.v.abs().sum().backward()
        grads.append(r.grad.clone())
    assert torch.allclose(grads[0], grads[1], rtol=1e-6, atol=1e-9)


def test_gradcheck_mixed_harmonic_flow():
    """Every per-order solve is refined too, so the harmonic gradient holds."""
    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)

    def fn(r):
        return solve_harmonic_flow(
            _two_bus(r, [[1e-6]], 2000.0, 500.0),
            [1, 5],
            dtype=CDT,
            precision="mixed",
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r,), eps=1e-6, atol=1e-5, rtol=1e-3)
