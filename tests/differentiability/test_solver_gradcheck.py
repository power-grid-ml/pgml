"""Solver-only differentiability: gradcheck of v w.r.t. Y, I and v_fixed."""

from __future__ import annotations

import torch

from pgml.solver import solve_harmonic

torch.manual_seed(1)


def _well_conditioned_y(n, dtype):
    """A diagonally dominant complex matrix (invertible, stable gradients)."""
    a = torch.randn(n, n, dtype=dtype)
    a = a + (n * 1.0) * torch.eye(n, dtype=dtype)
    return a


def test_gradcheck_norton_wrt_y_and_i():
    n = 3
    y = _well_conditioned_y(n, torch.complex128).requires_grad_(True)
    i = torch.randn(n, dtype=torch.complex128, requires_grad=True)

    def fn(y, i):
        return solve_harmonic(y, i)

    assert torch.autograd.gradcheck(fn, (y, i), eps=1e-6, atol=1e-6)


def test_gradcheck_batched_norton():
    h, n = 2, 3
    y = torch.stack([_well_conditioned_y(n, torch.complex128) for _ in range(h)])
    y = y.requires_grad_(True)
    i = torch.randn(h, n, dtype=torch.complex128, requires_grad=True)

    def fn(y, i):
        return solve_harmonic(y, i)

    assert torch.autograd.gradcheck(fn, (y, i), eps=1e-6, atol=1e-6)


def test_gradcheck_ideal_slack_wrt_y_i_vfixed():
    n = 4
    y = _well_conditioned_y(n, torch.complex128).requires_grad_(True)
    i = torch.randn(n, dtype=torch.complex128, requires_grad=True)
    fixed_rows = torch.tensor([0, 2], dtype=torch.int64)
    v_fixed = torch.randn(2, dtype=torch.complex128, requires_grad=True)

    def fn(y, i, v_fixed):
        return solve_harmonic(y, i, fixed_rows=fixed_rows, v_fixed=v_fixed)

    assert torch.autograd.gradcheck(fn, (y, i, v_fixed), eps=1e-6, atol=1e-6)


def test_finite_difference_solver_spot_check():
    n = 3
    y = _well_conditioned_y(n, torch.complex128)
    i = torch.randn(n, dtype=torch.complex128)

    y_leaf = y.clone().requires_grad_(True)
    out = solve_harmonic(y_leaf, i).abs().sum()
    out.backward()
    g = y_leaf.grad

    # Perturb one real entry of Y and central-difference |v|.sum().
    h = 1e-7
    idx = (0, 1)

    def loss(yp):
        return float(solve_harmonic(yp, i).abs().sum())

    yp = y.clone()
    yp[idx] = yp[idx] + h
    lp = loss(yp)
    ym = y.clone()
    ym[idx] = ym[idx] - h
    lm = loss(ym)
    fd = (lp - lm) / (2 * h)
    # Gradient of a real loss w.r.t. the real part of Y[idx].
    analytic = float(g[idx].real)
    assert abs(analytic - fd) < 1e-3 * max(1.0, abs(fd))
