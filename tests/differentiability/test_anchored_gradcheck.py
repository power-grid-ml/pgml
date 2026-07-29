"""Solver-only differentiability: gradcheck of the anchored solve w.r.t. every complex input."""

from __future__ import annotations

import torch

from pgml.solver import solve_anchored

torch.manual_seed(2)


def _well_conditioned_y(n, dtype):
    """A diagonally dominant complex matrix (invertible, stable gradients)."""
    a = torch.randn(n, n, dtype=dtype)
    a = a + (n * 1.0) * torch.eye(n, dtype=dtype)
    return a


def test_gradcheck_row_anchor_norton_wrt_y_i_target():
    n = 4
    y = _well_conditioned_y(n, torch.complex128).requires_grad_(True)
    i = torch.randn(1, n, dtype=torch.complex128, requires_grad=True)
    target = torch.randn(1, n, dtype=torch.complex128, requires_grad=True)
    weight = torch.rand(1, n, dtype=torch.float64)

    def fn(y, i, target):
        return solve_anchored(y, i, row_weight=weight, row_target=target)

    assert torch.autograd.gradcheck(fn, (y, i, target), eps=1e-6, atol=1e-6)


def test_gradcheck_full_anchors_ideal_slack():
    """Gradients through both anchor terms and the hard slack: y, i, targets, op, v_fixed."""
    n, k = 4, 2
    y = _well_conditioned_y(n, torch.complex128).requires_grad_(True)
    i = torch.randn(1, n, dtype=torch.complex128, requires_grad=True)
    row_target = torch.randn(1, n, dtype=torch.complex128, requires_grad=True)
    op = torch.randn(k, n, dtype=torch.complex128, requires_grad=True)
    op_target = torch.randn(1, k, dtype=torch.complex128, requires_grad=True)
    v_fixed = torch.randn(1, dtype=torch.complex128, requires_grad=True)
    row_weight = torch.rand(1, n, dtype=torch.float64)
    op_weight = torch.rand(1, k, dtype=torch.float64)
    fixed_rows = torch.tensor([0], dtype=torch.int64)

    def fn(y, i, row_target, op, op_target, v_fixed):
        return solve_anchored(
            y,
            i,
            row_weight=row_weight,
            row_target=row_target,
            op=op,
            op_weight=op_weight,
            op_target=op_target,
            fixed_rows=fixed_rows,
            v_fixed=v_fixed,
        )

    assert torch.autograd.gradcheck(
        fn, (y, i, row_target, op, op_target, v_fixed), eps=1e-6, atol=1e-6
    )
