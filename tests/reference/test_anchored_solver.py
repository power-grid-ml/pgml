"""solve_anchored vs an independent dense WLS oracle.

The oracle forms the normal equations of the stated objective directly on the free rows
(with the hard-fixed slack folded into the right-hand sides), so it shares no code with the
reduced-correction implementation. The measurements are deliberately INCONSISTENT with the
physics right-hand side: at a consistent optimum every residual vanishes and a sign or
conjugation error would go unnoticed, whereas the inconsistent trade-off pins the exact
weighted-least-squares algebra.
"""

from __future__ import annotations

import pytest
import torch

from pgml.errors import InputError
from pgml.solver import solve_anchored, solve_harmonic

torch.manual_seed(0)

N, B, K = 8, 3, 4


def _problem(dtype):
    """A well-conditioned random anchored system with some rows left unanchored."""
    rdt = torch.empty(0, dtype=dtype).real.dtype
    y = torch.randn(N, N, dtype=dtype) + 3.0 * torch.eye(N, dtype=dtype)
    i = torch.randn(B, N, dtype=dtype)
    row_weight = torch.rand(B, N, dtype=rdt)
    row_weight[:, ::3] = 0.0
    row_target = torch.randn(B, N, dtype=dtype)
    op = torch.randn(K, N, dtype=dtype)
    op_weight = torch.rand(B, K, dtype=rdt)
    op_target = torch.randn(B, K, dtype=dtype)
    fixed_rows = torch.tensor([0], dtype=torch.int64)
    v_fixed = torch.randn(1, dtype=dtype)
    return y, i, row_weight, row_target, op, op_weight, op_target, fixed_rows, v_fixed


def _oracle(
    y, i, row_weight, row_target, op, op_weight, op_target, fixed_rows, v_fixed
):
    """Per-sample normal equations of the anchored WLS objective on the free rows."""
    n = y.shape[-1]
    keep = torch.ones(n, dtype=torch.bool)
    if fixed_rows is not None:
        keep[fixed_rows] = False
    free = torch.nonzero(keep).squeeze(-1)
    y_ff = y[free][:, free]
    out = torch.zeros(i.shape[0], n, dtype=y.dtype)
    for b in range(i.shape[0]):
        rhs = i[b, free]
        op_t = op_target[b] if op_target is not None else torch.zeros(K, dtype=y.dtype)
        if fixed_rows is not None:
            vf = v_fixed.to(y.dtype)
            rhs = rhs - y[free][:, fixed_rows] @ vf
            op_t = op_t - op[:, fixed_rows] @ vf
            out[b, fixed_rows] = vf
        w_r = row_weight[b, free].to(y.dtype)
        w_o = op_weight[b].to(y.dtype)
        lhs = (
            y_ff.conj().T @ y_ff
            + torch.diag(w_r)
            + op[:, free].conj().T @ torch.diag(w_o) @ op[:, free]
        )
        rhs_n = (
            y_ff.conj().T @ rhs
            + w_r * row_target[b, free]
            + op[:, free].conj().T @ (w_o * op_t)
        )
        out[b, free] = torch.linalg.solve(lhs, rhs_n)
    return out


def test_inconsistent_wls_matches_oracle_ideal_slack():
    args = _problem(torch.complex128)
    got = solve_anchored(
        args[0],
        args[1],
        row_weight=args[2],
        row_target=args[3],
        op=args[4],
        op_weight=args[5],
        op_target=args[6],
        fixed_rows=args[7],
        v_fixed=args[8],
    )
    assert torch.allclose(got, _oracle(*args), atol=1e-12)


def test_inconsistent_wls_matches_oracle_norton():
    y, i, row_weight, row_target, *_ = _problem(torch.complex128)
    got = solve_anchored(y, i, row_weight=row_weight, row_target=row_target)
    ref = _oracle(
        y,
        i,
        row_weight,
        row_target,
        torch.zeros(K, N, dtype=y.dtype),
        torch.zeros(B, K),
        None,
        None,
        None,
    )
    assert torch.allclose(got, ref, atol=1e-12)


def test_no_anchors_reproduces_solve_harmonic():
    y, i, *_, fixed_rows, v_fixed = _problem(torch.complex128)
    got = solve_anchored(y, i, fixed_rows=fixed_rows, v_fixed=v_fixed)
    ref = solve_harmonic(y.unsqueeze(0), i, fixed_rows=fixed_rows, v_fixed=v_fixed)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_dtype_honored(dtype):
    """complex64 and complex128 inputs both solve, and the output keeps the input dtype."""
    args = _problem(dtype)
    got = solve_anchored(
        args[0],
        args[1],
        row_weight=args[2],
        row_target=args[3],
        op=args[4],
        op_weight=args[5],
        op_target=args[6],
        fixed_rows=args[7],
        v_fixed=args[8],
    )
    assert got.dtype == dtype
    ref = _oracle(*[a.to(torch.complex128) if a.is_complex() else a for a in args])
    assert torch.allclose(got.to(torch.complex128), ref, atol=1e-4)


def test_unbatched_rhs_returns_vector():
    y, i, row_weight, row_target, *_ = _problem(torch.complex128)
    got = solve_anchored(y, i[0], row_weight=row_weight[:1], row_target=row_target[:1])
    assert got.shape == (N,)


def test_batched_operator_rejected():
    y, i, row_weight, row_target, *_ = _problem(torch.complex128)
    with pytest.raises(InputError):
        solve_anchored(y.unsqueeze(0), i, row_weight=row_weight, row_target=row_target)
