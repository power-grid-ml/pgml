"""AnchoredSystem (factor-once anchored solve) vs solve_anchored on identical inputs.

The two implementations optimize the same identity-floored objective through different
algebra (dense correction build vs the push-through identity on the anchored rows), so
their results must agree to a few ulps of the conditioning. The measurements are
deliberately INCONSISTENT with the physics right-hand side (a consistent optimum has
vanishing residuals and would hide sign/conjugation errors), and the batches carry
HETEROGENEOUS anchor patterns per sample — the padding path of the row selection.
"""

from __future__ import annotations

import pytest
import torch

from pgml.errors import InputError
from pgml.solver import AnchoredSystem, solve_anchored

torch.manual_seed(0)

N, B, K = 9, 4, 5


def _problem(dtype):
    rdt = torch.empty(0, dtype=dtype).real.dtype
    y = torch.randn(N, N, dtype=dtype) + 3.0 * torch.eye(N, dtype=dtype)
    i = torch.randn(B, N, dtype=dtype)
    row_weight = torch.rand(B, N, dtype=rdt)
    # heterogeneous per-sample anchor sets, incl. one sample with a single anchored row
    row_weight[:, ::3] = 0.0
    row_weight[0, :] = 0.0
    row_weight[0, 1] = 0.7
    row_target = torch.randn(B, N, dtype=dtype)
    op = torch.randn(K, N, dtype=dtype)
    op_weight = torch.rand(B, K, dtype=rdt)
    op_weight[1, :] = 0.0
    op_target = torch.randn(B, K, dtype=dtype)
    fixed_rows = torch.tensor([0], dtype=torch.int64)
    v_fixed = torch.randn(1, dtype=dtype)
    return y, i, row_weight, row_target, op, op_weight, op_target, fixed_rows, v_fixed


@pytest.mark.parametrize("dtype", [torch.complex128])
def test_matches_solve_anchored_full(dtype):
    """Row + channel anchors, hard slack, heterogeneous per-sample patterns."""
    y, i, wr, tr, op, wo, to, fixed, vf = _problem(dtype)
    ref = solve_anchored(
        y,
        i,
        row_weight=wr,
        row_target=tr,
        op=op,
        op_weight=wo,
        op_target=to,
        fixed_rows=fixed,
        v_fixed=vf,
    )
    sys_ = AnchoredSystem(y, op=op, fixed_rows=fixed)
    out = sys_.solve(
        i, row_weight=wr, row_target=tr, op_weight=wo, op_target=to, v_fixed=vf
    )
    assert torch.allclose(out, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("dtype", [torch.complex128])
def test_matches_solve_anchored_rows_only_norton(dtype):
    y, i, wr, tr, *_ = _problem(dtype)
    ref = solve_anchored(y, i, row_weight=wr, row_target=tr)
    out = AnchoredSystem(y).solve(i, row_weight=wr, row_target=tr)
    assert torch.allclose(out, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("dtype", [torch.complex128])
def test_no_anchors_is_plain_solve(dtype):
    y, i, *_, fixed, vf = _problem(dtype)
    ref = solve_anchored(y, i, fixed_rows=fixed, v_fixed=vf)
    out = AnchoredSystem(y, fixed_rows=fixed).solve(i, v_fixed=vf)
    assert torch.allclose(out, ref, rtol=1e-9, atol=1e-11)


def test_zero_weight_sample_reduces_to_plain():
    """A sample whose every anchor weight is zero must solve as if unanchored."""
    y, i, wr, tr, *_ = _problem(torch.complex128)
    wr = wr.clone()
    wr[2, :] = 0.0
    out = AnchoredSystem(y).solve(i, row_weight=wr, row_target=tr)
    plain = AnchoredSystem(y).solve(i[2:3])
    assert torch.allclose(out[2], plain[0], rtol=1e-9, atol=1e-11)


def test_unbatched_roundtrip():
    y, i, wr, tr, *_ = _problem(torch.complex128)
    out = AnchoredSystem(y).solve(i[0], row_weight=wr[0], row_target=tr[0])
    ref = solve_anchored(y, i[0], row_weight=wr[0], row_target=tr[0])
    assert out.shape == (N,)
    assert torch.allclose(out, ref, rtol=1e-9, atol=1e-11)


def test_refuses_operator_on_tape():
    y = torch.randn(N, N, dtype=torch.complex128) + 3.0 * torch.eye(
        N, dtype=torch.complex128
    )
    with pytest.raises(InputError):
        AnchoredSystem(y.clone().requires_grad_(True))


def test_gradcheck_wrt_rhs_and_targets():
    """Gradients flow through the cached solve w.r.t. i, targets and v_fixed."""
    y, _, wr, _, op, wo, _, fixed, _ = _problem(torch.complex128)
    sys_ = AnchoredSystem(y, op=op, fixed_rows=fixed)
    i = torch.randn(2, N, dtype=torch.complex128, requires_grad=True)
    tr = torch.randn(2, N, dtype=torch.complex128, requires_grad=True)
    to = torch.randn(2, K, dtype=torch.complex128, requires_grad=True)
    vf = torch.randn(1, dtype=torch.complex128, requires_grad=True)

    def fn(i, tr, to, vf):
        return sys_.solve(
            i,
            row_weight=wr[:2],
            row_target=tr,
            op_weight=wo[:2],
            op_target=to,
            v_fixed=vf,
        )

    assert torch.autograd.gradcheck(fn, (i, tr, to, vf), eps=1e-6, atol=1e-6)
