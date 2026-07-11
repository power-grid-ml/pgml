"""float64 gradcheck of the sparse factorization backend (the linear-solve adjoint).

The sparse backend routes ``solve_factored`` through ``_SparseSolveFn``, whose
backward is one solve with the conjugate-transposed factors plus the ``-λ·conj(V)``
outer-product for ``grad_Y``. Gradcheck verifies that adjoint against numerical
derivatives w.r.t. ``Y``, the RHS, and the ideal-slack ``v_fixed`` — and that the
end-to-end power-flow gradients are backend-independent (the IFT differentiates at
the SAME converged ``V*`` regardless of how the forward factored).
"""

from __future__ import annotations

import torch

from pgml.grids import synthetic_feeder
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored


def _spd_like(n: int) -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(1, n, n, dtype=torch.complex128) + n * torch.eye(
        n, dtype=torch.complex128
    )


def test_gradcheck_sparse_norton():
    n = 6
    y = _spd_like(n).requires_grad_(True)
    i = torch.randn(2, 1, n, dtype=torch.complex128).requires_grad_(True)

    def f(y_, i_):
        return solve_factored(lu_factor_system(y_, backend="sparse"), i_)

    assert torch.autograd.gradcheck(f, (y, i), eps=1e-6, atol=1e-8)


def test_gradcheck_sparse_ideal_slack():
    n = 6
    y = _spd_like(n).requires_grad_(True)
    i = torch.randn(2, 1, n, dtype=torch.complex128).requires_grad_(True)
    fixed_rows = torch.tensor([1, 4])
    v_fixed = torch.randn(2, dtype=torch.complex128).requires_grad_(True)

    def f(y_, i_, vf_):
        fac = lu_factor_system(y_, fixed_rows=fixed_rows, backend="sparse")
        return solve_factored(fac, i_, v_fixed=vf_)

    assert torch.autograd.gradcheck(f, (y, i, v_fixed), eps=1e-6, atol=1e-8)


def test_gradcheck_sparse_multi_factorization_batch():
    """Distinct factorizations along a leading (frequency) axis, scenario RHS."""
    n = 5
    torch.manual_seed(5)
    y = (
        torch.randn(3, n, n, dtype=torch.complex128)
        + n * torch.eye(n, dtype=torch.complex128)
    ).requires_grad_(True)
    i = torch.randn(2, 3, n, dtype=torch.complex128).requires_grad_(True)

    def f(y_, i_):
        return solve_factored(lu_factor_system(y_, backend="sparse"), i_)

    assert torch.autograd.gradcheck(f, (y, i), eps=1e-6, atol=1e-8)


def test_power_flow_gradients_backend_independent():
    """IFT parameter gradients are identical for dense and sparse forwards."""
    grid = synthetic_feeder(8)
    grads = {}
    for backend in ("dense", "sparse"):
        p = torch.tensor(2.0e5, dtype=torch.float64, requires_grad=True)
        load_id = next(a.id for a in grid.appliances if a.id >= 20000)
        res = solve_power_flow(
            grid, operating_point={load_id: {"p_w": p}}, linear_solver=backend
        )
        assert res.converged
        res.v.abs().sum().backward()
        grads[backend] = p.grad.clone()
    assert torch.allclose(grads["dense"], grads["sparse"], rtol=1e-9)
    assert float(grads["dense"].abs()) > 0.0
