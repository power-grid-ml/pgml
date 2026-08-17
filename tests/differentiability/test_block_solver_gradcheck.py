"""float64 gradcheck of the block-diagonal factorization backend.

The block backend gathers each diagonal block out of ``Y``, factors the equal-sized
blocks with one batched ``torch.linalg.lu_factor`` and scatters the block solutions
back — all pure torch ops, so autograd carries the linear-solve adjoint (the
``lu_solve`` backward re-uses the SAME factors, transposed) without a hand-written
backward. Gradcheck verifies that against numerical derivatives at the linear-system
level (w.r.t. ``Y``, the right-hand side and the ideal-slack ``v_fixed``) and end to
end through a merged multi-grid ``solve_power_flow`` w.r.t. the MEMBER grids' own
parameter leaves.
"""

from __future__ import annotations

import torch

from pgml.multigrid import merge_grids
from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored

CDT = torch.complex128
torch.manual_seed(0)

BLOCK_SIZES = (3, 3, 2)
BLOCK_ROWS = (torch.arange(0, 3), torch.arange(3, 6), torch.arange(6, 8))


def _block_diagonal(sizes=BLOCK_SIZES, *, lead: int = 1) -> torch.Tensor:
    """A well-conditioned block-diagonal ``[lead, N, N]`` complex system."""
    n = sum(sizes)
    mats = []
    for k in range(lead):
        y = torch.zeros(n, n, dtype=CDT)
        off = 0
        for s in sizes:
            blk = torch.randn(s, s, dtype=CDT) + (s + k) * torch.eye(s, dtype=CDT)
            y[off : off + s, off : off + s] = blk
            off += s
        mats.append(y)
    return torch.stack(mats)


def test_gradcheck_block_norton():
    y = _block_diagonal().requires_grad_(True)
    i = torch.randn(2, 1, 8, dtype=CDT).requires_grad_(True)

    def fn(y_, i_):
        return solve_factored(
            lu_factor_system(y_, backend="block", block_rows=BLOCK_ROWS), i_
        )

    assert torch.autograd.gradcheck(fn, (y, i), eps=1e-6, atol=1e-8)


def test_gradcheck_block_ideal_slack():
    y = _block_diagonal().requires_grad_(True)
    i = torch.randn(2, 1, 8, dtype=CDT).requires_grad_(True)
    fixed_rows = torch.tensor([0, 3, 6])  # one fixed row per block
    v_fixed = torch.randn(3, dtype=CDT).requires_grad_(True)

    def fn(y_, i_, vf_):
        fac = lu_factor_system(
            y_, fixed_rows=fixed_rows, backend="block", block_rows=BLOCK_ROWS
        )
        return solve_factored(fac, i_, v_fixed=vf_)

    assert torch.autograd.gradcheck(fn, (y, i, v_fixed), eps=1e-6, atol=1e-8)


def test_gradcheck_block_multi_factorization_batch():
    """Distinct factorizations along a leading (frequency) axis, scenario RHS."""
    y = _block_diagonal(lead=3).requires_grad_(True)
    i = torch.randn(2, 3, 8, dtype=CDT).requires_grad_(True)

    def fn(y_, i_):
        return solve_factored(
            lu_factor_system(y_, backend="block", block_rows=BLOCK_ROWS), i_
        )

    assert torch.autograd.gradcheck(fn, (y, i), eps=1e-6, atol=1e-8)


def _chain(n_bus: int, r, *, base: int = 0, p: float = 2000.0) -> Grid:
    """Single-phase radial chain (base + i node ids) with a tensor-capable line R."""
    nodes = [
        Node(id=base + i, u_rated_v=230.0, phases=(Phase.A,)) for i in range(n_bus)
    ]
    branches = [
        Line(
            id=base + 100 + i,
            from_node=base + i,
            to_node=base + i + 1,
            from_phases=(Phase.A,),
            to_phases=(Phase.A,),
            length_m=100.0,
            series_resistance_ohm_per_m=r,
            series_inductance_h_per_m=[[1.0e-6]],
            shunt_capacitance_f_per_m=[[1.0e-9]],
        )
        for i in range(n_bus - 1)
    ]
    appliances = [
        Source(
            id=base + 200,
            node=base,
            phases=(Phase.A,),
            u_ref_v=(230.0,),
            u_angle_deg=(0.0,),
            resistance_ohm=[[0.1]],
            inductance_h=[[1.0e-3]],
        )
    ]
    appliances += [
        Load(
            id=base + 300 + i,
            node=base + i,
            phases=(Phase.A,),
            p_nom_w=p,
            q_nom_var=500.0,
        )
        for i in range(1, n_bus)
    ]
    return Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=branches, appliances=appliances
    )


def test_gradcheck_merged_power_flow_member_leaves():
    """IFT gradients through a merged block-backend solve reach both members."""
    r_a = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    r_b = torch.tensor([[2.0e-3]], dtype=torch.float64, requires_grad=True)

    def fn(a, b):
        merged = merge_grids([_chain(2, a), _chain(3, b, base=1000)])
        return solve_power_flow(
            merged.grid,
            dtype=CDT,
            linear_solver="block",
            block_rows=merged.block_rows(),
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r_a, r_b), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_merged_power_flow_batched_operating_point():
    """A scenario-batched member load stays differentiable on the block path."""
    p = torch.tensor([2000.0, 3000.0], dtype=torch.float64, requires_grad=True)

    def fn(pp):
        merged = merge_grids([_chain(2, [[1.0e-3]]), _chain(3, [[2.0e-3]], base=1000)])
        op = merged.operating_point([{301: {"p_w": pp}}, None])
        return solve_power_flow(
            merged.grid,
            dtype=CDT,
            operating_point=op,
            linear_solver="block",
            block_rows=merged.block_rows(),
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_power_flow_gradients_backend_independent():
    """IFT parameter gradients are identical for the dense and block forwards."""
    grads = {}
    for backend in ("dense", "block"):
        p = torch.tensor(2.0e3, dtype=torch.float64, requires_grad=True)
        merged = merge_grids([_chain(3, [[1.0e-3]]), _chain(4, [[2.0e-3]], base=1000)])
        op = merged.operating_point([{301: {"p_w": p}}, None])
        kwargs = (
            {"linear_solver": "block", "block_rows": merged.block_rows()}
            if backend == "block"
            else {}
        )
        res = solve_power_flow(merged.grid, dtype=CDT, operating_point=op, **kwargs)
        assert res.converged
        res.v.abs().sum().backward()
        grads[backend] = p.grad.clone()
    assert torch.allclose(grads["dense"], grads["block"], rtol=1e-9)
    assert float(grads["dense"].abs()) > 0.0
