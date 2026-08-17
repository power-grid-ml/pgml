"""CPU-vs-CUDA parity for the block-diagonal factorization backend (needs CUDA).

The block backend exists for CUDA: a merged ensemble's union has no sparse direct
solve on the GPU, so its only union alternative is a dense LU of ``Σ N`` rows. These
checks assert that the block path runs unchanged on CUDA — factors and index tensors
land on the input's device, the solved voltages honor the requested dtype, and the
result matches the CPU solve — for both slack modes, a scenario batch and
``complex64``.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_ybus
from pgml.grids import synthetic_feeder
from pgml.multigrid import merge_grids
from pgml.solver import prepare_power_flow, solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - CPU-only host
    pytest.skip("CUDA not available", allow_module_level=True)

CUDA = torch.device("cuda")


@pytest.fixture(scope="module")
def members():
    return [
        synthetic_feeder(6, n_feeders=2),
        synthetic_feeder(9, n_feeders=3),
        synthetic_feeder(6, n_feeders=2),
        synthetic_feeder(4, n_feeders=1),
        synthetic_feeder(12, n_feeders=4),
    ]


@pytest.fixture(scope="module")
def merged(members):
    return merge_grids(members)


def _op(grid, b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        a.id: {"p_w": float(a.p_nom_w) * (0.5 + torch.rand(b, generator=g))}
        for a in grid.appliances
        if a.id >= 20000
    }


def test_block_factor_lives_on_the_input_device(merged):
    y = assemble_ybus(merged.grid, [50.0]).Y.to(CUDA)
    fac = lu_factor_system(y, backend="block", block_rows=merged.block_rows())
    assert fac.backend == "block"
    for pos, lu, piv in fac.block.buckets:
        assert pos.device.type == lu.device.type == piv.device.type == "cuda"
        assert lu.dtype == y.dtype
    rhs = torch.randn(4, 1, y.shape[-1], dtype=y.dtype, device=CUDA)
    v = solve_factored(fac, rhs)
    assert v.device.type == "cuda" and v.dtype == y.dtype
    v_cpu = solve_factored(
        lu_factor_system(y.cpu(), backend="block", block_rows=merged.block_rows()),
        rhs.cpu(),
    )
    assert torch.allclose(v.cpu(), v_cpu, atol=1e-9 * float(v_cpu.abs().max()))


@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_block_power_flow_parity(merged, slack):
    rows = merged.block_rows()
    r_cpu = solve_power_flow(
        merged.grid, slack=slack, linear_solver="block", block_rows=rows
    )
    r_gpu = solve_power_flow(
        merged.grid,
        slack=slack,
        device=CUDA,
        linear_solver="block",
        block_rows=[r.to(CUDA) for r in rows],
    )
    assert r_cpu.converged and r_gpu.converged
    assert r_gpu.v.device.type == "cuda" and r_gpu.v.dtype == torch.complex128
    assert torch.allclose(r_cpu.v, r_gpu.v.cpu(), atol=1e-6)


def test_block_batched_operating_point_parity(merged, members):
    ops = [_op(g, 8, seed=i) for i, g in enumerate(members)]
    op = merged.operating_point(ops)
    rows = merged.block_rows()
    r_cpu = solve_power_flow(merged.grid, operating_point=op, linear_solver="dense")
    r_gpu = solve_power_flow(
        merged.grid,
        operating_point=op,
        device=CUDA,
        linear_solver="block",
        block_rows=rows,
    )
    assert r_gpu.converged and r_gpu.v.shape == (8, merged.n_rows)
    assert torch.allclose(r_cpu.v, r_gpu.v.cpu(), atol=1e-6)


def test_block_complex64_honors_dtype(merged):
    rows = merged.block_rows()
    system = prepare_power_flow(
        merged.grid,
        dtype=torch.complex64,
        device=CUDA,
        linear_solver="block",
        block_rows=rows,
    )
    for _, lu, _ in system.factorization.block.buckets:
        assert lu.dtype == torch.complex64 and lu.device.type == "cuda"
    res = solve_power_flow(
        merged.grid, dtype=torch.complex64, device=CUDA, system=system
    )
    assert res.converged and res.v.dtype == torch.complex64
    ref = solve_power_flow(merged.grid, linear_solver="block", block_rows=rows)
    rel = (
        (res.v.abs().cpu().to(torch.float64) - ref.v.abs()).abs() / ref.v.abs()
    ).max()
    assert float(rel) < 1e-4  # single-precision floor, not a parity claim


def test_block_gradients_flow_on_cuda(merged):
    """The IFT backward reaches a member leaf through the CUDA block forward."""
    load = next(a for a in merged.grid.appliances if a.id >= 20000)
    p = torch.tensor(2.0e5, dtype=torch.float64, device=CUDA, requires_grad=True)
    res = solve_power_flow(
        merged.grid,
        device=CUDA,
        operating_point={load.id: {"p_w": p}},
        linear_solver="block",
        block_rows=merged.block_rows(),
    )
    res.v.abs().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert float(p.grad.abs()) > 0.0
