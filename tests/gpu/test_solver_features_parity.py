"""CPU-vs-CUDA parity for the solver performance features (skips without CUDA).

Covers: the ``linear_solver="auto"`` backend selection (CUDA must stay dense and
match the CPU sparse result), batched ``branch_states`` topology solves, the
prepared :class:`~pgml.solver.PowerFlowSystem` reuse, and the two formulations of the
convergence floor's row scale (read from the matrix's nonzeros on CPU, densely on CUDA).
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_network_ybus, node_phase_index
from pgml.grids import synthetic_feeder
from pgml.solver import prepare_power_flow, solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - CPU-only host
    pytest.skip("CUDA not available", allow_module_level=True)

CUDA = torch.device("cuda")


def _op(grid, b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        a.id: {"p_w": float(a.p_nom_w) * (0.5 + torch.rand(b, generator=g))}
        for a in grid.appliances
        if a.id >= 20000
    }


def test_auto_backend_is_dense_on_cuda():
    n = 600  # above the CPU sparse threshold
    y = (torch.eye(n, dtype=torch.complex128) * 2.0).cuda()
    assert lu_factor_system(y, backend="auto").backend == "dense"


def test_interleaved_shared_rhs_axis_parity():
    """CUDA preserves ``Y=[B,1,H,N,N]`` / ``I=[B,T,H,N]`` batch alignment."""
    b, steps, h, n = 2, 4, 3, 5
    torch.manual_seed(19)
    y = torch.randn(b, 1, h, n, n, dtype=torch.complex128)
    y = y + (n + 2) * torch.eye(n, dtype=torch.complex128)
    rhs = torch.randn(b, steps, h, n, dtype=torch.complex128)
    cpu = solve_factored(lu_factor_system(y, equilibrate="off"), rhs)
    gpu = solve_factored(lu_factor_system(y.to(CUDA), equilibrate="off"), rhs.to(CUDA))
    assert gpu.shape == rhs.shape and gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-12, atol=1e-12)


def test_power_flow_parity_large_grid_auto_backend():
    """CPU (sparse auto) and CUDA (dense auto) converge to the same voltages."""
    grid = synthetic_feeder(250)  # 750 rows -> CPU auto picks sparse
    op = _op(grid, 16)
    r_cpu = solve_power_flow(grid, operating_point=op)
    r_gpu = solve_power_flow(grid, operating_point=op, device=CUDA)
    assert r_cpu.converged and r_gpu.converged
    assert torch.allclose(r_cpu.v, r_gpu.v.cpu(), atol=1e-6)


def test_branch_states_parity():
    grid = synthetic_feeder(40, n_feeders=4, tie_switches=3)
    states = {
        30000: torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64),
        30001: torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
    }
    r_cpu = solve_power_flow(grid, branch_states=states)
    states_gpu = {k: v.cuda() for k, v in states.items()}
    r_gpu = solve_power_flow(grid, branch_states=states_gpu, device=CUDA)
    assert r_cpu.converged and r_gpu.converged
    assert torch.allclose(r_cpu.v, r_gpu.v.cpu(), atol=1e-6)


def test_prepared_system_parity():
    grid = synthetic_feeder(60)
    op = _op(grid, 8)
    system = prepare_power_flow(grid, device=CUDA)
    ref = solve_power_flow(grid, operating_point=op, device=CUDA)
    res = solve_power_flow(grid, operating_point=op, device=CUDA, system=system)
    assert torch.allclose(ref.v, res.v)


def test_mismatch_floor_row_scale_parity():
    """The floor's row scale is read from the nonzeros on CPU and densely on CUDA.

    Two formulations of one quantity, so they must agree: the cancellation scale itself,
    its cheap upper bound, and the per-row threshold a solve reports from them.
    """
    from pgml.solver.power_flow import _abs_row_scale, _row_scale_bound

    grid = synthetic_feeder(250, tie_switches=1)  # 750 rows
    index = node_phase_index(grid)
    y = assemble_network_ybus(grid, float(grid.base_frequency_hz)).Y.reshape(
        index.size, index.size
    )
    v = torch.full((index.size,), 11547.0, dtype=torch.float64)
    cpu_exact, cpu_bound = _abs_row_scale(y, v), _row_scale_bound(y, v)
    gpu_exact = _abs_row_scale(y.to(CUDA), v.to(CUDA)).cpu()
    gpu_bound = _row_scale_bound(y.to(CUDA), v.to(CUDA)).cpu()
    assert torch.allclose(cpu_exact, gpu_exact, rtol=1e-13, atol=0.0)
    assert torch.allclose(cpu_bound, gpu_bound, rtol=1e-13, atol=0.0)
    assert bool((gpu_bound >= gpu_exact).all())

    op = _op(grid, 4)
    r_cpu = solve_power_flow(grid, operating_point=op)
    r_gpu = solve_power_flow(grid, operating_point=op, device=CUDA)
    assert r_cpu.diagnostics.mismatch_floor_pu == pytest.approx(
        r_gpu.diagnostics.mismatch_floor_pu, rel=1e-12
    )
