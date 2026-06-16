"""GPU gate: CPU-vs-CUDA parity of assemble + solve (skips cleanly without CUDA)."""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.solver import solve_harmonic

from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

GRIDS = [single_phase_chain, three_phase_two_bus]


@pytest.mark.parametrize("grid_fn", GRIDS)
@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_parity(grid_fn, dtype):
    grid = grid_fn()
    freqs = [50.0, 250.0]

    # CPU path.
    idx_cpu = node_phase_index(grid)
    f_cpu = torch.tensor(
        freqs, dtype=torch.float64 if dtype == torch.complex128 else torch.float32
    )
    yb_cpu = assemble_ybus(grid, f_cpu, dtype=dtype, device=torch.device("cpu"))
    i_cpu = build_injections(
        grid, f_cpu, idx_cpu, dtype=dtype, device=torch.device("cpu")
    )
    v_cpu = solve_harmonic(yb_cpu.Y, i_cpu)

    # CUDA path.
    dev = torch.device("cuda")
    f_cuda = f_cpu.to(dev)
    idx_cuda = node_phase_index(grid)
    yb_cuda = assemble_ybus(grid, f_cuda, dtype=dtype, device=dev)
    i_cuda = build_injections(grid, f_cuda, idx_cuda, dtype=dtype, device=dev)
    v_cuda = solve_harmonic(yb_cuda.Y, i_cuda)

    assert yb_cuda.Y.device.type == "cuda"
    assert v_cuda.device.type == "cuda"
    assert yb_cuda.Y.dtype == dtype

    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=tol, atol=tol)
    torch.testing.assert_close(v_cuda.cpu(), v_cpu, rtol=tol, atol=tol)


def test_cuda_ideal_slack_parity():
    grid = single_phase_chain()
    dev = torch.device("cuda")
    f = [50.0]
    idx = node_phase_index(grid)
    slack_row = idx.row(1, grid.nodes[0].phases[0])

    yb_cpu = assemble_ybus(grid, f, dtype=torch.complex128, device=torch.device("cpu"))
    i_cpu = build_injections(
        grid, f, idx, dtype=torch.complex128, device=torch.device("cpu")
    )
    fixed = torch.tensor([slack_row], dtype=torch.int64)
    vfix = torch.tensor([230.0 + 0.0j], dtype=torch.complex128)
    v_cpu = solve_harmonic(yb_cpu.Y, i_cpu, fixed_rows=fixed, v_fixed=vfix)

    yb_cuda = assemble_ybus(
        grid, torch.tensor(f, device=dev), dtype=torch.complex128, device=dev
    )
    i_cuda = build_injections(
        grid, torch.tensor(f, device=dev), idx, dtype=torch.complex128, device=dev
    )
    v_cuda = solve_harmonic(
        yb_cuda.Y, i_cuda, fixed_rows=fixed.to(dev), v_fixed=vfix.to(dev)
    )

    torch.testing.assert_close(v_cuda.cpu(), v_cpu, rtol=1e-9, atol=1e-9)
