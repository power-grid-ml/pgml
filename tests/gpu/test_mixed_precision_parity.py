"""CPU-vs-CUDA parity of mixed precision and the per-unit criteria (skips without CUDA).

Mixed precision is above all a CUDA lever: on a consumer card double precision runs at a
fraction of the single-precision rate, so factoring at complex64 and refining against
complex128 residuals is where the dense GPU path earns its throughput. These checks pin
that the option behaves identically on both devices — same convergence verdict, same
iteration count, same voltages to the refined accuracy — and that the per-unit
convergence measures are device-independent.
"""

from __future__ import annotations

import pytest
import torch

from pgml.grids import cigre_lv_full_grid, synthetic_feeder
from pgml.schemas.grid_schema import Load
from pgml.solver import prepare_power_flow, solve_harmonic_flow, solve_power_flow
from pgml.solver.harmonic import estimate_condition, lu_factor_system, solve_factored

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - CPU-only host
    pytest.skip("CUDA not available", allow_module_level=True)

CUDA = torch.device("cuda")
CDT = torch.complex128


def _op(grid, b, seed=0):
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    g = torch.Generator().manual_seed(seed)
    s = 0.8 + 0.4 * torch.rand(b, len(loads), generator=g, dtype=torch.float64)
    return {
        ld.id: {
            "p_w": float(ld.p_nom_w) * s[:, i],
            "q_var": float(getattr(ld, "q_nom_var", 0.0) or 0.0) * s[:, i],
        }
        for i, ld in enumerate(loads)
    }


def _to(op, device):
    return {
        cid: {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in e.items()
        }
        for cid, e in op.items()
    }


def test_mixed_precision_linear_solve_parity():
    """The refined solve is the same arithmetic on both devices."""
    n = 24
    torch.manual_seed(0)
    a = torch.randn(n, n, dtype=CDT) + n * torch.eye(n, dtype=CDT)
    a[0] = a[0] * 1.0e4
    b = torch.randn(n, dtype=CDT)
    x_cpu = solve_factored(lu_factor_system(a, precision="mixed"), b)
    x_gpu = solve_factored(
        lu_factor_system(a.cuda(), precision="mixed"), b.cuda()
    ).cpu()
    assert torch.allclose(x_cpu, x_gpu, rtol=1e-10, atol=1e-12)


def test_mixed_precision_power_flow_parity():
    grid, _ = cigre_lv_full_grid()
    op = _op(grid, 8)
    r_cpu = solve_power_flow(grid, operating_point=op, dtype=CDT, precision="mixed")
    r_gpu = solve_power_flow(
        grid, operating_point=_to(op, CUDA), dtype=CDT, precision="mixed", device=CUDA
    )
    assert r_cpu.converged and r_gpu.converged
    assert r_cpu.iterations == r_gpu.iterations
    assert torch.allclose(r_cpu.v, r_gpu.v.cpu(), rtol=1e-9, atol=1e-7)


def test_mixed_precision_matches_full_precision_on_cuda():
    """On the GPU too, mixed precision lands on the complex128 solution."""
    grid = synthetic_feeder(120)
    op = _to(_op(grid, 8), CUDA)
    full = solve_power_flow(grid, operating_point=op, dtype=CDT, device=CUDA)
    mixed = solve_power_flow(
        grid, operating_point=op, dtype=CDT, device=CUDA, precision="mixed"
    )
    assert full.converged and mixed.converged
    rel = (mixed.v - full.v).abs().max() / full.v.abs().max()
    assert float(rel) < 1e-9


def test_per_unit_diagnostics_are_device_independent():
    grid, _ = cigre_lv_full_grid()
    r_cpu = solve_power_flow(grid, dtype=CDT)
    r_gpu = solve_power_flow(grid, dtype=CDT, device=CUDA)
    for attr in ("mismatch_max_pu", "update_max_pu", "s_base_va"):
        assert getattr(r_cpu.diagnostics, attr) == pytest.approx(
            getattr(r_gpu.diagnostics, attr), rel=1e-6, abs=1e-14
        )
    assert r_cpu.iterations == r_gpu.iterations


def test_prepared_mixed_system_parity():
    grid = synthetic_feeder(80)
    op = _op(grid, 4)
    sys_cpu = prepare_power_flow(grid, dtype=CDT, precision="mixed")
    sys_gpu = prepare_power_flow(grid, dtype=CDT, precision="mixed", device=CUDA)
    r_cpu = solve_power_flow(
        grid, system=sys_cpu, operating_point=op, dtype=CDT, precision="mixed"
    )
    r_gpu = solve_power_flow(
        grid,
        system=sys_gpu,
        operating_point=_to(op, CUDA),
        dtype=CDT,
        precision="mixed",
        device=CUDA,
    )
    assert torch.allclose(r_cpu.v, r_gpu.v.cpu(), rtol=1e-9, atol=1e-7)


def test_harmonic_flow_mixed_precision_parity():
    grid, _ = cigre_lv_full_grid()
    cpu = solve_harmonic_flow(grid, [1, 5, 7], dtype=CDT, precision="mixed")
    gpu = solve_harmonic_flow(
        grid, [1, 5, 7], dtype=CDT, precision="mixed", device=CUDA
    )
    assert torch.allclose(cpu.v, gpu.v.cpu(), rtol=1e-9, atol=1e-7)


def test_condition_estimate_parity():
    grid, _ = cigre_lv_full_grid()
    from pgml.assembly import node_phase_index
    from pgml.solver.power_flow import _slack_rows_and_vref, _y_eff_and_islack

    idx = node_phase_index(grid)
    est = []
    for device in (torch.device("cpu"), CUDA):
        y, _ = _y_eff_and_islack(
            grid, float(grid.base_frequency_hz), idx, CDT, device, "ideal", None
        )
        rows, _ = _slack_rows_and_vref(grid, idx, torch.float64, CDT, device)
        est.append(estimate_condition(lu_factor_system(y, fixed_rows=rows)))
    assert est[0] == pytest.approx(est[1], rel=1e-6)
