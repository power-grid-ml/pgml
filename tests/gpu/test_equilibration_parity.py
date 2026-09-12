"""CPU-vs-CUDA parity of the equilibrated factorizations (skips without CUDA).

Equilibration is where the GPU path needs it most: CUDA is always dense, torch's dense LU
has no scaling option of its own (LAPACK's simple driver does not equilibrate, unlike the
sparse solvers the reference tools use), and single precision is the reason to be on the
card at all. These checks pin that the scaling is device-independent — the same scale
factors, the same solved voltages, the same condition estimate, the same behaviour on the
block-diagonal backend that only exists for the GPU ensemble case — and that the gradient
path is unchanged on the device as it is on the host.
"""

from __future__ import annotations

import pytest
import torch

from pgml.grids import synthetic_feeder
from pgml.multigrid import merge_grids
from pgml.solver import solve_harmonic_flow, solve_power_flow
from pgml.solver.equilibration import equilibration_scales
from pgml.solver.harmonic import estimate_condition, lu_factor_system, solve_factored

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - CPU-only host
    pytest.skip("CUDA not available", allow_module_level=True)

CUDA = torch.device("cuda")
CDT = torch.complex128


def _spread_system(n=16, spread=1.0e6, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=CDT) + 3.0 * torch.eye(n, dtype=CDT)
    d = torch.logspace(0.0, 6.0, n, dtype=torch.float64)
    return a * d.unsqueeze(-1) * d.unsqueeze(-2), torch.randn(n, dtype=CDT)


@pytest.mark.parametrize("mode", ["symmetric", "row_column"])
def test_scale_factors_are_device_independent(mode):
    a, _ = _spread_system()
    cpu = equilibration_scales(a, mode=mode)
    cuda = equilibration_scales(a.to(CUDA), mode=mode)
    for dc, dg in zip(cpu, cuda):
        assert dg.device.type == "cuda"
        # Powers of two on both devices: the scaling is exact, so it must be identical.
        assert torch.equal(dc, dg.cpu())


@pytest.mark.parametrize("mode", ["off", "symmetric", "row_column"])
def test_factored_solve_matches_across_devices(mode):
    a, b = _spread_system()
    ref = solve_factored(lu_factor_system(a, equilibrate=mode), b)
    got = solve_factored(
        lu_factor_system(a.to(CUDA), equilibrate=mode), b.to(CUDA)
    ).cpu()
    assert float((got - ref).abs().max() / ref.abs().max()) < 1e-12


@pytest.mark.parametrize("mode", ["off", "symmetric"])
def test_condition_estimate_matches_across_devices(mode):
    a, _ = _spread_system()
    cpu = estimate_condition(lu_factor_system(a, equilibrate=mode))
    cuda = estimate_condition(lu_factor_system(a.to(CUDA), equilibrate=mode))
    assert cuda == pytest.approx(cpu, rel=1e-9)


def test_power_flow_and_harmonic_flow_match_across_devices():
    grid = synthetic_feeder(8)
    orders = [1, 13]
    for mode in ("off", "symmetric"):
        cpu = solve_power_flow(grid, equilibrate=mode, tol_update_pu=1e-12)
        gpu = solve_power_flow(grid, device=CUDA, equilibrate=mode, tol_update_pu=1e-12)
        assert cpu.converged and gpu.converged
        assert float((gpu.v.cpu() - cpu.v).abs().max()) < 1e-6  # volts
        hc = solve_harmonic_flow(grid, orders, equilibrate=mode)
        hg = solve_harmonic_flow(grid, orders, device=CUDA, equilibrate=mode)
        assert float((hg.v.cpu() - hc.v).abs().max()) < 1e-6


def test_block_backend_is_equilibrated_on_cuda():
    """The block backend is the CUDA ensemble path; its scale is built per block."""
    merged = merge_grids([synthetic_feeder(6), synthetic_feeder(6)])
    rows = [r.to(CUDA) for r in merged.block_rows()]
    kw = dict(linear_solver="block", block_rows=rows, device=CUDA, tol_update_pu=1e-12)
    off = solve_power_flow(merged.grid, equilibrate="off", **kw)
    sym = solve_power_flow(merged.grid, equilibrate="symmetric", **kw)
    assert off.converged and sym.converged
    assert float((sym.v - off.v).abs().max()) < 1e-6  # volts on a 20 kV feeder
    with pytest.raises(Exception):  # row_column needs a matrix this backend never forms
        solve_power_flow(merged.grid, equilibrate="row_column", **kw)


def test_gradient_is_unchanged_on_cuda():
    grid = synthetic_feeder(6)
    line = grid.branches[0]
    r0 = torch.as_tensor(line.series_resistance_ohm_per_m, dtype=torch.float64)
    grads = {}
    for mode in ("off", "symmetric"):
        r = r0.to(CUDA).clone().requires_grad_(True)
        res = solve_power_flow(
            grid,
            device=CUDA,
            param_overrides={("line", int(line.id), "series_resistance_ohm_per_m"): r},
            equilibrate=mode,
            tol_update_pu=1e-12,
        )
        res.v.abs().sum().backward()
        grads[mode] = r.grad.clone()
    assert torch.allclose(
        grads["symmetric"],
        grads["off"],
        rtol=1e-9,
        atol=1e-9 * float(grads["off"].abs().max()),
    )
