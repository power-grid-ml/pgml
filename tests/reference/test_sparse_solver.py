"""Sparse (scipy SuperLU) factorization backend: parity with dense + error paths.

The sparse backend must be numerically interchangeable with the dense batched LU
(:func:`pgml.solver.harmonic.lu_factor_system` / :func:`solve_factored`) on real
assembled feeder systems, in both slack modes, for factorization batches and
scenario batches — and a singular system must fail with an actionable message.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_network_ybus, assemble_ybus
from pgml.errors import ComputationError, InputError
from pgml.grids import synthetic_feeder
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import _SPARSE_MIN_ROWS, lu_factor_system, solve_factored


@pytest.fixture(scope="module")
def feeder():
    return synthetic_feeder(40)  # 120 node-phase rows


@pytest.fixture(scope="module")
def y_linear(feeder):
    """Const-Z linear Y (source folded -> non-singular, Norton-solvable)."""
    return assemble_ybus(feeder, [50.0, 250.0, 350.0]).Y  # [3, N, N]


def test_norton_parity_multi_frequency(y_linear):
    n = y_linear.shape[-1]
    rhs = torch.randn(5, 3, n, dtype=y_linear.dtype)  # 5 scenarios x 3 frequencies
    vd = solve_factored(lu_factor_system(y_linear, backend="dense"), rhs)
    vs = solve_factored(lu_factor_system(y_linear, backend="sparse"), rhs)
    assert torch.allclose(vd, vs, atol=1e-9 * float(vd.abs().max()))


def test_ideal_slack_parity(feeder):
    y = assemble_network_ybus(feeder, [50.0]).Y  # [1, N, N] passive network
    n = y.shape[-1]
    fixed_rows = torch.tensor([0, 1, 2])  # the station bus rows
    v_fixed = 11547.0 * torch.exp(
        1j * torch.tensor([0.0, -2.0943951, 2.0943951], dtype=torch.float64)
    )
    rhs = torch.randn(4, 1, n, dtype=y.dtype)
    vd = solve_factored(
        lu_factor_system(y, fixed_rows=fixed_rows, backend="dense"),
        rhs,
        v_fixed=v_fixed,
    )
    vs = solve_factored(
        lu_factor_system(y, fixed_rows=fixed_rows, backend="sparse"),
        rhs,
        v_fixed=v_fixed,
    )
    assert torch.allclose(vd, vs, atol=1e-9 * float(vd.abs().max()))


@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_solve_power_flow_sparse_matches_dense(feeder, slack):
    rd = solve_power_flow(feeder, slack=slack, linear_solver="dense")
    rs = solve_power_flow(feeder, slack=slack, linear_solver="sparse")
    assert rd.converged and rs.converged
    assert torch.allclose(rd.v, rs.v, atol=1e-6)  # V scale is ~1e4 V


def test_solve_power_flow_sparse_matches_dense_batched(feeder):
    g = torch.Generator().manual_seed(7)
    op = {
        a.id: {"p_w": float(a.p_nom_w) * (0.5 + torch.rand(6, generator=g))}
        for a in feeder.appliances
        if a.id >= 20000
    }
    rd = solve_power_flow(feeder, operating_point=op, linear_solver="dense")
    rs = solve_power_flow(feeder, operating_point=op, linear_solver="sparse")
    assert rd.converged and rs.converged
    assert rd.v.shape == rs.v.shape == (6, rd.index.size)
    assert torch.allclose(rd.v, rs.v, atol=1e-6)


def test_auto_selects_sparse_above_threshold_on_cpu():
    n_small = torch.randn(4, 4, dtype=torch.complex128) + 8 * torch.eye(
        4, dtype=torch.complex128
    )
    assert lu_factor_system(n_small, backend="auto").backend == "dense"
    m = _SPARSE_MIN_ROWS
    big = torch.eye(m, dtype=torch.complex128) * 3.0
    assert lu_factor_system(big, backend="auto").backend == "sparse"


def test_singular_sparse_system_raises_actionable_error():
    y = torch.eye(6, dtype=torch.complex128)
    y[3, 3] = 0.0  # a dead row: no admittance anywhere
    with pytest.raises(ComputationError, match="check_connectivity"):
        lu_factor_system(y, backend="sparse")


def test_newton_rejects_sparse():
    grid = synthetic_feeder(5)
    with pytest.raises(InputError, match="newton"):
        solve_power_flow(grid, method="newton", linear_solver="sparse")


@pytest.mark.gpu
def test_sparse_backend_rejects_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    y = (torch.eye(8, dtype=torch.complex128) * 2.0).cuda()
    with pytest.raises(InputError, match="CPU"):
        lu_factor_system(y, backend="sparse")


def test_sparse_complex64_batch_converges_at_backend_floor():
    """A large complex64 scenario batch terminates under the sparse backend.

    SuperLU's single-precision back-substitution leaves more rounding noise per
    iterate than the dense torch LU (measured on this batch: the per-row voltage
    update plateaus FLAT at 5.7e-6 per unit, where the dense backend is still
    contracting at 8.8e-7), so marginal scenarios of a large batch oscillate above a
    dense-calibrated floor and come back flagged unconverged although their voltages
    sit at the single-precision floor. The backend-aware floor
    (:func:`pgml.solver.power_flow._rel_convergence_floor`, 1.2e-5 per unit for the
    sparse float32 path) must let the batch terminate like the dense backend does,
    with the full mask converged and floor-level accuracy against the
    double-precision dense reference.
    """
    pp = pytest.importorskip("pandapower")
    import numpy as np
    import pandapower.networks as pn

    from pgml.convert.pandapower import to_grid
    from pgml.schemas import Load

    net = pn.case33bw()
    pp.runpp(net, numba=False)
    grid, _ = to_grid(net)
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    scen = np.random.default_rng(0).uniform(0.8, 1.2, size=(5000, len(loads)))

    def op_for(rdt):
        s = torch.tensor(scen, dtype=rdt)
        return {
            ld.id: {
                "p_w": float(ld.p_nom_w) * s[:, i],
                "q_var": float(getattr(ld, "q_nom_var", 0.0) or 0.0) * s[:, i],
            }
            for i, ld in enumerate(loads)
        }

    r64 = solve_power_flow(
        grid,
        slack="ideal",
        operating_point=op_for(torch.float32),
        dtype=torch.complex64,
        linear_solver="sparse",
    )
    assert bool(r64.converged_mask.all()), (
        f"{int((~r64.converged_mask).sum())} scenarios flagged unconverged "
        f"after {r64.iterations} iterations"
    )
    assert r64.iterations <= 20, (
        f"sparse complex64 batch took {r64.iterations} iterations "
        "(floor not terminating the fixed point)"
    )

    ref = solve_power_flow(
        grid,
        slack="ideal",
        operating_point=op_for(torch.float64),
        dtype=torch.complex128,
        linear_solver="dense",
    )
    rel = ((r64.v.abs().to(torch.float64) - ref.v.abs()).abs() / ref.v.abs()).max()
    assert float(rel) < 5e-5, f"relative |V| error {float(rel):.2e} above c64 floor"
