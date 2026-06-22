"""Batched-solve robustness: dtype-aware convergence, Newton batching, partial failure.

Covers the solver behaviour that makes large scenario batches usable:

- a ``complex64`` batch converges at the dtype's resolvable precision (a relative floor)
  instead of spinning to ``max_iter`` against an unreachable absolute ``tol``, and a
  warning is logged when ``tol`` is below that floor;
- ``solve_power_flow(method="newton")`` accepts a batched operating point (solved per
  scenario) and matches the current-injection fixed point;
- a batch with infeasible scenarios does NOT raise — every scenario's best-effort
  voltage is returned, the failures are listed in ``failed_states`` + logged, and the
  single-grid criticality SVD is skipped for a batch;
- a sweep that varies only SOME devices (loads but not generators) still assembles.
"""

from __future__ import annotations

import logging

import torch

from pgml.schemas.grid_schema import Generator, Load
from pgml.solver.power_flow import solve_power_flow
from tests.fixtures.tiny_grids import single_phase_chain

CDT = torch.complex128
CF = torch.complex64


def _batched_chain_op(p_values, q_value=300.0):
    """The single-phase chain + an operating point batching the node-3 load's P."""
    grid = single_phase_chain()
    p = torch.tensor(p_values, dtype=torch.float64)
    q = torch.full((len(p_values),), float(q_value), dtype=torch.float64)
    return grid, {30: {"p_w": p, "q_var": q}}


class TestBatchedNewton:
    def test_newton_accepts_batched_operating_point(self) -> None:
        grid, op = _batched_chain_op([1500.0, 3000.0, 6000.0, 9000.0])
        r = solve_power_flow(grid, operating_point=op, method="newton", dtype=CDT)
        assert r.converged
        assert tuple(r.v.shape) == (4, 3)  # [B, N]; the chain has 3 single-phase nodes

    def test_newton_matches_current_injection(self) -> None:
        grid, op = _batched_chain_op([1500.0, 3000.0, 6000.0, 9000.0])
        ci = solve_power_flow(
            grid, operating_point=op, method="current_injection", dtype=CDT
        )
        nt = solve_power_flow(grid, operating_point=op, method="newton", dtype=CDT)
        assert ci.converged and nt.converged
        assert ci.v.shape == nt.v.shape
        assert torch.max(torch.abs(ci.v - nt.v)).item() < 1e-6

    def test_batched_newton_gradients_flow(self) -> None:
        """Sequential forward + shared IFT backward differentiate the batch w.r.t. a grid
        parameter (the documented leaf; the batch is supplied by ``operating_point``),
        and the gradient matches the current-injection path."""
        op = {
            30: {
                "p_w": torch.tensor([2000.0, 4000.0], dtype=torch.float64),
                "q_var": torch.tensor([300.0, 300.0], dtype=torch.float64),
            }
        }

        def grad_for(method: str) -> torch.Tensor:
            grid = single_phase_chain()
            # A 2-D TENSOR (tensor duality) — a python list would be coerced to floats.
            r_leaf = torch.tensor([[2.0e-3]], dtype=torch.float64, requires_grad=True)
            grid.branches[0].series_resistance_ohm_per_m = r_leaf
            res = solve_power_flow(grid, operating_point=op, method=method, dtype=CDT)
            res.v.abs().sum().backward()
            return r_leaf.grad.detach().clone()

        g_newton = grad_for("newton")
        g_ci = grad_for("current_injection")
        assert torch.isfinite(g_newton).all() and g_newton.abs() > 0
        assert torch.allclose(g_newton, g_ci, rtol=1e-6, atol=1e-9)


class TestDtypeFloor:
    def test_complex64_batch_converges_not_maxiter(self) -> None:
        """float32 settles at its floor rather than spinning to max_iter."""
        grid, op = _batched_chain_op([1500.0, 3000.0, 4500.0])
        r = solve_power_flow(
            grid, operating_point=op, method="current_injection", dtype=CF, max_iter=100
        )
        assert r.converged
        assert r.iterations < 100
        assert tuple(r.converged_mask.shape) == (3,)

    def test_complex64_close_to_complex128(self) -> None:
        grid, op = _batched_chain_op([1500.0, 3000.0, 4500.0])
        r64 = solve_power_flow(grid, operating_point=op, dtype=CF)
        r128 = solve_power_flow(grid, operating_point=op, dtype=CDT)
        rel = torch.max(torch.abs(r64.v - r128.v)) / torch.max(torch.abs(r128.v))
        assert float(rel) < 1e-4  # float32 precision, plenty for data generation

    def test_warns_when_tol_below_dtype_floor(self, caplog) -> None:
        grid, op = _batched_chain_op([2000.0, 3000.0])
        with caplog.at_level(logging.WARNING, logger="pgml"):
            solve_power_flow(grid, operating_point=op, dtype=CF, tol=1e-12)
        assert any("precision floor" in rec.message for rec in caplog.records), (
            "expected a dtype-floor warning for tol below float32 precision"
        )

    def test_no_warning_for_complex128(self, caplog) -> None:
        grid, op = _batched_chain_op([2000.0, 3000.0])
        with caplog.at_level(logging.WARNING, logger="pgml"):
            solve_power_flow(grid, operating_point=op, dtype=CDT, tol=1e-12)
        assert not any("precision floor" in rec.message for rec in caplog.records)


class TestPartialFailure:
    def test_batch_with_infeasible_states_does_not_raise(self, caplog) -> None:
        # Mix feasible loads with loads far past the chain's loadability nose.
        grid, op = _batched_chain_op([2000.0, 3000.0, 8.0e5, 1.0e6])
        with caplog.at_level(logging.ERROR, logger="pgml"):
            r = solve_power_flow(
                grid, operating_point=op, method="current_injection", max_iter=60
            )
        assert not r.converged
        assert r.failed_states == (2, 3)  # the two infeasible scenarios
        assert tuple(r.v.shape) == (4, 3)
        assert torch.isfinite(r.v[0]).all() and torch.isfinite(r.v[1]).all()
        # criticality is a single-grid diagnostic -> skipped for a batch (no crash)
        assert r.diagnostics is not None and r.diagnostics.criticality is None
        assert any("did not converge" in rec.message for rec in caplog.records), (
            "expected an error log naming the failed states"
        )

    def test_all_converged_reports_clean(self) -> None:
        grid, op = _batched_chain_op([1500.0, 2500.0, 3500.0])
        r = solve_power_flow(grid, operating_point=op)
        assert r.converged
        assert r.failed_states == ()
        assert bool(r.converged_mask.all())

    def test_unbatched_result_is_unchanged(self) -> None:
        r = solve_power_flow(single_phase_chain())
        assert r.converged
        assert r.converged_mask is None  # no batch dim
        assert r.failed_states == ()
        assert r.v.ndim == 1


class TestMixedBatchedDevices:
    def test_varying_only_some_devices_assembles(self) -> None:
        """A sweep over loads on a grid that also has a generator must not fail the
        per-element stack (the unbatched generator broadcasts across the batch)."""
        grid = single_phase_chain()
        from pgml.schemas.grid_schema import Phase

        grid.appliances.append(
            Generator(id=40, node=2, phases=(Phase.A,), p_nom_w=500.0, q_nom_var=0.0)
        )
        # batch ONLY the load; the generator keeps its (unbatched) nameplate
        p = torch.tensor([1500.0, 3000.0, 4500.0], dtype=torch.float64)
        r = solve_power_flow(grid, operating_point={30: {"p_w": p, "q_var": p * 0.2}})
        assert r.converged
        assert tuple(r.v.shape) == (3, 3)
        # the loads differ per scenario, so the node-3 voltage must differ across the batch
        assert torch.std(r.v[:, -1].abs()).item() > 0.0
        _ = Load  # keep the import meaningful for readers
