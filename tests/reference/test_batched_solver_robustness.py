"""Batched-solve robustness: dtype-aware convergence, Newton batching, partial failure.

Covers the solver behaviour that makes large scenario batches usable:

- a ``complex64`` batch converges at the dtype's resolvable precision (a relative floor)
  instead of spinning to ``max_iter`` against an unreachable absolute ``tol``, and a
  warning is logged when ``tol`` is below that floor;
- ``solve_power_flow(method="newton")`` accepts a batched operating point (solved per
  scenario) and matches the current-injection fixed point;
- a batch with infeasible scenarios does NOT raise — every scenario's best-effort
  voltage is returned, the failures are listed in ``failed_states`` + logged, and the
  criticality SVD analyses the HARDEST scenario (naming it) instead of raising or
  reporting nothing;
- a sweep that varies only SOME devices (loads but not generators) still assembles.
"""

from __future__ import annotations

import logging

import pytest
import torch

from pgml.schemas.grid_schema import Generator, Load
from pgml.solver.harmonic import lu_factor_system, solve_factored, solve_harmonic
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

    def test_newton_batched_matches_loop_of_singles(self) -> None:
        """A batched Newton solve equals a python loop of single-scenario solves
        (the batched==loop invariant, previously pinned only for the
        current-injection method)."""
        p_vals = [1500.0, 3000.0, 6000.0, 9000.0]
        grid, op = _batched_chain_op(p_vals)
        batched = solve_power_flow(grid, operating_point=op, method="newton", dtype=CDT)
        assert batched.converged
        for k, p in enumerate(p_vals):
            single = solve_power_flow(
                grid,
                operating_point={30: {"p_w": float(p), "q_var": 300.0}},
                method="newton",
                dtype=CDT,
            )
            assert single.converged
            assert torch.max(torch.abs(batched.v[k] - single.v)).item() < 1e-9

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
        assert any("did not converge" in rec.message for rec in caplog.records), (
            "expected an error log naming the failed states"
        )
        # Each infeasible scenario is held at the iterate where its own iteration
        # stopped making progress, so the solve reports the stall instead of spending
        # every remaining iteration on a diverging scenario.
        assert r.iterations < 60
        assert r.diagnostics.n_stalled == 2
        assert r.diagnostics.likely_cause.startswith("the iteration stopped making")
        # The criticality diagnostic RUNS on a batch and names the scenario it analysed
        # (the one with the largest nodal mismatch — one of the two infeasible ones).
        crit = r.diagnostics.criticality
        assert crit is not None and "skipped" not in crit
        assert crit["batch"] in (2, 3)
        assert crit["min_singular_value"] > 0.0
        assert len(crit["critical_nodes"]) > 0

    def test_batched_criticality_equals_the_single_scenario_analysis(self) -> None:
        """The batched diagnostic analyses THAT scenario's system, not the batch's.

        The Jacobian of the worst scenario is built from its own admittance, slack
        current, injection powers and setpoints, so the figures must equal the ones the
        same scenario produces when it is solved alone.
        """
        p_vals = [2000.0, 3000.0, 8.0e5, 1.0e6]
        grid, op = _batched_chain_op(p_vals)
        batched = solve_power_flow(
            grid, operating_point=op, max_iter=60
        ).diagnostics.criticality
        worst = batched["batch"]
        assert worst in (2, 3)  # one of the two infeasible scenarios
        single = solve_power_flow(
            single_phase_chain(),
            operating_point={30: {"p_w": p_vals[worst], "q_var": 300.0}},
            max_iter=60,
        ).diagnostics.criticality
        for key in ("min_singular_value", "max_singular_value", "condition_number"):
            assert batched[key] == pytest.approx(single[key], rel=1e-9)
        assert [c["node_id"] for c in batched["critical_nodes"]] == [
            c["node_id"] for c in single["critical_nodes"]
        ]

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


class TestFactoredSolve:
    """factor-once-solve-many (lu_factor_system + solve_factored) == solve_harmonic."""

    @staticmethod
    def _spd_like(h, n):
        torch.manual_seed(0)
        return torch.randn(h, n, n, dtype=CDT) + torch.eye(n, dtype=CDT) * (n + 5)

    def test_norton_matches_and_reuses(self) -> None:
        y = self._spd_like(3, 5)  # [H, N, N], one matrix per harmonic
        i = torch.randn(7, 3, 5, dtype=CDT)  # [B, H, N] — batch shares Y
        ref = solve_harmonic(y, i)
        fac = solve_factored(lu_factor_system(y), i)
        assert torch.max(torch.abs(ref - fac)).item() < 1e-12

    def test_ideal_slack_matches(self) -> None:
        y = self._spd_like(3, 6)
        i = torch.randn(4, 3, 6, dtype=CDT)
        fixed = torch.tensor([0, 3], dtype=torch.int64)
        vfix = torch.tensor([1.0 + 0j, 0.5 + 0j], dtype=CDT)
        ref = solve_harmonic(y, i, fixed_rows=fixed, v_fixed=vfix)
        fac = solve_factored(lu_factor_system(y, fixed_rows=fixed), i, v_fixed=vfix)
        assert torch.max(torch.abs(ref - fac)).item() < 1e-12
        assert torch.max(torch.abs(fac[..., fixed] - vfix)).item() < 1e-12

    def test_factored_gradients_match(self) -> None:
        y = self._spd_like(1, 5).requires_grad_(True)
        i = torch.randn(6, 1, 5, dtype=CDT)
        g_ref = torch.autograd.grad(
            solve_harmonic(y, i).abs().sum(), y, retain_graph=True
        )[0]
        g_fac = torch.autograd.grad(
            solve_factored(lu_factor_system(y), i).abs().sum(), y
        )[0]
        assert torch.max(torch.abs(g_ref - g_fac)).item() < 1e-10

    @pytest.mark.parametrize("backend", ["dense", "sparse"])
    def test_interleaved_shared_rhs_axis_matches_direct_solve(self, backend) -> None:
        """A singleton factor axis may expand to a non-singleton RHS step axis.

        This is the harmonic-flow layout when each scenario has its own device shunt
        but several coherent-spectrum steps share that shunt: ``Y=[B, 1, H, N, N]``
        and ``I=[B, T, H, N]``. Each ``(B, H)`` factor must serve all ``T`` columns.
        """
        b, steps, h, n = 2, 3, 2, 4
        y = self._spd_like(b * h, n).reshape(b, 1, h, n, n)
        i = torch.randn(b, steps, h, n, dtype=CDT)
        y_full = y.broadcast_to(b, steps, h, n, n)
        ref = torch.linalg.solve(y_full, i.unsqueeze(-1)).squeeze(-1)
        got = solve_factored(lu_factor_system(y, backend=backend, equilibrate="off"), i)
        assert got.shape == i.shape
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


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


class TestTrailingSingletonBatch:
    """An operating point batched ``[B, 1]`` keeps its trailing dim as a SCENARIO dim.

    A per-step recipe with one step produces exactly this shape, and the size-one
    dim sits where the assembly's singleton frequency axis used to sit — a
    value-based squeeze there mixes the ``B`` scenarios into a ``[B, B]`` broadcast
    instead of solving them independently. The contract: ``v``'s batch shape equals
    the operating point's batch shape, for every method and for the harmonic solve.
    """

    def _refs(self, grid, p, q):
        """Independent single-scenario solutions ``[B, N]``."""
        return torch.stack(
            [
                solve_power_flow(
                    grid,
                    operating_point={30: {"p_w": float(p[i]), "q_var": float(q[i])}},
                    dtype=CDT,
                ).v
                for i in range(len(p))
            ],
            0,
        )

    def test_batch_shape_is_preserved(self) -> None:
        grid = single_phase_chain()
        for shape in [(5,), (5, 1), (5, 6), (5, 1, 1), (1,)]:
            op = {
                30: {
                    "p_w": torch.full(shape, 2000.0, dtype=torch.float64),
                    "q_var": torch.full(shape, 300.0, dtype=torch.float64),
                }
            }
            r = solve_power_flow(grid, operating_point=op, dtype=CDT)
            assert r.converged
            assert tuple(r.v.shape) == (*shape, 3), (
                f"op batch {shape} must survive into v, got {tuple(r.v.shape)}"
            )

    def test_trailing_singleton_matches_loop_of_singles(self) -> None:
        grid = single_phase_chain()
        p = torch.tensor([500.0, 2000.0, 3500.0, 5000.0], dtype=torch.float64)
        q = torch.tensor([0.0, 150.0, 300.0, 450.0], dtype=torch.float64)
        ref = self._refs(grid, p, q)
        for method in ("current_injection", "newton"):
            r = solve_power_flow(
                grid,
                operating_point={
                    30: {"p_w": p.reshape(4, 1), "q_var": q.reshape(4, 1)}
                },
                method=method,
                dtype=CDT,
            )
            assert r.converged
            assert tuple(r.v.shape) == (4, 1, 3)
            err = torch.max(torch.abs(r.v.squeeze(1) - ref)).item()
            assert err < 1e-7, f"{method}: [B, 1] deviates from singles by {err:.3e}"

    def test_harmonic_solve_carries_the_step_axis(self) -> None:
        from pgml.solver import solve_harmonic_flow

        grid = single_phase_chain()
        p = torch.tensor([500.0, 2000.0, 3500.0], dtype=torch.float64)
        q = 0.2 * p
        inj = {30: {3: (0.04, 10.0), 5: (0.02, -30.0)}}
        flat = solve_harmonic_flow(
            grid,
            [1, 3, 5],
            operating_point={30: {"p_w": p, "q_var": q}},
            harmonic_injection=inj,
            dtype=CDT,
        ).v  # [B, H, N]
        stepped = solve_harmonic_flow(
            grid,
            [1, 3, 5],
            operating_point={30: {"p_w": p.reshape(3, 1), "q_var": q.reshape(3, 1)}},
            harmonic_injection=inj,
            dtype=CDT,
        ).v  # [B, 1, H, N]
        assert tuple(stepped.shape) == (3, 1, 3, 3)
        assert torch.equal(stepped.squeeze(1), flat)

    def test_scenario_shunt_is_shared_across_deeper_injection_steps(self) -> None:
        """A ``[B]`` operating point and ``[B, T]`` spectrum solve in one call."""
        from pgml.solver import solve_harmonic_flow

        grid = single_phase_chain()
        p = torch.tensor([1200.0, 3200.0], dtype=torch.float64)
        q = 0.2 * p
        mag3 = torch.tensor(
            [[0.02, 0.04, 0.06], [0.03, 0.05, 0.07]], dtype=torch.float64
        )
        phase3 = torch.tensor(
            [[-10.0, 0.0, 10.0], [15.0, 25.0, 35.0]], dtype=torch.float64
        )
        batched = solve_harmonic_flow(
            grid,
            [1, 3],
            operating_point={30: {"p_w": p, "q_var": q}},
            harmonic_injection={30: {3: (mag3, phase3)}},
            load_shunt="opendss",
            load_shunt_basis="operating_point",
            linear_solver="dense",
            dtype=CDT,
        ).v
        assert batched.shape == (2, 3, 2, 3)

        singles = []
        for scenario in range(2):
            steps = []
            for step in range(3):
                steps.append(
                    solve_harmonic_flow(
                        grid,
                        [1, 3],
                        operating_point={
                            30: {
                                "p_w": float(p[scenario]),
                                "q_var": float(q[scenario]),
                            }
                        },
                        harmonic_injection={
                            30: {
                                3: (
                                    float(mag3[scenario, step]),
                                    float(phase3[scenario, step]),
                                )
                            }
                        },
                        load_shunt="opendss",
                        load_shunt_basis="operating_point",
                        linear_solver="dense",
                        dtype=CDT,
                    ).v
                )
            singles.append(torch.stack(steps))
        torch.testing.assert_close(
            batched, torch.stack(singles), rtol=1e-12, atol=1e-12
        )

    def test_gradients_flow_through_the_singleton_batch(self) -> None:
        """The IFT backward returns a gradient of the operating point's own shape."""
        grid = single_phase_chain()
        p = torch.tensor(
            [[1500.0], [3000.0], [4500.0]], dtype=torch.float64, requires_grad=True
        )
        r = solve_power_flow(
            grid,
            operating_point={30: {"p_w": p, "q_var": torch.zeros_like(p.detach())}},
            dtype=CDT,
        )
        assert r.converged
        r.v.abs().sum().backward()
        assert p.grad is not None
        assert tuple(p.grad.shape) == (3, 1)
        assert bool(torch.isfinite(p.grad).all())
        assert torch.max(torch.abs(p.grad)).item() > 0.0
