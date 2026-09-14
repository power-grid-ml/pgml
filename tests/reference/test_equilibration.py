"""Diagonal equilibration of the factored systems: same answers, better conditioning.

An SI-unit nodal admittance spans decades (a stiff source row near 1e5 S next to a
low-voltage cable row near 1e-2 S, and one more decade per harmonic order), so it is
badly SCALED rather than intrinsically ill-conditioned. The solvers therefore factor the
equilibrated matrix ``D_r A D_c`` and undo the scaling on the solution
(:mod:`pgml.solver.equilibration`, on by default).

What these tests pin:

- the scaling is INVISIBLE in the result: every entry point returns what the unscaled
  solve returns, to round-off, for both slack modes, all three factorization backends,
  mixed precision and the Woodbury low-rank path;
- the conditioning measurably improves on real networks (the numbers in the docstrings
  are the measured ones, CPU complex128);
- the scale factors are powers of two, so the scaled matrix is exact in binary floating
  point and the equilibration introduces no rounding error of its own;
- gradients are unchanged (bit-for-bit on a small system), and the equilibrated path
  passes a float64 gradcheck;
- the mode is the documented default and remains overridable per call.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml import defaults
from pgml.assembly import node_phase_index
from pgml.assembly._params import phase_voltage_magnitude
from pgml.errors import InputError
from pgml.grids import cigre_lv_full_grid, ieee33_geometry_grid, synthetic_feeder
from pgml.multigrid import merge_grids
from pgml.schemas.grid_schema import Switch
from pgml.solver import solve_harmonic_flow, solve_power_flow
from pgml.solver.equilibration import (
    EQUILIBRATION_MODES,
    equilibrate_matrix,
    equilibration_scales,
    resolve_equilibration,
)
from pgml.solver.harmonic import (
    estimate_condition,
    lu_factor_system,
    solve_factored,
    solve_harmonic,
)
from pgml.solver.lowrank import low_rank_update, solve_factored_updated

CDT = torch.complex128


def _bases(grid, index) -> torch.Tensor:
    nb = {int(n.id): n for n in grid.nodes}
    return torch.tensor(
        [
            phase_voltage_magnitude(float(nb[int(i)].u_rated_v), len(nb[int(i)].phases))
            for i in index.node_ids.tolist()
        ],
        dtype=torch.float64,
    )


def _spread_system(n: int = 12, spread: float = 1.0e6, seed: int = 0):
    """A complex system whose rows and columns span ``spread`` decades of scale."""
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=CDT) + 3.0 * torch.eye(n, dtype=CDT)
    d = torch.logspace(0.0, math.log10(spread), n, dtype=torch.float64)
    return a * d.unsqueeze(-1) * d.unsqueeze(-2), torch.randn(n, dtype=CDT)


@pytest.fixture(scope="module")
def ieee33():
    grid, _ = ieee33_geometry_grid()
    return grid


class TestScaleFactors:
    def test_symmetric_scale_is_the_inverse_square_root_of_the_diagonal(self):
        a, _ = _spread_system()
        d_row, d_col = equilibration_scales(a, mode="symmetric", power_of_two=False)
        assert d_row is d_col  # a congruence: one scale, both sides
        want = a.diagonal().abs().rsqrt()
        assert torch.allclose(d_row, want, rtol=1e-12)

    def test_power_of_two_scaling_is_exact_in_floating_point(self):
        """``D A D`` with power-of-two factors introduces NO rounding error.

        Dividing the scaled matrix back by the scales must reproduce the original
        bit-for-bit, which is why the rounding is the default: the equilibration cannot
        move a solution it was only meant to condition.
        """
        a, _ = _spread_system()
        a_hat, d_row, d_col = equilibrate_matrix(
            a, mode="symmetric", power_of_two=True
        )
        for d in (d_row, d_col):
            assert torch.equal(torch.log2(d), torch.log2(d).round())
        back = a_hat / d_row.unsqueeze(-1) / d_col.unsqueeze(-2)
        assert torch.equal(back, a)

    def test_unrounded_scaling_is_not_exact(self):
        """The contrast: the unrounded scaling does perturb the matrix (by ~eps)."""
        a, _ = _spread_system()
        a_hat, d_row, d_col = equilibrate_matrix(
            a, mode="symmetric", power_of_two=False
        )
        back = a_hat / d_row.unsqueeze(-1) / d_col.unsqueeze(-2)
        assert not torch.equal(back, a)
        assert torch.allclose(back, a, rtol=1e-14)

    def test_a_zero_diagonal_row_keeps_scale_one(self):
        """A structurally empty row must not become an infinity."""
        a, _ = _spread_system()
        a = a.clone()
        a[3, 3] = 0.0
        d_row, _ = equilibration_scales(a, mode="symmetric")
        assert torch.isfinite(d_row).all()
        assert float(d_row[3]) == 1.0

    def test_mode_resolution_and_validation(self):
        documented = str(defaults.get("solver.equilibration.mode"))
        assert resolve_equilibration(None) == documented
        assert resolve_equilibration(True) == documented
        assert resolve_equilibration(False) == "off"
        for mode in EQUILIBRATION_MODES:
            assert resolve_equilibration(mode) == mode
        with pytest.raises(InputError, match="symmetric"):
            resolve_equilibration("van_der_sluis")
        with pytest.raises(InputError, match="symmetric"):
            resolve_equilibration("row_column")

    def test_the_documented_default_is_symmetric(self):
        assert defaults.get("solver.equilibration.mode") == "symmetric"
        assert defaults.get("solver.equilibration.power_of_two") is True


class TestLinearSolveIsUnchanged:
    """Every factored path answers the SI system, whatever the scaling."""

    @pytest.mark.parametrize("backend", ["dense", "sparse"])
    def test_factored_solve_matches_the_unscaled_solve(self, backend):
        a, b = _spread_system()
        ref = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        v = solve_factored(
            lu_factor_system(a, backend=backend, equilibrate="symmetric"), b
        )
        assert float((v - ref).abs().max() / ref.abs().max()) < 1e-13

    def test_direct_solve_matches_in_both_slack_modes(self):
        a, b = _spread_system()
        fixed = torch.tensor([0, 1])
        vf = torch.tensor([1.0 + 0j, 0.5 + 0j], dtype=CDT)
        for kw in ({}, {"fixed_rows": fixed, "v_fixed": vf}):
            ref = solve_harmonic(
                a.unsqueeze(0), b.unsqueeze(0), equilibrate="off", **kw
            )
            got = solve_harmonic(
                a.unsqueeze(0), b.unsqueeze(0), equilibrate="symmetric", **kw
            )
            assert float((got - ref).abs().max() / ref.abs().max()) < 1e-13

    def test_mixed_precision_composes_with_equilibration(self):
        a, b = _spread_system()
        ref = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        for mode in ("off", "symmetric"):
            fac = lu_factor_system(a, precision="mixed", equilibrate=mode)
            v = solve_factored(fac, b)
            assert float((v - ref).abs().max() / ref.abs().max()) < 1e-12

    def test_woodbury_update_matches_a_fresh_factorization(self):
        """The low-rank path reads ``A^{-1}U`` through the same scaled solve."""
        a, b = _spread_system()
        n = a.shape[-1]
        u = torch.zeros(n, 2, dtype=CDT)
        u[3, 0] = 1.0
        u[4, 1] = 1.0
        c = torch.tensor([[3.0e2 + 1e2j, 0.0], [0.0, -2.0e2 + 5e1j]], dtype=CDT)
        ref = torch.linalg.solve(a + u @ c @ u.conj().T, b.unsqueeze(-1)).squeeze(-1)
        for mode in ("off", "symmetric"):
            upd = low_rank_update(lu_factor_system(a, equilibrate=mode), u, c)
            v = solve_factored_updated(upd, b)
            assert float((v - ref).abs().max() / ref.abs().max()) < 1e-12

    def test_block_backend_matches(self):
        a1, b1 = _spread_system(5, seed=1)
        a2, b2 = _spread_system(7, seed=2)
        big = torch.zeros(12, 12, dtype=CDT)
        big[:5, :5] = a1
        big[5:, 5:] = a2
        rhs = torch.cat([b1, b2])
        blocks = [torch.arange(5), torch.arange(5, 12)]
        ref = torch.linalg.solve(big, rhs.unsqueeze(-1)).squeeze(-1)
        v = solve_factored(
            lu_factor_system(
                big, backend="block", block_rows=blocks, equilibrate="symmetric"
            ),
            rhs,
        )
        assert float((v - ref).abs().max() / ref.abs().max()) < 1e-13


class TestConditioning:
    """What the equilibration is for, measured on real networks."""

    def test_condition_estimate_drops_on_a_scaled_system(self):
        a, _ = _spread_system(spread=1.0e6)
        plain = estimate_condition(lu_factor_system(a, equilibrate="off"))
        scaled = estimate_condition(lu_factor_system(a, equilibrate="symmetric"))
        assert plain > 1.0e10  # measured 1.3e12
        assert scaled < 1.0e3  # measured 9.6e1

    def test_ieee33_harmonic_system_is_mostly_a_scaling_artefact(self, ieee33):
        """Order 13 on IEEE-33: κ 8.0e8 as assembled, 1.5e3 equilibrated (1-norm est)."""
        from pgml.solver.harmonic_flow import assemble_harmonic_ybus

        y, _ = assemble_harmonic_ybus(ieee33, [13], dtype=CDT)
        y = y.reshape(y.shape[-1], y.shape[-1])
        plain = estimate_condition(lu_factor_system(y, equilibrate="off"))
        scaled = estimate_condition(lu_factor_system(y, equilibrate="symmetric"))
        assert plain > 1.0e8
        assert scaled < 1.0e4
        assert plain / scaled > 1.0e4

    def test_cigre_three_phase_fundamental_improves(self):
        """Three-phase CIGRE LV, fundamental free block: 4.8e3 -> 2.1e3 (1-norm est).

        The equilibrated figure does not depend on how the benchmark's three ideal
        bus-bus switches are treated, and the unscaled one does: with the switches
        collapsed (the default) the estimate is 4.8e3, and with the documented near-ideal
        1e-04 Ohm stand-in stamped instead it is 5.5e4 — while both equilibrate to the
        same 2.1e3 (to five digits: 2097.92 fused against 2097.95 stamped, two matrices
        of 123 and 132 rows). Exact bus fusion and the equilibration remove the same
        scaling artefact, one from the data and one from the factorization.
        """
        from pgml.convert.pandapower import PhaseMode

        from pgml.solver import prepare_power_flow

        grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
        plain = estimate_condition(
            prepare_power_flow(grid, equilibrate="off").factorization
        )
        scaled = estimate_condition(
            prepare_power_flow(grid, equilibrate="symmetric").factorization
        )
        assert plain > 2.0e3
        assert scaled < plain / 2.0

        stand_in, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
        stand_in.branches = [
            b.model_copy(update={"resistance_ohm": 1.0e-4})
            if isinstance(b, Switch)
            else b
            for b in stand_in.branches
        ]
        plain_stamped = estimate_condition(
            prepare_power_flow(stand_in, equilibrate="off").factorization
        )
        scaled_stamped = estimate_condition(
            prepare_power_flow(stand_in, equilibrate="symmetric").factorization
        )
        assert plain_stamped > 10.0 * plain
        assert scaled_stamped == pytest.approx(scaled, rel=1e-4)


class TestPowerFlowIsUnchanged:
    def test_fundamental_solution_is_unchanged(self, ieee33):
        index = node_phase_index(ieee33)
        bases = _bases(ieee33, index)
        ref = solve_power_flow(ieee33, equilibrate="off", tol_update_pu=1e-12)
        got = solve_power_flow(ieee33, equilibrate="symmetric", tol_update_pu=1e-12)
        assert got.converged
        assert float(((got.v - ref.v).abs() / bases).max()) < 1e-11
        assert got.iterations == ref.iterations

    def test_newton_solution_is_unchanged(self, ieee33):
        index = node_phase_index(ieee33)
        bases = _bases(ieee33, index)
        ref = solve_power_flow(
            ieee33, method="newton", equilibrate="off", tol_update_pu=1e-12
        )
        got = solve_power_flow(
            ieee33, method="newton", equilibrate="symmetric", tol_update_pu=1e-12
        )
        assert got.converged and ref.converged
        assert float(((got.v - ref.v).abs() / bases).max()) < 1e-11

    def test_harmonic_orders_are_unchanged(self, ieee33):
        index = node_phase_index(ieee33)
        bases = _bases(ieee33, index)
        orders = [1, 5, 13]
        ref = solve_harmonic_flow(ieee33, orders, equilibrate="off")
        got = solve_harmonic_flow(ieee33, orders, equilibrate="symmetric")
        assert float(((got.v - ref.v).abs() / bases).max()) < 1e-10

    def test_block_backend_ensemble_is_unchanged(self):
        merged = merge_grids([synthetic_feeder(6), synthetic_feeder(6)])
        rows = merged.block_rows()
        kw = dict(linear_solver="block", block_rows=rows, tol_update_pu=1e-12)
        ref = solve_power_flow(merged.grid, equilibrate="off", **kw)
        got = solve_power_flow(merged.grid, equilibrate="symmetric", **kw)
        assert got.converged and ref.converged
        assert float((got.v - ref.v).abs().max()) < 1e-6  # volts, on a 20 kV feeder

    def test_prepared_system_must_agree_with_the_solve(self, ieee33):
        from pgml.solver import prepare_power_flow

        system = prepare_power_flow(ieee33, equilibrate="off")
        with pytest.raises(InputError, match="equilibrate"):
            solve_power_flow(ieee33, system=system, equilibrate="symmetric")
        assert solve_power_flow(ieee33, system=system, equilibrate="off").converged


class TestGradientsAreUnchanged:
    def test_linear_solve_gradients_match_the_unscaled_path(self):
        a0, b0 = _spread_system(8, spread=1.0e3)
        grads = {}
        for mode in ("off", "symmetric"):
            a = a0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            v = solve_factored(lu_factor_system(a, equilibrate=mode), b)
            v.abs().square().sum().backward()
            grads[mode] = (a.grad.clone(), b.grad.clone())
        for g, ref in zip(grads["symmetric"], grads["off"]):
            assert torch.allclose(
                g, ref, rtol=1e-11, atol=1e-11 * float(ref.abs().max())
            )

    def test_gradcheck_through_an_equilibrated_factored_solve(self):
        a0, b0 = _spread_system(6, spread=1.0e3)

        def f(mat, rhs):
            return solve_factored(lu_factor_system(mat, equilibrate="symmetric"), rhs)

        assert torch.autograd.gradcheck(
            f,
            (a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)),
            eps=1e-6,
            atol=1e-6,
        )

    def test_power_flow_parameter_gradient_is_unchanged(self):
        grid = synthetic_feeder(6)
        line = grid.branches[0]
        r0 = torch.as_tensor(line.series_resistance_ohm_per_m, dtype=torch.float64)
        grads = {}
        for mode in ("off", "symmetric"):
            r = r0.clone().requires_grad_(True)
            res = solve_power_flow(
                grid,
                param_overrides={
                    ("line", int(line.id), "series_resistance_ohm_per_m"): r
                },
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
