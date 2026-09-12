"""Newton power-flow method (`solve_power_flow(method="newton")`).

Newton on the real power-balance residual, warm-started by the LINEAR const-Z solution
(OpenDSS-style). Asserts:

- it reproduces the current-injection fixed point where both converge (CIGRE LV, 1-phase
  and 3-phase) — same solution, far fewer iterations (quadratic);
- it converges right at the loadability nose, where the current-injection fixed point
  oscillates and fails (a clean 2-bus radial with a closed-form nose);
- it still fails (no fabricated solution) past the nose, where no solution exists.

Differentiability of the Newton path is covered by ``tests/differentiability``.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.oracles.grids import cigre_lv_full_grid
from pgml.errors import InputError
from pgml.schemas.grid_schema import Generator, Grid, Line, Load, Node, Phase, Source
from pgml.solver.power_flow import loadability_limit, solve_power_flow

CDT = torch.complex128
_E, _R, _X = 230.0, 0.5, 0.5  # 2-bus radial: source EMF [V], line R/X [Ohm]


def _two_bus(p_w: float) -> Grid:
    """Stiff source --R+jX-- const-P (unity-PF) load; closed-form P-V nose."""
    l_h = _X / (2.0 * math.pi * 50.0)
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=_E, phases=(Phase.A,)),
            Node(id=2, u_rated_v=_E, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1.0,
                series_resistance_ohm_per_m=[[_R]],
                series_inductance_h_per_m=[[l_h]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(_E,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[1.0e-9]],
                inductance_h=[[1.0e-12]],
            ),
            Load(id=30, node=2, phases=(Phase.A,), p_nom_w=p_w, q_nom_var=0.0),
        ],
    )


def _two_bus_with_generation(p_load_w: float, p_gen_w: float) -> Grid:
    """:func:`_two_bus` plus a generator at the load bus (for the λ-ramp choice)."""
    grid = _two_bus(p_load_w)
    gen = Generator(id=40, node=2, phases=(Phase.A,), p_nom_w=p_gen_w, q_nom_var=0.0)
    return grid.model_copy(update={"appliances": [*grid.appliances, gen]})


def _nose_power() -> float:
    return _E**2 * (math.sqrt(_R**2 + _X**2) - _R) / (2.0 * _X**2)


def _v2_pu(res) -> float:
    return abs(res.v.reshape(-1)[res.index.row(2, Phase.A)].item()) / _E


class TestNewtonPowerFlow:
    def test_matches_current_injection_cigre(self) -> None:
        for mode in (PhaseMode.SINGLE_PHASE_EQUIV, PhaseMode.THREE_PHASE):
            grid, _ = cigre_lv_full_grid(phase_mode=mode)
            # Agreement is asserted in VOLTS at the 1e-8 level, which is three orders
            # tighter than the engine's per-unit power-mismatch default allows on a
            # 20 kV/400 V grid, so both solves are driven by the per-unit VOLTAGE
            # criterion (1e-13 pu ~ 4e-11 V on the LV rows). The power-mismatch
            # tolerance stays at its default: on this grid it bottoms out at ~1e-9 pu
            # (the cancellation scale of Y·V at the stiff 20 kV source).
            ci = solve_power_flow(
                grid,
                slack="ideal",
                method="current_injection",
                tol_update_pu=1e-13,
                max_iter=200,
                dtype=CDT,
            )
            nt = solve_power_flow(
                grid,
                slack="ideal",
                method="newton",
                tol_update_pu=1e-13,
                max_iter=50,
                dtype=CDT,
            )
            assert ci.converged and nt.converged
            assert (ci.v - nt.v).abs().max().item() < 1e-8
            assert nt.iterations < ci.iterations  # quadratic vs linear

    def test_converges_at_the_nose_where_fixed_point_fails(self) -> None:
        p_max = _nose_power()
        grid = _two_bus(0.999 * p_max)  # right at the loadability limit
        ci = solve_power_flow(
            grid,
            slack="ideal",
            method="current_injection",
            tol=1e-8,
            max_iter=100,
            dtype=CDT,
        )
        nt = solve_power_flow(
            grid,
            slack="ideal",
            method="newton",
            tol=1e-8,
            max_iter=50,
            dtype=CDT,
        )
        # The current-injection fixed point does not settle this close to the nose;
        # Newton (linear const-Z init + quadratic steps) does.
        assert not ci.converged
        assert nt.converged
        assert nt.iterations < 20
        # Converged onto the stable (upper) branch with a ~zero power mismatch.
        assert nt.diagnostics.mismatch_max_a < 1e-5
        assert 0.5 < _v2_pu(nt) < 0.65  # upper-branch nose voltage (~0.55 pu)

    def test_does_not_fabricate_a_solution_past_the_nose(self) -> None:
        grid = _two_bus(1.10 * _nose_power())  # beyond the nose: no solution exists
        nt = solve_power_flow(
            grid,
            slack="ideal",
            method="newton",
            tol=1e-8,
            max_iter=50,
            dtype=CDT,
        )
        assert not nt.converged

    def test_fast_quadratic_convergence(self) -> None:
        grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        nt = solve_power_flow(
            grid,
            slack="ideal",
            method="newton",
            tol=1e-10,
            max_iter=50,
            dtype=CDT,
        )
        assert nt.converged and nt.iterations <= 8  # quadratic from the const-Z init

    def test_matrix_free_matches_dense(self) -> None:
        """Jacobian-free Newton-Krylov reaches the same solution as the dense Jacobian."""
        grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        de = solve_power_flow(
            grid,
            slack="ideal",
            method="newton",
            linear_solver="dense",
            tol=1e-10,
            max_iter=50,
            dtype=CDT,
        )
        mf = solve_power_flow(
            grid,
            slack="ideal",
            method="newton",
            linear_solver="matrix_free",
            tol=1e-8,
            max_iter=50,
            dtype=CDT,
        )
        assert de.converged and mf.converged
        assert (de.v - mf.v).abs().max().item() < 1e-6

    def test_invalid_linear_solver_raises(self) -> None:
        import pgml

        grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        with pytest.raises(pgml.errors.InputError):
            solve_power_flow(grid, method="newton", linear_solver="krylov")


class TestLoadabilityContinuation:
    def test_feasible_margin_matches_nose(self) -> None:
        # load = 0.8 * nose  ->  it breaks at λ* = nose / load = 1.25 (margin 0.25).
        res = loadability_limit(
            _two_bus(0.8 * _nose_power()),
            slack="ideal",
            lambda_max=3.0,
            lambda_step=0.1,
            dtype=CDT,
        )
        assert res.feasible
        assert not res.capped  # a genuine nose was found inside the ramp
        assert res.breaking_lambda == pytest.approx(1.25, abs=0.05)
        assert res.margin == pytest.approx(0.25, abs=0.05)
        # the load bus is the collapse point AND the limiting injection
        assert res.critical_nodes[0]["node_id"] == 2
        assert res.limiting_loads[0]["appliance_id"] == 30
        assert res.limiting_loads[0]["responsibility"] == pytest.approx(1.0)

    def test_capped_ramp_reports_lower_bound(self) -> None:
        # load = 0.1 * nose -> the true nose sits at λ* = 10, far past lambda_max:
        # every ramp step converges, so the result is a censored LOWER BOUND.
        res = loadability_limit(
            _two_bus(0.1 * _nose_power()),
            slack="ideal",
            lambda_max=2.0,
            lambda_step=0.5,
            dtype=CDT,
        )
        assert res.capped
        assert res.feasible
        assert res.breaking_lambda == pytest.approx(2.0)

    def test_infeasible_load_past_nose(self) -> None:
        # load = 1.2 * nose  ->  infeasible: it breaks at λ* = 1/1.2 ≈ 0.833 < 1.
        res = loadability_limit(
            _two_bus(1.2 * _nose_power()),
            slack="ideal",
            lambda_max=3.0,
            lambda_step=0.1,
            dtype=CDT,
        )
        assert not res.feasible
        assert res.breaking_lambda == pytest.approx(0.833, abs=0.05)
        assert res.margin < 0.0

    def test_the_ramp_option_decides_what_lambda_multiplies(self) -> None:
        """``ramp="load"`` holds generation at nameplate; ``"all"`` scales it too.

        Both limits are known in closed form on this two-bus feeder. With a load of
        ``0.8 P*`` and a generator of ``0.3 P*`` at the same bus, the net load reaches the
        nose ``P*`` at

        - ``λ_load = (P* + P_gen) / P_load = 1.625``  (generation fixed), and
        - ``λ_all  = P* / (P_load − P_gen)  = 2.0``   (generation scaled with the load),

        so the choice is worth 23 % of the reported margin on a feeder with modest
        generation — which is why it is named in the result.
        """
        nose = _nose_power()
        p_load, p_gen = 0.8 * nose, 0.3 * nose
        grid = _two_bus_with_generation(p_load, p_gen)
        got = {
            ramp: loadability_limit(
                grid,
                slack="ideal",
                lambda_max=4.0,
                lambda_step=0.1,
                dtype=CDT,
                ramp=ramp,
            )
            for ramp in ("all", "load")
        }
        assert got["load"].breaking_lambda == pytest.approx(
            (nose + p_gen) / p_load, abs=0.05
        )
        assert got["all"].breaking_lambda == pytest.approx(
            nose / (p_load - p_gen), abs=0.05
        )
        assert got["load"].breaking_lambda < got["all"].breaking_lambda
        for ramp, res in got.items():
            assert res.ramp == ramp  # the result records what it measured
            assert not res.capped and res.feasible

    def test_the_two_ramps_agree_without_generation(self) -> None:
        """With loads only there is nothing to hold fixed, so both ramps coincide."""
        grid = _two_bus(0.8 * _nose_power())
        kw = dict(slack="ideal", lambda_max=3.0, lambda_step=0.1, dtype=CDT)
        assert loadability_limit(
            grid, ramp="load", **kw
        ).breaking_lambda == pytest.approx(
            loadability_limit(grid, ramp="all", **kw).breaking_lambda
        )

    def test_unknown_ramp_raises(self) -> None:
        with pytest.raises(InputError):
            loadability_limit(_two_bus(1000.0), ramp="generation")

    def test_the_default_ramp_is_the_load_only_one(self) -> None:
        """``ramp=None`` resolves the documented default, which is the textbook ramp.

        On a feeder with generation the default decides a reported number, so it is
        pinned here and in ``solver.loadability.ramp`` rather than in the signature.
        """
        from pgml import defaults

        assert defaults.get("solver.loadability.ramp") == "load"
        nose = _nose_power()
        grid = _two_bus_with_generation(0.8 * nose, 0.3 * nose)
        kw = dict(slack="ideal", lambda_max=4.0, lambda_step=0.1, dtype=CDT)
        res = loadability_limit(grid, **kw)
        assert res.ramp == "load"
        assert res.breaking_lambda == pytest.approx(
            loadability_limit(grid, ramp="load", **kw).breaking_lambda
        )

    def test_breaking_lambda_is_a_lower_bound_on_the_nose(self) -> None:
        """The corrector fails before the singularity, so tightening it moves λ* UP.

        This is the documented bias of a step-and-bisect on Newton feasibility against a
        true arc-length continuation: the reported limit is the largest λ whose corrector
        converged, never more than the nose.
        """
        grid = _two_bus(0.8 * _nose_power())
        kw = dict(slack="ideal", lambda_max=3.0, lambda_step=0.1, dtype=CDT)
        loose = loadability_limit(grid, bisect_tol=1e-1, **kw)
        tight = loadability_limit(grid, bisect_tol=1e-4, **kw)
        exact = 1.25  # the closed-form nose of this feeder at this load
        assert loose.breaking_lambda <= tight.breaking_lambda + 1e-9
        assert tight.breaking_lambda <= exact + 1e-3
        assert (
            abs(tight.breaking_lambda - exact)
            < abs(loose.breaking_lambda - exact) + 1e-9
        )

    @pytest.mark.slow
    def test_cigre_has_positive_margin_and_localizes(self) -> None:
        grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        res = loadability_limit(
            grid,
            slack="ideal",
            lambda_max=5.0,
            lambda_step=1.0,
            dtype=CDT,
        )
        assert res.feasible and res.margin > 0.0
        assert res.critical_nodes and res.limiting_loads
        assert 0.0 <= res.limiting_loads[0]["responsibility"] <= 1.0
        assert res.converged_lambdas[0] == 0.0  # ramp starts at the feasible base
