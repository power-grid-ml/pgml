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

import torch

from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.oracles.grids import cigre_lv_full_grid
from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver.power_flow import solve_power_flow

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
                id=20, from_node=1, to_node=2, from_phases=(Phase.A,),
                to_phases=(Phase.A,), length_m=1.0,
                series_resistance_ohm_per_m=[[_R]],
                series_inductance_h_per_m=[[l_h]], shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=10, node=1, phases=(Phase.A,), u_ref_v=(_E,), u_angle_deg=(0.0,),
                resistance_ohm=[[1.0e-9]], inductance_h=[[1.0e-12]],
            ),
            Load(id=30, node=2, phases=(Phase.A,), p_nom_w=p_w, q_nom_var=0.0),
        ],
    )


def _nose_power() -> float:
    return _E**2 * (math.sqrt(_R**2 + _X**2) - _R) / (2.0 * _X**2)


def _v2_pu(res) -> float:
    return abs(res.v.reshape(-1)[res.index.row(2, Phase.A)].item()) / _E


class TestNewtonPowerFlow:
    def test_matches_current_injection_cigre(self) -> None:
        for mode in (PhaseMode.SINGLE_PHASE_EQUIV, PhaseMode.THREE_PHASE):
            grid, _ = cigre_lv_full_grid(phase_mode=mode)
            ci = solve_power_flow(
                grid, slack="ideal", method="current_injection",
                tol=1e-10, max_iter=200, dtype=CDT,
            )
            nt = solve_power_flow(
                grid, slack="ideal", method="newton", tol=1e-10, max_iter=50, dtype=CDT,
            )
            assert ci.converged and nt.converged
            assert (ci.v - nt.v).abs().max().item() < 1e-8
            assert nt.iterations < ci.iterations  # quadratic vs linear

    def test_converges_at_the_nose_where_fixed_point_fails(self) -> None:
        p_max = _nose_power()
        grid = _two_bus(0.999 * p_max)  # right at the loadability limit
        ci = solve_power_flow(
            grid, slack="ideal", method="current_injection",
            tol=1e-8, max_iter=100, dtype=CDT,
        )
        nt = solve_power_flow(
            grid, slack="ideal", method="newton", tol=1e-8, max_iter=50, dtype=CDT,
        )
        # The current-injection fixed point does not settle this close to the nose;
        # Newton (linear const-Z init + quadratic steps) does.
        assert not ci.converged
        assert nt.converged
        assert nt.iterations < 20
        # Converged onto the stable (upper) branch with a ~zero power mismatch.
        assert nt.diagnostics.power_mismatch_max < 1e-5
        assert 0.5 < _v2_pu(nt) < 0.65  # upper-branch nose voltage (~0.55 pu)

    def test_does_not_fabricate_a_solution_past_the_nose(self) -> None:
        grid = _two_bus(1.10 * _nose_power())  # beyond the nose: no solution exists
        nt = solve_power_flow(
            grid, slack="ideal", method="newton", tol=1e-8, max_iter=50, dtype=CDT,
        )
        assert not nt.converged

    def test_fast_quadratic_convergence(self) -> None:
        grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        nt = solve_power_flow(
            grid, slack="ideal", method="newton", tol=1e-10, max_iter=50, dtype=CDT,
        )
        assert nt.converged and nt.iterations <= 8  # quadratic from the const-Z init
