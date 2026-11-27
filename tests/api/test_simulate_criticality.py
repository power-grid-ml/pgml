"""``simulate(criticality=...)``: when the Jacobian criticality analysis runs.

The keyword is passed to the fundamental solve of both calculations. It controls the
non-convergence diagnostic only, so the solved state (voltages, convergence flag,
iterations, residual) must not depend on it, and the default must keep today's ``"auto"``
behaviour: the analysis runs on a non-converging solve and is absent on a converged one.
"""

from __future__ import annotations

import pytest
import torch

from pgml.errors import ConvergenceError, InputError
from pgml.simulation import SimulationConfig, simulate, simulate_serializable
from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus

CALCULATIONS = ("power_flow", "harmonic")
GRIDS = (single_phase_chain, three_phase_two_bus)


def _config(calculation: str, max_iter: int = 100) -> SimulationConfig:
    return SimulationConfig(
        calculation=calculation, harmonic_orders=[1, 5, 7], max_iter=max_iter
    )


def _state_fields(state) -> tuple:
    return state.converged, state.iterations, state.residual


def _diagnostics(grid_fn, calculation: str, **kwargs) -> dict:
    """The diagnostics of a strict solve that is cut off after one iteration."""
    with pytest.raises(ConvergenceError) as excinfo:
        simulate(grid_fn(), _config(calculation, max_iter=1), **kwargs)
    return excinfo.value.diagnostics


@pytest.mark.parametrize("grid_fn", GRIDS)
@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_default_is_auto(grid_fn, calculation):
    converged_default = simulate(grid_fn(), _config(calculation))
    converged_auto = simulate(grid_fn(), _config(calculation), criticality="auto")
    assert torch.equal(converged_default.v, converged_auto.v)
    assert _state_fields(converged_default) == _state_fields(converged_auto)

    default = _diagnostics(grid_fn, calculation)
    auto = _diagnostics(grid_fn, calculation, criticality="auto")
    # The non-converging default still runs the analysis.
    assert isinstance(default["criticality"], dict)
    assert default == auto


@pytest.mark.parametrize("grid_fn", GRIDS)
@pytest.mark.parametrize("calculation", CALCULATIONS)
@pytest.mark.parametrize("max_iter", [100, 1], ids=["converging", "non_converging"])
def test_never_leaves_the_solved_state_unchanged(grid_fn, calculation, max_iter):
    cfg = _config(calculation, max_iter=max_iter)
    auto = simulate(grid_fn(), cfg, strict=False)
    never = simulate(grid_fn(), cfg, strict=False, criticality="never")
    assert auto.converged is (max_iter == 100)
    assert torch.equal(auto.v, never.v)
    assert _state_fields(auto) == _state_fields(never)
    for bc_auto, bc_never in zip(auto.branch_currents(), never.branch_currents()):
        assert torch.equal(bc_auto.i_from, bc_never.i_from)
        assert torch.equal(bc_auto.i_to, bc_never.i_to)


@pytest.mark.parametrize("grid_fn", GRIDS)
@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_never_skips_the_analysis_and_keeps_the_cheap_diagnostics(grid_fn, calculation):
    auto = _diagnostics(grid_fn, calculation)
    never = _diagnostics(grid_fn, calculation, criticality="never")
    assert isinstance(auto["criticality"], dict)
    assert never["criticality"] is None
    for key in ("converged", "iterations", "mismatch_max_pu", "update_max_pu"):
        assert never[key] == auto[key], key
    assert never["residual_history"] == auto["residual_history"]


@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_always_runs_the_analysis_without_changing_the_state(calculation):
    auto = simulate(single_phase_chain(), _config(calculation))
    always = simulate(single_phase_chain(), _config(calculation), criticality="always")
    assert torch.equal(auto.v, always.v)
    assert _state_fields(auto) == _state_fields(always)


def test_invalid_criticality_raises_before_solving():
    with pytest.raises(InputError, match="criticality"):
        simulate(single_phase_chain(), criticality="sometimes")


def test_simulate_serializable_forwards_criticality():
    cfg = _config("power_flow")
    auto = simulate_serializable(single_phase_chain(), cfg)
    never = simulate_serializable(single_phase_chain(), cfg, criticality="never")
    assert auto.diagnostics == never.diagnostics
    assert auto.nodes == never.nodes
    assert auto.branches == never.branches
    with pytest.raises(InputError, match="criticality"):
        simulate_serializable(single_phase_chain(), cfg, criticality="sometimes")
