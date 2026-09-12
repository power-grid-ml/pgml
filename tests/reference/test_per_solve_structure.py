"""What a solve pays ONCE per grid rather than once per call.

Structural quantities — which zero-impedance branches collapse, the node-phase layout,
the per-row voltage bases, the voltage-regulating terminals — are properties of the
network, not of the operating point. They are resolved once per solve and, where a
:class:`~pgml.solver.PowerFlowSystem` carries them, once per grid. These tests pin the
COUNT of those resolutions, because nothing in a result reveals that one of them was
repeated: a regression here is invisible except in the time a small grid takes, where a
repeated walk of the branch list is a large fraction of the solve.
"""

from __future__ import annotations

import pytest
import torch

import pgml.assembly._fusion as fusion_mod
import pgml.geometry.sequence as seq_mod
import pgml.solver.harmonic_flow as hf_mod
import pgml.solver.power_flow as pf_mod
from pgml.grids import synthetic_feeder
from pgml.schemas.grid_schema import Switch
from pgml.solver import prepare_power_flow, solve_harmonic_flow, solve_power_flow

CDT = torch.complex128


@pytest.fixture(scope="module")
def grid():
    return synthetic_feeder(20)


@pytest.fixture(scope="module")
def near_ideal_switch_grid():
    """A feeder whose closed micro-ohm tie switch lifts the mismatch floor above ``tol``.

    ``Σ_j |Y_ij| V_base,j`` of the switch's terminal rows is then ~1e11 VA, so the
    per-row precision floor (4 eps of that, in per unit) is 2e-7 pu — two decades above
    the default tolerance, which is the case the exact row scale exists for.
    """
    g = synthetic_feeder(20, tie_switches=1)
    return g.model_copy(
        update={
            "branches": [
                b.model_copy(update={"closed": True, "resistance_ohm": 1.0e-6})
                if isinstance(b, Switch)
                else b
                for b in g.branches
            ]
        }
    )


@pytest.fixture(scope="module")
def positive_sequence_grid():
    """A feeder whose lines carry the skin-effect positive-sequence harmonic model."""
    g = synthetic_feeder(20)
    for ln in g.branches:
        ln.harmonic_line_model = "positive_sequence"
        ln.harmonic_skin_effect = True
    return g


def _count(monkeypatch, name, *modules):
    """Count calls to ``name`` through every module that binds it.

    A module that imports the function by name holds its own reference, so the counter
    has to replace the name in each of them to see every call.
    """
    calls = {"n": 0}
    for module in modules:
        orig = getattr(module, name)

        def counted(*a, _orig=orig, **kw):
            calls["n"] += 1
            return _orig(*a, **kw)

        monkeypatch.setattr(module, name, counted)
    return calls


def _count_branch_walks(monkeypatch):
    return _count(monkeypatch, "zero_impedance_branches", fusion_mod, pf_mod, hf_mod)


def test_the_branch_list_is_walked_once_per_solve(monkeypatch, grid):
    """The fusion map and the modeling gate share one ``zero_impedance_branches`` walk.

    Both answer questions about the same list, and the assembly that follows re-resolved
    it a third time when it was handed ``None`` instead of the resolved result.
    """
    calls = _count_branch_walks(monkeypatch)
    res = solve_power_flow(grid, dtype=CDT)
    assert res.converged
    assert calls["n"] == 1, f"{calls['n']} branch walks in one solve"


def test_a_prepared_system_walks_the_branch_list_only_when_prepared(monkeypatch, grid):
    """Preparing resolves the structure; the solves that reuse it resolve nothing."""
    system = prepare_power_flow(grid, dtype=CDT)
    calls = _count_branch_walks(monkeypatch)
    for _ in range(3):
        assert solve_power_flow(grid, dtype=CDT, system=system).converged
    assert calls["n"] == 0, f"{calls['n']} branch walks on the prepared path"


def test_a_harmonic_study_walks_the_branch_list_once_per_order_set(monkeypatch, grid):
    """One walk for the study plus one for its inner fundamental solve, not one per order.

    The study resolves the structure itself and hands the result to every order's
    assembly.
    """
    calls = _count_branch_walks(monkeypatch)
    hf = solve_harmonic_flow(grid, [1, 5, 7, 11, 13], dtype=CDT)
    assert hf.pf.converged
    assert calls["n"] <= 2, f"{calls['n']} branch walks for five orders"


def test_the_fundamental_solve_runs_no_skin_effect_fit(
    monkeypatch, positive_sequence_grid
):
    """``m(f0) = 1`` is known, so the fundamental never fits an equivalent ``Rdc``."""
    calls = _count(monkeypatch, "fit_equivalent_rdc", seq_mod)
    assert solve_power_flow(positive_sequence_grid, dtype=CDT).converged
    assert calls["n"] == 0, f"{calls['n']} skin-effect fits at the fundamental"

    # Above the fundamental the fit IS the model and must still run.
    hf = solve_harmonic_flow(positive_sequence_grid, [1, 5], dtype=CDT)
    assert hf.pf.converged
    assert calls["n"] >= 1


def test_the_row_scale_of_the_mismatch_floor_is_computed_once_per_solve(
    monkeypatch, near_ideal_switch_grid
):
    """The ``|Y|`` pass behind the per-row mismatch floor is not per iteration.

    It reads the whole matrix, so paying it per iteration would dominate a large
    system's solve; a prepared system pays it once per grid instead. This grid's
    near-ideal switch raises its terminal rows' cancellation scale until the floor
    really governs, which is the case that needs the exact scale.
    """
    calls = _count(monkeypatch, "_abs_row_scale", pf_mod)
    res = solve_power_flow(near_ideal_switch_grid, dtype=CDT)
    assert res.converged and res.diagnostics.mismatch_floor_pu > 1.0e-8
    assert calls["n"] == 1, f"{calls['n']} |Y| passes for {res.iterations} iterations"

    system = prepare_power_flow(near_ideal_switch_grid, dtype=CDT)
    calls["n"] = 0
    for _ in range(3):
        solve_power_flow(near_ideal_switch_grid, dtype=CDT, system=system)
    assert calls["n"] == 0, f"{calls['n']} |Y| passes on the prepared path"


def test_no_y_pass_where_the_precision_floor_cannot_reach_the_tolerance(
    monkeypatch, grid
):
    """A floor far below the requested tolerance is decided by a bound, not by ``|Y|``.

    Both mismatch thresholds are ``max(tol, floor · s_scale_row)``, so on a grid whose
    cancellation scale cannot lift the floor to the tolerance they are exactly the
    tolerance — and one fused reduction over the matrix proves that for every row at a
    fraction of the exact pass. The thresholds must come out the same as the exact route's:
    the prepared system carries the exact scale, so its voltages pin it.
    """
    exact = _count(monkeypatch, "_abs_row_scale", pf_mod)
    bound = _count(monkeypatch, "_row_scale_bound", pf_mod)
    res = solve_power_flow(grid, dtype=CDT)
    assert res.converged and res.iterations >= 3
    assert res.diagnostics.mismatch_floor_pu == pytest.approx(1.0e-8)
    assert exact["n"] == 0, f"{exact['n']} |Y| passes for an inert floor"
    assert bound["n"] == 1, f"{bound['n']} row-scale bounds per solve"

    system = prepare_power_flow(grid, dtype=CDT)
    prepared = solve_power_flow(grid, dtype=CDT, system=system)
    assert torch.equal(prepared.v, res.v)
    assert prepared.diagnostics.mismatch_floor_pu == res.diagnostics.mismatch_floor_pu


def test_a_prepared_system_carries_the_structural_quantities(monkeypatch, grid):
    """The node layout and the per-row voltage bases come from the system, not the grid."""
    system = prepare_power_flow(grid, dtype=CDT)
    layout = _count(monkeypatch, "node_phase_index", pf_mod)
    bases = _count(monkeypatch, "_node_voltage_bases", pf_mod)
    ref = solve_power_flow(grid, dtype=CDT)
    assert layout["n"] >= 1 and bases["n"] == 1  # the one-shot path builds both
    layout["n"] = bases["n"] = 0
    res = solve_power_flow(grid, dtype=CDT, system=system)
    assert layout["n"] == 0 and bases["n"] == 0
    assert torch.equal(res.v, ref.v)
