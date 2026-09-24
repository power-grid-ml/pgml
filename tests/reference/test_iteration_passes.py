"""What the current-injection fixed point evaluates once per iteration.

Each iteration walks a scenarios-by-rows tensor several times: the device
currents, the residual and the two per-unit convergence criteria. Three of those
passes carry no information, and dropping them must not move a single digit:

1. The convergence test reads both criteria off the row MAXIMUM wherever the
   threshold is one number for every row, instead of comparing every row twice
   and reducing four times. ``max_i x_i <= t`` and ``x_i <= t for all i`` are the
   same statement — including when a row is not finite.
2. A finished scenario is held at the iterate that finished it. Until one
   finishes there is nothing to hold, and the select over the whole batch is
   skipped.
3. A group of constant-power devices has the ZIP law as its identity, so the
   terminal magnitude, the per-unit ratio and both polynomials are not formed.

Each test compares the economical form against the explicit one it replaces.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pgml.assembly import build_injection_plan, injections_from_plan, node_phase_index
from pgml.grids import synthetic_feeder
from pgml.schemas.grid_schema import LoadModel, ZipCoefficients
from pgml.solver.power_flow import _BatchIterationState, _PuConvergence


def _criterion(y_eff, n, *, tol=1e-8, floor=1e-15, device="cpu"):
    return _PuConvergence(
        v_base=torch.full((n,), 230.0, dtype=torch.float64),
        s_base=1.0e6,
        tol_mismatch_pu=tol,
        tol_update_pu=1e-8,
        floor_update=1e-15,
        floor_mismatch=floor,
        fixed_rows=torch.tensor([0]),
        y_eff=y_eff,
        n=n,
        device=torch.device(device),
        rdt=torch.float64,
        warn=False,
    )


@pytest.mark.parametrize("floor", [1e-15, 1e-3])
def test_the_check_agrees_with_the_explicit_per_row_comparison(floor):
    """Both threshold regimes: uniform (floor inert) and per row (floor binds)."""
    n = 12
    y = torch.eye(n, dtype=torch.complex128) * 5.0
    ctest = _criterion(y, n, floor=floor)
    assert ctest.uniform_mismatch == (floor == 1e-15)

    gen = torch.Generator().manual_seed(7)
    mism = torch.rand(64, n, generator=gen, dtype=torch.float64) * 4e-8
    upd = torch.rand(64, n, generator=gen, dtype=torch.float64) * 4e-8
    mism[3, 5] = float("nan")
    upd[4, 2] = float("inf")

    ok, in_band, mism_max, upd_max = ctest.check(mism, upd)
    want_ok = (mism <= ctest.thr_mismatch).all(dim=-1) & (upd <= ctest.thr_update).all(
        dim=-1
    )
    want_band = (mism <= ctest.ceil_mismatch).all(dim=-1) & (
        upd <= ctest.ceil_update
    ).all(dim=-1)
    assert torch.equal(ok, want_ok)
    assert torch.equal(in_band, want_band)
    for got, want in ((mism_max, mism.amax(dim=-1)), (upd_max, upd.amax(dim=-1))):
        assert bool(((got == want) | (got.isnan() & want.isnan())).all())


def test_nothing_is_held_until_a_scenario_finishes():
    n, b = 6, 8
    y = torch.eye(n, dtype=torch.complex128) * 5.0
    state = _BatchIterationState(
        shape=(b,),
        device=torch.device("cpu"),
        rdt=torch.float64,
        ctest=_criterion(y, n),
    )
    v_new = torch.randn(b, n, dtype=torch.complex128)
    v_old = torch.randn(b, n, dtype=torch.complex128)
    assert state.hold(v_new, v_old) is v_new

    tight = torch.zeros(b, dtype=torch.float64)
    ok = torch.zeros(b, dtype=torch.bool)
    ok[2] = True
    state.step(ok, ok, tight, tight)
    assert state.any_finished
    held = state.hold(v_new, v_old)
    assert torch.equal(
        held, torch.where(state.finished_mask.unsqueeze(-1), v_old, v_new)
    )
    assert torch.equal(held[2], v_old[2])
    assert torch.equal(held[3], v_new[3])


def _plan_for(grid, v):
    index = node_phase_index(grid)
    return build_injection_plan(
        grid, index, [grid.base_frequency_hz], dtype=torch.complex128
    )


def test_the_constant_power_shortcut_is_the_zip_law_itself():
    """The shortcut and the polynomial it replaces agree bit for bit."""
    grid = synthetic_feeder(8)
    n = node_phase_index(grid).size
    plan = _plan_for(grid, None)
    assert plan.uncontrolled and all(g.constant_power for g in plan.uncontrolled)

    torch.manual_seed(0)
    v = torch.polar(
        torch.full((16, n), 230.0, dtype=torch.float64)
        + torch.randn(16, n, dtype=torch.float64),
        torch.randn(16, n, dtype=torch.float64) * 0.01,
    ).to(torch.complex128)

    fast = injections_from_plan(plan, v)
    spelled_out = replace(
        plan,
        uncontrolled=tuple(replace(g, constant_power=False) for g in plan.uncontrolled),
    )
    assert torch.equal(fast, injections_from_plan(spelled_out, v))


def test_a_voltage_dependent_group_keeps_the_polynomial():
    grid = synthetic_feeder(8)
    for appliance in grid.appliances:
        if hasattr(appliance, "load_model"):
            appliance.zip_coefficients = ZipCoefficients(
                z_p=1.0, i_p=0.0, p_p=0.0, z_q=1.0, i_q=0.0, p_q=0.0
            )
            appliance.load_model = LoadModel.ZIP
    plan = _plan_for(grid, None)
    assert not any(g.constant_power for g in plan.uncontrolled)

    n = node_phase_index(grid).size
    v_nominal = torch.full((n,), 230.0, dtype=torch.float64).to(torch.complex128)
    half = injections_from_plan(plan, 0.5 * v_nominal)
    full = injections_from_plan(plan, v_nominal)
    # A constant-impedance device draws a current proportional to its voltage.
    assert torch.allclose(half, 0.5 * full, atol=1e-9 * float(full.abs().max()))
