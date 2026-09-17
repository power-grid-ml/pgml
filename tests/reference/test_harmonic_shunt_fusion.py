"""Low-rank harmonic device shunt on a grid with fused (zero-impedance) branches.

The fundamental is reported on the grid's full row layout while a fused harmonic
system lives on the reduced rows. A voltage-dependent device derives its harmonic
shunt from its own fundamental terminal voltage, so both solve paths (the batched
low-rank update and the assembled per-scenario factorization) have to read that
voltage on the same layout.
"""

from __future__ import annotations

import pytest
import torch

import pgml.solver.harmonic_flow as harmonic_flow
from pgml.errors import InputError
from pgml.schemas.grid_schema import Line, LoadModel, Switch
from pgml.solver import solve_harmonic_flow

from .test_harmonic_shunt_woodbury import _operating_point, _sparse_load_grid

ORDERS = [1, 5, 7]
POWER = torch.tensor([1_500.0, 2_000.0, 2_500.0], dtype=torch.float64)


def _grid(load_model: LoadModel, *, fused: bool):
    """The 5-node feeder with 1 km segments; ``fused`` replaces line 2-3 by a switch.

    The fused group precedes the load in node order, so the load's reduced row
    differs from its full row.
    """
    base = _sparse_load_grid()
    phases = base.nodes[0].phases
    branches = [
        b.model_copy(update={"length_m": 1_000.0}) if isinstance(b, Line) else b
        for b in base.branches
    ]
    if fused:
        branches[1] = Switch(
            id=branches[1].id,
            from_node=2,
            to_node=3,
            from_phases=phases,
            to_phases=phases,
            closed=True,
        )
    load = base.appliances[1].model_copy(update={"load_model": load_model})
    return base.model_copy(
        update={"branches": branches, "appliances": [base.appliances[0], load]}
    )


def _count_lowrank(monkeypatch) -> list:
    calls: list = []
    original = harmonic_flow.low_rank_update

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(harmonic_flow, "low_rank_update", counted)
    return calls


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize(
    "load_model",
    [LoadModel.CONST_POWER, LoadModel.CONST_IMPEDANCE, LoadModel.CONST_CURRENT],
    ids=lambda m: m.value,
)
def test_batched_lowrank_matches_per_scenario_solve(monkeypatch, load_model, fused):
    grid = _grid(load_model, fused=fused)
    calls = _count_lowrank(monkeypatch)
    batched = solve_harmonic_flow(
        grid, ORDERS, operating_point=_operating_point(POWER), dtype=torch.complex128
    )
    assert calls, "the batched solve is expected to take the low-rank path"
    assert batched.pf.converged

    calls.clear()
    single = torch.stack(
        [
            solve_harmonic_flow(
                grid,
                ORDERS,
                operating_point=_operating_point(p.reshape(1)),
                dtype=torch.complex128,
            ).v[0]
            for p in POWER
        ]
    )
    assert not calls, "a single scenario is expected to take the assembled path"

    scale = single[:, 1:].abs().amax()
    # The fundamental differs by the nonlinear solve's tolerance, not by roundoff.
    assert (batched.v[:, 1:] - single[:, 1:]).abs().amax() <= 1e-9 * scale


@pytest.mark.parametrize(
    "load_model",
    [LoadModel.CONST_IMPEDANCE, LoadModel.CONST_CURRENT],
    ids=lambda m: m.value,
)
def test_lowrank_matches_assembled_system_at_the_same_fundamental(
    monkeypatch, load_model
):
    """With one shared fundamental the two paths agree to roundoff."""
    grid = _grid(load_model, fused=True)
    operating_point = _operating_point(POWER)
    calls = _count_lowrank(monkeypatch)
    result = solve_harmonic_flow(
        grid, ORDERS, operating_point=operating_point, dtype=torch.complex128
    )
    assert calls
    y_bus, current, _ = harmonic_flow.assemble_harmonic_system(
        grid,
        [5, 7],
        result.v[..., 0, :],
        operating_point=operating_point,
        dtype=torch.complex128,
    )
    direct = result.fusion.prolong(torch.linalg.solve(y_bus, current))
    scale = direct.abs().amax()
    assert (result.v[..., 1:, :] - direct).abs().amax() <= 1e-12 * scale


@pytest.mark.parametrize(
    "load_model",
    [LoadModel.CONST_IMPEDANCE, LoadModel.CONST_CURRENT],
    ids=lambda m: m.value,
)
def test_gradcheck_through_fused_lowrank_path(monkeypatch, load_model):
    grid = _grid(load_model, fused=True)
    calls = _count_lowrank(monkeypatch)

    def harmonic_voltages(power):
        result = solve_harmonic_flow(
            grid,
            ORDERS,
            operating_point=_operating_point(power),
            dtype=torch.complex128,
            tol=1e-13,
            tol_update_pu=1e-13,
        )
        return torch.view_as_real(result.v[:, 1:, :])

    power = POWER[:2].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        harmonic_voltages, (power,), eps=1e-2, atol=1e-9, rtol=1e-6
    )
    assert calls


def test_full_layout_fundamental_with_reduced_index_is_refused():
    grid = _grid(LoadModel.CONST_IMPEDANCE, fused=True)
    result = solve_harmonic_flow(
        grid, [1], operating_point=_operating_point(POWER), dtype=torch.complex128
    )
    v1_full = result.v[..., 0, :]
    with pytest.raises(InputError, match="same layout"):
        harmonic_flow.harmonic_injections(
            grid,
            v1_full,
            [5],
            operating_point=_operating_point(POWER),
            index=result.fusion.index,
        )
