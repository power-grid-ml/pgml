"""prepare_power_flow / PowerFlowSystem: reuse across solves is result-identical.

The system caches the operating-point-independent half of a solve (assembly,
slack rows, factorization, grid leaves); passing it must change performance
only — voltages, convergence reporting, gradients, and mismatch validation are
covered here.
"""

from __future__ import annotations

import pytest
import torch

from pgml.errors import ConnectivityError, InputError
from pgml.grids import synthetic_feeder
from pgml.solver import prepare_power_flow, solve_power_flow


@pytest.fixture(scope="module")
def grid():
    return synthetic_feeder(30)


def _op(grid, b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        a.id: {"p_w": float(a.p_nom_w) * (0.5 + torch.rand(b, generator=g))}
        for a in grid.appliances
        if a.id >= 20000
    }


@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_system_solve_matches_plain_solve(grid, slack):
    system = prepare_power_flow(grid, slack=slack)
    for op in (None, _op(grid, 5)):
        ref = solve_power_flow(grid, slack=slack, operating_point=op)
        res = solve_power_flow(grid, slack=slack, operating_point=op, system=system)
        assert res.converged == ref.converged
        assert res.iterations == ref.iterations
        assert torch.allclose(res.v, ref.v)


def test_system_reuse_across_chunks_matches_full_batch(grid):
    op = _op(grid, 8)
    system = prepare_power_flow(grid)
    full = solve_power_flow(grid, operating_point=op, system=system)
    parts = []
    for start in (0, 4):
        op_c = {k: {"p_w": v["p_w"][start : start + 4]} for k, v in op.items()}
        parts.append(solve_power_flow(grid, operating_point=op_c, system=system).v)
    assert torch.allclose(torch.cat(parts, 0), full.v)


def test_gradients_unchanged_with_system(grid):
    load_id = next(a.id for a in grid.appliances if a.id >= 20000)
    grads = {}
    system = prepare_power_flow(grid)
    for use_system in (False, True):
        p = torch.tensor(2.0e5, dtype=torch.float64, requires_grad=True)
        res = solve_power_flow(
            grid,
            operating_point={load_id: {"p_w": p}},
            system=system if use_system else None,
        )
        res.v.abs().sum().backward()
        grads[use_system] = p.grad.clone()
    assert torch.allclose(grads[False], grads[True], rtol=1e-10)


def test_mismatched_system_raises(grid):
    system = prepare_power_flow(grid, slack="norton")
    with pytest.raises(InputError, match="PowerFlowSystem"):
        solve_power_flow(grid, slack="ideal", system=system)


def test_system_from_different_network_rejected(grid):
    """A same-size grid with changed parameters must not reuse the stale system."""
    system = prepare_power_flow(grid)
    other = grid.model_copy(deep=True)
    line = next(b for b in other.branches if hasattr(b, "series_resistance_ohm_per_m"))
    line.series_resistance_ohm_per_m[0][0] *= 2.0
    with pytest.raises(InputError, match="different network"):
        solve_power_flow(other, system=system)


def test_prepare_runs_the_connectivity_check():
    grid = synthetic_feeder(10, n_feeders=2, tie_switches=1)
    with pytest.raises(ConnectivityError):
        prepare_power_flow(grid, branch_states={10001: 0.0})


def test_run_scenarios_chunked_uses_one_system(grid):
    """run_scenarios chunking (which now shares one prepared system) == unchunked."""
    from pgml.scenarios import run_scenarios
    from pgml.scenarios.config import (
        ParameterSpec,
        ScenarioConfig,
        Selector,
        Uniform,
    )

    cfg = ScenarioConfig(
        n_samples=6,
        seed=7,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
                field="pq",
                mode="scale",
                per="each",
            )
        ],
    )
    ref = run_scenarios(grid, cfg)
    chunked = run_scenarios(grid, cfg, chunk_size=2)
    assert torch.allclose(ref.v, chunked.v)


def test_assemble_ybus_batched_operating_point(grid):
    """The const-Z fold broadcasts a batched operating point: batched Y == loop."""
    from pgml.assembly import assemble_ybus

    load_id = next(a.id for a in grid.appliances if a.id >= 20000)
    p = torch.tensor([1.0e5, 2.0e5, 3.0e5], dtype=torch.float64)
    yb = assemble_ybus(grid, [50.0], operating_point={load_id: {"p_w": p}}).Y
    n = yb.shape[-1]
    assert yb.shape == (3, 1, n, n)
    for i in range(3):
        yi = assemble_ybus(
            grid, [50.0], operating_point={load_id: {"p_w": float(p[i])}}
        ).Y
        assert torch.allclose(yb[i], yi)
