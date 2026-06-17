"""run_scenarios: batched solve == loop-of-individual, harmonic, reproducibility."""

from __future__ import annotations

import torch

from pgml.scenarios import (
    CartesianAxis,
    CartesianConfig,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    cartesian_sample,
    run_scenarios,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128


def _cfg(n=16, seed=7, method="sobol"):
    return ScenarioConfig(
        n_samples=n,
        seed=seed,
        method=method,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.6, high=1.4),
                field="pq",
                mode="scale",
                per="each",
            )
        ],
    )


def test_power_flow_shape_and_matches_loop(grid3):
    cfg = _cfg(n=12)
    res = run_scenarios(grid3, cfg, calculation="power_flow", slack="ideal", dtype=CDT)
    n_nodes = res.index.size
    assert res.v.shape == (12, n_nodes)

    # Independent reference: solve each scenario's operating point one at a time.
    op = res.sampled.operating_point
    for b in range(12):
        op_b = {cid: {k: v[b] for k, v in d.items()} for cid, d in op.items()}
        v_b = solve_power_flow(grid3, slack="ideal", operating_point=op_b, dtype=CDT).v
        torch.testing.assert_close(res.v[b], v_b, rtol=1e-7, atol=1e-9)


def test_harmonic_shape(grid3):
    res = run_scenarios(
        grid3,
        _cfg(n=8),
        calculation="harmonic",
        harmonic_orders=[1, 5, 7],
        slack="norton",
        dtype=CDT,
    )
    assert res.v.shape == (8, 3, res.index.size)
    assert res.frequencies_hz.tolist() == [50.0, 250.0, 350.0]


def test_reproducible_results(grid3):
    cfg = _cfg(n=10, seed=123)
    v1 = run_scenarios(grid3, cfg, calculation="power_flow", dtype=CDT).v
    v2 = run_scenarios(grid3, cfg, calculation="power_flow", dtype=CDT).v
    torch.testing.assert_close(v1, v2, rtol=0, atol=0)


def test_cartesian_product_batch(grid3):
    """Cartesian sweep: 3 levels (load 10) x 2 levels (load 11) -> B=6."""
    cfg = CartesianConfig(
        axes=[
            CartesianAxis(
                name="l10",
                selector=Selector(component="load", ids=[10]),
                values=[0.5, 1.0, 1.5],
                field="pq",
                mode="scale",
            ),
            CartesianAxis(
                name="l11",
                selector=Selector(component="load", ids=[11]),
                values=[0.8, 1.2],
                field="pq",
                mode="scale",
            ),
        ]
    )
    sampled = cartesian_sample(grid3, cfg)
    assert sampled.n_samples == 6
    # Every (level10, level11) combination is present exactly once.
    combos = {
        (
            round(float(sampled.operating_point[10]["p_w"][b] / 2000.0), 3),
            round(float(sampled.operating_point[11]["p_w"][b] / 3000.0), 3),
        )
        for b in range(6)
    }
    assert combos == {(a, c) for a in (0.5, 1.0, 1.5) for c in (0.8, 1.2)}
    res = run_scenarios(grid3, cfg, calculation="power_flow", slack="ideal", dtype=CDT)
    assert res.v.shape == (6, res.index.size)


def test_differentiable_through_batch(grid3):
    """A scalar loss over the whole batch backprops to a line R tensor in the grid."""
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    grid3.branches[0].series_resistance_ohm_per_m = r
    res = run_scenarios(
        grid3, _cfg(n=6), calculation="power_flow", slack="ideal", dtype=CDT
    )
    res.v.abs().sum().backward()
    assert (
        r.grad is not None and torch.isfinite(r.grad).all() and r.grad.abs().sum() > 0
    )
