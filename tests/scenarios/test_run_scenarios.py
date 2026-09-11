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


def test_chunked_equals_whole_power_flow(grid3):
    """chunk_size streams the batch and concatenates -> the same solution (within the
    solver tolerance) as the whole solve, incl. an uneven last chunk (12 / 5 -> 5,5,2).
    The tiny sub-tol difference is the max-over-batch iteration coupling: an easy
    scenario takes fewer iterations in a small chunk than alongside a hard one."""
    cfg = _cfg(n=12)
    whole = run_scenarios(grid3, cfg, calculation="power_flow", dtype=CDT)
    chunked = run_scenarios(
        grid3, cfg, calculation="power_flow", dtype=CDT, chunk_size=5
    )
    assert chunked.v.shape == whole.v.shape
    torch.testing.assert_close(chunked.v, whole.v, rtol=1e-7, atol=1e-9)


def test_chunked_equals_whole_harmonic_with_size1_tail(grid3):
    """Harmonic chunking with a size-1 tail (10 / 3 -> 3,3,3,1) re-adds the batch axis."""
    cfg = _cfg(n=10)
    whole = run_scenarios(
        grid3, cfg, calculation="harmonic", harmonic_orders=[1, 5, 7], dtype=CDT
    )
    chunked = run_scenarios(
        grid3,
        cfg,
        calculation="harmonic",
        harmonic_orders=[1, 5, 7],
        dtype=CDT,
        chunk_size=3,
    )
    assert chunked.v.shape == whole.v.shape == (10, 3, grid3_n(grid3))
    torch.testing.assert_close(chunked.v, whole.v, rtol=1e-7, atol=1e-9)


def test_chunked_equals_whole_coherent(grid3):
    """Coherent (node-coherent ``[B, T, H, N]``) chunking slices the SCENARIO axis ``B`` and
    concatenates -> the same sequences as the whole solve, incl. a size-1 tail (7 / 3 ->
    3,3,1) that the solver returns without the leading scenario axis."""
    from pgml.scenarios import CoherentSpectrumConfig

    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=4,
        n_scenarios=7,
        n_modes=2,
        seed=0,
    )
    whole = run_scenarios(grid3, cfg, dtype=CDT)
    chunked = run_scenarios(grid3, cfg, dtype=CDT, chunk_size=3)
    assert whole.v.ndim == 4  # [B, T, H, N]
    assert chunked.v.shape == whole.v.shape == (7, 4, 3, grid3_n(grid3))
    torch.testing.assert_close(chunked.v, whole.v, rtol=1e-7, atol=1e-9)


def test_output_device_collects_off_solve_device(grid3):
    """output_device collects the result there (CPU here) without changing the values, so a
    large GPU batch can stream its result to host memory instead of accumulating in VRAM."""
    cfg = _cfg(n=10)
    whole = run_scenarios(
        grid3, cfg, calculation="harmonic", harmonic_orders=[1, 5], dtype=CDT
    )
    streamed = run_scenarios(
        grid3,
        cfg,
        calculation="harmonic",
        harmonic_orders=[1, 5],
        dtype=CDT,
        chunk_size=3,
        output_device="cpu",
    )
    assert streamed.v.device.type == "cpu"
    torch.testing.assert_close(streamed.v, whole.v.cpu(), rtol=1e-7, atol=1e-9)


def test_chunked_is_differentiable(grid3):
    """A scalar loss over a chunked batch backprops to a grid line-R tensor."""
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    grid3.branches[0].series_resistance_ohm_per_m = r
    res = run_scenarios(
        grid3, _cfg(n=8), calculation="power_flow", dtype=CDT, chunk_size=3
    )
    res.v.abs().sum().backward()
    assert (
        r.grad is not None and torch.isfinite(r.grad).all() and r.grad.abs().sum() > 0
    )


def grid3_n(grid3) -> int:
    from pgml.assembly import node_phase_index

    return node_phase_index(grid3).size


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


def test_scenario_failures_map_step_indices_to_scenarios():
    """Sequence [B, T] convergence masks flatten over B*T; failed indices must
    come back as SCENARIO indices (deduplicated), not step indices."""
    from pgml.scenarios.run import _scenario_failures

    v_sequence = torch.zeros(4, 3, 2, 5, dtype=CDT)  # [B=4, T=3, H, N]
    # steps 0..2 -> scenario 0; steps 3..5 -> scenario 1; step 11 -> scenario 3
    assert _scenario_failures((0, 2, 4, 11), v_sequence, 3) == (0, 1, 3)
    # a single-scenario sequence [T, H, N]: any failed step fails scenario 0
    v_single = torch.zeros(3, 2, 5, dtype=CDT)
    assert _scenario_failures((1,), v_single, 3) == (0,)
    assert _scenario_failures((), v_single, 3) == ()
    # snapshot batches pass through untouched
    v_flat = torch.zeros(4, 5, dtype=CDT)
    assert _scenario_failures((1, 3), v_flat, 1) == (1, 3)


def test_any_object_with_sample_is_a_spec(grid3):
    """``run_scenarios`` plugs into a spec protocol, not a closed set of config classes."""
    from pgml.scenarios import ScenarioSpec, batch_from_values

    class HalfAndDouble:
        """A minimal generator: two scenarios around the nameplate at one order."""

        harmonic_orders = (1, 5)

        def sample(self, grid):
            b = 2
            return batch_from_values(
                grid,
                n_samples=b,
                p_w={10: torch.tensor([1.0e3, 4.0e3], dtype=torch.float64)},
                harmonic_injection={10: {5: (torch.full((b,), 0.05), torch.zeros(b))}},
            )

    spec = HalfAndDouble()
    assert isinstance(spec, ScenarioSpec)
    res = run_scenarios(grid3, spec, dtype=CDT)
    # the hint switched the calculation to harmonic and supplied the order set
    assert res.frequencies_hz is not None and res.v.shape == (2, 2, res.v.shape[-1])


def test_a_spec_without_sample_is_rejected(grid3):
    """A wrong argument must name the contract it failed, not fail deep in the solver."""
    import pytest

    from pgml.errors import InputError

    with pytest.raises(InputError, match="scenario spec"):
        run_scenarios(grid3, object())
