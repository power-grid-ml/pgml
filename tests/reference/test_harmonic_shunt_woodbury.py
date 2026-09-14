"""Low-rank harmonic device-shunt solve against exact assembled systems."""

from __future__ import annotations

import torch

from pgml.assembly import node_phase_index
from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.solver import solve_harmonic_flow
from pgml.solver.harmonic import solve_harmonic
from pgml.solver.harmonic_flow import assemble_harmonic_system


def _sparse_load_grid() -> Grid:
    phase = (Phase.A,)
    spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
                HarmonicComponent(order=7, magnitude_pu=0.1, phase_deg=0.0),
            ]
        )
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=230.0, phases=phase) for i in range(1, 6)],
        branches=[
            Line(
                id=20 + i,
                from_node=i,
                to_node=i + 1,
                from_phases=phase,
                to_phases=phase,
                length_m=100.0,
                series_resistance_ohm_per_m=[[1.0e-3]],
                series_inductance_h_per_m=[[1.0e-6]],
                shunt_capacitance_f_per_m=[[1.0e-9]],
            )
            for i in range(1, 5)
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=phase,
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1.0e-3]],
            ),
            Load(
                id=30,
                node=5,
                phases=phase,
                p_nom_w=2_000.0,
                q_nom_var=500.0,
                spectrum=spectrum,
            ),
        ],
    )


def _operating_point(power: torch.Tensor) -> dict:
    return {30: {"p_w": power, "q_var": torch.full_like(power, 500.0)}}


def _direct_harmonics(grid, result, operating_point):
    y_bus, current, _ = assemble_harmonic_system(
        grid,
        [5, 7],
        result.v[..., 0, :],
        operating_point=operating_point,
        dtype=torch.complex128,
    )
    return solve_harmonic(y_bus, current)


def test_sparse_scenario_shunt_matches_exact_assembled_factorization():
    grid = _sparse_load_grid()
    power = torch.tensor([1_500.0, 2_000.0, 2_500.0], dtype=torch.float64)
    operating_point = _operating_point(power)

    result = solve_harmonic_flow(
        grid, [1, 5, 7], operating_point=operating_point, dtype=torch.complex128
    )
    direct = _direct_harmonics(grid, result, operating_point)

    assert torch.allclose(result.v[..., 1:, :], direct, rtol=2e-12, atol=2e-12)


def test_selector_is_built_without_a_full_identity(monkeypatch):
    import pgml.solver.harmonic_flow as harmonic_flow

    grid = _sparse_load_grid()
    power = torch.tensor([1_500.0, 2_000.0, 2_500.0], dtype=torch.float64)
    operating_point = _operating_point(power)
    v1 = solve_harmonic_flow(
        grid, [1], operating_point=operating_point, dtype=torch.complex128
    ).v[..., 0, :]
    original = torch.eye

    def reject_full_identity(n, *args, **kwargs):
        if n == node_phase_index(grid).size:
            raise AssertionError("selector must not allocate an N-by-N identity")
        return original(n, *args, **kwargs)

    monkeypatch.setattr(harmonic_flow.torch, "eye", reject_full_identity)
    u, core = harmonic_flow._harmonic_shunt_lowrank_terms(
        grid,
        v1,
        node_phase_index(grid),
        [5, 7],
        operating_point,
        "opendss",
        None,
        True,
        torch.complex128,
        torch.float64,
        torch.device("cpu"),
    )
    assert u.shape == (5, 1)
    assert core.shape == (3, 2, 1, 1)


def test_bad_lowrank_residual_falls_back_to_exact_assembly(monkeypatch, caplog):
    import pgml.solver.harmonic_flow as harmonic_flow

    grid = _sparse_load_grid()
    power = torch.tensor(
        [1_500.0, 2_000.0, 2_500.0], dtype=torch.float64, requires_grad=True
    )
    operating_point = _operating_point(power)
    original = harmonic_flow.solve_factored_updated

    def corrupt(*args, **kwargs):
        return torch.zeros_like(original(*args, **kwargs))

    monkeypatch.setattr(harmonic_flow, "solve_factored_updated", corrupt)
    result = solve_harmonic_flow(
        grid, [1, 5, 7], operating_point=operating_point, dtype=torch.complex128
    )
    direct = _direct_harmonics(grid, result, operating_point)

    assert torch.allclose(result.v[..., 1:, :], direct, rtol=2e-12, atol=2e-12)
    assert "falling back to exact assembled factorization" in caplog.text
    result.v.abs().sum().backward()
    assert power.grad is not None
    assert torch.isfinite(power.grad).all()
    assert torch.count_nonzero(power.grad) == power.numel()


def test_high_rank_population_keeps_direct_factorization(monkeypatch):
    """A two-row network with one touched row is beyond the rank crossover."""
    import pgml.solver.harmonic_flow as harmonic_flow

    grid = _sparse_load_grid()
    compact = grid.model_copy(
        update={
            "nodes": grid.nodes[:2],
            "branches": grid.branches[:1],
            "appliances": [
                grid.appliances[0],
                grid.appliances[1].model_copy(update={"node": 2}),
            ],
        }
    )
    power = torch.tensor([1_500.0, 2_000.0, 2_500.0], dtype=torch.float64)

    def unexpected(*args, **kwargs):
        raise AssertionError("high-rank shunt must keep the direct path")

    monkeypatch.setattr(harmonic_flow, "low_rank_update", unexpected)
    result = solve_harmonic_flow(
        compact,
        [1, 5, 7],
        operating_point=_operating_point(power),
        dtype=torch.complex128,
    )
    assert torch.isfinite(result.v).all()


def test_deeper_injection_batch_keeps_aligned_direct_path(monkeypatch):
    """A ``[scenario, step]`` spectrum needs Y's singleton step axis."""
    import pgml.solver.harmonic_flow as harmonic_flow

    grid = _sparse_load_grid()
    power = torch.tensor([1_500.0, 2_000.0, 2_500.0], dtype=torch.float64)
    magnitude = torch.full((3, 2), 0.2, dtype=torch.float64)

    def unexpected(*args, **kwargs):
        raise AssertionError("deeper injection batch must keep the aligned direct path")

    monkeypatch.setattr(harmonic_flow, "low_rank_update", unexpected)
    result = solve_harmonic_flow(
        grid,
        [1, 5],
        operating_point=_operating_point(power),
        harmonic_injection={30: {1: (1.0, 0.0), 5: (magnitude, 0.0)}},
        dtype=torch.complex128,
    )
    assert result.v.shape == (3, 2, 2, 5)
    assert torch.isfinite(result.v).all()
