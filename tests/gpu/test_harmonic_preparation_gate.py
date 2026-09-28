"""GPU gate: the harmonic preparation row gate applies to the CPU only.

An accelerator gains from the preparation at every grid size measured, because the
network assembly and factorization it removes are a larger share of the call there,
so a chunked run on CUDA prepares whatever the row count (skips cleanly without CUDA).
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import node_phase_index
from pgml.grids import synthetic_feeder
from pgml.scenarios import (
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    run_scenarios,
)
from pgml.scenarios import run as run_module

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _cfg() -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=8,
        seed=7,
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


def test_small_cuda_grid_still_prepares(monkeypatch):
    grid = synthetic_feeder(20)
    assert node_phase_index(grid).size < 256  # gated on the CPU, not here

    built: list = []
    real = run_module.HarmonicFlowSystem

    def record(**kwargs):
        system = real(**kwargs)
        built.append(system)
        return system

    monkeypatch.setattr(run_module, "HarmonicFlowSystem", record)
    kwargs = dict(
        calculation="harmonic",
        harmonic_orders=[1, 5],
        slack="norton",
        dtype=torch.complex128,
    )
    on_cuda = run_scenarios(grid, _cfg(), device="cuda", chunk_size=3, **kwargs)
    assert len(built) == 1
    assert built[0].stats["harmonic_network_hits"] > 0

    reference = run_scenarios(grid, _cfg(), **kwargs)
    torch.testing.assert_close(on_cuda.v.cpu(), reference.v, rtol=1e-7, atol=1e-9)
