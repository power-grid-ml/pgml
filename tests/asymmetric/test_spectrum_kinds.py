"""Stored spectrum kinds: what the solve, the numpy oracle and the sampler read.

Only ``StaticSpectrum`` has evaluation semantics. The other schema kinds must raise a
``ModelingError`` naming the device and the kind instead of injecting nothing, in every
reader (``pgml.assembly._spectrum`` is the one they share). ``spectrum_per_phase`` must
reach the numpy oracle per element exactly as it reaches the solve, and the sampler must
refuse to override a per-phase device with a single seeded spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.errors import ModelingError
from pgml.evaluation.oracles.numpy_oracle import numpy_harmonic_voltages
from pgml.scenarios import ParameterSpec, ScenarioConfig, Selector, Uniform, sample
from pgml.schemas.grid_schema import (
    DistributionSpectrum,
    Load,
    LoadModel,
    LoadVaryingSpectrum,
    Phase,
    RandomSpectrum,
    TimeVaryingSpectrum,
)
from pgml.solver import solve_harmonic_flow

from .test_harmonic_per_phase import ABC, _grid, _spec

POINT = _spec([(1, 1.0, 0.0), (5, 0.1, 10.0), (7, 0.05, -20.0)]).spectrum
NON_STATIC = {
    "load_varying": LoadVaryingSpectrum(
        loading_levels_pu=[0.0, 1.0], spectra=[POINT, POINT]
    ),
    "time_varying": TimeVaryingSpectrum(schedule={0: POINT}),
    "random": RandomSpectrum(
        order_magnitude_mean_pu={5: 0.1}, order_magnitude_std_pu={5: 0.0}
    ),
    "distribution": DistributionSpectrum(distribution_ref="fitted"),
}


def _load(**spectrum_kw):
    return Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        load_model=LoadModel.CONST_POWER,
        **spectrum_kw,
    )


@pytest.mark.parametrize("kind", sorted(NON_STATIC))
@pytest.mark.parametrize("field", ["spectrum", "spectrum_per_phase"])
def test_non_static_kind_raises_in_solve_and_oracle(kind, field):
    spec = NON_STATIC[kind]
    value = spec if field == "spectrum" else {Phase.B: spec}
    grid = _grid(_load(**{field: value}), ABC)
    with pytest.raises(ModelingError, match=rf"Device 30: spectrum kind '{kind}'"):
        solve_harmonic_flow(grid, [1, 5, 7], slack="norton")
    with pytest.raises(ModelingError, match=rf"Device 30: spectrum kind '{kind}'"):
        numpy_harmonic_voltages(grid, None, [1, 5], slack="norton")


def test_oracle_reads_spectrum_per_phase_like_the_solve():
    """Asymmetric per-phase spectra (A and C differ, B none) agree solve vs oracle."""
    spp = {
        Phase.A: _spec([(1, 1.0, 0.0), (5, 0.3, 10.0), (7, 0.1, 5.0)]),
        Phase.C: _spec([(1, 1.0, 0.0), (5, 0.05, -40.0)]),
    }
    grid = _grid(_load(spectrum_per_phase=spp), ABC)
    orders = [1, 5, 7]
    res = solve_harmonic_flow(
        grid, orders, slack="norton", load_shunt="none", dtype=torch.complex128
    )
    v = res.v.detach().cpu().numpy()
    vo = numpy_harmonic_voltages(
        grid, None, orders, slack="norton", v1=v[0], load_shunt="none"
    )
    assert np.abs(v[1]).max() > 1e-3  # the per-phase emission reaches the grid
    np.testing.assert_allclose(vo, v, rtol=0, atol=1e-10 * np.abs(v).max())


def _hcfg(**kw) -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=4,
        seed=0,
        parameters=[
            ParameterSpec(
                name="hm",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
                field="h_mag",
                orders=[5],
                **kw,
            )
        ],
    )


def test_sampler_refuses_to_override_a_per_phase_device():
    grid = _grid(_load(spectrum_per_phase={Phase.A: _spec([(5, 0.2, 0.0)])}), ABC)
    with pytest.raises(ModelingError, match="spectrum_per_phase"):
        sample(grid, _hcfg(mode="absolute"))


@pytest.mark.parametrize("kind", sorted(NON_STATIC))
def test_sampler_refuses_a_non_static_kind(kind):
    grid = _grid(_load(spectrum=NON_STATIC[kind]), ABC)
    with pytest.raises(ModelingError, match=rf"spectrum kind '{kind}'"):
        sample(grid, _hcfg(mode="scale"))
