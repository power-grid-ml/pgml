"""Legacy hybrid harmonic oracles must not omit explicit DER impedances."""

import pytest
import torch

from pgml.assembly import node_phase_index
from pgml.evaluation.oracles.numpy_oracle import (
    numpy_harmonic_profiles,
    numpy_harmonic_voltages,
)
from pgml.evaluation.oracles.opendss_oracle import (
    opendss_dyn_transformer_harmonic_voltages,
    opendss_geometry_harmonic_profiles,
    opendss_harmonic_voltages,
)
from tests.differentiability.test_der_harmonic_impedance_gradcheck import _grid


@pytest.mark.parametrize(
    "invoke",
    [
        lambda grid: numpy_harmonic_profiles(
            grid,
            torch.ones(2, dtype=torch.complex128),
            node_phase_index(grid),
            [5],
        ),
        lambda grid: numpy_harmonic_voltages(grid, None, [5]),
        lambda grid: opendss_harmonic_voltages(grid, None, [5]),
        lambda grid: opendss_dyn_transformer_harmonic_voltages(grid, None, [5]),
        lambda grid: opendss_geometry_harmonic_profiles(grid, object(), [5]),
    ],
)
def test_hybrid_oracle_rejects_explicit_der_impedance(invoke):
    with pytest.raises(ValueError, match="opendss_scenario_oracle"):
        invoke(_grid(0.4, 1.5e-3))
