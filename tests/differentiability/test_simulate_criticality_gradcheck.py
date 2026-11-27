"""float64 gradcheck of ``simulate`` under every ``criticality`` mode.

The criticality analysis is an autograd-free diagnostic of the fundamental solve; the
gradient of the solved voltages w.r.t. a grid parameter must be the same whether it runs
(``"always"``), runs only on non-convergence (``"auto"``) or never runs (``"never"``).
"""

from __future__ import annotations

import pytest
import torch

from pgml.simulation import SimulationConfig, simulate
from tests.fixtures.tiny_grids import single_phase_chain

_R_KEY = ("line", 20, "series_resistance_ohm_per_m")


def _voltages(calculation: str, criticality: str):
    grid = single_phase_chain()
    cfg = SimulationConfig(calculation=calculation, harmonic_orders=[1, 5])

    def fn(r: torch.Tensor) -> torch.Tensor:
        state = simulate(
            grid, cfg, param_overrides={_R_KEY: r}, criticality=criticality
        )
        return torch.view_as_real(state.node_voltages())

    return fn


@pytest.mark.parametrize("calculation", ["power_flow", "harmonic"])
@pytest.mark.parametrize("criticality", ["never", "always"])
def test_gradcheck_simulate_with_criticality(calculation, criticality):
    r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        _voltages(calculation, criticality), (r,), eps=1e-7, atol=1e-6, rtol=1e-4
    )


@pytest.mark.parametrize("calculation", ["power_flow", "harmonic"])
def test_gradient_does_not_depend_on_criticality(calculation):
    grads = {}
    for mode in ("auto", "always", "never"):
        r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
        _voltages(calculation, mode)(r).abs().sum().backward()
        grads[mode] = r.grad
    torch.testing.assert_close(grads["never"], grads["auto"], rtol=1e-12, atol=0.0)
    torch.testing.assert_close(grads["always"], grads["auto"], rtol=1e-12, atol=0.0)
