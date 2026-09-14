"""CPU/CUDA parity for differentiable DER harmonic impedance assembly."""

import pytest
import torch

from tests.differentiability.test_der_harmonic_impedance_gradcheck import _grid
from pgml.solver import solve_harmonic_flow


@pytest.mark.gpu
def test_cpu_cuda_internal_voltage_impedance_and_gradient_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    outputs = []
    gradients = []
    for device in (torch.device("cpu"), torch.device("cuda")):
        resistance = torch.tensor(
            0.4, dtype=torch.float64, device=device, requires_grad=True
        )
        inductance = torch.tensor(
            1.5e-3, dtype=torch.float64, device=device, requires_grad=True
        )
        result = solve_harmonic_flow(
            _grid(resistance, inductance),
            [1, 5],
            load_shunt="none",
            dtype=torch.complex128,
            device=device,
        ).v
        result.abs().sum().backward()
        outputs.append(result.detach().cpu())
        gradients.append(torch.stack([resistance.grad, inductance.grad]).detach().cpu())

    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-8, atol=2e-8)
