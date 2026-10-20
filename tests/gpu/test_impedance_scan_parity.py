"""GPU gate: CPU-vs-CUDA parity of the impedance scan (assembly, factored column solve,
sequence transform), complex128 and complex64. Skips cleanly without CUDA."""

from __future__ import annotations

import pytest
import torch

from pgml.studies import impedance_scan

from tests.fixtures.tiny_grids import three_phase_two_bus

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


@pytest.mark.parametrize("load_shunt", [None, "const_z"])
@pytest.mark.parametrize(
    ("dtype", "rtol"), [(torch.complex128, 1e-10), (torch.complex64, 1e-4)]
)
def test_impedance_scan_cpu_cuda_parity(dtype, rtol, load_shunt):
    grid = three_phase_two_bus()
    f = torch.linspace(50.0, 2500.0, 50, dtype=torch.float64)
    kw = dict(transfer=True, dtype=dtype, load_shunt=load_shunt)
    cpu = impedance_scan(grid, frequencies_hz=f, **kw)
    gpu = impedance_scan(grid, frequencies_hz=f.cuda(), **kw)
    assert gpu.z_transfer_ohm.device.type == "cpu"  # results are host tensors
    assert gpu.z_transfer_ohm.dtype == dtype
    torch.testing.assert_close(
        gpu.z_transfer_ohm, cpu.z_transfer_ohm, rtol=rtol, atol=0
    )
    torch.testing.assert_close(
        gpu.z_positive_ohm, cpu.z_positive_ohm, rtol=rtol, atol=0
    )
