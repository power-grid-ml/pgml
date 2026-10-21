"""GPU gate: CPU-vs-CUDA parity of the inverter Norton models and the emission study,
complex128 and complex64. Skips cleanly without CUDA."""

from __future__ import annotations

import pytest
import torch

from pgml.studies import emission_flow

from tests.inverter.grids import default_models, default_states, lv_feeder

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


@pytest.mark.parametrize(
    ("dtype", "rtol"), [(torch.complex128, 1e-9), (torch.complex64, 2e-3)]
)
def test_emission_flow_cpu_cuda_parity(dtype, rtol):
    grid = lv_feeder()
    kw = dict(
        states=default_states(),
        harmonic_orders=range(2, 41),
        f_max_hz=40000.0,
        dtype=dtype,
    )
    cpu = emission_flow(grid, default_models(), **kw)
    gpu = emission_flow(grid, default_models(), device=torch.device("cuda"), **kw)
    assert gpu.v.device.type == "cuda" and gpu.v.dtype == dtype
    scale = cpu.v.abs().max()
    torch.testing.assert_close(gpu.v.cpu(), cpu.v, rtol=rtol, atol=rtol * float(scale))
    torch.testing.assert_close(
        gpu.v_rms.cpu(), cpu.v_rms, rtol=rtol, atol=rtol * float(scale)
    )
    for k in cpu.stability_margin:
        torch.testing.assert_close(
            gpu.stability_margin[k].cpu(), cpu.stability_margin[k], rtol=rtol, atol=rtol
        )
