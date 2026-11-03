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


def test_three_phase_margin_gradient_matches_cpu():
    """The multi-element margin's SVD runs on the host for CUDA inputs; the result and
    its gradient must come back on the input device and equal the CPU path."""
    grid = lv_feeder()

    def margin_grad(device):
        p = torch.tensor(9000.0, dtype=torch.float64, device=device, requires_grad=True)
        res = emission_flow(
            grid,
            default_models(),
            states=default_states(),
            operating_point={101: {"p_w": p}},
            harmonic_orders=range(2, 41),
            f_max_hz=2000.0,
            device=device,
        )
        m = res.stability_margin[101]
        assert m.device.type == torch.device(device).type
        (g,) = torch.autograd.grad(m.sum(), p)
        return m.detach().cpu(), g.cpu()

    m_cpu, g_cpu = margin_grad("cpu")
    m_gpu, g_gpu = margin_grad("cuda")
    torch.testing.assert_close(m_gpu, m_cpu, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(g_gpu, g_cpu, rtol=1e-7, atol=1e-15)
