"""GPU gate: CPU-vs-CUDA parity of the P-V and Q-V studies (skips cleanly without CUDA)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.studies import pv_curve, qv_curve
from tests.fixtures.tiny_grids import three_phase_two_bus

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


@pytest.mark.parametrize(
    ("dtype", "rtol"), [(torch.complex128, 1e-9), (torch.complex64, 1e-3)]
)
def test_pv_curve_parity(dtype, rtol):
    grid = three_phase_two_bus()
    cpu = pv_curve(grid, dtype=dtype, device="cpu", stop_at_nose=True)
    gpu = pv_curve(grid, dtype=dtype, device="cuda", stop_at_nose=True)
    assert gpu.lambda_max == pytest.approx(cpu.lambda_max, rel=rtol)
    assert gpu.critical_node == cpu.critical_node


def test_qv_curve_parity():
    grid = three_phase_two_bus()
    cpu = qv_curve(grid, 2, v_range_pu=(0.85, 1.03), steps=7, device="cpu")
    gpu = qv_curve(grid, 2, v_range_pu=(0.85, 1.03), steps=7, device="cuda")
    np.testing.assert_allclose(gpu.q_var, cpu.q_var, rtol=1e-9, atol=1e-6)
