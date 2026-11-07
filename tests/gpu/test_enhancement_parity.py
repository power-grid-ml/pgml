"""GPU gate: enhancement sensitivities agree on CPU and CUDA.

The device-sensitive parts are the parameter overrides on the device, the solve, the
branch-current readout, the stress and the single backward pass through the IFT. Skips
cleanly without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.studies import enhancement_sensitivities
from tests.studies.grids import stressed_feeder

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

_ARRAYS = (
    "line_impedance_elasticity",
    "line_rating_elasticity",
    "transformer_impedance_elasticity",
    "transformer_rating_elasticity",
    "transformer_tap_gradient",
    "d_stress_d_p_per_kw",
    "d_stress_d_q_per_kvar",
)


@pytest.mark.parametrize(
    "dtype, rtol", [(torch.complex128, 1e-8), (torch.complex64, 1e-2)]
)
def test_enhancement_sensitivities_cpu_cuda_parity(dtype, rtol):
    grid = stressed_feeder(tap=1.02, magnetizing=True)
    kw = dict(stress_onset=0.5, dtype=dtype)
    cpu = enhancement_sensitivities(grid, **kw)
    gpu = enhancement_sensitivities(grid, device="cuda", **kw)
    assert gpu.stress == pytest.approx(cpu.stress, rel=rtol)
    for name in _ARRAYS:
        np.testing.assert_allclose(
            getattr(gpu, name), getattr(cpu, name), rtol=rtol, atol=1e-9, err_msg=name
        )
