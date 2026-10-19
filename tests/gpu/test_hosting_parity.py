"""GPU gate: a hosting-capacity search lands on the same powers on CPU and CUDA.

The search is a sequence of batched power flows plus host-side bisection bookkeeping; the
device-sensitive parts are the batched operating point, the solve and the branch-current
readout through the fusion map of an ideal switch. Skips cleanly without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.studies import HostingLimits, hosting_capacity
from tests.studies.grids import lv_feeder_with_switch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_hosting_capacity_cpu_cuda_parity(dtype):
    grid = lv_feeder_with_switch(i_max_a=150.0)
    kw = dict(
        limits=HostingLimits(v_max_pu=1.03, vuf_max=0.02),
        p_max_w=200.0e3,
        tol_w=50.0,
        dtype=dtype,
    )
    cpu = hosting_capacity(grid, [3, 4], **kw)
    gpu = hosting_capacity(grid, [3, 4], device="cuda", **kw)
    # identical bisection paths unless a verdict flips within working precision of a
    # limit, which moves the answer by at most one tolerance step
    assert np.all(np.abs(cpu.p_max_w - gpu.p_max_w) <= 50.0)
    assert [b["kind"] for b in cpu.binding] == [b["kind"] for b in gpu.binding]
