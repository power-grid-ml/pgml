"""GPU gate: CPU-vs-CUDA parity of the anchored solve (skips cleanly without CUDA)."""

from __future__ import annotations

import pytest
import torch

from pgml.solver import solve_anchored

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

N, B, K = 8, 3, 4


def _problem(dtype, device):
    torch.manual_seed(3)
    rdt = torch.empty(0, dtype=dtype).real.dtype
    y = torch.randn(N, N, dtype=dtype) + 3.0 * torch.eye(N, dtype=dtype)
    i = torch.randn(B, N, dtype=dtype)
    row_weight = torch.rand(B, N, dtype=rdt)
    row_target = torch.randn(B, N, dtype=dtype)
    op = torch.randn(K, N, dtype=dtype)
    op_weight = torch.rand(B, K, dtype=rdt)
    op_target = torch.randn(B, K, dtype=dtype)
    fixed_rows = torch.tensor([0], dtype=torch.int64)
    v_fixed = torch.randn(1, dtype=dtype)
    args = (y, i)
    kwargs = dict(
        row_weight=row_weight,
        row_target=row_target,
        op=op,
        op_weight=op_weight,
        op_target=op_target,
        fixed_rows=fixed_rows,
        v_fixed=v_fixed,
    )
    args = tuple(a.to(device) for a in args)
    kwargs = {k: v.to(device) for k, v in kwargs.items()}
    return args, kwargs


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_parity(dtype):
    args_cpu, kwargs_cpu = _problem(dtype, torch.device("cpu"))
    args_gpu, kwargs_gpu = _problem(dtype, torch.device("cuda"))

    v_cpu = solve_anchored(*args_cpu, **kwargs_cpu)
    v_gpu = solve_anchored(*args_gpu, **kwargs_gpu)

    assert v_gpu.device.type == "cuda"
    assert v_gpu.dtype == dtype
    tol = 1e-10 if dtype == torch.complex128 else 1e-4
    assert torch.allclose(v_cpu, v_gpu.cpu(), atol=tol, rtol=tol)
