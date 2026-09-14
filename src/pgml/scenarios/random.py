"""Reusable stochastic processes for scenario generation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def ar1_noise(
    shape: tuple[int, ...] | Sequence[int],
    rho: float | Tensor,
    generator: torch.Generator,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Draw stationary standard-normal AR(1) noise along the last axis.

    The recurrence is ``e[t] = rho * e[t-1] + sqrt(1-rho**2) * eta[t]``.
    Its initial value is standard normal, so every step is marginally standard
    normal. ``rho`` may be a scalar or a tensor broadcastable against
    ``shape[:-1]``; tensor gradients are preserved.

    The output device defaults to the generator's device. A generator cannot
    drive a random draw on another device, so an explicit mismatch raises before
    drawing. The three positional arguments intentionally form the stable public
    contract; ``dtype`` and ``device`` are keyword-only.
    """
    dims = tuple(int(dim) for dim in shape)
    if not dims or dims[-1] < 1 or any(dim < 0 for dim in dims):
        raise ValueError("shape must be non-empty with a positive last dimension")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point dtype")

    generator_device = torch.device(generator.device)
    output_device = generator_device if device is None else torch.device(device)
    if output_device != generator_device:
        raise ValueError(
            f"generator is on {generator_device}, but device={output_device}; "
            "use a generator created on the output device"
        )
    if isinstance(rho, Tensor) and rho.device != output_device:
        raise ValueError(
            f"rho is on {rho.device}, but the generator and output are on "
            f"{output_device}; place rho and the generator on the same device"
        )

    rho_t = torch.as_tensor(rho, dtype=dtype, device=output_device)
    try:
        torch.broadcast_shapes(tuple(rho_t.shape), dims[:-1])
    except RuntimeError as exc:
        raise ValueError(
            f"rho shape {tuple(rho_t.shape)} does not broadcast against {dims[:-1]}"
        ) from exc

    eta = torch.randn(dims, generator=generator, dtype=dtype, device=output_device)
    innovation_scale = torch.sqrt(torch.clamp(1.0 - rho_t.square(), min=0.0))
    states = [eta[..., 0]]
    for step in range(1, dims[-1]):
        states.append(rho_t * states[-1] + innovation_scale * eta[..., step])
    return torch.stack(states, dim=-1)


__all__ = ["ar1_noise"]
