"""Differentiable primitive-admittance stamp builders (internal).

These helpers turn the materialised, per-phase real parameter matrices of each
component kind into complex primitive admittance blocks, batched over branches
and over H frequencies, ready to scatter into the global Y-bus.

All math here is autograd-safe torch: no ``.item()/.detach()/.numpy()``, no
in-place ops on tracked tensors, no Python control flow on tensor values, and no
Python loop over branches/phases inside the differentiable path. Loops over the
small, fixed set of component KINDS and over the (Python-side) collection of
parameters are fine — they do not iterate per branch element on the tape.

Shape conventions
-----------------
- ``f``: real tensor ``[H]`` absolute frequencies.
- Series parameter matrices stack to ``[K, P, P]`` real (K branches of a kind,
  P phases). Frequency scaling broadcasts to ``[H, K, P, P]``.
- A series primitive 2x2 block expands to a ``[H, K, 2P, 2P]`` complex tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _cdtype(dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real dtype (for building complex from real parts)."""
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


def _rdtype(dtype: torch.dtype) -> torch.dtype:
    """Real dtype matching a (possibly complex) dtype."""
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def series_admittance_matrix(
    r: Tensor,
    ind: Tensor,
    f: Tensor,
    cdtype: torch.dtype,
    *,
    r_mult: Tensor | None = None,
    r_unscaled: Tensor | None = None,
) -> Tensor:
    """Series admittance ``Ys(f) = (R(f) + jX(f))^-1`` per branch per frequency.

    ``R(f) = r * r_mult(f) + r_unscaled`` and ``X(f) = 2*pi*f*ind``.

    Parameters
    ----------
    r, ind:
        Real tensors ``[K, P, P]`` — series resistance and inductance matrices.
    f:
        Real tensor ``[H]`` absolute frequencies.
    cdtype:
        Target complex dtype.
    r_mult:
        Optional real tensor broadcastable to ``[H, K, 1, 1]`` (or ``[H,K,P,P]``)
        skin-effect multiplier applied to ``r``. Defaults to 1.
    r_unscaled:
        Optional real ``[K, P, P]`` resistance ADDED after the multiplier — the part of
        the resistance matrix the skin effect does not scale (the earth-return mutual
        terms of a multi-phase line). Defaults to 0.

    Returns
    -------
    Tensor
        Complex ``[H, K, P, P]`` admittance matrices (matrix inverse over the last
        two dims).
    """
    # X(h) = 2*pi*f*L  -> [H, K, P, P]
    two_pi_f = (2.0 * torch.pi) * f  # [H]
    x = two_pi_f[:, None, None, None] * ind[None]  # [H,K,P,P]
    r_b = r[None]  # [1,K,P,P]
    if r_mult is not None:
        r_b = r_b * r_mult
    if r_unscaled is not None:
        r_b = r_b + r_unscaled[None]
    z = torch.complex(r_b.expand_as(x).to(_rdtype(cdtype)), x.to(_rdtype(cdtype))).to(
        cdtype
    )  # [H,K,P,P]
    return torch.linalg.inv(z)


def shunt_admittance_matrix(
    g: Tensor, c: Tensor, f: Tensor, cdtype: torch.dtype
) -> Tensor:
    """Shunt admittance ``Y_sh(f) = G + jB`` with ``B = 2*pi*f*C`` per frequency.

    ``g, c`` are real ``[K, P, P]``; returns complex ``[H, K, P, P]``.
    """
    two_pi_f = (2.0 * torch.pi) * f  # [H]
    b = two_pi_f[:, None, None, None] * c[None]  # [H,K,P,P]
    g_b = g[None].expand_as(b)  # [H,K,P,P]
    rdt = _rdtype(cdtype)
    return torch.complex(g_b.to(rdt), b.to(rdt)).to(cdtype)


def pi_series_blocks(ys: Tensor) -> Tensor:
    """Expand series admittance ``Ys`` ``[H,K,P,P]`` into the 2P x 2P primitive.

    ``[[Ys, -Ys], [-Ys, Ys]]`` stacked block matrix, complex ``[H, K, 2P, 2P]``.
    """
    top = torch.cat([ys, -ys], dim=-1)  # [H,K,P,2P]
    bot = torch.cat([-ys, ys], dim=-1)  # [H,K,P,2P]
    return torch.cat([top, bot], dim=-2)  # [H,K,2P,2P]


__all__ = [
    "series_admittance_matrix",
    "shunt_admittance_matrix",
    "pi_series_blocks",
    "_cdtype",
    "_rdtype",
]
