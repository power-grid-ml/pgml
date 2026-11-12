"""Private helpers shared by the inverter models: tensors, frequencies, sequences.

Every model function evaluates a transfer function at the Laplace variable
``s = j 2 pi f`` for a real frequency tensor ``f``. Negative frequencies are meaningful
for the three-phase models: a complex space vector rotating at ``-f`` is the negative
sequence at ``f`` (see :func:`sequence_to_element`).
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor

from pgml.errors import InputError

Scalar = Union[float, int, Tensor]

TWO_PI = 2.0 * math.pi


def as_real(x: Scalar, rdt: torch.dtype, device) -> Tensor:
    """A real tensor on ``device``; a tensor keeps its autograd graph."""
    if isinstance(x, Tensor):
        return x.to(dtype=rdt, device=device)
    return torch.as_tensor(float(x), dtype=rdt, device=device)


def as_param(x: Scalar, s: Tensor) -> Tensor:
    """A physical parameter aligned for a product with the Laplace variable ``s``.

    Parameters may carry a scenario batch ``[*batch]`` (a tensor), so a fleet of units with
    different values is evaluated in one call. ``s`` has the frequency axis (and possibly an
    element axis) last; the parameter gains that many trailing singleton axes, so the
    product broadcasts to ``[*batch, *s.shape]``. A scalar stays 0-d.
    """
    rdt = s.real.dtype
    t = as_real(x, rdt, s.device)
    if t.dim() == 0:
        return t
    return t.reshape(*t.shape, *([1] * s.dim()))


def real_dtype(cdt: torch.dtype) -> torch.dtype:
    """The real dtype matching a complex dtype."""
    if cdt == torch.complex64:
        return torch.float32
    if cdt == torch.complex128:
        return torch.float64
    raise InputError(f"expected complex64 or complex128, got {cdt}.")


def laplace(f_hz: Tensor, cdt: torch.dtype) -> Tensor:
    """``s = j 2 pi f`` as a complex tensor of the shape of ``f_hz``."""
    return torch.complex(torch.zeros_like(f_hz), TWO_PI * f_hz).to(cdt)


def series_admittance(r: Tensor, ind: Tensor, s: Tensor) -> Tensor:
    """``1 / (R + s L)``; an all-zero branch raises (it is a short circuit)."""
    z = r.to(s.dtype) + s * ind.to(s.dtype)
    return 1.0 / z


def capacitor_branch_admittance(c: Tensor, r: Tensor, s: Tensor) -> Tensor:
    """Admittance of a capacitor ``C`` in series with ``R``: ``sC / (1 + sRC)``.

    ``C = 0`` gives an exact zero (an absent branch) without a division by zero.
    """
    sc = s * c.to(s.dtype)
    return sc / (1.0 + sc * r.to(s.dtype))


# Symmetrical components of a three-phase element vector (phases a, b, c in order):
# x_abc = A @ [x0, x+, x-], with a = exp(j 2 pi / 3).
def _sequence_matrix(cdt: torch.dtype, device) -> Tensor:
    a = complex(math.cos(TWO_PI / 3.0), math.sin(TWO_PI / 3.0))
    a2 = a * a
    return torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, a2, a], [1.0, a, a2]], dtype=cdt, device=device
    )


def sequence_to_element(diag: Tensor) -> Tensor:
    """Element-domain operator ``A diag(d0, d+, d-) A^-1`` from ``diag[..., 3]``.

    ``diag[..., 0]`` acts on the zero sequence, ``[..., 1]`` on the positive and
    ``[..., 2]`` on the negative sequence. The result is ``[..., 3, 3]``.
    """
    a = _sequence_matrix(diag.dtype, diag.device)
    a_inv = torch.linalg.inv(a)
    return torch.einsum("ik,...k,kj->...ij", a, diag, a_inv)


def element_to_sequence(x: Tensor) -> Tensor:
    """``[x0, x+, x-] = A^-1 x_abc`` over the trailing axis of length 3."""
    a_inv = torch.linalg.inv(_sequence_matrix(x.dtype, x.device))
    return torch.einsum("ij,...j->...i", a_inv, x)


def sequence_to_vector(seq: Tensor) -> Tensor:
    """Element vectors ``x_abc = A [x0, x+, x-]`` over the trailing axis of length 3."""
    return torch.einsum("ij,...j->...i", _sequence_matrix(seq.dtype, seq.device), seq)


def apply_sequence_operator(diag: Tensor, x: Tensor) -> Tensor:
    """Apply ``A diag(d) A^-1`` to element vectors ``x[..., 3]`` (broadcasting)."""
    return torch.einsum("...ij,...j->...i", sequence_to_element(diag), x)
