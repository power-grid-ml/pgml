"""Differentiable parameter extraction from materialised schema objects.

These helpers turn schema model fields (Python lists / floats / tuples) into
batched torch tensors that enter the differentiable Y-bus path. The conversion
from python numbers to a tensor is the START of the autograd tape; downstream
math is torch and gradients flow back to these tensors (a test wishing to
gradcheck w.r.t. R/L/C/Z replaces the relevant tensor with a leaf — see the
``param_overrides`` hook on the public assembly API).

No differentiable in-place ops; loops here are over the (small, fixed) python
collection of components, building lists that are stacked ONCE into a tensor.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor


def matrix_to_tensor(
    mat: Sequence[Sequence[float]], dtype: torch.dtype, device
) -> Tensor:
    """Convert a row-major PerPhaseMatrix to a real tensor ``[P, P]``."""
    return torch.as_tensor(mat, dtype=dtype, device=device)


def diag_from_tuple(values: Sequence[float], dtype: torch.dtype, device) -> Tensor:
    """Build a diagonal ``[P, P]`` real tensor from a per-phase tuple."""
    v = torch.as_tensor(values, dtype=dtype, device=device)
    return torch.diag(v)


def phase_voltage_magnitude(u_rated_v: float, n_phases: int) -> float:
    """Line-to-neutral voltage magnitude used for the const-Z load model.

    The schema stores ``Node.u_rated_v`` as line-to-line for 3-phase nodes and
    line-to-neutral for 1-phase nodes. For the per-phase const-Z conversion we use
    a line-to-neutral magnitude: divide by sqrt(3) when 3 phases are present.
    """
    if n_phases >= 3:
        return u_rated_v / math.sqrt(3.0)
    return u_rated_v


def const_z_shunt_admittance(
    p_per_phase: Sequence[float],
    q_per_phase: Sequence[float],
    u_ln_v: float,
    sign: float,
    dtype: torch.dtype,
    device,
) -> Tensor:
    """Per-phase complex const-Z shunt admittance ``y = conj(P + jQ)/|U|^2``.

    Parameters
    ----------
    p_per_phase, q_per_phase:
        Per-phase active/reactive operating-point power (W / var).
    u_ln_v:
        Line-to-neutral voltage magnitude.
    sign:
        ``+1`` for a load (consumes), ``-1`` for a generator (injects), applied to
        both P and Q before forming the admittance.
    Returns
    -------
    Tensor
        Complex ``[P]`` per-phase shunt admittance (diagonal stamp values).
    """
    rdt = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    p = sign * torch.as_tensor(p_per_phase, dtype=rdt, device=device)
    q = sign * torch.as_tensor(q_per_phase, dtype=rdt, device=device)
    u2 = u_ln_v * u_ln_v
    # y = conj(P + jQ) / |U|^2 = (P - jQ) / |U|^2
    cdt = torch.complex128 if rdt == torch.float64 else torch.complex64
    s = torch.complex(p, -q).to(cdt)
    return s / u2


def resolve_operating_power(
    appliance,
    operating_point: Optional[dict],
) -> tuple[list[float], list[float]]:
    """Per-phase (P, Q) operating point for a Load/Generator, length == phases.

    Resolution order:
    1. ``operating_point[appliance.id]`` if given, a dict with ``p_w`` / ``q_var``
       (totals) and/or ``p_per_phase_w`` / ``q_per_phase_var``.
    2. The appliance's ``*_per_phase_*`` nameplate split if present.
    3. The total nameplate ``p_nom_w`` / ``q_nom_var`` split equally across phases.
    """
    n = len(appliance.phases)
    p_total = appliance.p_nom_w
    q_total = appliance.q_nom_var
    p_per = appliance.p_nom_per_phase_w
    q_per = appliance.q_nom_per_phase_var

    if operating_point is not None and appliance.id in operating_point:
        op = operating_point[appliance.id]
        if "p_per_phase_w" in op:
            p_per = op["p_per_phase_w"]
        elif "p_w" in op:
            p_total = op["p_w"]
            p_per = None
        if "q_per_phase_var" in op:
            q_per = op["q_per_phase_var"]
        elif "q_var" in op:
            q_total = op["q_var"]
            q_per = None

    p_list = list(p_per) if p_per is not None else [p_total / n] * n
    q_list = list(q_per) if q_per is not None else [q_total / n] * n
    return p_list, q_list


__all__ = [
    "matrix_to_tensor",
    "diag_from_tuple",
    "phase_voltage_magnitude",
    "const_z_shunt_admittance",
    "resolve_operating_power",
]
