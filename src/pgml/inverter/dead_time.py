"""Bridge voltage error caused by the switch blanking (dead) time.

During the blanking time ``t_d`` both switches of a leg are off and the leg voltage
follows the current direction. Averaged over a switching period, the leg voltage drops
by ``(t_d / T_s) U_dc`` for a positive leg current and rises by the same amount for a
negative one; for a single-phase full bridge the output voltage error is twice that
(Mohan, Undeland and Robbins, *Power Electronics*, sec. 8-5, eqs. 8-75 to 8-77). The
error is a square wave locked to the current: odd harmonics in a single-phase unit,
orders ``6k +- 1`` in the line-to-line (and differential phase) voltages of a three-phase
unit (same section). Its amplitude does not depend on the magnitude of the current,
which is why the resulting low-order current harmonics do not scale with the power.

With the hard sign the series is exact: ``-dV sgn(cos(w t + phi)) = -dV (4 / pi)
sum_{h odd} (-1)^((h-1)/2) / h cos(h (w t + phi))``. A finite current ripple makes the
current cross zero within a switching period near the fundamental zero crossing, which
softens the sign; ``ripple_a`` replaces ``sgn(i)`` by ``clip(i / ripple_a, -1, 1)`` and the
spectrum is then taken by a DFT over one fundamental period.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor


def dead_time_voltage(
    i1: Tensor,
    u_dc: Tensor,
    dead_time_s: float,
    switching_hz: float,
    orders: Sequence[int],
    *,
    full_bridge: bool,
    three_phase: bool,
    ripple_a: Optional[float] = None,
    n_samples: int = 2048,
) -> Tensor:
    """Dead-time bridge voltage disturbance at integer ``orders`` (RMS phasors).

    Parameters
    ----------
    i1:
        Fundamental bridge-side current per element, RMS phasor in the grid's phase
        reference, ``[*batch, E]`` (current leaving the bridge).
    u_dc:
        DC-link voltage ``[*batch]``.
    dead_time_s, switching_hz:
        Blanking time per commutation and leg switching frequency.
    orders:
        Integer harmonic orders (>= 2) to return.
    full_bridge:
        ``True`` for a single-phase H bridge (two legs), ``False`` for one leg per
        element.
    three_phase:
        Remove the common-mode part of the three leg errors (a three-wire unit drives
        only the differential part).
    ripple_a:
        Optional current below which the sign softens linearly (see module docstring).
    n_samples:
        Samples per fundamental period for the DFT used with ``ripple_a``.

    Returns
    -------
    Tensor
        Complex ``[*batch, H, E]`` disturbance added to the bridge voltage.
    """
    rdt = u_dc.dtype
    dv = (dead_time_s * switching_hz) * u_dc.to(rdt) * (2.0 if full_bridge else 1.0)
    phi = torch.angle(i1)  # [*batch, E]
    h = torch.as_tensor([float(o) for o in orders], dtype=rdt, device=u_dc.device)
    if ripple_a is None:
        odd = (torch.remainder(h, 2.0) == 1.0).to(rdt)
        sign = torch.where(torch.remainder(h - 1.0, 4.0) == 0.0, 1.0, -1.0).to(rdt)
        coeff = odd * sign * (4.0 / math.pi) / h  # [H]
        amp = -dv.unsqueeze(-1) * coeff / math.sqrt(2.0)  # [*batch, H]
        rotation = torch.polar(
            torch.ones_like(phi).unsqueeze(-2), h.unsqueeze(-1) * phi.unsqueeze(-2)
        )  # [*batch, H, E]
        d = amp.unsqueeze(-1).to(rotation.dtype) * rotation
    else:
        t = torch.arange(n_samples, dtype=rdt, device=u_dc.device) * (
            2.0 * math.pi / n_samples
        )  # w t over one period
        mag = math.sqrt(2.0) * torch.abs(i1)  # peak
        wave = mag.unsqueeze(-1) * torch.cos(t + phi.unsqueeze(-1))  # [*batch, E, N]
        err = -dv.unsqueeze(-1).unsqueeze(-1) * torch.clamp(wave / ripple_a, -1.0, 1.0)
        basis = torch.exp(-1j * h.unsqueeze(-1) * t)  # [H, N]
        d = math.sqrt(2.0) * torch.einsum("...en,hn->...he", err.to(basis.dtype), basis)
        d = d / n_samples
    if three_phase:
        d = d - d.mean(dim=-1, keepdim=True)
    return d
