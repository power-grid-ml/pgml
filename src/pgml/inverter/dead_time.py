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
sum_{h odd} (-1)^((h-1)/2) / h cos(h (w t + phi))``.

Ripple dead zone (``ripple="local"``): a leg commutes at the extremes of the inductor's
switching ripple. Where the fundamental current is smaller than half the local
peak-to-peak ripple, the current at the two commutations of a period has opposite signs
and the two blanking errors cancel, so the error vanishes there and keeps its full value
elsewhere. The half ripple follows the local modulation ``m = M cos(w t + t_b)`` of the
bridge-side inductor ``L1`` at carrier frequency ``f_c``:

- one leg against a stiff point (three-phase): ``U_dc (1 - m^2) / (8 L1 f_c)``;
- bipolar full bridge: ``U_dc (1 - m^2) / (4 L1 f_c)``;
- unipolar full bridge (pulses between 0 and ``+-U_dc`` at ``2 f_c``):
  ``U_dc |m| (1 - |m|) / (4 L1 f_c)``.

This replaces the hard sign by ``sgn(i) [|i| > half ripple]`` and takes the spectrum by
a DFT over one fundamental period. Compared with switching simulations, the hard sign
overstates the low-order error once the current is comparable to the ripple (low
power), where the dead zone follows the reference much more closely; see the inverter
modelling page for the measured agreement.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml.errors import InputError


def dead_time_voltage(
    i1: Tensor,
    u_dc: Tensor,
    dead_time_s: float,
    switching_hz: float,
    orders: Sequence[int],
    *,
    full_bridge: bool,
    three_phase: bool,
    ripple: Optional[str] = None,
    l1_h: Optional[float] = None,
    modulation_index: Optional[Tensor] = None,
    bridge_angle: Optional[Tensor] = None,
    topology: Optional[str] = None,
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
    ripple:
        ``None`` (hard sign) or ``"local"`` (ripple dead zone, see module docstring),
        which needs ``l1_h``, ``modulation_index`` ``[*batch]``, ``bridge_angle``
        ``[*batch, E]`` (angle of each element's bridge fundamental) and the PWM
        ``topology``.
    n_samples:
        Samples per fundamental period for the DFT of the dead-zone model.

    Returns
    -------
    Tensor
        Complex ``[*batch, H, E]`` disturbance added to the bridge voltage.
    """
    rdt = u_dc.dtype
    dv = (dead_time_s * switching_hz) * u_dc.to(rdt) * (2.0 if full_bridge else 1.0)
    phi = torch.angle(i1)  # [*batch, E]
    h = torch.as_tensor([float(o) for o in orders], dtype=rdt, device=u_dc.device)
    if ripple is None:
        odd = (torch.remainder(h, 2.0) == 1.0).to(rdt)
        sign = torch.where(torch.remainder(h - 1.0, 4.0) == 0.0, 1.0, -1.0).to(rdt)
        coeff = odd * sign * (4.0 / math.pi) / h  # [H]
        amp = -dv.unsqueeze(-1) * coeff / math.sqrt(2.0)  # [*batch, H]
        rotation = torch.polar(
            torch.ones_like(phi).unsqueeze(-2), h.unsqueeze(-1) * phi.unsqueeze(-2)
        )  # [*batch, H, E]
        d = amp.unsqueeze(-1).to(rotation.dtype) * rotation
    elif ripple == "local":
        if l1_h is None or modulation_index is None or bridge_angle is None:
            raise InputError(
                "dead_time_voltage(ripple='local') needs l1_h, modulation_index and "
                "bridge_angle."
            )
        t = torch.arange(n_samples, dtype=rdt, device=u_dc.device) * (
            2.0 * math.pi / n_samples
        )  # w t over one period
        mag = math.sqrt(2.0) * torch.abs(i1)  # peak
        wave = mag.unsqueeze(-1) * torch.cos(t + phi.unsqueeze(-1))  # [*batch, E, N]
        m = modulation_index.to(rdt).unsqueeze(-1).unsqueeze(-1) * torch.cos(
            t + bridge_angle.to(rdt).unsqueeze(-1)
        )
        scale = u_dc.to(rdt).unsqueeze(-1).unsqueeze(-1) / (l1_h * switching_hz)
        if topology == "single_phase_unipolar":
            half = scale * m.abs() * (1.0 - m.abs()).clamp(min=0.0) / 4.0
        elif topology == "single_phase_bipolar":
            half = scale * (1.0 - m * m).clamp(min=0.0) / 4.0
        else:
            half = scale * (1.0 - m * m).clamp(min=0.0) / 8.0
        conducts = (wave.abs() > half).to(rdt)
        err = -dv.unsqueeze(-1).unsqueeze(-1) * torch.sign(wave) * conducts
        basis = torch.exp(-1j * h.unsqueeze(-1) * t)  # [H, N]
        d = math.sqrt(2.0) * torch.einsum("...en,hn->...he", err.to(basis.dtype), basis)
        d = d / n_samples
    else:
        raise InputError(f"dead_time_voltage: unknown ripple model {ripple!r}.")
    if three_phase:
        d = d - d.mean(dim=-1, keepdim=True)
    return d
