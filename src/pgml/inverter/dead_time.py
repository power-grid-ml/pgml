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
peak-to-peak ripple ``h``, the current at the two commutations of a period has opposite
signs and the two blanking errors cancel, so the error vanishes there. During the
blanking interval the current keeps ramping with slope ``s`` away from the ripple
extreme, so the transition is not a step: the fraction of the full error is
``clip((|i| - h + s t_d) / (s t_d), 0, 1)``. With ``K = U_dc / (8 L1 f_c)`` and the
local modulation ``m = M cos(w t + t_b)`` of the bridge-side inductor ``L1``:

- bipolar full bridge: ``h = 2 K (1 - m^2)``, ``s = U_dc (1 -+ m) / L1`` (minus for a
  positive current);
- unipolar full bridge (pulses between 0 and ``+-U_dc`` at ``2 f_c``):
  ``h = 2 K |m| (1 - |m|)``, ``s = U_dc (1 - |m|) / L1`` when the current and ``m`` have
  the same sign, else ``U_dc |m| / L1``. Near ``m = 0`` the pulses become shorter than the
  blanking time and the ramp removes the dead zone the threshold alone would predict;
- three-phase three-wire: the phase ripple superposes the three legs,
  ``h_a = K / 3 |2 (1 - m_a^2) - g(m_a, m_b) - g(m_a, m_c)|`` with
  ``g(a, x) = (1 - max(a, x)) (1 + min(a, x))``, and ``s = U_dc / 2 [(2 -+ (s_b + s_c)) / 3
  -+ m_a] / L1`` (upper signs for a positive current), ``s_x = sgn(m_x - m_a)``.

The spectrum is then taken by a DFT over one fundamental period. The expressions hold for
the bridge's own switching ripple on a sinusoidal current; the measured agreement with
switching simulations is on the inverter modelling page. Not covered: current clamping
at zero within the blanking interval and inductor resistance.
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
        udc = u_dc.to(rdt).unsqueeze(-1).unsqueeze(-1)
        k = udc / (8.0 * l1_h * switching_hz)
        pos = wave > 0
        if topology == "single_phase_unipolar":
            am = m.abs()
            half = 2.0 * k * am * (1.0 - am).clamp(min=0.0)
            same = torch.sign(wave) == torch.sign(m)
            slope = torch.where(same, udc * (1.0 - am), udc * am) / l1_h
        elif topology == "single_phase_bipolar":
            half = 2.0 * k * (1.0 - m * m).clamp(min=0.0)
            slope = torch.where(pos, udc * (1.0 - m), udc * (1.0 + m)) / l1_h
        else:
            if m.shape[-2] != 3:
                raise InputError("the three-phase dead zone needs three elements.")
            m_b = torch.roll(m, shifts=-1, dims=-2)
            m_c = torch.roll(m, shifts=-2, dims=-2)

            def g(a, x):
                return (1.0 - torch.maximum(a, x)) * (1.0 + torch.minimum(a, x))

            half = k / 3.0 * (2.0 * (1.0 - m * m) - g(m, m_b) - g(m, m_c)).abs()
            s_bc = torch.sign(m_b - m) + torch.sign(m_c - m)
            slope = (
                torch.where(
                    pos,
                    0.5 * udc * ((2.0 - s_bc) / 3.0 - m),
                    0.5 * udc * ((2.0 + s_bc) / 3.0 + m),
                )
                / l1_h
            )
        ramp = (slope * dead_time_s).clamp(min=1e-12)
        fraction = ((wave.abs() - half + ramp) / ramp).clamp(0.0, 1.0)
        err = -dv.unsqueeze(-1).unsqueeze(-1) * torch.sign(wave) * fraction
        basis = torch.exp(-1j * h.unsqueeze(-1) * t)  # [H, N]
        d = math.sqrt(2.0) * torch.einsum("...en,hn->...he", err.to(basis.dtype), basis)
        d = d / n_samples
    else:
        raise InputError(f"dead_time_voltage: unknown ripple model {ripple!r}.")
    if three_phase:
        d = d - d.mean(dim=-1, keepdim=True)
    return d
