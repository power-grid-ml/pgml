"""PWM bridge voltage spectrum of carrier-based sine-triangle modulation.

One inverter leg, naturally sampled against a triangular carrier, has the double Fourier
series (leg voltage against the DC-link midpoint, reference ``M cos(w_0 t + t_0)``,
carrier phase ``t_c``)

.. math::

    v(t) = \\frac{U_{dc}}{2} M \\cos(w_0 t + t_0)
    + \\frac{2 U_{dc}}{\\pi} \\sum_{m \\ge 1} \\sum_{n=-\\infty}^{\\infty}
      \\frac{1}{m} J_n\\!\\left(m \\frac{\\pi}{2} M\\right) \\sin\\!\\left((m+n)\\frac{\\pi}{2}\\right)
      \\cos\\!\\big(m (w_c t + t_c) + n (w_0 t + t_0)\\big)

(Holmes and Lipo, *Pulse Width Modulation for Power Converters*, the naturally sampled
two-level leg). A topology is a combination of legs:

- ``"single_phase_bipolar"``: two complementary legs, ``v_AB = 2 v_a``;
- ``"single_phase_unipolar"``: two legs with references shifted by ``pi`` on one
  carrier, ``v_AB = v_a(t_0) - v_a(t_0 + pi)``; the odd carrier groups cancel and the
  remaining groups at ``2 m f_c`` carry the sidebands of Klatt et al. (IET GTD 13(24),
  2019, eq. 3);
- ``"three_phase"``: three legs with references shifted by ``2 pi / 3``; the element
  voltage is the leg voltage minus the mean of the three (the differential part a
  three-wire connection can drive).

The modulation index is ``M = peak bridge fundamental / U_dc`` for the two single-phase
topologies and ``M = peak leg fundamental / (U_dc / 2)`` for the three-phase leg (so a
phase-to-neutral bridge fundamental of peak ``M U_dc / 2``).

Only the switching residues are returned: the fundamental component is the controlled
bridge voltage and belongs to the operating point. Lines are RMS phasors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from pgml.errors import InputError, ModelingError

from ._common import Scalar, as_real

Topology = Literal["single_phase_bipolar", "single_phase_unipolar", "three_phase"]


def bessel_j(n: Tensor, x: Tensor, n_points: int = 0) -> Tensor:
    """Bessel function of the first kind ``J_n(x)`` for integer ``n``, differentiable in ``x``.

    Evaluated from Bessel's integral ``J_n(x) = 1/(2 pi) int_0^{2 pi} cos(n t - x sin t)
    dt`` with the trapezoidal rule, which is spectrally accurate for this periodic
    integrand once the number of points exceeds ``|n| + |x|`` by a margin (the default
    takes ``2 (|n| + |x|) + 64``, read from the values off the tape; callers with a
    static bound pass ``n_points``). ``n`` and ``x`` broadcast.
    """
    n = n.to(x.dtype)
    if n_points <= 0:
        with torch.no_grad():
            bound = float(n.abs().max()) + float(x.abs().max()) if x.numel() else 0.0
        n_points = int(2 * math.ceil(bound) + 64)
    t = torch.arange(n_points, dtype=x.dtype, device=x.device) * (
        2.0 * math.pi / n_points
    )
    arg = n.unsqueeze(-1) * t - x.unsqueeze(-1) * torch.sin(t)
    return torch.cos(arg).mean(dim=-1)


@dataclass(frozen=True)
class PWMModulator:
    """Carrier-based PWM of the bridge.

    Attributes
    ----------
    topology:
        ``"single_phase_bipolar"``, ``"single_phase_unipolar"`` or ``"three_phase"``.
    carrier_hz:
        Triangular carrier frequency ``f_c``. The switching frequency of a leg equals
        ``f_c``; unipolar modulation places the first residue group at ``2 f_c``.
    carrier_groups:
        Number of carrier groups ``m = 1 .. carrier_groups`` kept.
    sidebands:
        Sideband orders ``|n| <= sidebands`` kept in each group.
    """

    topology: Topology = "single_phase_unipolar"
    carrier_hz: float = 16000.0
    carrier_groups: int = 4
    sidebands: int = 12

    def __post_init__(self):
        if self.topology not in (
            "single_phase_bipolar",
            "single_phase_unipolar",
            "three_phase",
        ):
            raise InputError(f"unknown PWM topology {self.topology!r}.")
        if self.carrier_groups < 1 or self.sidebands < 0:
            raise InputError("carrier_groups must be >= 1 and sidebands >= 0.")

    @property
    def n_elements(self) -> int:
        return 3 if self.topology == "three_phase" else 1

    def line_orders(self) -> tuple[list[int], list[int]]:
        """The ``(m, n)`` pairs of the residue lines kept by this modulator."""
        ms, ns = [], []
        for m in range(1, self.carrier_groups + 1):
            for n in range(-self.sidebands, self.sidebands + 1):
                ms.append(m)
                ns.append(n)
        return ms, ns

    def line_frequencies(self, f1_hz: float) -> list[float]:
        """Frequencies ``m f_c + n f_1`` of the kept lines (positive ones only)."""
        ms, ns = self.line_orders()
        return [m * self.carrier_hz + n * f1_hz for m, n in zip(ms, ns)]

    def bridge_spectrum(
        self,
        modulation_index: Tensor,
        u_dc: Tensor,
        f1_hz: float,
        reference_angle: Tensor,
        carrier_angle: Scalar = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Residue lines of the bridge element voltages.

        Parameters
        ----------
        modulation_index, u_dc, reference_angle:
            ``[*batch]`` tensors: ``M`` (as defined in the module docstring), the
            DC-link voltage, and the angle ``t_0`` of the bridge fundamental reference
            (of element 0) in the grid's phase reference [rad].
        f1_hz:
            Fundamental frequency.
        carrier_angle:
            Carrier phase ``t_c`` [rad] (a free parameter: carriers of different units
            are not synchronised).

        Returns
        -------
        (frequencies_hz, phasors):
            ``[L]`` line frequencies (python-float based, positive) and the RMS element
            voltage phasors ``[*batch, L, E]`` (``E`` elements of the topology). Lines
            of equal frequency are not merged here.
        """
        rdt = u_dc.dtype
        dev = u_dc.device
        cdt = torch.complex128 if rdt == torch.float64 else torch.complex64
        ms, ns = self.line_orders()
        m_t = torch.as_tensor(ms, dtype=rdt, device=dev)
        n_t = torch.as_tensor(ns, dtype=rdt, device=dev)
        freqs = m_t * self.carrier_hz + n_t * f1_hz
        if bool((freqs <= 0).any()):
            raise InputError(
                "PWM residue lines below 0 Hz: carrier_hz must exceed sidebands * f1."
            )
        with torch.no_grad():
            if bool((modulation_index > 1.0).any()):
                raise ModelingError(
                    "PWM modulation index above 1: the naturally sampled sine-triangle "
                    "series holds for M <= 1 only (overmodulation adds baseband "
                    "harmonics it does not describe)."
                )
        mi = modulation_index.to(rdt).unsqueeze(-1)  # [*batch, 1]
        # Quadrature points from the static bound |n| + m pi M / 2 with M <= 1.
        n_points = int(2 * (self.sidebands + self.carrier_groups * math.pi * 0.75) + 64)
        jn = bessel_j(n_t, m_t * (math.pi / 2.0) * mi, n_points)  # [*batch, L]
        amp = (2.0 / math.pi) * u_dc.to(rdt).unsqueeze(-1) * jn / m_t
        amp = amp * torch.sin((m_t + n_t) * (math.pi / 2.0))
        tc = as_real(carrier_angle, rdt, dev)
        t0 = reference_angle.to(rdt).unsqueeze(-1)

        def leg(offset: float) -> Tensor:
            # cos(m (w_c t + t_c) + n (w_0 t + t_0 + offset)) as an RMS phasor.
            phase = m_t * tc + n_t * (t0 + offset)
            # The Bessel amplitude is signed: multiply a unit phasor (torch.polar
            # expects a nonnegative magnitude and differentiates wrongly otherwise).
            unit = torch.polar(torch.ones_like(phase), phase)
            return (amp / math.sqrt(2.0)).to(cdt) * unit.to(cdt)  # [*batch, L]

        if self.topology == "single_phase_bipolar":
            out = (2.0 * leg(0.0)).unsqueeze(-1)
        elif self.topology == "single_phase_unipolar":
            out = (leg(0.0) - leg(math.pi)).unsqueeze(-1)
        else:
            legs = torch.stack(
                [leg(0.0), leg(-2.0 * math.pi / 3.0), leg(2.0 * math.pi / 3.0)], dim=-1
            )
            out = legs - legs.mean(dim=-1, keepdim=True)
        return freqs, out

    def modulation_index(self, bridge_peak_v: Tensor, u_dc: Tensor) -> Tensor:
        """``M`` from the peak bridge fundamental per element and the DC-link voltage."""
        if self.topology == "three_phase":
            return bridge_peak_v / (0.5 * u_dc)
        return bridge_peak_v / u_dc
