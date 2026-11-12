"""DC-link voltage loop: MPPT interharmonics and the double-frequency ripple.

The DC-link voltage controller sets the amplitude ``|i_g|`` of the grid-current
reference, which the PLL angle turns into a sinusoid. Any oscillation of ``|i_g|`` at
``f_n`` therefore appears in the grid current at ``f_1 - f_n`` and ``f_1 + f_n``.

MPPT
----
Perturb-and-observe MPPT steps the DC-link reference periodically; in steady state the
reference cycles through ``{0, +v_step, 0, -v_step}``, each level held for
``T_MPPT = 1 / f_MPPT``, so its period is ``T_0 = 4 T_MPPT`` and only odd Fourier orders
``n`` survive at ``f_n = n f_MPPT / 4`` with amplitude ``A_n = 2 sqrt 2 v_step / (pi n)``
(Sangwongwanich et al., IEEE TPEL 33(10), 2018, doi:10.1109/TPEL.2017.2778025, eqs. 1-3).
The closed DC-link loop maps the reference to ``|i_g|`` by (same paper, eq. 4)

``G_cl = G_PI G_cc / (1 + G_notch G_plant G_PI G_cc)``, ``G_PI = k_p + k_i / s``,
``G_cc = 1 / (1 + 3 T_s s)``, ``G_plant = V_g / (sqrt 2 V_dc C_dc s)``,
``G_notch = (s^2 + w_h^2) / (s^2 + k_h s + w_h^2)`` with ``w_h = 2 w_1``,

and each ``|i_g|`` line ``A'_n`` gives two grid-current lines of amplitude ``A'_n / 2``
(eqs. 5-7). The phases use the quadrant-correct ``atan2`` and the exact product
``A' sin(w_n t + p) sin(w_1 t) = A'/2 [cos((w_1 - w_n) t - p) - cos((w_1 + w_n) t + p)]``:
the printed eq. 2 describes the waveform shifted by half a period and eqs. 6-7 carry a
sign slip in the lower sideband (both found by a numerical Fourier analysis of the
four-level waveform and of the product). The MPPT clocks of different units are independent, so these lines
have no common phase reference between devices.

These lines are the grid current of eq. 4, whose inner current loop is the first-order
``G_cc``; they are used as Norton currents directly, i.e. with ideal tracking of the
reference by the inner loop (the closed loop's ``G_ref`` of
:func:`~pgml.inverter.control.closed_loop` is not applied a second time; it is 1.10 in
magnitude at 100 Hz for a typical single-phase unit).

The plant above is single-phase. For an ``n``-phase unit the power is ``n`` times that
of one element, so ``G_plant`` is scaled by ``n`` (an extension of the single-phase
model).

Double-frequency ripple
-----------------------
A single-phase unit's AC power pulsates at ``2 f_1``. With terminal voltage
``sqrt 2 V cos(w t + a)`` and current ``sqrt 2 I cos(w t + b)`` the lossless DC-link
balance ``C U du/dt = -V I cos(2 w t + a + b)`` gives the ripple
``u = -V I / (2 w C U) sin(2 w t + a + b)``, of amplitude ``P / (4 pi f_1 C U)`` at unity
power factor. The measured ripple passes the notch and the PI controller into
``|i_g|``, whose ``2 f_1`` component multiplies the fundamental reference into a third
harmonic. With an ideal notch at ``2 f_1`` the path is blocked. This component is locked
to the fundamental and scales with the power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ._common import TWO_PI, Scalar, as_param, as_real, laplace


@dataclass(frozen=True)
class DCLinkControl:
    """DC-link capacitor and voltage controller.

    Attributes
    ----------
    c_dc_f:
        DC-link capacitance.
    kp, ki:
        PI gains from the DC-voltage error [V] to the peak grid-current amplitude [A]
        (``kp`` in A/V, ``ki`` in A/(V s)).
    sampling_hz:
        Controller sampling frequency; the inner current loop is ``1 / (1 + 3 T_s s)``.
    notch_quality_rad_s:
        ``k_h`` of the ``2 f_1`` notch in the feedback path; ``None`` removes the notch.
    """

    c_dc_f: Scalar
    kp: Scalar
    ki: Scalar
    sampling_hz: float = 16000.0
    notch_quality_rad_s: Optional[Scalar] = 100.0

    def _blocks(
        self, s: Tensor, f1_hz: float, v_g: Tensor, u_dc: Tensor, n_phases: int
    ):
        g_pi = as_param(self.kp, s).to(s.dtype) + as_param(self.ki, s).to(s.dtype) / s
        g_cc = 1.0 / (1.0 + 3.0 / self.sampling_hz * s)
        c = as_param(self.c_dc_f, s).to(s.dtype)
        plant = (
            n_phases
            * v_g.to(s.dtype).unsqueeze(-1)
            / (math.sqrt(2.0) * u_dc.to(s.dtype).unsqueeze(-1) * c * s)
        )
        if self.notch_quality_rad_s is None:
            notch = torch.ones_like(s)
        else:
            wh2 = (2.0 * TWO_PI * f1_hz) ** 2
            kh = as_param(self.notch_quality_rad_s, s).to(s.dtype)
            notch = (s * s + wh2) / (s * s + kh * s + wh2)
        return g_pi, g_cc, plant, notch

    def reference_to_amplitude(
        self, s: Tensor, f1_hz: float, v_g: Tensor, u_dc: Tensor, n_phases: int = 1
    ) -> Tensor:
        """``G_cl(s)`` from the DC-link reference to ``|i_g|`` (``[*batch, F]``)."""
        g_pi, g_cc, plant, notch = self._blocks(s, f1_hz, v_g, u_dc, n_phases)
        return g_pi * g_cc / (1.0 + notch * plant * g_pi * g_cc)

    def ripple_to_amplitude(
        self, s: Tensor, f1_hz: float, v_g: Tensor, u_dc: Tensor, n_phases: int = 1
    ) -> Tensor:
        """From a DC-link voltage disturbance to ``|i_g|`` (``[*batch, F]``)."""
        g_pi, g_cc, plant, notch = self._blocks(s, f1_hz, v_g, u_dc, n_phases)
        return notch * g_pi * g_cc / (1.0 + notch * plant * g_pi * g_cc)


@dataclass(frozen=True)
class PerturbObserveMPPT:
    """Perturb-and-observe MPPT in steady state (three-level oscillation).

    Attributes
    ----------
    v_step_v:
        Perturbation step of the DC-link voltage reference.
    rate_hz:
        MPPT sampling rate ``f_MPPT``.
    max_line_hz:
        Highest line frequency returned. A line at ``f`` collects an upper sideband
        from order ``4 (f - f_1) / f_MPPT`` and a folded lower one from
        ``4 (f + f_1) / f_MPPT``, and the reference amplitudes fall only as ``1 / n``,
        so the orders are taken up to ``4 (max_line_hz + f_1) / f_MPPT`` and every
        returned line is complete. (A fixed count of 50 orders left lines at 2 Hz and
        5 Hz MPPT rates up to 63 % off against a time-domain reference.)
    n_terms:
        Optional fixed number of odd orders instead (lines are then complete only as
        far as the orders reach).
    """

    v_step_v: Scalar
    rate_hz: float
    max_line_hz: float = 500.0
    n_terms: Optional[int] = None

    def orders(self, f1_hz: float = 50.0) -> list[int]:
        if self.n_terms is not None:
            return [2 * k + 1 for k in range(self.n_terms)]
        n_max = int(math.ceil(4.0 * (self.max_line_hz + f1_hz) / self.rate_hz))
        return list(range(1, n_max + 1, 2))

    def line_frequencies(self, f1_hz: float) -> list[float]:
        """``|f_1 - n f_MPPT / 4|`` and ``f_1 + n f_MPPT / 4`` for the kept orders.

        A lower sideband below 0 Hz is the same real signal at ``n f_MPPT / 4 - f_1``
        (folded); one exactly at 0 Hz is a DC offset and dropped.
        """
        out = []
        for n in self.orders(f1_hz):
            fn = n * self.rate_hz / 4.0
            if abs(f1_hz - fn) > 1e-9 and abs(f1_hz - fn) <= self.max_line_hz:
                out.append(abs(f1_hz - fn))
            if f1_hz + fn <= self.max_line_hz:
                out.append(f1_hz + fn)
        return out

    def reference_lines(
        self, rdt, device, f1_hz: float = 50.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``(f_n, A_n, phi_n)`` of the DC-link reference (peak amplitudes)."""
        n = torch.as_tensor(self.orders(f1_hz), dtype=rdt, device=device)
        v = as_param(self.v_step_v, n)  # [*batch, 1] for a batched step
        a_n = -(2.0 * v / (math.pi * n)) * torch.sin(math.pi * n / 2.0)
        b_n = (v / (math.pi * n)) * (1.0 - torch.cos(math.pi * n))
        amp = torch.sqrt(a_n * a_n + b_n * b_n)
        phi = torch.atan2(a_n, b_n)
        return n * self.rate_hz / 4.0, amp, phi


def mppt_current_lines(
    mppt: PerturbObserveMPPT,
    dc: DCLinkControl,
    *,
    f1_hz: float,
    v_g: Tensor,
    u_dc: Tensor,
    current_angle: Tensor,
    n_phases: int = 1,
    cdt: torch.dtype = torch.complex128,
) -> tuple[list[float], Tensor]:
    """Grid-current lines of the MPPT perturbation.

    Parameters
    ----------
    v_g, u_dc, current_angle:
        ``[*batch]``: RMS grid voltage per element, DC-link voltage, angle of the
        fundamental grid current (element 0) in the grid's phase reference.
    n_phases:
        Elements of the unit; a three-phase unit's lines are positive sequence.

    Returns
    -------
    (frequencies_hz, phasors):
        Line frequencies ``[L]`` and RMS current phasors ``[*batch, L, E]`` leaving the
        device. A lower sideband below 0 Hz is folded to ``f_n - f_1`` with the conjugate
        phasor (``cos(-w t + p) = cos(w t - p)``; for a three-phase unit the folded line
        is negative sequence). Lines may share a frequency; the caller sums them.
    """
    rdt = u_dc.dtype
    dev = u_dc.device
    fn, a_n, phi_n = mppt.reference_lines(rdt, dev, f1_hz)
    s = laplace(fn, cdt)
    # A higher DC-link reference means less exported power: |i_g| = -G_cl v*_dc, the
    # loop convention of the ripple path below (|i_g| = G_cc G_PI (G_notch v_dc - v*)).
    g = -dc.reference_to_amplitude(s, f1_hz, v_g, u_dc, n_phases)  # [*batch, Nn]
    a_p = a_n * g.abs()
    p = phi_n + torch.angle(g)
    beta = (
        current_angle.to(rdt).unsqueeze(-1) + math.pi / 2.0
    )  # i = |i| sin(w t + beta')
    lower = torch.polar(a_p / (2.0 * math.sqrt(2.0)), beta - p)
    upper = torch.polar(a_p / (2.0 * math.sqrt(2.0)), beta + p + math.pi)
    freqs: list[float] = []
    cols = []
    fold = []
    for k, n in enumerate(mppt.orders(f1_hz)):
        f_n = n * mppt.rate_hz / 4.0
        if abs(f1_hz - f_n) > 1e-9 and abs(f1_hz - f_n) <= mppt.max_line_hz:
            freqs.append(abs(f1_hz - f_n))
            cols.append(lower[..., k])
            fold.append(f1_hz - f_n < 0)
        if f1_hz + f_n <= mppt.max_line_hz:
            freqs.append(f1_hz + f_n)
            cols.append(upper[..., k])
            fold.append(False)
    lines = torch.stack(cols, dim=-1).to(cdt).unsqueeze(-1)  # [*batch, L, 1]
    if n_phases == 3:
        shift = torch.polar(
            torch.ones(3, dtype=rdt, device=dev),
            torch.as_tensor([0.0, -TWO_PI / 3.0, TWO_PI / 3.0], dtype=rdt, device=dev),
        ).to(cdt)
        lines = lines * shift
    folded = torch.as_tensor(fold, dtype=torch.bool, device=dev).unsqueeze(-1)
    return freqs, torch.where(folded, torch.conj(lines), lines)


def ripple_third_harmonic(
    dc: DCLinkControl,
    *,
    f1_hz: float,
    v_t: Tensor,
    i_t: Tensor,
    u_dc: Tensor,
    cdt: torch.dtype = torch.complex128,
) -> Tensor:
    """Third-harmonic grid-current reference of the ``2 f_1`` DC-link ripple.

    ``v_t`` and ``i_t`` are the fundamental voltage and current (leaving the device) of
    a single-phase unit, RMS phasors ``[*batch]``. Returns the RMS third-harmonic
    current phasor ``[*batch]`` (locked to the fundamental).
    """
    w = TWO_PI * f1_hz
    c = as_real(dc.c_dc_f, u_dc.dtype, u_dc.device)
    amp = torch.abs(v_t) * torch.abs(i_t) / (2.0 * w * c * u_dc)
    # u(t) = Re(U exp(j 2 w t)), U = j A exp(j (a + b)) (peak phasor).
    u_ph = torch.polar(amp, torch.angle(v_t) + torch.angle(i_t) + math.pi / 2.0).to(cdt)
    s2 = laplace(
        torch.as_tensor([2.0 * f1_hz], dtype=u_dc.dtype, device=u_dc.device), cdt
    )
    h = dc.ripple_to_amplitude(s2, f1_hz, torch.abs(v_t), u_dc, 1)[..., 0]
    delta = h * u_ph  # peak phasor of |i_g| at 2 f_1
    # |i_g|(t) cos(w t + b) -> third harmonic peak phasor delta exp(j b) / 2.
    return (
        delta
        * torch.polar(torch.ones_like(amp), torch.angle(i_t)).to(cdt)
        / (2.0 * math.sqrt(2.0))
    )
