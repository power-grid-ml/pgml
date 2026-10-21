"""Closed-loop current control of a grid-following inverter in the frequency domain.

The model is the small-signal, linear time-invariant (decoupled) part of a current-
controlled voltage-source converter behind an LCL filter, following the structure of
Wang, Harnefors and Blaabjerg (IEEE TPEL 33(2), 2018, doi:10.1109/TPEL.2017.2684906):
the closed loop ``i2 = G_ref i_ref - Y v`` (eqs. 6-7), a digital delay
``exp(-1.5 T_s s)`` (eq. 5), and the synchronous-reference-frame PLL, linearised as
``dtheta = H_PLL(s) dV_q`` with ``H_PLL = H_PI / (s + H_PI V_1d)`` (eq. 16), which
perturbs the current reference by ``j dtheta I_1`` (eqs. 17, 42) and, for dq-frame
current control, the modulating voltage by ``j dtheta V_c1`` (eq. 18). The paper's
L filter is generalised here to the LCL filter of :class:`~pgml.inverter.LCLFilter`,
and grid-voltage feedforward and capacitor-current active damping are added.

Space-vector frequencies
------------------------
Three-phase quantities are complex space vectors: a positive-sequence phasor at ``f`` is
the vector at ``+f``, a negative-sequence phasor at ``f`` the conjugate of the vector at
``-f`` (so its admittance is ``conj(Y(-j 2 pi f))``). The PLL makes the dq-frame model
asymmetric; in the stationary frame this couples a vector at ``f`` to its mirror at
``2 f_1 - f`` (Wang et al., eq. 34). :func:`closed_loop` returns the direct
(same-frequency) term ``Y_plus`` and the coupling coefficient ``Y_mirror`` of that
mirror vector, i.e. ``i(f) = G_ref i_ref(f) + G_d d(f) - Y_plus(f) v(f)
- Y_mirror(f) conj(v(2 f_1 - f))``. The harmonic solve uses the decoupled part only.

Unknowns and equations (per frequency, solved exactly)
------------------------------------------------------
``x = [i1, i2, v_c, v_b]``: bridge-side and grid-side inductor currents, capacitor
voltage, bridge voltage. With ``v`` the inner-port voltage and ``d`` an additive bridge
voltage disturbance (dead time, PWM residues):

- ``v_b - Z1 i1 - v_c = 0``
- ``v_c - Z2 i2 = v``
- ``i1 - i2 - Yc v_c = 0``
- ``v_b = G_del [G_c (i_ref + dI_PLL - i_meas) - k_c Yc v_c + H_ff v + dV_PLL] + d``

``i_meas`` is ``i2`` (grid-current feedback) or ``i1`` (converter-current feedback).
Above the Nyquist frequency of the controller (``f_s / 2``) the control has no authority
and every control term is set to zero: the bridge is a voltage source behind the
passive filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

import torch
from torch import Tensor

from pgml.errors import InputError

from ._common import TWO_PI, Scalar, as_real, laplace
from .filters import LCLFilter

Feedback = Literal["grid", "converter"]
ControllerKind = Literal["pr", "pi_dq"]


@dataclass(frozen=True)
class CurrentController:
    """Current controller of a grid-following inverter.

    Attributes
    ----------
    kind:
        ``"pr"``: proportional-resonant control in the stationary frame (real transfer
        function, acts on both sequences and on single-phase units),
        ``G_c = k_p + sum_h R_h(s)`` with one resonator per entry of ``resonant``.
        ``"pi_dq"``: PI control in the synchronous frame, ``G_c = k_p + k_i / s`` in dq,
        i.e. ``k_p + k_i / (s - j w_1)`` for the positive-sequence space vector in the
        stationary frame (Wang et al. 2018, eq. 4). Three-phase only.
    kp:
        Proportional gain [V/A].
    ki:
        Integral gain of ``"pi_dq"`` [V/(A s)].
    resonant:
        ``{order: k_r}`` resonators of ``"pr"`` at ``order * f_1`` [V/A], each the
        non-ideal resonator ``2 k_r w_c s / (s^2 + 2 w_c s + (h w_1)^2)`` with gain
        ``k_r`` at its centre frequency.
    resonant_bandwidth_rad_s:
        ``w_c`` of the resonators (positive). An ideal resonator (``w_c = 0``) has an
        infinite gain exactly at its centre frequency, which every harmonic frequency
        grid contains, so it is not accepted.
    sampling_hz:
        Controller sampling frequency ``f_s``; the control acts below ``f_s / 2``.
    delay_samples:
        Total delay in sampling periods, ``exp(-delay_samples / f_s * s)``; 1.5 is one
        computation period plus half a period of PWM (Wang et al. 2018, eq. 5).
    feedback:
        Which inductor current is measured and controlled (``"grid"`` or
        ``"converter"``).
    active_damping_kc:
        Capacitor-current feedback gain [V/A] subtracted from the bridge voltage
        command (0 disables it).
    feedforward_gain:
        Gain of the inner-port voltage feedforward onto the bridge voltage command
        (0 disables it, 1 is full feedforward).
    feedforward_cutoff_hz:
        Cut-off of a first-order low-pass on the feedforward path (``None``: no filter).
    """

    kind: ControllerKind = "pr"
    kp: Scalar = 10.0
    ki: Scalar = 0.0
    resonant: Mapping[int, Scalar] = field(default_factory=lambda: {1: 400.0})
    resonant_bandwidth_rad_s: Scalar = 5.0
    sampling_hz: float = 16000.0
    delay_samples: float = 1.5
    feedback: Feedback = "grid"
    active_damping_kc: Scalar = 0.0
    feedforward_gain: Scalar = 0.0
    feedforward_cutoff_hz: Optional[float] = None

    def __post_init__(self):
        if self.kind not in ("pr", "pi_dq"):
            raise InputError(
                f"CurrentController.kind must be 'pr' or 'pi_dq', got {self.kind!r}."
            )
        if self.feedback not in ("grid", "converter"):
            raise InputError(
                f"CurrentController.feedback must be 'grid' or 'converter', got {self.feedback!r}."
            )
        if (
            self.kind == "pr"
            and self.resonant
            and not (
                isinstance(self.resonant_bandwidth_rad_s, Tensor)
                or float(self.resonant_bandwidth_rad_s) > 0.0
            )
        ):
            raise InputError(
                "CurrentController.resonant_bandwidth_rad_s must be positive: an ideal "
                "resonator has an infinite gain at its centre frequency."
            )
        if not self.sampling_hz > 0:
            raise InputError("CurrentController.sampling_hz must be positive.")

    def gain(self, s: Tensor, f1_hz: float) -> Tensor:
        """``G_c(s)`` for the stationary-frame space vector (or single-phase signal)."""
        rdt = s.real.dtype
        dev = s.device
        g = as_real(self.kp, rdt, dev).to(s.dtype) + torch.zeros_like(s)
        w1 = TWO_PI * f1_hz
        if self.kind == "pi_dq":
            ki = as_real(self.ki, rdt, dev).to(s.dtype)
            return g + ki / (s - 1j * w1)
        wc = as_real(self.resonant_bandwidth_rad_s, rdt, dev).to(s.dtype)
        for order, kr in self.resonant.items():
            kr_t = as_real(kr, rdt, dev).to(s.dtype)
            wh2 = (float(order) * w1) ** 2
            g = g + 2.0 * kr_t * wc * s / (s * s + 2.0 * wc * s + wh2)
        return g

    def delay(self, s: Tensor) -> Tensor:
        """``exp(-delay_samples / f_s * s)``."""
        return torch.exp(-(self.delay_samples / self.sampling_hz) * s)

    def feedforward(self, s: Tensor) -> Tensor:
        """``H_ff(s)``: the feedforward gain behind its optional low-pass."""
        k = as_real(self.feedforward_gain, s.real.dtype, s.device).to(s.dtype)
        if self.feedforward_cutoff_hz is None:
            return k + torch.zeros_like(s)
        return k / (1.0 + s / (TWO_PI * float(self.feedforward_cutoff_hz)))


@dataclass(frozen=True)
class PhaseLockedLoop:
    """Synchronous-reference-frame PLL with a PI loop filter.

    ``dtheta = H_PI(s) / s * V_q`` with ``H_PI = k_p + k_i / s``, closed to
    ``H_PLL(s) = H_PI(s) / (s + H_PI(s) V_1d)`` (Wang et al. 2018, eqs. 15-16). ``V_1d``
    is the steady-state d-axis (peak, amplitude-invariant) voltage at the inner port.
    The gains act on the phase error in volts, so ``k_p`` is in rad/(V s) and ``k_i`` in
    rad/(V s^2).
    """

    kp: Scalar
    ki: Scalar

    def transfer(self, s: Tensor, v1d: Tensor) -> Tensor:
        """``H_PLL(s)`` [rad/V] for the dq-frame Laplace variable ``s``."""
        rdt = s.real.dtype
        h_pi = (
            as_real(self.kp, rdt, s.device).to(s.dtype)
            + as_real(self.ki, rdt, s.device).to(s.dtype) / s
        )
        return h_pi / (s + h_pi * v1d.to(s.dtype))


@dataclass(frozen=True)
class ClosedLoop:
    """Closed-loop transfer functions at the inner port, per frequency.

    All are complex tensors ``[*batch, F]`` evaluated at the requested space-vector
    frequencies (signed for three-phase models). ``i2 = g_ref i_ref + g_dist d - y v -
    y_mirror conj(v(2 f_1 - f))``.

    Attributes
    ----------
    y:
        Direct output admittance (decoupled part), including the passive filter.
    y_passive:
        The admittance with the bridge shorted and no control (``Yo`` of the filter):
        the passive part of ``y``; ``y - y_passive`` is the control's contribution.
    g_ref:
        Current reference to grid current.
    g_dist:
        Bridge voltage disturbance to grid current (inner port shorted).
    y_mirror:
        Coupling coefficient of the mirror-frequency vector from the PLL (zero without
        a PLL); a diagnostic of how large the neglected coupling is.
    loop_gain:
        Open-loop gain of the current loop ``T(s)`` (zero above ``f_s / 2``).
    """

    y: Tensor
    y_passive: Tensor
    g_ref: Tensor
    g_dist: Tensor
    y_mirror: Tensor
    loop_gain: Tensor


def closed_loop(
    filt: LCLFilter,
    ctrl: CurrentController,
    f_hz: Tensor,
    *,
    f1_hz: float,
    pll: Optional[PhaseLockedLoop] = None,
    v1d: Optional[Tensor] = None,
    i1dq: Optional[Tensor] = None,
    vb1dq: Optional[Tensor] = None,
    control_active: Optional[Tensor] = None,
    cdt: torch.dtype = torch.complex128,
) -> ClosedLoop:
    """Evaluate the closed loop of ``filt`` + ``ctrl`` at space-vector frequencies.

    Parameters
    ----------
    filt, ctrl:
        Filter and current controller.
    f_hz:
        Real frequencies ``[F]`` (signed for space vectors).
    f1_hz:
        Fundamental frequency (resonator centres, dq frame translation).
    pll, v1d, i1dq, vb1dq:
        Optional PLL and the steady state it is linearised at: ``v1d`` the d-axis
        inner-port voltage (peak), ``i1dq`` the grid-side current and ``vb1dq`` the
        bridge (modulating) voltage as complex dq values (peak, aligned to the inner
        port voltage), each ``[*batch]``. ``vb1dq`` is used by ``"pi_dq"`` control only.
    control_active:
        Optional bool ``[*batch]``; ``False`` evaluates the passive filter with the
        bridge shorted (no control), used for a unit whose bridge does not switch.
    cdt:
        Complex dtype.

    Returns
    -------
    ClosedLoop
    """
    rdt = f_hz.dtype
    dev = f_hz.device
    s = laplace(f_hz, cdt)  # [F]
    z1, yc, z2 = filt.impedances(s)
    yp, yo = filt.plant(s)

    nyquist = 0.5 * ctrl.sampling_hz
    authority = (f_hz.abs() < nyquist).to(cdt)  # [F]
    gc = ctrl.gain(s, f1_hz) * authority
    gdel = ctrl.delay(s)
    kc = as_real(ctrl.active_damping_kc, rdt, dev).to(cdt) * authority
    hff = ctrl.feedforward(s) * authority

    # PLL perturbations, all decoupled into a direct term (on v) and a mirror term (on
    # the conjugate mirror vector), both evaluated at the dq-frame frequency s - j w_1.
    w1 = TWO_PI * f1_hz
    batch = torch.Size([])
    y_pll = torch.zeros_like(s)
    g_pll = torch.zeros_like(s)
    if pll is not None:
        if v1d is None or i1dq is None:
            raise InputError("closed_loop with a PLL needs v1d and i1dq.")
        v1d = v1d.to(rdt)
        batch = torch.broadcast_shapes(v1d.shape, i1dq.shape)
        h = pll.transfer(s - 1j * w1, v1d.unsqueeze(-1))  # [*batch, F]
        y_pll = h * i1dq.to(cdt).unsqueeze(-1) / 2.0 * authority
        if ctrl.kind == "pi_dq":
            if vb1dq is None:
                raise InputError(
                    "closed_loop with pi_dq control and a PLL needs vb1dq."
                )
            g_pll = h * vb1dq.to(cdt).unsqueeze(-1) / 2.0 * authority
    if control_active is not None:
        active = control_active.to(cdt).unsqueeze(-1)  # [*batch, 1]
        batch = torch.broadcast_shapes(batch, control_active.shape)
        gc = gc * active
        kc = kc * active
        hff = hff * active
        y_pll = y_pll * active
        g_pll = g_pll * active

    shape = (*batch, s.shape[-1])

    def full(x):
        return x.broadcast_to(shape) if x.dim() <= len(shape) else x

    z1b, ycb, z2b = full(z1), full(yc), full(z2)
    gcb, gdelb, kcb = full(gc), full(gdel), full(kc)
    one = torch.ones(shape, dtype=cdt, device=dev)
    zero = torch.zeros(shape, dtype=cdt, device=dev)
    meas_i1 = ctrl.feedback == "converter"
    # Rows: KVL bridge side, KVL grid side, KCL capacitor, control law.
    a = torch.stack(
        [
            torch.stack([-z1b, zero, -one, one], dim=-1),
            torch.stack([zero, -z2b, one, zero], dim=-1),
            torch.stack([one, -one, -ycb, zero], dim=-1),
            torch.stack(
                [
                    gdelb * gcb if meas_i1 else zero,
                    zero if meas_i1 else gdelb * gcb,
                    gdelb * kcb * ycb,
                    one,
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )  # [*batch, F, 4, 4]
    # Right-hand sides: unit v (direct, incl. PLL and feedforward), unit i_ref, unit d,
    # unit mirror vector.
    rhs_v = torch.stack(
        [zero, one, zero, gdelb * (gcb * full(y_pll) + full(hff) + full(g_pll))], dim=-1
    )
    rhs_ref = torch.stack([zero, zero, zero, gdelb * gcb], dim=-1)
    rhs_d = torch.stack([zero, zero, zero, one], dim=-1)
    rhs_m = torch.stack(
        [zero, zero, zero, -gdelb * (gcb * full(y_pll) + full(g_pll))], dim=-1
    )
    rhs = torch.stack([rhs_v, rhs_ref, rhs_d, rhs_m], dim=-1)  # [*batch, F, 4, 4]
    x = torch.linalg.solve(a, rhs)
    i2 = x[..., 1, :]  # [*batch, F, 4]
    y = -i2[..., 0]
    loop = gdel * gc * (yp if not meas_i1 else (1.0 + z2 * yc) * yp)
    y_passive = (yo * torch.ones_like(y)) if y.dim() else yo
    return ClosedLoop(
        y=y,
        y_passive=y_passive,
        g_ref=i2[..., 1],
        g_dist=i2[..., 2],
        y_mirror=-i2[..., 3],
        loop_gain=loop * torch.ones_like(y),
    )


def sequence_admittances(cl_plus: ClosedLoop, cl_minus: ClosedLoop) -> dict:
    """Positive- and negative-sequence quantities from evaluations at ``+f`` and ``-f``.

    A negative-sequence phasor at ``f`` is the conjugate of the space vector at ``-f``,
    so every transfer function ``G`` of the negative sequence is ``conj(G(-j w))``.
    """
    out = {}
    for name in ("y", "y_passive", "g_ref", "g_dist", "loop_gain"):
        out[name] = (getattr(cl_plus, name), torch.conj(getattr(cl_minus, name)))
    return out


def peak(x: Tensor) -> Tensor:
    """RMS phasor to peak space-vector magnitude scale (``sqrt 2``)."""
    return x * math.sqrt(2.0)
