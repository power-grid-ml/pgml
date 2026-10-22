"""Generic grid-following PV inverters designed from their rating, and parameter ranges.

The internal design of a commercial inverter (filter values, controller structure and
gains) is rarely published. :class:`GenericInverterDesign` therefore builds a *generic*
unit from its rating with textbook design rules, so a study can start from a physically
consistent device; it describes no specific product. Every design knob is a field that
accepts a float or a tensor with a scenario batch, so a whole fleet of variants is one
batched model (:mod:`pgml.inverter` evaluates batched parameters in one call).

:data:`GENERIC_PARAMETER_RANGES` lists, per knob, a plausible range and its basis: either a
published value or design rule, or a stated design assumption. Sampling over these ranges
(for example to cover the unknown products of a population) is left to the caller.

Design rules (per connection element, ``U`` the rms phase voltage, ``P_e`` the rated power
per element, ``w_1 = 2 pi f_1``):

- converter-side inductance from the peak-to-peak current ripple ``r`` (a fraction of the
  rated peak current) at the worst modulation index of the topology:
  ``L1 = U_dc / (c f_c dI_pp)`` with ``c = 8`` (unipolar), ``2`` (bipolar), ``4`` (one
  three-phase leg against the DC midpoint), from the local ripple expressions of
  :mod:`pgml.inverter.dead_time`;
- filter capacitor from its reactive power as a fraction ``q`` of the rated active power,
  ``C_f = q P_e / (w_1 U^2)``, ``q <= 0.05`` (Liserre, Blaabjerg and Hansen, IEEE TIA
  41(5), 2005, design constraint a); grid-side inductance ``L2 = l2_ratio L1``
  (their index ``r``);
- damping resistor one third of the capacitor impedance at the resonance (their design
  example, after their reference [26]);
- current controller ``k_p = 2 pi f_cb (L1 + L2)`` for a crossover ``f_cb`` (a fraction of
  the sampling frequency); stationary-frame resonators at orders 1, 3, 5, 7 with gains of
  20 and 5 times ``k_p`` (single-phase) or dq PI with ``k_i = 2 pi (f_cb / 10) k_p``
  (three-phase);
- SRF-PLL for a damping ratio 0.707 and natural frequency ``2 pi f_pll`` of
  ``H_PLL = H_PI / (s + H_PI V_1d)``: ``k_p = 2 zeta w_n / V_1d``, ``k_i = w_n^2 / V_1d``;
- DC-link capacitor for a double-frequency ripple amplitude ``d`` (fraction of ``U_dc``),
  ``C_dc = P / (4 pi f_1 U_dc^2 d)``, and a DC-voltage PI for a bandwidth ``f_dc`` on the
  plant ``n V / (sqrt 2 U_dc C_dc s)`` of Sangwongwanich et al. (2018, eq. 4).

The controller rules (crossover, resonator gains, PI zero, PLL damping, DC-loop bandwidth)
are common design practice, not taken from a specific product; their ranges are marked
as design assumptions in :data:`GENERIC_PARAMETER_RANGES`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pgml.errors import InputError

from ._common import Scalar, as_param
from .control import CurrentController, PhaseLockedLoop
from .dc_link import DCLinkControl, PerturbObserveMPPT
from .filters import LCLFilter, SeriesStage, ShuntStage
from .model import GridFollowingInverter
from .pwm import PWMModulator

_TOPOLOGY_RIPPLE = {
    "single_phase_unipolar": 8.0,
    "single_phase_bipolar": 2.0,
    "three_phase": 4.0,
}

#: Plausible ranges of the design knobs of :class:`GenericInverterDesign`, with their basis.
#: ``scale`` says whether a range is best sampled linearly or logarithmically.
GENERIC_PARAMETER_RANGES: dict = {
    "ripple_fraction": {
        "low": 0.10,
        "high": 0.40,
        "scale": "linear",
        "basis": "design assumption; 0.10 is the design example of Liserre et al. 2005",
    },
    "cf_reactive_fraction": {
        "low": 0.01,
        "high": 0.05,
        "scale": "linear",
        "basis": "Liserre et al. 2005, constraint a): filter reactive power below 5 % of rated power",
    },
    "l2_ratio": {
        "low": 0.1,
        "high": 1.0,
        "scale": "log",
        "basis": "design assumption (Liserre et al. 2005 index r between the inductors)",
    },
    "carrier_hz": {
        "low": 4000.0,
        "high": 20000.0,
        "scale": "log",
        "structural": True,
        "basis": "4-20 kHz listed for PV inverters by Peiris, Elphick and Robinson 2025",
    },
    "dead_time_s": {
        "low": 0.5e-6,
        "high": 3e-6,
        "scale": "log",
        "basis": "design assumption (typical blanking times of IGBT and MOSFET legs)",
    },
    "current_bandwidth_fraction": {
        "low": 0.03,
        "high": 0.10,
        "scale": "log",
        "basis": "design assumption: current-loop crossover as a fraction of the sampling frequency",
    },
    "pll_bandwidth_hz": {
        "low": 5.0,
        "high": 50.0,
        "scale": "log",
        "basis": "design assumption; Wang, Harnefors and Blaabjerg 2018 analyse 20 to 330 Hz",
    },
    "feedforward_gain": {
        "low": 0.0,
        "high": 1.0,
        "scale": "linear",
        "basis": "structural option: grid-voltage feedforward off (0) or full (1)",
    },
    "emc_cx_f": {
        "low": 0.5e-6,
        "high": 10e-6,
        "scale": "log",
        "basis": "0.5-10 uF per 1-3 kW inverter, Enslin and Heskes 2004 as quoted by Cuk et al. 2012",
    },
    "dc_ripple_fraction": {
        "low": 0.01,
        "high": 0.05,
        "scale": "log",
        "basis": "design assumption: double-frequency DC-link ripple amplitude",
    },
    "dc_bandwidth_hz": {
        "low": 2.0,
        "high": 20.0,
        "scale": "log",
        "basis": "design assumption: DC-voltage loop bandwidth",
    },
    "mppt_step_v": {
        "low": 2.0,
        "high": 18.0,
        "scale": "log",
        "basis": "Sangwongwanich et al. 2018 tested 6, 12 and 18 V steps",
    },
    "mppt_rate_hz": {
        "low": 1.0,
        "high": 20.0,
        "scale": "log",
        "structural": True,
        "basis": "typical MPPT sampling rates 1-20 Hz, Sangwongwanich et al. 2018",
    },
}


def current_loop_stability(
    filt: LCLFilter,
    ctrl: CurrentController,
    *,
    f1_hz: float = 50.0,
    n_points: int = 6001,
) -> tuple[Tensor, Tensor]:
    """Internal stability of a unit's current loop on a stiff grid.

    Counts the closed-loop poles in the right half-plane by the Nyquist principle, for the
    capacitor-current damping loop and the current loop around it (both evaluated on a
    contour shifted 2 rad/s into the right half-plane, which steps around the marginal
    poles of integrators and resonators), and returns ``(n_unstable, margin)``: the number of
    unstable poles (0 for a stable loop) and the smallest ``|1 + T|`` of the current loop
    over ``|f| < f_s / 2``. Both carry the batch of batched parameters.
    """
    fs = ctrl.sampling_hz
    f = torch.linspace(-0.499 * fs, 0.499 * fs, n_points, dtype=torch.float64)
    s = torch.complex(torch.full_like(f, 2.0), 2.0 * math.pi * f)
    z1, yc, z2 = filt.impedances(s)
    yp, _ = filt.plant(s)
    gdel = ctrl.delay(s)
    gc = ctrl.gain(s, f1_hz)
    kc = as_param(ctrl.active_damping_kc, s).to(s.dtype)
    i_meas = (1.0 + z2 * yc) * yp if ctrl.feedback == "converter" else yp
    t_inner = gdel * kc * (yc * z2 * yp)
    t_outer = gdel * gc * i_meas / (1.0 + t_inner)

    def turns(z: Tensor) -> Tensor:
        d = torch.diff(torch.angle(z), dim=-1)
        d = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
        return d.sum(dim=-1) / (2.0 * math.pi)

    # Counter-clockwise turns of 1 + T along increasing frequency are P - Z; the inner
    # loop has no unstable open-loop poles, the outer loop's are the inner loop's zeros.
    n_unstable = -(turns(1.0 + t_inner) + turns(1.0 + t_outer))
    margin = (1.0 + t_outer).abs().amin(dim=-1)
    return torch.round(n_unstable), margin


#: Candidate capacitor-current damping gains, as multiples of ``w_res L1``.
_DAMPING_FACTORS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class GenericInverterDesign:
    """A generic grid-following PV inverter designed from its rating.

    Attributes
    ----------
    p_rated_w:
        Rated active power of the unit (all phases).
    phases:
        1 (single-phase, unipolar PWM, PR control) or 3 (three-phase, two-level PWM, dq PI).
    u_ln_v, f1_hz:
        Rated phase-to-neutral voltage and fundamental frequency.
    u_dc_v:
        Nominal DC-link voltage (``None``: 400 V single-phase, 700 V three-phase).
    carrier_hz, sampling_hz:
        Carrier frequency and controller sampling frequency (``None``: twice the carrier),
        scalars (they fix the frequency grid).
    mppt_rate_hz:
        MPPT sampling rate (scalar).
    ripple_fraction, cf_reactive_fraction, l2_ratio, dead_time_s, current_bandwidth_fraction,
    pll_bandwidth_hz, feedforward_gain, emc_cx_f, dc_ripple_fraction, dc_bandwidth_hz,
    mppt_step_v:
        Design knobs (floats or ``[*batch]`` tensors), see the module docstring and
        :data:`GENERIC_PARAMETER_RANGES`.
    """

    p_rated_w: Scalar
    phases: int = 1
    u_ln_v: float = 230.0
    f1_hz: float = 50.0
    u_dc_v: Optional[Scalar] = None
    carrier_hz: float = 16000.0
    sampling_hz: Optional[float] = None
    mppt_rate_hz: float = 5.0
    ripple_fraction: Scalar = 0.2
    cf_reactive_fraction: Scalar = 0.03
    l2_ratio: Scalar = 0.3
    dead_time_s: Scalar = 1e-6
    current_bandwidth_fraction: Scalar = 0.05
    pll_bandwidth_hz: Scalar = 20.0
    feedforward_gain: Scalar = 1.0
    emc_cx_f: Scalar = 1e-6
    dc_ripple_fraction: Scalar = 0.02
    dc_bandwidth_hz: Scalar = 10.0
    mppt_step_v: Scalar = 4.0

    def __post_init__(self):
        if self.phases not in (1, 3):
            raise InputError("GenericInverterDesign.phases must be 1 or 3.")

    def _t(self, x: Scalar) -> Tensor:
        return (
            x.to(torch.float64)
            if isinstance(x, Tensor)
            else torch.tensor(float(x), dtype=torch.float64)
        )

    @property
    def topology(self) -> str:
        return "single_phase_unipolar" if self.phases == 1 else "three_phase"

    def nominal_u_dc(self) -> Tensor:
        """The nominal DC-link voltage."""
        if self.u_dc_v is not None:
            return self._t(self.u_dc_v)
        return self._t(400.0 if self.phases == 1 else 700.0)

    def filter_values(self) -> dict:
        """``{l1_h, c_f, l2_h, r_d_ohm, f_res_hz}`` from the design rules."""
        n = self.phases
        p_e = self._t(self.p_rated_w) / n
        w1 = 2.0 * math.pi * self.f1_hz
        u = self.u_ln_v
        i_peak = math.sqrt(2.0) * p_e / u
        di_pp = self._t(self.ripple_fraction) * i_peak
        l1 = self.nominal_u_dc() / (
            _TOPOLOGY_RIPPLE[self.topology] * self.carrier_hz * di_pp
        )
        c_f = self._t(self.cf_reactive_fraction) * p_e / (w1 * u * u)
        l2 = self._t(self.l2_ratio) * l1
        w_res = torch.sqrt((l1 + l2) / (l1 * l2 * c_f))
        r_d = 1.0 / (3.0 * w_res * c_f)
        return {
            "l1_h": l1,
            "c_f": c_f,
            "l2_h": l2,
            "r_d_ohm": r_d,
            "f_res_hz": w_res / (2 * math.pi),
        }

    def _filter(self, fv: dict) -> LCLFilter:
        w1 = 2.0 * math.pi * self.f1_hz
        # Winding resistances for an X/R of 10 at the fundamental (design assumption).
        return LCLFilter(
            l1_h=fv["l1_h"],
            c_f=fv["c_f"],
            l2_h=fv["l2_h"],
            r1_ohm=w1 * fv["l1_h"] / 10.0,
            r_d_ohm=fv["r_d_ohm"],
            r2_ohm=w1 * fv["l2_h"] / 10.0,
        )

    def _tune_damping(self, control: CurrentController, fv: dict):
        """Capacitor-current damping gain: among multiples of ``w_res L1``, the stable one
        with the largest current-loop margin (per design of a batch).

        With the digital delay, grid-current control of an LCL filter is stable without
        active damping only for a resonance above about a sixth of the sampling frequency,
        and capacitor-current damping turns into a negative resistance above it, so no
        single gain fits every design; a manufacturer tunes it per product. The choice is
        a discrete selection (off the gradient); the gain itself stays differentiable.
        """
        filt = self._filter(fv)
        base = 2.0 * math.pi * fv["f_res_hz"] * fv["l1_h"]
        best_margin, best_kc, stable_any = None, None, None
        with torch.no_grad():
            for factor in _DAMPING_FACTORS:
                trial = dataclasses.replace(control, active_damping_kc=base * factor)
                n_bad, margin = current_loop_stability(filt, trial, f1_hz=self.f1_hz)
                ok = n_bad == 0
                score = torch.where(ok, margin, torch.full_like(margin, -1.0))
                if best_margin is None:
                    best_margin = score
                    best_kc = torch.full_like(margin, factor)
                    stable_any = ok
                else:
                    better = score > best_margin
                    best_margin = torch.where(better, score, best_margin)
                    best_kc = torch.where(
                        better, torch.full_like(margin, factor), best_kc
                    )
                    stable_any = stable_any | ok
        factor = best_kc if best_kc.dim() else float(best_kc)
        tuned = dataclasses.replace(control, active_damping_kc=base * factor)
        return tuned, stable_any, best_margin

    def report(self) -> dict:
        """Design summary: ``{"stable", "margin", "f_res_hz", "f_res_over_fs",
        "active_damping_kc", "l1_h", "c_f", "l2_h", "r_d_ohm"}``.

        ``stable`` is ``False`` where no tried damping gain gives a stable current loop on a
        stiff grid: such a design (a corner of the parameter ranges) is not a product and
        should be left out of a population; ``margin`` is the current loop's smallest
        ``|1 + T|``.
        """
        model = self.build()
        fv = self.filter_values()
        n_bad, margin = current_loop_stability(
            model.filter, model.control, f1_hz=self.f1_hz
        )
        return {
            "stable": n_bad == 0,
            "margin": margin,
            "f_res_hz": fv["f_res_hz"],
            "f_res_over_fs": fv["f_res_hz"] / model.control.sampling_hz,
            "active_damping_kc": model.control.active_damping_kc,
            **{k: fv[k] for k in ("l1_h", "c_f", "l2_h", "r_d_ohm")},
        }

    def build(self) -> GridFollowingInverter:
        """The :class:`~pgml.inverter.GridFollowingInverter` of this design."""
        fv = self.filter_values()
        l1, l2 = fv["l1_h"], fv["l2_h"]
        fs = self.sampling_hz if self.sampling_hz is not None else 2.0 * self.carrier_hz
        kp = 2.0 * math.pi * self._t(self.current_bandwidth_fraction) * fs * (l1 + l2)
        if self.phases == 1:
            control = CurrentController(
                "pr",
                kp=kp,
                resonant={1: 20.0 * kp, 3: 5.0 * kp, 5: 5.0 * kp, 7: 5.0 * kp},
                resonant_bandwidth_rad_s=5.0,
                sampling_hz=fs,
                feedforward_gain=self.feedforward_gain,
                feedforward_cutoff_hz=1000.0,
            )
        else:
            ki = (
                2.0
                * math.pi
                * (self._t(self.current_bandwidth_fraction) * fs / 10.0)
                * kp
            )
            control = CurrentController(
                "pi_dq",
                kp=kp,
                ki=ki,
                sampling_hz=fs,
                feedforward_gain=self.feedforward_gain,
                feedforward_cutoff_hz=1000.0,
            )
        control, _, _ = self._tune_damping(control, fv)
        v1d = math.sqrt(2.0) * self.u_ln_v
        wn = 2.0 * math.pi * self._t(self.pll_bandwidth_hz)
        pll = PhaseLockedLoop(kp=2.0 * 0.707 * wn / v1d, ki=wn * wn / v1d)
        u_dc = self.nominal_u_dc()
        p = self._t(self.p_rated_w)
        c_dc = p / (
            4.0 * math.pi * self.f1_hz * u_dc * u_dc * self._t(self.dc_ripple_fraction)
        )
        k_plant = self.phases * self.u_ln_v / (math.sqrt(2.0) * u_dc * c_dc)
        kp_dc = 2.0 * math.pi * self._t(self.dc_bandwidth_hz) / k_plant
        ki_dc = kp_dc * 2.0 * math.pi * self._t(self.dc_bandwidth_hz) / 5.0
        return GridFollowingInverter(
            filter=self._filter(fv),
            control=control,
            pwm=PWMModulator(
                self.topology,
                carrier_hz=self.carrier_hz,
                carrier_groups=4,
                sidebands=12,
            ),
            pll=pll,
            emc=(
                SeriesStage(l_h=20e-6, r_ohm=0.01),
                ShuntStage(c_f=self.emc_cx_f, c_ground_f=4.7e-9),
            ),
            dead_time_s=self.dead_time_s,
            dc_link=DCLinkControl(c_dc_f=c_dc, kp=kp_dc, ki=ki_dc, sampling_hz=fs),
            mppt=PerturbObserveMPPT(
                v_step_v=self.mppt_step_v, rate_hz=self.mppt_rate_hz
            ),
        )


def generic_single_phase_inverter(
    p_rated_w: Scalar = 4600.0, **knobs
) -> GridFollowingInverter:
    """A generic single-phase PV inverter of the given rating (see :class:`GenericInverterDesign`)."""
    return GenericInverterDesign(p_rated_w=p_rated_w, phases=1, **knobs).build()


def generic_three_phase_inverter(
    p_rated_w: Scalar = 10000.0, **knobs
) -> GridFollowingInverter:
    """A generic three-phase PV inverter of the given rating (see :class:`GenericInverterDesign`)."""
    knobs.setdefault("carrier_hz", 10000.0)
    return GenericInverterDesign(p_rated_w=p_rated_w, phases=3, **knobs).build()
