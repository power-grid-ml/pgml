"""Grid-following inverter as a frequency-domain Norton equivalent.

At every frequency the unit is ``i = I_N - Y v`` at its terminals (``i`` leaving the
device, per connection element). Both terms are built from the physical blocks of the
unit and depend on its operating state:

- ``Y`` is the closed-loop output admittance of the current-controlled bridge behind
  its LCL filter (:func:`~pgml.inverter.control.closed_loop`), carried through the EMC
  stages. Its passive part (the filter with the bridge shorted) is reported separately,
  so the contributions of the control and of the filter can be compared.
- ``I_N`` collects the primary emission mechanisms, each mapped to the terminals by the
  closed loop: the PWM residues of the bridge voltage
  (:class:`~pgml.inverter.pwm.PWMModulator`), the dead-time voltage error
  (:func:`~pgml.inverter.dead_time.dead_time_voltage`), the MPPT interharmonics and the
  double-frequency DC-link ripple (:mod:`pgml.inverter.dc_link`).

Locked and unlocked sources
---------------------------
Dead-time harmonics and the ripple harmonic follow the device's own fundamental current,
so they carry a phase in the grid's reference (``source_locked``). PWM residues follow
the carrier and MPPT lines follow the MPPT clock, neither synchronised between units
(``source_unlocked``): their phase is a free parameter, and their sum over many units is
a random-phase sum.

Three-phase units are modelled in symmetrical components: the closed loop is evaluated
for the positive-sequence space vector at ``+f`` and the negative sequence at ``-f``; the
zero sequence of a three-wire unit has no inner path, only the ground capacitance of
the EMC stages. The element-domain admittance is ``A diag(Y0, Y+, Y-) A^-1``. A
single-phase unit takes the mean of the two space-vector evaluations, which equals the
real-coefficient transfer function whenever the model has no frequency coupling (no PLL,
stationary-frame control).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor

from pgml.errors import InputError, ModelingError

from ._common import (
    element_to_sequence,
    laplace,
    sequence_to_element,
    sequence_to_vector,
)
from .control import ClosedLoop, CurrentController, PhaseLockedLoop, closed_loop
from .dc_link import (
    DCLinkControl,
    PerturbObserveMPPT,
    mppt_current_lines,
    ripple_third_harmonic,
)
from .dead_time import dead_time_voltage
from .filters import LCLFilter, inner_port_state, propagate_norton
from .operating import InverterState
from .pwm import PWMModulator

Mechanism = Literal["pwm", "dead_time", "mppt", "dc_ripple"]
MECHANISMS: tuple[str, ...] = ("pwm", "dead_time", "mppt", "dc_ripple")
_LOCKED = {"dead_time", "dc_ripple"}


@dataclass(frozen=True)
class NortonEquivalent:
    """Terminal Norton equivalent of a device over a frequency grid.

    Attributes
    ----------
    frequencies_hz:
        Real ``[F]``.
    admittance:
        Complex ``[*batch, F, E, E]`` element-domain admittance ``Y``.
    admittance_passive:
        The same with the bridge shorted and no control (filter and EMC stages only).
    source_locked, source_unlocked:
        Complex ``[*batch, F, E]`` Norton currents, locked to the fundamental or not.
    components:
        ``{mechanism: [*batch, F, E]}`` each mechanism's Norton current.
    y_mirror:
        Complex ``[*batch, F]``: the positive-sequence mirror-frequency coupling
        coefficient of the PLL at the inner port (zero without a PLL), a diagnostic of
        the coupling the decoupled model neglects.
    loop_gain:
        Complex ``[*batch, F]``: the current loop's open-loop gain (positive sequence).
    """

    frequencies_hz: Tensor
    admittance: Tensor
    admittance_passive: Tensor
    source_locked: Tensor
    source_unlocked: Tensor
    components: dict = field(default_factory=dict)
    y_mirror: Optional[Tensor] = None
    loop_gain: Optional[Tensor] = None

    @property
    def source(self) -> Tensor:
        """``source_locked + source_unlocked`` (the coherent sum at the given phases)."""
        return self.source_locked + self.source_unlocked


def _place_lines(
    values: Tensor, lines_hz: Sequence[float], grid_hz: Sequence[float]
) -> Tensor:
    """Sum line values ``[*batch, L, E]`` onto the frequency grid ``[*batch, F, E]``.

    Lines above the grid's highest frequency are outside the study and dropped; a line
    inside the grid's range that the grid does not contain raises (matching to 1e-6 Hz).
    """
    lookup = {}
    for k, f in enumerate(grid_hz):
        lookup.setdefault(round(float(f), 6), k)
    f_top = max(grid_hz) + 1e-6
    keep, positions = [], []
    for j, f in enumerate(lines_hz):
        if f > f_top:
            continue
        key = round(float(f), 6)
        if key not in lookup:
            raise InputError(
                f"emission line at {f} Hz is not in the frequency grid; build the grid "
                "with emission_frequencies() or include the line."
            )
        keep.append(j)
        positions.append(lookup[key])
    out = torch.zeros(
        (*values.shape[:-2], len(grid_hz), values.shape[-1]),
        dtype=values.dtype,
        device=values.device,
    )
    if not keep:
        return out
    sel = torch.as_tensor(keep, dtype=torch.int64, device=values.device)
    idx = torch.as_tensor(positions, dtype=torch.int64, device=values.device)
    return out.index_add(-2, idx, values.index_select(-2, sel))


@dataclass(frozen=True)
class GridFollowingInverter:
    """A current-controlled grid-following inverter with an LCL and EMC filter.

    Attributes
    ----------
    filter:
        The LCL stage between bridge and inner port.
    control:
        The current controller.
    pwm:
        The modulator; its topology fixes the number of elements (1 or 3).
    pll:
        Optional PLL; ``None`` neglects the synchronisation dynamics.
    emc:
        EMC stages from the inner port to the terminals.
    emc_on_grid_side:
        Whether the EMC stages stay on the grid when the relay is open (standby, night).
    dead_time_s:
        Blanking time per commutation (0 disables the mechanism).
    dead_time_ripple:
        ``None`` (hard sign) or ``"local"`` (the ripple dead zone of
        :func:`~pgml.inverter.dead_time.dead_time_voltage`).
    dc_link, mppt:
        Optional DC-link control and P&O MPPT (needed for the ``"mppt"`` and
        ``"dc_ripple"`` mechanisms).
    harmonic_orders:
        Integer orders at which the locked mechanisms are evaluated.
    mechanisms:
        Which primary emission mechanisms contribute to ``I_N``.
    """

    filter: LCLFilter
    control: CurrentController
    pwm: PWMModulator
    pll: Optional[PhaseLockedLoop] = None
    emc: tuple = ()
    emc_on_grid_side: bool = True
    dead_time_s: float = 0.0
    dead_time_ripple: Optional[str] = None
    dc_link: Optional[DCLinkControl] = None
    mppt: Optional[PerturbObserveMPPT] = None
    harmonic_orders: tuple = tuple(range(2, 51))
    mechanisms: tuple = MECHANISMS

    def __post_init__(self):
        unknown = set(self.mechanisms) - set(MECHANISMS)
        if unknown:
            raise InputError(f"unknown emission mechanisms {sorted(unknown)}.")
        if self.control.kind == "pi_dq" and self.n_elements != 3:
            raise InputError("dq-frame PI current control needs a three-phase unit.")
        for stage in self.emc:
            if not hasattr(stage, "admittance") and not hasattr(stage, "impedance"):
                raise InputError(f"unsupported EMC stage {stage!r}.")

    @property
    def n_elements(self) -> int:
        return self.pwm.n_elements

    # ---- frequency grid --------------------------------------------------------
    def emission_frequencies(
        self, f1_hz: float, f_max_hz: Optional[float] = None
    ) -> list[float]:
        """Every frequency at which this unit's sources have a line, ascending."""
        lines: list[float] = []
        if "dead_time" in self.mechanisms or "dc_ripple" in self.mechanisms:
            lines += [h * f1_hz for h in self.harmonic_orders]
        if "pwm" in self.mechanisms:
            lines += self.pwm.line_frequencies(f1_hz)
        if "mppt" in self.mechanisms and self.mppt is not None:
            lines += self.mppt.line_frequencies(f1_hz)
        uniq = sorted({round(f, 6) for f in lines if f > 0})
        if f_max_hz is not None:
            uniq = [f for f in uniq if f <= f_max_hz]
        return uniq

    # ---- evaluation ------------------------------------------------------------
    def _loop(
        self, f: Tensor, f1_hz: float, ss: dict, active: Tensor, cdt
    ) -> ClosedLoop:
        return closed_loop(
            self.filter,
            self.control,
            f,
            f1_hz=f1_hz,
            pll=self.pll,
            v1d=ss.get("v1d"),
            i1dq=ss.get("i1dq"),
            vb1dq=ss.get("vb1dq"),
            control_active=active,
            cdt=cdt,
        )

    def norton(
        self,
        frequencies_hz: Tensor,
        *,
        f1_hz: float,
        v1: Tensor,
        i1: Tensor,
        state: InverterState,
    ) -> NortonEquivalent:
        """Evaluate the Norton equivalent on a frequency grid.

        Parameters
        ----------
        frequencies_hz:
            Real ``[F]`` grid (positive); every source line must be on it
            (:meth:`emission_frequencies`).
        f1_hz:
            Fundamental frequency.
        v1, i1:
            Fundamental terminal voltage and current leaving the device, RMS phasors
            ``[*batch, E]`` in the grid's phase reference (from the solved power flow).
        state:
            The unit's :class:`~pgml.inverter.operating.InverterState`.
        """
        cdt = v1.dtype
        rdt = frequencies_hz.dtype
        dev = frequencies_hz.device
        e = self.n_elements
        if v1.shape[-1] != e or i1.shape[-1] != e:
            raise InputError(
                f"inverter with {e} element(s) got fundamental values of shape "
                f"{tuple(v1.shape)} / {tuple(i1.shape)}."
            )
        f = frequencies_hz
        n_f = f.shape[0]
        grid_list = [float(x) for x in f.detach().cpu().tolist()]
        u_dc = (
            torch.as_tensor(state.u_dc_v, dtype=rdt, device=dev)
            if not isinstance(state.u_dc_v, Tensor)
            else state.u_dc_v.to(dtype=rdt, device=dev)
        )
        bridge_on, connected = state.masks(rdt, dev)
        # A unit whose bridge does not switch may report any DC-link voltage (0 at
        # night); its sources are masked, so a placeholder keeps them finite.
        u_dc = torch.where(bridge_on & (u_dc > 0), u_dc, torch.ones_like(u_dc))

        # Fundamental state at the inner port and the bridge (positive sequence for
        # a three-phase unit).
        s1 = laplace(torch.as_tensor([f1_hz], dtype=rdt, device=dev), cdt)[0]
        v_p, i_p = inner_port_state(v1, i1, self.emc, s1)
        z1, yc, z2 = self.filter.impedances(s1)
        v_c = v_p + z2 * i_p
        i_b = i_p + yc * v_c  # bridge-side current
        v_b = v_c + z1 * i_b
        if e == 3:
            ref_v = element_to_sequence(v_p)[..., 1]
            ref_i = element_to_sequence(i_p)[..., 1]
            ref_b = element_to_sequence(v_b)[..., 1]
        else:
            ref_v, ref_i, ref_b = v_p[..., 0], i_p[..., 0], v_b[..., 0]
        rot = torch.polar(torch.ones_like(ref_v.real), -torch.angle(ref_v))
        ss = {
            "v1d": math.sqrt(2.0) * torch.abs(ref_v),
            "i1dq": math.sqrt(2.0) * ref_i * rot,
            "vb1dq": math.sqrt(2.0) * ref_b * rot,
        }

        cl_p = self._loop(f, f1_hz, ss, bridge_on, cdt)
        cl_m = self._loop(-f, f1_hz, ss, bridge_on, cdt)
        y_plus, y_minus = cl_p.y, torch.conj(cl_m.y)
        yp_plus, yp_minus = cl_p.y_passive, torch.conj(cl_m.y_passive)
        g_plus, g_minus = cl_p.g_dist, torch.conj(cl_m.g_dist)
        # A unit with its relay open has no inner path at all.
        conn = connected.to(cdt).unsqueeze(-1)
        y_plus, y_minus = y_plus * conn, y_minus * conn
        yp_plus, yp_minus = yp_plus * conn, yp_minus * conn

        batch = torch.broadcast_shapes(y_plus.shape[:-1], v1.shape[:-1], u_dc.shape)

        # ---- primary sources at the inner port: bridge disturbances and reference lines
        i_ref: dict[str, Tensor] = {}
        parts: dict[str, Tensor] = {}
        on_mask = bridge_on.unsqueeze(-1).unsqueeze(-1)  # [*batch, 1, 1]

        def masked(x: Tensor) -> Tensor:
            return torch.where(on_mask, x, torch.zeros_like(x))

        peak_b = (
            math.sqrt(2.0) * torch.abs(v_b if e == 1 else ref_b.unsqueeze(-1))[..., 0]
        )
        mi = self.pwm.modulation_index(peak_b, u_dc)
        with torch.no_grad():
            over = bridge_on & (mi > 1.0)
            if bool(over.any()):
                raise ModelingError(
                    "inverter bridge overmodulated (modulation index "
                    f"{float(mi[over].max()):.3f} > 1): the DC-link voltage is below "
                    "the peak bridge voltage the operating point needs. The sine-"
                    "triangle spectrum does not hold there; raise u_dc_v or model the "
                    "state as not switching."
                )
        # Scenarios whose bridge does not switch are masked below; a zero index keeps
        # their (unused) spectra finite and in range.
        mi = torch.where(bridge_on, mi, torch.zeros_like(mi))
        if "pwm" in self.mechanisms:
            lines_hz, lines = self.pwm.bridge_spectrum(
                mi, u_dc, f1_hz, torch.angle(v_b[..., 0]), state.carrier_angle_rad
            )
            parts["pwm"] = masked(
                _place_lines(lines.to(cdt), lines_hz.tolist(), grid_list)
            )
        if "dead_time" in self.mechanisms and self.dead_time_s > 0:
            d = dead_time_voltage(
                i_b,
                u_dc,
                self.dead_time_s,
                self.pwm.carrier_hz,
                self.harmonic_orders,
                full_bridge=e == 1,
                three_phase=e == 3,
                ripple=self.dead_time_ripple,
                l1_h=float(self.filter.l1_h),
                modulation_index=mi,
                bridge_angle=torch.angle(v_b),
                topology=self.pwm.topology,
            ).to(cdt)
            parts["dead_time"] = masked(
                _place_lines(d, [h * f1_hz for h in self.harmonic_orders], grid_list)
            )
        if "mppt" in self.mechanisms and self.mppt is not None:
            if self.dc_link is None:
                raise InputError(
                    "the 'mppt' mechanism needs dc_link control parameters."
                )
            v_rms = torch.abs(ref_v)
            m_hz, m_lines = mppt_current_lines(
                self.mppt,
                self.dc_link,
                f1_hz=f1_hz,
                v_g=v_rms,
                u_dc=u_dc,
                current_angle=torch.angle(i_p[..., 0]),
                n_phases=e,
                cdt=cdt,
            )
            i_ref["mppt"] = masked(_place_lines(m_lines, m_hz, grid_list))
        if "dc_ripple" in self.mechanisms and self.dc_link is not None and e == 1:
            i3 = ripple_third_harmonic(
                self.dc_link,
                f1_hz=f1_hz,
                v_t=v_p[..., 0],
                i_t=i_p[..., 0],
                u_dc=u_dc,
                cdt=cdt,
            )
            i_ref["dc_ripple"] = masked(
                _place_lines(i3.unsqueeze(-1).unsqueeze(-1), [3.0 * f1_hz], grid_list)
            )

        def to_terminal_current(x: Tensor, bridge_voltage: bool) -> Tensor:
            """Inner-port Norton current of one mechanism, carried to the terminals."""
            if e == 1:
                if bridge_voltage:
                    x = x * (0.5 * (g_plus + g_minus)).unsqueeze(-1)
                i_n, _ = propagate_norton(
                    x,
                    (0.5 * (y_plus + y_minus)).unsqueeze(-1),
                    self.emc,
                    laplace(f, cdt).unsqueeze(-1),
                )
                return i_n
            seq = element_to_sequence(x)  # [*batch, F, 3]
            zero = torch.zeros_like(y_plus)
            if bridge_voltage:
                gain = torch.stack([zero, g_plus, g_minus], dim=-1)
                seq = seq * gain
            y_seq = torch.stack([zero, y_plus, y_minus], dim=-1)
            s_col = laplace(f, cdt).unsqueeze(-1)
            i0, _ = propagate_norton(
                seq[..., :1], y_seq[..., :1], self.emc, s_col, True
            )
            i12, _ = propagate_norton(seq[..., 1:], y_seq[..., 1:], self.emc, s_col)
            return sequence_to_vector(torch.cat([i0, i12], dim=-1))

        components = {}
        for name, x in parts.items():
            components[name] = to_terminal_current(x, True)
        for name, x in i_ref.items():
            components[name] = to_terminal_current(x, False)

        # ---- terminal admittance (inner closed loop through the EMC stages)
        s_col = laplace(f, cdt)
        emc_conn = torch.ones_like(conn) if self.emc_on_grid_side else conn

        def terminal_admittance(yp_: Tensor, ym_: Tensor) -> Tensor:
            if e == 1:
                _, y = propagate_norton(
                    torch.zeros_like(yp_), 0.5 * (yp_ + ym_), self.emc, s_col
                )
                # With the relay open only the grid-side EMC stages remain.
                return (y * emc_conn).unsqueeze(-1).unsqueeze(-1)
            zero = torch.zeros_like(yp_)
            _, y0 = propagate_norton(zero, zero, self.emc, s_col, True)
            _, y1 = propagate_norton(zero, yp_, self.emc, s_col)
            _, y2 = propagate_norton(zero, ym_, self.emc, s_col)
            diag = torch.stack([y0 * emc_conn, y1 * emc_conn, y2 * emc_conn], dim=-1)
            return sequence_to_element(diag)

        admittance = terminal_admittance(y_plus, y_minus)
        admittance_passive = terminal_admittance(yp_plus, yp_minus)
        target = (*batch, n_f, e)
        locked = torch.zeros(target, dtype=cdt, device=dev)
        unlocked = torch.zeros(target, dtype=cdt, device=dev)
        conn_mask = connected.unsqueeze(-1).unsqueeze(-1)
        for name, x in components.items():
            x = torch.where(conn_mask, x, torch.zeros_like(x))
            components[name] = x
            if name in _LOCKED:
                locked = locked + x
            else:
                unlocked = unlocked + x
        return NortonEquivalent(
            frequencies_hz=f,
            admittance=admittance.broadcast_to((*batch, n_f, e, e)),
            admittance_passive=admittance_passive.broadcast_to((*batch, n_f, e, e)),
            source_locked=locked,
            source_unlocked=unlocked,
            components=components,
            y_mirror=cl_p.y_mirror,
            loop_gain=cl_p.loop_gain,
        )
