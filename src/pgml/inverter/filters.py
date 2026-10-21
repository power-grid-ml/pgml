"""Passive output filters of an inverter: the LCL stage and the EMC stages.

Conventions
-----------
All values are per connection element (one WYE phase of a three-phase unit, or the
single L-N element of a single-phase unit), in SI units. Inductors and capacitors store
``L`` and ``C``; impedances are formed at the Laplace variable ``s``.

The LCL stage sits between the PWM bridge (the inverter side) and the inner port where
the current controller measures the grid current and the voltage. EMC stages sit between
that inner port and the device terminals, listed from the inner port outward, and are
outside the control loop.

A Norton equivalent ``i = I_N - Y v`` (current ``i`` leaving the device into the grid,
terminal voltage ``v``) is carried through an outer stage as follows. A shunt
admittance ``y`` at the outer node leaves ``I_N`` unchanged and adds ``y`` to ``Y``.
A series impedance ``z`` gives ``Y' = Y / (1 + z Y)`` and ``I_N' = I_N / (1 + z Y)``,
which follows from the Thevenin form ``v_open = I_N / Y`` behind ``1/Y + z``; both
expressions stay finite for ``Y = 0`` (a disconnected inner port).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

from torch import Tensor

from pgml.errors import InputError

from ._common import Scalar, as_real, capacitor_branch_admittance, series_admittance


@dataclass(frozen=True)
class LCLFilter:
    """The LCL output filter between the PWM bridge and the inner port.

    Attributes
    ----------
    l1_h, r1_ohm:
        Bridge-side inductor and its series resistance.
    c_f, r_d_ohm:
        Filter capacitor and its series (passive damping) resistance. ``c_f = 0`` gives
        an L filter.
    l2_h, r2_ohm:
        Grid-side inductor and its series resistance. ``l2_h = r2_ohm = 0`` places the
        capacitor directly at the inner port (an LC filter).
    """

    l1_h: Scalar
    c_f: Scalar = 0.0
    l2_h: Scalar = 0.0
    r1_ohm: Scalar = 0.0
    r_d_ohm: Scalar = 0.0
    r2_ohm: Scalar = 0.0

    def impedances(self, s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(Z1, Yc, Z2)`` at ``s``: bridge-side impedance, capacitor-branch admittance,
        grid-side impedance."""
        rdt = s.real.dtype
        dev = s.device
        z1 = as_real(self.r1_ohm, rdt, dev).to(s.dtype) + s * as_real(
            self.l1_h, rdt, dev
        ).to(s.dtype)
        yc = capacitor_branch_admittance(
            as_real(self.c_f, rdt, dev), as_real(self.r_d_ohm, rdt, dev), s
        )
        z2 = as_real(self.r2_ohm, rdt, dev).to(s.dtype) + s * as_real(
            self.l2_h, rdt, dev
        ).to(s.dtype)
        return z1, yc, z2

    def plant(self, s: Tensor) -> tuple[Tensor, Tensor]:
        """Open-loop transfer functions of the filter at ``s``.

        Returns ``(Yp, Yo)`` with ``i2 = Yp v_b - Yo v``: ``Yp`` maps the bridge voltage
        to the grid-side current with the inner port shorted, ``Yo`` is the admittance
        seen from the inner port with the bridge shorted.
        """
        z1, yc, z2 = self.impedances(s)
        # i2 = (v_b - (1 + Z1 Yc) v) / (Z1 + Z2 + Z1 Yc Z2), from the node equation at
        # the capacitor: (v_b - v_c)/Z1 = Yc v_c + (v_c - v)/Z2.
        den = z1 + z2 + z1 * yc * z2
        return 1.0 / den, (1.0 + z1 * yc) / den


@dataclass(frozen=True)
class SeriesStage:
    """A series inductor (differential-mode choke) of an EMC filter, per element."""

    l_h: Scalar
    r_ohm: Scalar = 0.0

    def __post_init__(self):
        if not isinstance(self.l_h, Tensor) and not isinstance(self.r_ohm, Tensor):
            if float(self.l_h) == 0.0 and float(self.r_ohm) == 0.0:
                raise InputError(
                    "SeriesStage needs a nonzero inductance or resistance."
                )

    def impedance(self, s: Tensor) -> Tensor:
        rdt = s.real.dtype
        return 1.0 / series_admittance(
            as_real(self.r_ohm, rdt, s.device), as_real(self.l_h, rdt, s.device), s
        )


@dataclass(frozen=True)
class ShuntStage:
    """A shunt capacitor stage of an EMC filter, per element.

    Attributes
    ----------
    c_f:
        Differential capacitance per element (an X capacitor as its star equivalent),
        which carries no zero-sequence current in a three-phase unit.
    c_ground_f:
        Capacitance from each element to ground (Y capacitors), which acts in every
        sequence. For a single-phase element both add.
    r_ohm:
        Series resistance of the capacitor branches.
    """

    c_f: Scalar = 0.0
    c_ground_f: Scalar = 0.0
    r_ohm: Scalar = 0.0

    def admittance(self, s: Tensor, zero_sequence: bool = False) -> Tensor:
        rdt = s.real.dtype
        dev = s.device
        r = as_real(self.r_ohm, rdt, dev)
        y_ground = capacitor_branch_admittance(as_real(self.c_ground_f, rdt, dev), r, s)
        if zero_sequence:
            return y_ground
        return y_ground + capacitor_branch_admittance(as_real(self.c_f, rdt, dev), r, s)


Stage = Union[SeriesStage, ShuntStage]


def propagate_norton(
    i_n: Tensor,
    y: Tensor,
    stages: Sequence[Stage],
    s: Tensor,
    zero_sequence: bool = False,
) -> tuple[Tensor, Tensor]:
    """Carry a Norton equivalent ``(I_N, Y)`` through ``stages`` (inner to outer)."""
    for stage in stages:
        if isinstance(stage, ShuntStage):
            y = y + stage.admittance(s, zero_sequence=zero_sequence)
        else:
            z = stage.impedance(s)
            k = 1.0 / (1.0 + z * y)
            y = y * k
            i_n = i_n * k
    return i_n, y


def inner_port_state(
    v_t: Tensor, i_t: Tensor, stages: Sequence[Stage], s: Tensor
) -> tuple[Tensor, Tensor]:
    """Voltage and current at the inner port from the terminal values.

    ``i_t`` leaves the device at the terminals. The stages are walked from the terminal
    inward, so ``v_t`` and ``i_t`` are mapped back through each one.
    """
    v, i = v_t, i_t
    for stage in reversed(list(stages)):
        if isinstance(stage, ShuntStage):
            i = i + stage.admittance(s) * v
        else:
            v = v + stage.impedance(s) * i
    return v, i
