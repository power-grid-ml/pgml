"""Harmonic emission models of grid-following inverters.

A unit is described by its physical blocks (LCL and EMC filter, current control, PLL,
PWM modulator, dead time, DC-link control and MPPT) and its operating state, and
evaluates to a frequency-domain Norton equivalent ``i = I_N - Y v`` at its terminals
(:meth:`GridFollowingInverter.norton`). The harmonic network solve with these
equivalents is :func:`pgml.studies.emission_flow`.

The models are per-solve inputs, not grid data: they are keyed by device id and leave
the :class:`~pgml.schemas.grid_schema.Grid` contract unchanged. Every function is
written in torch and differentiable in its tensor-valued parameters.
"""

from __future__ import annotations

from .control import (
    ClosedLoop,
    CurrentController,
    PhaseLockedLoop,
    closed_loop,
)
from .dc_link import (
    DCLinkControl,
    PerturbObserveMPPT,
    mppt_current_lines,
    ripple_third_harmonic,
)
from .dead_time import dead_time_voltage
from .filters import (
    LCLFilter,
    SeriesStage,
    ShuntStage,
    inner_port_state,
    propagate_norton,
)
from .model import MECHANISMS, GridFollowingInverter, NortonEquivalent
from .operating import InverterState, PVArray
from .pwm import PWMModulator, bessel_j

__all__: list[str] = [
    "ClosedLoop",
    "CurrentController",
    "DCLinkControl",
    "GridFollowingInverter",
    "InverterState",
    "LCLFilter",
    "MECHANISMS",
    "NortonEquivalent",
    "PVArray",
    "PWMModulator",
    "PerturbObserveMPPT",
    "PhaseLockedLoop",
    "SeriesStage",
    "ShuntStage",
    "bessel_j",
    "closed_loop",
    "dead_time_voltage",
    "inner_port_state",
    "mppt_current_lines",
    "propagate_norton",
    "ripple_third_harmonic",
]
