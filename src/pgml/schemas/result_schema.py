"""
result_schema.py — simulation OUTPUT contract (rev 1).

Per-component results for the differentiable harmonic power-flow simulation:
voltages per node, currents per branch terminal, currents per injection, with
optional powers. Companion to grid_schema.py (input) and scenario_schema.py
(realized inputs). Validation/compliance metrics (EN 50160 etc.) are derived and
live in a separate, later artifact; this schema is results only.

================================================================================
CONVENTIONS
================================================================================

PER-COMPONENT, FULL STATE. Results are stored per node / per branch / per
injection (not per monitor), giving the complete grid state with unambiguous
location and direction. This is the chosen model over emulating physical meters.

PHASORS AS REAL/IMAG. Every complex phasor (voltage, current) is stored as a
(real, imag) pair in SI units, NOT magnitude/angle. This is the solver's native
differentiable output, is lossless, and avoids the 2*pi angle-wrap discontinuity
that harms ML targets (state estimation). Magnitude/angle, P/Q/S aggregates,
THD, unbalance and EN 50160 compliance are all DERIVED VIEWS computed on demand;
they are not part of this contract (except the optional P/Q/S below, kept for
convenience).

PER FREQUENCY, ALWAYS. Every record is indexed by `frequency_hz` (float), not by
integer harmonic order, so interharmonics are representable. The harmonic order
h = frequency_hz / base_frequency_hz is derivable (need not be integer). Results
are never aggregated across frequency in this schema; THD and other cross-
frequency metrics are computed downstream.

POWERS ARE OPTIONAL, PER FREQUENCY. P [W], Q [var] and S [VA] may be stored
alongside V and I (per phase, per frequency) when convenient, but are derivable
from V and I and may be omitted. At harmonic h, P_h=Re(V_h conj(I_h)),
Q_h=Im(V_h conj(I_h)), S_h=|V_h||I_h|.

REFERENCE DIRECTIONS.
  * Branch terminals: i_from / i_to (and their powers) are positive when flowing
    FROM the respective node INTO the branch. Hence p_from + p_to over all phases
    equals the branch losses. Direction is anchored to the component's
    from_node/to_node from grid_schema.
  * Injections (load/generator/source/shunt): current and power are positive when
    flowing FROM the node INTO the appliance (LOAD reference). A consuming load
    reports positive P; a generator injecting into the grid reports negative P.
    One rule for all appliance kinds; the sign carries the direction.

PER-PHASE. Each record carries an explicit `phases` tuple (from grid_schema) and
parallel per-phase value tuples aligned to it, supporting 1/2/3/4-phase uniformly
(branches may have different from/to phase sets). The ML tensor materialisation
uses a fixed (A,B,C,N) layout with masking; this object/columnar form is variable.

MATERIALISATIONS. Object (these models) / columnar (parquet, one table per record
type) / tensor (dense complex arrays V[step, freq, node, phase],
I_from/I_to[step, freq, branch, phase]). Bulk storage is columnar/tensor; the
object form is for interchange and small/interactive use. Loose coupling to the
grid and scenario is by integer id (result_set_id, *_id), as in the prior design.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field, model_validator

from grid_schema import GridModel, Phase, si_field


def _check_phase_lengths(phases: tuple, **named) -> None:
    n = len(phases)
    for name, val in named.items():
        if val is not None and len(val) != n:
            raise ValueError(f"`{name}` length ({len(val)}) must match phase count ({n}).")


# =============================================================================
# 1. Result-set metadata & solver diagnostics
# =============================================================================
class ResultSet(GridModel):
    """Metadata for one coherent set of simulation results (a 'dataset'). Loose
    references to the grid, its topology, and the scenario that produced it."""

    id: int = Field(description="Unique result-set id.")
    description: Optional[str] = Field(default=None)
    experiment_id: Optional[str] = Field(default=None)
    experiment_details: dict = Field(default_factory=dict, description="Free-form experiment config.")
    grid_id: Optional[int] = Field(default=None, description="Loose ref to the grid description.")
    grid_topology_id: Optional[int] = Field(default=None, description="Loose ref to grid topology.")
    scenario_id: Optional[int] = Field(
        default=None, description="Loose ref to the Scenario (scenario_schema) of realized inputs."
    )
    set: Optional[str] = Field(default=None, description="ML split label, e.g. train/val/test.")
    base_frequency_hz: float = si_field("System fundamental f0 used in this set.", short="Hz",
                                        long="hertz", gt=0.0, default=50.0)
    step_size_ms: Optional[float] = si_field("Time between steps.", short="ms", long="millisecond",
                                            default=None)
    date_created: datetime = Field(default_factory=datetime.now)


class SolverDiagnostics(GridModel):
    """Per-step solver telemetry (one row per step). For the decoupled linear
    harmonic solve `residual_norm` reflects solve tolerance/conditioning; for a
    future coupled/iterative model it reflects true iteration convergence."""

    result_set_id: int = Field(description="Loose ref to ResultSet.id.")
    step: int = Field(description="Time/scenario step index.")
    converged: bool = Field(default=True)
    iterations: Optional[int] = Field(default=None)
    process_time_us: Optional[float] = si_field("Wall-clock solve time.", short="us",
                                                long="microsecond", default=None)
    residual_norm: Optional[float] = Field(default=None, description="Final residual norm.")
    convergence_details: dict = Field(default_factory=dict)


# =============================================================================
# 2. Raw per-component results
# =============================================================================
class NodeResult(GridModel):
    """Per-node voltage phasor (and optional net injected power) at one frequency
    and step."""

    result_set_id: int = Field(description="Loose ref to ResultSet.id.")
    node_id: int = Field(description="Loose ref to the grid node id.")
    frequency_hz: float = si_field("Frequency of this record.", short="Hz", long="hertz", gt=0.0)
    step: int = Field(description="Time/scenario step index.")
    phases: tuple[Phase, ...] = Field(description="Phases present, fixing per-phase array order.")
    v_re: tuple[float, ...] = si_field("Per-phase voltage, real part.", short="V", long="volt")
    v_im: tuple[float, ...] = si_field("Per-phase voltage, imaginary part.", short="V", long="volt")
    p_w: Optional[tuple[float, ...]] = si_field("Per-phase net active power injected at node.",
                                               short="W", long="watt", default=None)
    q_var: Optional[tuple[float, ...]] = si_field("Per-phase net reactive power at node.",
                                                 short="var", long="var", default=None)
    s_va: Optional[tuple[float, ...]] = si_field("Per-phase apparent power at node.", short="VA",
                                                long="volt-ampere", default=None)

    @model_validator(mode="after")
    def _check(self) -> "NodeResult":
        _check_phase_lengths(self.phases, v_re=self.v_re, v_im=self.v_im, p_w=self.p_w,
                             q_var=self.q_var, s_va=self.s_va)
        return self


class BranchResult(GridModel):
    """Per-branch terminal currents (both ends) and optional terminal powers at one
    frequency and step. from_/to_ are anchored to the component's from_node/to_node;
    currents are positive flowing INTO the branch at each terminal."""

    result_set_id: int = Field(description="Loose ref to ResultSet.id.")
    branch_id: int = Field(description="Loose ref to the grid branch id.")
    branch_kind: Literal["line", "transformer", "switch", "shunt_reactor", "generic_branch"] = Field(
        description="Branch component kind (mirrors grid_schema discriminator)."
    )
    frequency_hz: float = si_field("Frequency of this record.", short="Hz", long="hertz", gt=0.0)
    step: int = Field(description="Time/scenario step index.")
    from_phases: tuple[Phase, ...] = Field(description="From-terminal phases, array order.")
    to_phases: tuple[Phase, ...] = Field(description="To-terminal phases, array order.")
    i_from_re: tuple[float, ...] = si_field("From-terminal current, real part.", short="A",
                                           long="ampere")
    i_from_im: tuple[float, ...] = si_field("From-terminal current, imaginary part.", short="A",
                                           long="ampere")
    i_to_re: tuple[float, ...] = si_field("To-terminal current, real part.", short="A",
                                         long="ampere")
    i_to_im: tuple[float, ...] = si_field("To-terminal current, imaginary part.", short="A",
                                         long="ampere")
    p_from_w: Optional[tuple[float, ...]] = si_field("From-terminal active power (into branch).",
                                                    short="W", long="watt", default=None)
    q_from_var: Optional[tuple[float, ...]] = si_field("From-terminal reactive power.", short="var",
                                                      long="var", default=None)
    s_from_va: Optional[tuple[float, ...]] = si_field("From-terminal apparent power.", short="VA",
                                                     long="volt-ampere", default=None)
    p_to_w: Optional[tuple[float, ...]] = si_field("To-terminal active power (into branch).",
                                                  short="W", long="watt", default=None)
    q_to_var: Optional[tuple[float, ...]] = si_field("To-terminal reactive power.", short="var",
                                                    long="var", default=None)
    s_to_va: Optional[tuple[float, ...]] = si_field("To-terminal apparent power.", short="VA",
                                                   long="volt-ampere", default=None)

    @model_validator(mode="after")
    def _check(self) -> "BranchResult":
        _check_phase_lengths(self.from_phases, i_from_re=self.i_from_re, i_from_im=self.i_from_im,
                             p_from_w=self.p_from_w, q_from_var=self.q_from_var, s_from_va=self.s_from_va)
        _check_phase_lengths(self.to_phases, i_to_re=self.i_to_re, i_to_im=self.i_to_im,
                             p_to_w=self.p_to_w, q_to_var=self.q_to_var, s_to_va=self.s_to_va)
        return self


class InjectionResult(GridModel):
    """Per-appliance (load/generator/source/shunt) current and optional power at one
    frequency and step. LOAD reference: positive = into the appliance."""

    result_set_id: int = Field(description="Loose ref to ResultSet.id.")
    injection_id: int = Field(description="Loose ref to the grid appliance id.")
    injection_kind: Literal["load", "generator", "source", "shunt"] = Field(
        description="Appliance kind (mirrors grid_schema discriminator)."
    )
    frequency_hz: float = si_field("Frequency of this record.", short="Hz", long="hertz", gt=0.0)
    step: int = Field(description="Time/scenario step index.")
    phases: tuple[Phase, ...] = Field(description="Connected phases, array order.")
    i_re: tuple[float, ...] = si_field("Per-phase current, real part.", short="A", long="ampere")
    i_im: tuple[float, ...] = si_field("Per-phase current, imaginary part.", short="A", long="ampere")
    p_w: Optional[tuple[float, ...]] = si_field("Per-phase active power (into appliance).",
                                               short="W", long="watt", default=None)
    q_var: Optional[tuple[float, ...]] = si_field("Per-phase reactive power.", short="var",
                                                 long="var", default=None)
    s_va: Optional[tuple[float, ...]] = si_field("Per-phase apparent power.", short="VA",
                                                long="volt-ampere", default=None)

    @model_validator(mode="after")
    def _check(self) -> "InjectionResult":
        _check_phase_lengths(self.phases, i_re=self.i_re, i_im=self.i_im, p_w=self.p_w,
                             q_var=self.q_var, s_va=self.s_va)
        return self


__all__ = [
    "ResultSet", "SolverDiagnostics", "NodeResult", "BranchResult", "InjectionResult",
]
