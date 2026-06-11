"""
scenario_schema.py — realized simulation INPUTS (rev 1).

The per-step operating points, realized harmonic spectra, and deliberate
parameter perturbations that were fed into a simulation. Kept SEPARATE from
results (result_schema.py) and from the static grid (grid_schema.py), linked by
integer ids, so a surrogate / inverse model can join inputs to outputs by
(scenario, component, step) without the grid description carrying time-varying
state.

================================================================================
CONVENTIONS
================================================================================

INPUTS, NOT RESULTS. These are the realized values applied during simulation: the
actual operating-point P/Q (from the profile service, NOT the nameplate ratings
in grid_schema), the actual harmonic spectrum injected at each step (after any
random/distribution sampling), and any injected parameter errors. They are the
ground-truth INPUT side for learning input->output relations and for parameter
recovery.

AUTHORING-NATURAL FORMS. Unlike results (which use real/imag for ML), inputs are
stored in their natural setpoint forms: P [W] / Q [var] for loads & generators;
reference voltage magnitude + angle for sources; spectra as magnitude relative to
the fundamental + phase, per harmonic (consistent with grid_schema spectra). A
downstream adapter converts to whatever encoding a model needs.

PER PHASE, PER STEP, PER FREQUENCY (spectra). Operating points are per step;
realized spectra add a `frequency_hz` axis (interharmonic-ready, matching the
result schema). All per-phase arrays align to an explicit `phases` tuple.

SIGN/REFERENCE. Operating-point P/Q use the appliance's natural sense: a load's
p_w is consumption (positive), a generator's p_w is production (positive) — the
component kind disambiguates, since these are setpoints, not signed flows.
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
# 1. Scenario metadata
# =============================================================================
class Scenario(GridModel):
    """One coherent set of time-stepped inputs applied to a grid. A ResultSet
    references the Scenario that produced it."""

    id: int = Field(description="Unique scenario id.")
    description: Optional[str] = Field(default=None)
    grid_id: Optional[int] = Field(default=None, description="Loose ref to the grid description.")
    grid_topology_id: Optional[int] = Field(default=None, description="Loose ref to grid topology.")
    n_steps: int = Field(description="Number of time/scenario steps.", ge=1)
    step_size_ms: Optional[float] = si_field("Time between steps.", short="ms", long="millisecond",
                                            default=None)
    base_frequency_hz: float = si_field("System fundamental f0.", short="Hz", long="hertz", gt=0.0,
                                        default=50.0)
    set: Optional[str] = Field(default=None, description="ML split label, e.g. train/val/test.")
    provenance: dict = Field(default_factory=dict,
                             description="How inputs were generated (profile-service config, seeds).")
    date_created: datetime = Field(default_factory=datetime.now)


# =============================================================================
# 2. Realized operating points (per step)
# =============================================================================
class LoadOperatingPoint(GridModel):
    """Realized per-phase P/Q of a load at one step (the actual operating point,
    not the nameplate rating)."""

    scenario_id: int = Field(description="Loose ref to Scenario.id.")
    load_id: int = Field(description="Loose ref to the grid load id.")
    step: int = Field(description="Step index.")
    phases: tuple[Phase, ...] = Field(description="Connected phases, array order.")
    p_w: tuple[float, ...] = si_field("Per-phase active power consumed.", short="W", long="watt")
    q_var: tuple[float, ...] = si_field("Per-phase reactive power consumed.", short="var", long="var")

    @model_validator(mode="after")
    def _check(self) -> "LoadOperatingPoint":
        _check_phase_lengths(self.phases, p_w=self.p_w, q_var=self.q_var)
        return self


class GeneratorOperatingPoint(GridModel):
    """Realized per-phase P/Q of a generator at one step (production positive)."""

    scenario_id: int = Field(description="Loose ref to Scenario.id.")
    generator_id: int = Field(description="Loose ref to the grid generator id.")
    step: int = Field(description="Step index.")
    phases: tuple[Phase, ...] = Field(description="Connected phases, array order.")
    p_w: tuple[float, ...] = si_field("Per-phase active power produced.", short="W", long="watt")
    q_var: tuple[float, ...] = si_field("Per-phase reactive power produced.", short="var", long="var")

    @model_validator(mode="after")
    def _check(self) -> "GeneratorOperatingPoint":
        _check_phase_lengths(self.phases, p_w=self.p_w, q_var=self.q_var)
        return self


class SourceOperatingPoint(GridModel):
    """Realized per-phase reference voltage setpoint of a source at one step."""

    scenario_id: int = Field(description="Loose ref to Scenario.id.")
    source_id: int = Field(description="Loose ref to the grid source id.")
    step: int = Field(description="Step index.")
    phases: tuple[Phase, ...] = Field(description="Connected phases, array order.")
    u_ref_v: tuple[float, ...] = si_field("Per-phase reference voltage magnitude.", short="V",
                                         long="volt")
    u_angle_deg: tuple[float, ...] = si_field("Per-phase reference voltage angle.", short="deg",
                                             long="degree")

    @model_validator(mode="after")
    def _check(self) -> "SourceOperatingPoint":
        _check_phase_lengths(self.phases, u_ref_v=self.u_ref_v, u_angle_deg=self.u_angle_deg)
        return self


# =============================================================================
# 3. Realized harmonic spectra (per step, per frequency)
# =============================================================================
class RealizedSpectrumPoint(GridModel):
    """The harmonic spectrum actually injected by a parent (load/generator/source)
    at one step and frequency — i.e. the concrete content after any random /
    distribution sampling of the grid_schema Spectrum. Magnitudes are relative to
    the fundamental injection, per phase (consistent with grid_schema)."""

    scenario_id: int = Field(description="Loose ref to Scenario.id.")
    parent_kind: Literal["load", "generator", "source"] = Field(description="Owning component kind.")
    parent_id: int = Field(description="Loose ref to the parent component id.")
    step: int = Field(description="Step index.")
    frequency_hz: float = si_field("Frequency of this spectral line.", short="Hz", long="hertz",
                                   gt=0.0)
    phases: tuple[Phase, ...] = Field(description="Connected phases, array order.")
    magnitude_pu: tuple[float, ...] = si_field(
        "Per-phase magnitude as a fraction of the fundamental injection.",
        short="pu", long="per unit of fundamental",
    )
    phase_deg: tuple[float, ...] = si_field("Per-phase phase relative to the fundamental.",
                                           short="deg", long="degree")

    @model_validator(mode="after")
    def _check(self) -> "RealizedSpectrumPoint":
        _check_phase_lengths(self.phases, magnitude_pu=self.magnitude_pu, phase_deg=self.phase_deg)
        return self


# =============================================================================
# 4. Injected parameter perturbations (ground truth for inverse problems)
# =============================================================================
class ParameterPerturbation(GridModel):
    """A deliberate perturbation of a grid parameter recorded as ground truth, for
    the gradient-based parameter-recovery / error-detection use case. Generalises
    the prior 'injected error' table to any component parameter."""

    scenario_id: int = Field(description="Loose ref to Scenario.id.")
    step: Optional[int] = Field(default=None, description="Step index, or None if static.")
    component_kind: str = Field(description="e.g. 'line', 'transformer', 'source', 'node'.")
    component_id: int = Field(description="Loose ref to the perturbed component id.")
    parameter_path: str = Field(
        description="Dotted path of the perturbed parameter, e.g. "
        "'series_resistance_ohm_per_m' or 'zero_sequence.x0_ohm'."
    )
    nominal_value: float = Field(description="True/unperturbed value (SI of the parameter).")
    perturbed_value: float = Field(description="Applied (perturbed) value (SI of the parameter).")
    unit_short: Optional[str] = Field(default=None, description="Unit short code, e.g. 'Ohm/m'.")


__all__ = [
    "Scenario", "LoadOperatingPoint", "GeneratorOperatingPoint", "SourceOperatingPoint",
    "RealizedSpectrumPoint", "ParameterPerturbation",
]
