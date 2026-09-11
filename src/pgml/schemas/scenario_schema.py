"""Realized simulation INPUTS: the scenario identity and its injected perturbations.

What was fed into a simulation, kept SEPARATE from results
(:mod:`pgml.schemas.result_schema`) and from the static grid
(:mod:`pgml.schemas.grid_schema`) and linked by integer ids, so a surrogate or inverse
model joins inputs to outputs by (scenario, component, step) without the grid description
carrying time-varying state.

**Conventions**

*Inputs, not results.* These are the values applied during simulation, not derived from
it: the ground-truth INPUT side for learning input/output relations and for parameter
recovery.

*The realized values themselves are tensors, not rows.* A batch's realized operating
points and harmonic spectra are carried by
:class:`pgml.scenarios.SampledScenarios` and persisted as the columnar
``samples.parquet`` of :func:`pgml.scenarios.write_dataset`, which is the form a training
pipeline reads. This module holds what does not fit a tensor column: the scenario's own
identity and provenance, and the deliberate parameter perturbations that are ground truth
for the inverse problem.

*Sign/reference.* Setpoints use the appliance's natural sense: a load's ``p_w`` is
consumption (positive), a generator's ``p_w`` is production (positive) — the component
kind disambiguates, since these are setpoints, not signed flows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .grid_schema import GridModel, si_field


# =============================================================================
# 1. Scenario metadata
# =============================================================================
class Scenario(GridModel):
    """One coherent set of time-stepped inputs applied to a grid. A ResultSet
    references the Scenario that produced it."""

    id: int = Field(description="Unique scenario id.")
    description: Optional[str] = Field(default=None)
    grid_id: Optional[int] = Field(
        default=None, description="Loose ref to the grid description."
    )
    grid_topology_id: Optional[int] = Field(
        default=None, description="Loose ref to grid topology."
    )
    n_steps: int = Field(description="Number of time/scenario steps.", ge=1)
    step_size_ms: Optional[float] = si_field(
        "Time between steps.", short="ms", long="millisecond", default=None
    )
    base_frequency_hz: float = si_field(
        "System fundamental f0.", short="Hz", long="hertz", gt=0.0, default=50.0
    )
    set: Optional[str] = Field(
        default=None, description="ML split label, e.g. train/val/test."
    )
    provenance: dict = Field(
        default_factory=dict,
        description="How inputs were generated (profile-service config, seeds).",
    )
    date_created: datetime = Field(default_factory=datetime.now)


# =============================================================================
# 2. Injected parameter perturbations (ground truth for inverse problems)
# =============================================================================
class ParameterPerturbation(GridModel):
    """A deliberate perturbation of a grid parameter recorded as ground truth, for
    the gradient-based parameter-recovery / error-detection use case. Generalises
    the prior 'injected error' table to any component parameter."""

    scenario_id: int = Field(description="Loose ref to Scenario.id.")
    step: Optional[int] = Field(
        default=None, description="Step index, or None if static."
    )
    component_kind: str = Field(
        description="e.g. 'line', 'transformer', 'source', 'node'."
    )
    component_id: int = Field(description="Loose ref to the perturbed component id.")
    parameter_path: str = Field(
        description="Dotted path of the perturbed parameter, e.g. "
        "'series_resistance_ohm_per_m' or 'zero_sequence.x0_ohm'."
    )
    nominal_value: float = Field(
        description="True/unperturbed value (SI of the parameter)."
    )
    perturbed_value: float = Field(
        description="Applied (perturbed) value (SI of the parameter)."
    )
    unit_short: Optional[str] = Field(
        default=None, description="Unit short code, e.g. 'Ohm/m'."
    )


__all__ = [
    "Scenario",
    "ParameterPerturbation",
]
