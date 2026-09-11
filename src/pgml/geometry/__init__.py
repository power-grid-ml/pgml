"""pgml.geometry — conductor geometry -> differentiable line constants (Carson/Deri).

- ``carson``: torch Carson/Deri series impedance + skin effect + Maxwell capacitance,
  bit-exact vs OpenDSS (see ``docs/pgml/modeling/references/opendss/carson.md``), differentiable & batched.
- ``synthesis``: build a :class:`~pgml.schemas.grid_schema.LineGeometry` that reproduces
  a line's R/X at fundamental (for R/X-defined feeders that lack conductor geometry),
  with provenance tracking.

A :class:`~pgml.schemas.grid_schema.Line` carrying a ``conductor_geometry`` gets its
``Z(h)``/``Yc(h)`` from this model during assembly.
"""

from __future__ import annotations

from .carson import (
    i0_over_i1,
    internal_impedance,
    kron_reduce,
    line_constants,
    potential_coefficients,
    series_impedance,
)
from .sequence import (
    carson_earth_resistance,
    fit_equivalent_rdc,
    phase_to_sequence,
    positive_sequence_z,
    sequence_aware_phase_z,
    sequence_impedances,
    sequence_to_phase_z,
    skin_resistance_multiplier,
    two_conductor_geometry,
    two_conductor_loop_z,
    zero_sequence_harmonic_z,
)
from .synthesis import (
    apply_default_harmonic_model,
    apply_positive_sequence_harmonic_model,
    apply_sequence_aware_harmonic_model,
    positive_sequence_resistance_model,
    resolve_harmonic_line_models,
    strip_grid_geometry,
    synthesize_grid_geometry,
    synthesize_line_geometry,
    synthesize_three_phase_geometry,
)

__all__ = [
    # carson (geometry -> Z/Yc, full earth return)
    "series_impedance",
    "internal_impedance",
    "potential_coefficients",
    "kron_reduce",
    "line_constants",
    "i0_over_i1",
    # sequence (positive-sequence harmonic model, no earth floor)
    "positive_sequence_z",
    "skin_resistance_multiplier",
    "fit_equivalent_rdc",
    "two_conductor_geometry",
    "two_conductor_loop_z",
    "phase_to_sequence",
    "sequence_impedances",
    # sequence-aware harmonic model (unbalanced / 4-wire: earth return in Z0)
    "carson_earth_resistance",
    "zero_sequence_harmonic_z",
    "sequence_to_phase_z",
    "sequence_aware_phase_z",
    # synthesis (R/X -> geometry / harmonic model)
    "synthesize_line_geometry",
    "synthesize_three_phase_geometry",
    "synthesize_grid_geometry",
    "strip_grid_geometry",
    "positive_sequence_resistance_model",
    "apply_positive_sequence_harmonic_model",
    "apply_sequence_aware_harmonic_model",
    "apply_default_harmonic_model",
    "resolve_harmonic_line_models",
]
