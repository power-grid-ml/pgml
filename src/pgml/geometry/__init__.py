"""pgml.geometry — conductor geometry -> differentiable line constants (Carson/Deri).

- ``carson``: torch Carson/Deri series impedance + skin effect + Maxwell capacitance,
  bit-exact vs OpenDSS (see ``references/opendss/carson.md``), differentiable & batched.
- ``synthesis``: build a :class:`~pgml.schemas.grid_schema.LineGeometry` that reproduces
  a line's R/X at fundamental (for R/X-defined feeders that lack conductor geometry),
  with provenance tracking.

A :class:`~pgml.schemas.grid_schema.Line` carrying a ``conductor_geometry`` gets its
``Z(h)``/``Yc(h)`` from this model during assembly.
"""

from __future__ import annotations

from .carson import (
    i0_over_i1,
    kron_reduce,
    line_constants,
    potential_coefficients,
    series_impedance,
)
from .synthesis import synthesize_line_geometry, synthesize_grid_geometry

__all__ = [
    "series_impedance",
    "potential_coefficients",
    "kron_reduce",
    "line_constants",
    "i0_over_i1",
    "synthesize_line_geometry",
    "synthesize_grid_geometry",
]
