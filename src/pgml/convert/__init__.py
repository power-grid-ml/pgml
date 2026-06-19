"""pgml.convert — converters from external formats to our schema.

One subpackage per source:
- ``pandapower/``  — pandapower net -> Grid
- ``pgm/``         — power-grid-model input_data -> Grid
- ``opendss/``     — OpenDSS dss handle -> Grid

See each subpackage for the public ``to_grid`` signature. The shared scaffold in
:mod:`pgml.convert._common` (id allocation, the :class:`~pgml.convert._common.PhaseMode`
single-phase-equivalent / abc switch, sequence->phase line expansion, Thevenin
helpers, and the schema emit helpers) is the single place the per-library plumbing
lives.
"""

from pgml.convert._common import PhaseMode

__all__ = ["PhaseMode"]
