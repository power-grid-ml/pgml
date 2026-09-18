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

Every ``to_grid(..., return_report=True)`` and every ``from_grid(...).report`` yields a
:class:`ConversionReport` naming what was dropped, what was approximated and where
the two tools' default models differ.
"""

from pgml.convert._common import PhaseMode
from pgml.convert._model_differences import add_model_differences
from pgml.convert._report import (
    ConversionReport,
    ModelMatch,
    ReportCategory,
    ReportEntry,
)

__all__ = [
    "ConversionReport",
    "ModelMatch",
    "PhaseMode",
    "ReportCategory",
    "ReportEntry",
    "add_model_differences",
]
