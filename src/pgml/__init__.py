"""pgml — differentiable, GPU-ready harmonic power-flow + ML for power grids.

High-level entry points (the stable, public surface):

- :func:`simulate` ``(grid, config) -> SolvedState`` — run a simulation; the result is
  a complete, differentiable solved-grid snapshot (voltages eager; currents / flows /
  spectra derived on access). :func:`simulate_serializable` returns the JSON-ready
  :class:`ResultBundle` instead (REST / dashboard / persistence).
- :class:`SimulationConfig` — the serializable definition of WHAT to simulate.
- :class:`~pgml.schemas.grid_schema.Grid` — the input grid (full schema in
  :mod:`pgml.schemas`).
- the exception hierarchy from :mod:`pgml.errors` (``PgmlError`` and friends; ``PgmError`` is a deprecated alias).

Lower-level / optional surfaces are reached via their subpackages (so a minimal core
install need not import the extras): :mod:`pgml.solver` (raw differentiable tensors),
:mod:`pgml.scenarios` (batched data generation), :mod:`pgml.dispatch` (storage
state of charge), :mod:`pgml.convert`,
:mod:`pgml.evaluation`, :mod:`pgml.schemas`, :mod:`pgml.defaults` (modeling defaults),
:mod:`pgml.assembly`, :mod:`pgml.geometry`, :mod:`pgml.topology` (dependency-free graph
helpers), :mod:`pgml.grids` (reference grid builders, needs the ``convert`` extra).
"""

from __future__ import annotations

import logging as _logging

#: Library version (single source of truth; the build reads this).
__version__ = "0.4.0"

# Library logging convention: emit on the ``pgml`` logger and attach a NullHandler so
# importing pgml never prints "No handlers could be found". Applications opt in by
# configuring logging (e.g. ``logging.basicConfig(level=logging.INFO)``) to see the
# INFO modeling summary (calculation symmetry, neutral modeling, load connections).
_logging.getLogger("pgml").addHandler(_logging.NullHandler())

from .errors import (  # noqa: E402
    ComputationError,
    ConfigurationError,
    ConvergenceError,
    ConversionError,
    InputError,
    ModelingError,
    PgmError,
    PgmlError,
)
from .paths import experiments_root  # noqa: E402
from .schemas.grid_schema import Grid  # noqa: E402
from .simulation import (  # noqa: E402
    ResultBundle,
    SimulationConfig,
    SolvedState,
    simulate,
    simulate_serializable,
)

__all__ = [
    "__version__",
    # high-level entry points
    "simulate",
    "simulate_serializable",
    "SimulationConfig",
    "SolvedState",
    "ResultBundle",
    "Grid",
    # filesystem conventions
    "experiments_root",
    # exceptions
    "PgmError",
    "PgmlError",
    "InputError",
    "ComputationError",
    "ConfigurationError",
    "ConversionError",
    "ModelingError",
    "ConvergenceError",
]
