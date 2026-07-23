"""Public schema API for pgml.

These three modules are the FROZEN data contracts (orchestrator-only). Consumers
import the canonical types from here, e.g. ``from pgml.schemas import Grid, Node``.
"""

from __future__ import annotations

from . import grid_schema, result_schema, scenario_schema
from .grid_schema import *  # noqa: F401,F403
from .result_schema import *  # noqa: F401,F403
from .scenario_schema import *  # noqa: F401,F403

#: Version of the frozen data contracts (Grid / Result / Scenario), stamped into persisted
#: datasets (``meta.json``) so a reload can detect a schema change. During pre-1.0 development
#: the schema MAJOR tracks the library major (both stay ``0.x`` while the library is < 1.0.0);
#: bump the patch/minor on any contract change. A MAJOR mismatch on read is incompatible.
SCHEMA_VERSION = "0.0.4"

__all__ = [
    "SCHEMA_VERSION",
    *grid_schema.__all__,
    *result_schema.__all__,
    *scenario_schema.__all__,
]
