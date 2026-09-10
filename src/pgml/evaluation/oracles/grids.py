"""Compatibility re-export — the benchmark grid builders moved to :mod:`pgml.grids`.

The IEEE-33 / CIGRE LV builders are canonical input grids (training data, examples,
oracle tests alike), not oracle plumbing, so they live in the core :mod:`pgml.grids`.
This module re-exports them for existing importers.
"""

from pgml.grids import (  # noqa: F401
    CONVERTER_SPECTRUM,
    cigre_lv_full_grid,
    cigre_lv_geometry_grid,
    ieee33_geometry_grid,
)

__all__ = [
    "CONVERTER_SPECTRUM",
    "ieee33_geometry_grid",
    "cigre_lv_geometry_grid",
    "cigre_lv_full_grid",
]
