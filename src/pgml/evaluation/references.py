"""Compatibility shim — moved to :mod:`pgml.evaluation.oracles`.

The real implementations live in the ``oracles`` subpackage.  This module
re-exports the complete public API for backward compatibility with existing
importers; it will be removed in a future release.
"""

from pgml.evaluation.oracles import (  # noqa: F401
    CONVERTER_SPECTRUM,
    align_dss_systemy,
    build_opendss_geometry_circuit,
    cigre_lv_full_grid,
    cigre_lv_geometry_grid,
    dss_systemy,
    ieee33_geometry_grid,
    numpy_harmonic_profiles,
    numpy_harmonic_voltages,
    opendss_dyn_transformer_harmonic_voltages,
    opendss_geometry_harmonic_profiles,
    opendss_geometry_systemy,
    opendss_harmonic_voltages,
    opendss_ybus,
    pandapower_voltage_profile,
    pandapower_ybus,
)

__all__ = [
    "ieee33_geometry_grid",
    "cigre_lv_geometry_grid",
    "cigre_lv_full_grid",
    "pandapower_ybus",
    "pandapower_voltage_profile",
    "dss_systemy",
    "align_dss_systemy",
    "opendss_ybus",
    "build_opendss_geometry_circuit",
    "opendss_geometry_systemy",
    "opendss_geometry_harmonic_profiles",
    "numpy_harmonic_profiles",
    "numpy_harmonic_voltages",
    "opendss_harmonic_voltages",
    "opendss_dyn_transformer_harmonic_voltages",
]
