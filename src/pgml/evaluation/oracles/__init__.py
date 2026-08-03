"""pgml.evaluation.oracles — optional reference-grid builders and oracle adapters.

This subpackage provides the test-support and validation tooling used to compare
pgml's differentiable harmonic solver against independent reference implementations.
It is an OPTIONAL import: ``pgml.evaluation`` (the plotting/data side) does NOT
import this subpackage and works without pandapower or opendssdirect installed.

Modules
-------
- ``grids``           Reference grid builders (IEEE-33, CIGRE LV) using pandapower.
- ``numpy_oracle``    Pure-numpy independent harmonic oracle (no live OpenDSS required).
- ``pandapower_oracle`` pandapower Ybus + voltage-profile adapters.
- ``opendss_oracle``  Live OpenDSS harmonic oracle + geometry circuit builders.

Optional dependencies
---------------------
``pandapower`` is required by ``grids`` and ``pandapower_oracle``.
``opendssdirect`` is required by ``opendss_oracle``.
Both are imported lazily INSIDE functions, so ``import pgml.evaluation.oracles``
itself succeeds without these packages.

Install the extras group ``pgml[oracles]`` to pull in all oracle dependencies.
"""

from __future__ import annotations

from .grids import (
    CONVERTER_SPECTRUM,
    cigre_lv_full_grid,
    cigre_lv_geometry_grid,
    ieee33_geometry_grid,
)
from .numpy_oracle import (
    numpy_harmonic_profiles,
    numpy_harmonic_voltages,
)
from .opendss_oracle import (
    align_dss_systemy,
    build_opendss_geometry_circuit,
    dss_systemy,
    opendss_dyn_transformer_harmonic_voltages,
    opendss_geometry_harmonic_profiles,
    opendss_geometry_systemy,
    opendss_harmonic_voltages,
    opendss_ybus,
)
from .opendss_scenario_oracle import (
    ExportedCircuit,
    compare_to_pgml,
    export_grid_to_opendss,
    run_opendss_scenarios,
    write_opendss_dataset,
)
from .pandapower_oracle import (
    pandapower_voltage_profile,
    pandapower_ybus,
)

__all__ = [
    # grid builders
    "CONVERTER_SPECTRUM",
    "ieee33_geometry_grid",
    "cigre_lv_full_grid",
    "cigre_lv_geometry_grid",
    # numpy oracle
    "numpy_harmonic_profiles",
    "numpy_harmonic_voltages",
    # pandapower oracle
    "pandapower_ybus",
    "pandapower_voltage_profile",
    # opendss oracle
    "dss_systemy",
    "align_dss_systemy",
    "opendss_ybus",
    "build_opendss_geometry_circuit",
    "opendss_geometry_systemy",
    "opendss_geometry_harmonic_profiles",
    "opendss_harmonic_voltages",
    "opendss_dyn_transformer_harmonic_voltages",
    # opendss SCENARIO oracle (independent full-circuit export + batch scenario runs)
    "ExportedCircuit",
    "export_grid_to_opendss",
    "run_opendss_scenarios",
    "write_opendss_dataset",
    "compare_to_pgml",
]
