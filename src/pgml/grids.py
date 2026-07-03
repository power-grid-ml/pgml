"""Reference grid builders: pandapower -> pgml Grid with optional geometry and spectra.

Each function converts a well-known benchmark network (IEEE 33-bus, CIGRE LV) from
pandapower into a pgml :class:`~pgml.schemas.grid_schema.Grid`.  These are the
suite's canonical INPUT grids — training-data generation, examples, and the oracle
comparison tests all build on them — so they live in the core package rather than
the evaluation oracles.

Importing this module requires ``pandapower`` (optional dependency).  All pandapower
imports are deferred to function scope except for ``numpy``, which is a core dependency.
"""

from __future__ import annotations

from pgml.convert.pandapower import ensure_numpy_compat as _numpy_shim


# Typical 6-pulse converter line-current spectrum (fraction of fundamental).
CONVERTER_SPECTRUM = [
    (1, 1.0, 0.0),
    (5, 0.20, 0.0),
    (7, 0.14, 0.0),
    (11, 0.09, 0.0),
    (13, 0.07, 0.0),
]


def _attach_spectrum_farthest(grid, n_loads: int, spectrum) -> None:
    from pgml.schemas.grid_schema import (
        HarmonicComponent,
        Load,
        SpectrumPoint,
        StaticSpectrum,
    )

    from pgml.topology import distance_from_slack

    dist = distance_from_slack(grid)
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    loads.sort(key=lambda a: dist.get(int(a.node), 0.0), reverse=True)
    comps = [
        HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a) for o, m, a in spectrum
    ]
    for ld in loads[:n_loads]:
        ld.spectrum = StaticSpectrum(spectrum=SpectrumPoint(components=comps))


def ieee33_geometry_grid(*, n_harmonic_loads: int = 3, spectrum=None):
    """IEEE-33 as a pgml grid with synthesized Carson geometry + converter spectra."""
    _numpy_shim()
    import pandapower as pp
    import pandapower.networks as pn

    from pgml.convert.pandapower import to_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    net = pn.case33bw()
    pp.runpp(net, numba=False)
    grid, id_map = to_grid(net)
    synthesize_grid_geometry(grid)
    _attach_spectrum_farthest(grid, n_harmonic_loads, spectrum or CONVERTER_SPECTRUM)
    return grid, id_map


def cigre_lv_full_grid(*, phase_mode=None, source_impedance_ohm=None):
    """The FULL CIGRE LV benchmark grid (all 3 feeders + MV source + 3 transformers).

    Unlike :func:`cigre_lv_geometry_grid` (one residential feeder with synthesized
    Carson geometry), this returns the WHOLE pandapower CIGRE LV network converted to a
    pgml :class:`~pgml.schemas.grid_schema.Grid` with standard R/X lines and the three
    20/0.4 kV transformers intact, fed by the single MV ext-grid source. The shared
    entry point for the full-grid examples + the OpenDSS oracle.

    The stock pandapower ext-grid converts to a near-ideal source (R~1e-6 Ohm) which
    short-circuits the bus at harmonics; a FINITE series impedance is applied so the
    source does not fully absorb injected harmonics (``source_impedance_ohm`` |Z| at the
    source's rated voltage, ``source.rx_ratio`` for the X/R split — config defaults under
    ``source.*``). Larger = weaker upstream grid = more cross-feeder coupling.

    Parameters
    ----------
    phase_mode:
        ``PhaseMode.SINGLE_PHASE_EQUIV`` (default) or ``PhaseMode.THREE_PHASE``.
    source_impedance_ohm:
        Source series-impedance magnitude [Ohm]; ``None`` -> config
        ``source.series_impedance_ohm``. Pass ``0`` to keep the converted (stiff) source.

    Returns
    -------
    (Grid, id_map)
    """
    _numpy_shim()
    import math

    import pandapower.networks as pn

    from pgml import defaults as _defaults
    from pgml.convert.pandapower import PhaseMode, to_grid
    from pgml.schemas.grid_schema import Source

    mode = phase_mode if phase_mode is not None else PhaseMode.SINGLE_PHASE_EQUIV
    grid, id_map = to_grid(pn.create_cigre_network_lv(), phase_mode=mode)

    z = (
        _defaults.get("source.series_impedance_ohm")
        if source_impedance_ohm is None
        else float(source_impedance_ohm)
    )
    if z > 0.0:
        rx = float(_defaults.get("source.rx_ratio"))
        r = z / math.sqrt(1.0 + rx * rx)
        ll = (rx * r) / (2.0 * math.pi * float(grid.base_frequency_hz))  # X = 2*pi*f0*L
        for a in grid.appliances:
            if isinstance(a, Source):
                p = len(a.phases)
                a.resistance_ohm = [
                    [r if i == j else 0.0 for j in range(p)] for i in range(p)
                ]
                a.inductance_h = [
                    [ll if i == j else 0.0 for j in range(p)] for i in range(p)
                ]
    return grid, id_map


def cigre_lv_geometry_grid(*, n_harmonic_loads: int = 3, spectrum=None):
    """CIGRE LV residential feeder (fed by a Thévenin source at its LV busbar) as a
    pgml grid with synthesized Carson geometry + converter spectra.

    The 20/0.4 kV transformer + MV grid are abstracted to a stiff 0.4 kV source so the
    comparison isolates the LV line (Carson) model.
    """
    _numpy_shim()
    import networkx as nx
    import pandapower as pp
    import pandapower.networks as pn
    import pandapower.topology as top

    from pgml.convert.pandapower import to_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    net = pn.create_cigre_network_lv()
    mg = top.create_nxgraph(net, include_trafos=False)
    comp = list(nx.node_connected_component(mg, 2))  # residential LV busbar = bus 2
    sub = pp.select_subnet(net, comp, include_results=False)
    pp.create_ext_grid(sub, bus=2, vm_pu=1.0)
    pp.runpp(sub, numba=False)
    grid, id_map = to_grid(sub)
    synthesize_grid_geometry(grid)
    _attach_spectrum_farthest(grid, n_harmonic_loads, spectrum or CONVERTER_SPECTRUM)
    return grid, id_map


__all__ = [
    "CONVERTER_SPECTRUM",
    "ieee33_geometry_grid",
    "cigre_lv_full_grid",
    "cigre_lv_geometry_grid",
]
