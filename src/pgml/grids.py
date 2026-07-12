"""Reference grid builders: benchmark networks and synthetic feeders as pgml Grids.

Most functions convert a well-known benchmark network (IEEE 33-bus, CIGRE LV) from
pandapower into a pgml :class:`~pgml.schemas.grid_schema.Grid`.  These are the
suite's canonical INPUT grids — training-data generation, examples, and the oracle
comparison tests all build on them — so they live in the core package rather than
the evaluation oracles.  :func:`synthetic_feeder` additionally builds a
parameterizable radial MV feeder of ANY size directly on the schema (no external
dependency) — the scaling knob for solver benchmarks and topology studies.

The pandapower-backed builders require ``pandapower`` (optional dependency); its
imports are deferred to function scope, so importing this module (and using
:func:`synthetic_feeder`) works without it.
"""

from __future__ import annotations

import math

#: Typical 6-pulse converter line-current spectrum, as ``(order, magnitude_pu, phase_deg)``
#: relative to the fundamental. Attached to the farthest loads by :func:`ieee33_geometry_grid`
#: and :func:`cigre_lv_geometry_grid` to give the harmonic examples a realistic source.
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
    source does not fully absorb injected harmonics (``source_impedance_ohm`` is ``|Z|``
    at the source's rated voltage, ``source.rx_ratio`` for the X/R split — config
    defaults under ``source.*``). Larger = weaker upstream grid = more cross-feeder
    coupling.

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
    import networkx as nx
    import pandapower as pp
    import pandapower.networks as pn
    import pandapower.topology as top
    from pandapower.toolbox import select_subnet

    from pgml.convert.pandapower import to_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    net = pn.create_cigre_network_lv()
    mg = top.create_nxgraph(net, include_trafos=False)
    comp = list(nx.node_connected_component(mg, 2))  # residential LV busbar = bus 2
    sub = select_subnet(net, comp, include_results=False)
    pp.create_ext_grid(sub, bus=2, vm_pu=1.0)
    pp.runpp(sub, numba=False)
    grid, id_map = to_grid(sub)
    synthesize_grid_geometry(grid)
    _attach_spectrum_farthest(grid, n_harmonic_loads, spectrum or CONVERTER_SPECTRUM)
    return grid, id_map


#: Harmonic orders :func:`se_benchmark_scenario_config` randomizes for general LV loads
#: (the odd orders up to 13, including the triplen 3/9).
LOAD_HARMONIC_ORDERS = [3, 5, 7, 9, 11, 13]
#: Harmonic orders :func:`se_benchmark_scenario_config` randomizes for PV inverters
#: (the non-triplen 5/7/11/13 — a typical inverter switching signature).
PV_HARMONIC_ORDERS = [5, 7, 11, 13]


def add_pv_systems(grid, *, fraction: float = 0.5) -> int:
    """Attach a unity-power-factor PV generator to a fraction of the load nodes.

    Each PV unit mirrors its host load's node/phases and is rated at half the
    load's nameplate active power, tagged ``consumer_type="pv"`` so scenario
    selectors can target it. The harmonic content of the PV inverters is
    supplied by the scenario, so no stored spectrum is attached here. Returns
    the number of PV systems added.
    """
    from pgml.schemas.grid_schema import Generator, Load

    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    next_id = max((a.id for a in grid.appliances), default=0) + 1
    added = 0
    for k, ld in enumerate(loads):
        if (k % max(1, round(1.0 / fraction))) != 0:
            continue
        grid.appliances.append(
            Generator(
                id=next_id,
                name=f"pv_{ld.id}",
                node=ld.node,
                phases=ld.phases,
                p_nom_w=0.5 * float(ld.p_nom_w),
                q_nom_var=0.0,
                consumer_type="pv",
            )
        )
        next_id += 1
        added += 1
    return added


def synthetic_feeder(
    n_nodes: int,
    *,
    n_feeders: int = 8,
    segment_m: float = 200.0,
    total_load_w: float = 5.0e6,
    power_factor: float = 0.95,
    u_rated_v: float = 20.0e3,
    base_frequency_hz: float = 50.0,
    tie_switches: int = 0,
):
    """A synthetic 3-phase radial MV feeder of arbitrary size (schema-only, no deps).

    Node 0 is the station bus holding the :class:`~pgml.schemas.grid_schema.Source`
    (Thevenin ~250 MVA short-circuit strength); the remaining ``n_nodes - 1`` nodes
    are distributed over ``n_feeders`` radial chains of explicit-R/L/C lines
    (typical 20 kV overhead-line constants), each non-station node carrying an equal
    share of ``total_load_w`` as a constant-power WYE load at the given
    ``power_factor``. The result converges under the current-injection fixed point
    across sizes (light per-node loading by construction) and its node-phase system
    has ``3 * n_nodes`` rows — the scaling knob for solver benchmarks.

    ``tie_switches`` adds that many NORMALLY-OPEN tie switches between the far ends
    of adjacent feeders (ids ``30000 + k``) — closed/open combinations of these are
    the canonical switch-state batching study. They are open by default so the base
    grid stays radial and fully energized.

    Sizes below ``1 + n_feeders`` shrink the feeder count to ``n_nodes - 1``.
    """
    from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source, Switch

    if n_nodes < 2:
        raise ValueError("synthetic_feeder needs at least 2 nodes (station + 1).")
    abc = (Phase.A, Phase.B, Phase.C)
    n_feeders = max(1, min(n_feeders, n_nodes - 1))

    nodes = [Node(id=i, u_rated_v=u_rated_v, phases=abc) for i in range(n_nodes)]

    # Typical 20 kV overhead line constants (per meter), mutually coupled 3x3.
    r = [[0.3e-3 if i == j else 0.05e-3 for j in range(3)] for i in range(3)]
    ell = [[1.15e-6 if i == j else 0.4e-6 for j in range(3)] for i in range(3)]
    c = [[1.0e-11 if i == j else -2.0e-12 for j in range(3)] for i in range(3)]

    # Assign node i (i >= 1) to feeder (i-1) % n_feeders; chain order by index.
    heads = [0] * n_feeders  # current end node of each feeder chain
    tails: list[int] = []
    branches = []
    for i in range(1, n_nodes):
        f = (i - 1) % n_feeders
        branches.append(
            Line(
                id=10000 + i,
                from_node=heads[f],
                to_node=i,
                from_phases=abc,
                to_phases=abc,
                length_m=segment_m,
                series_resistance_ohm_per_m=r,
                series_inductance_h_per_m=ell,
                shunt_capacitance_f_per_m=c,
            )
        )
        heads[f] = i
    tails = heads

    for k in range(min(tie_switches, n_feeders - 1)):
        branches.append(
            Switch(
                id=30000 + k,
                from_node=tails[k],
                to_node=tails[k + 1],
                from_phases=abc,
                to_phases=abc,
                closed=False,
                resistance_ohm=1.0e-3,
            )
        )

    z_diag = [[0.16 if i == j else 0.0 for j in range(3)] for i in range(3)]
    l_diag = [[5.0e-3 if i == j else 0.0 for j in range(3)] for i in range(3)]
    u_ln = u_rated_v / math.sqrt(3.0)
    appliances = [
        Source(
            id=1,
            node=0,
            phases=abc,
            u_ref_v=(u_ln, u_ln, u_ln),
            u_angle_deg=(0.0, -120.0, 120.0),
            resistance_ohm=z_diag,
            inductance_h=l_diag,
        )
    ]
    p_node = total_load_w / (n_nodes - 1)
    q_node = p_node * math.tan(math.acos(power_factor))
    for i in range(1, n_nodes):
        appliances.append(
            Load(
                id=20000 + i,
                node=i,
                phases=abc,
                p_nom_w=p_node,
                q_nom_var=q_node,
            )
        )
    return Grid(
        base_frequency_hz=base_frequency_hz,
        nodes=nodes,
        branches=branches,
        appliances=appliances,
    )


def se_benchmark_scenario_config(grid, *, n_samples: int, seed: int):
    """The canonical randomized state-estimation benchmark sampling recipe.

    Sobol over: a per-phase-independent load apparent-power scale (asymmetric
    demand), a per-load harmonic current spectrum as a fraction of the EN 50160
    limit, and — when the grid carries PV (:func:`add_pv_systems`) — one SHARED
    irradiance scale for all PV plus a per-inverter harmonic signature. The
    single source of the recipe: the dataset-generation example and the pgl test
    fixtures both build from here, so what the tests train on cannot silently
    drift from what the documented benchmark generates.
    """
    from pgml.schemas.grid_schema import Generator
    from pgml.scenarios import ParameterSpec, ScenarioConfig, Selector, Uniform

    has_pv = any(isinstance(a, Generator) for a in grid.appliances)
    params = [
        # Loads: per-phase-independent apparent-power scale -> asymmetric demand.
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.3, high=1.0),
            field="pq",
            mode="scale",
            per="each",
            symmetry="independent",
        ),
        # Loads: per-load harmonic current spectrum (fraction of the EN 50160 limit).
        ParameterSpec(
            name="load_spectrum",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.0, high=1.0),
            field="h_mag",
            orders=LOAD_HARMONIC_ORDERS,
            harmonic_reference="en50160",
            per="each",
        ),
    ]
    if has_pv:
        params += [
            # PV: ONE shared irradiance factor scales every PV together (per="shared").
            ParameterSpec(
                name="pv_scale",
                selector=Selector(component="generator", consumer_type="pv"),
                distribution=Uniform(low=0.0, high=1.0),
                field="pq",
                mode="scale",
                per="shared",
            ),
            # PV: per-inverter harmonic signature (each inverter independent).
            ParameterSpec(
                name="pv_spectrum",
                selector=Selector(component="generator", consumer_type="pv"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                orders=PV_HARMONIC_ORDERS,
                harmonic_reference="en50160",
                per="each",
            ),
        ]
    return ScenarioConfig(
        n_samples=n_samples, seed=seed, method="sobol", parameters=params
    )


__all__ = [
    "CONVERTER_SPECTRUM",
    "LOAD_HARMONIC_ORDERS",
    "PV_HARMONIC_ORDERS",
    "ieee33_geometry_grid",
    "cigre_lv_full_grid",
    "cigre_lv_geometry_grid",
    "add_pv_systems",
    "se_benchmark_scenario_config",
    "synthetic_feeder",
]
