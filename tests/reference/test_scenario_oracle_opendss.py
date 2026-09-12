"""Oracle test: the OpenDSS SCENARIO ORACLE (independent full-circuit export + batch runs).

Validates ``pgml.evaluation.oracles.opendss_scenario_oracle`` end to end on a small
3-phase radial feeder (:func:`pgml.grids.synthetic_feeder`, explicit R/L/C matrix lines,
WYE loads, a diagonal-Thevenin Source — no transformer/geometry/delta, keeping the
exporter's PRIMARY path exercised tightly):

- the exported DSS circuit solves and its ``Load``/``Spectrum`` mapping is correct
  (:class:`~pgml.schemas.grid_schema.LoadModel` -> DSS ``Model=``);
- a :class:`~pgml.scenarios.SampledScenarios` batch (snapshot AND node-coherent) round-trips
  through :func:`~pgml.evaluation.oracles.opendss_scenario_oracle.write_opendss_dataset` /
  :func:`pgml.scenarios.read_dataset` with the OpenDSS provenance stamped;
- ``mode="matched"`` agrees with pgml's own solver to near machine precision (measured
  ~1e-8 to 1e-10 relative — see the docstrings below for the exact figures pinned);
- ``mode="default"`` is pinned on the NON-TRIPLEN orders only: OpenDSS's own device model
  (``NeglectLoadY=No`` with ``%SeriesRL=50``) is pgml's default, so those orders now agree
  to ~8e-9 relative, while OpenDSS's imperial-calibrated earth-return zero-sequence term
  still dominates the TRIPLEN order on this 3-wire, no-explicit-neutral feeder (documented
  divergence, a few hundred percent);
- the exporter raises a clear :class:`~pgml.errors.ConversionError` for the documented
  refusals (conductor-geometry lines, zigzag transformer windings, an off-diagonal/coupled
  Source Thevenin impedance) rather than silently exporting a wrong circuit.
"""

from __future__ import annotations

import json

import pytest
import torch

# ---------------------------------------------------------------------------
# Optional opendssdirect guard (matches existing reference test conventions)
# ---------------------------------------------------------------------------
try:
    import opendssdirect as dss  # noqa: E402

    _OPENDSS_AVAILABLE = True
except ImportError:
    _OPENDSS_AVAILABLE = False

if not _OPENDSS_AVAILABLE:
    pytest.skip("opendssdirect not installed", allow_module_level=True)

pytestmark = pytest.mark.opendss

from pgml.errors import ConversionError  # noqa: E402
from pgml.evaluation.oracles.opendss_scenario_oracle import (  # noqa: E402
    compare_to_pgml,
    export_grid_to_opendss,
    run_opendss_scenarios,
    write_opendss_dataset,
)
from pgml.grids import synthetic_feeder  # noqa: E402
from pgml.scenarios import (  # noqa: E402
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    batch_from_values,
    read_dataset,
    sample,
)
from pgml.schemas.grid_schema import (  # noqa: E402
    ComplexTap,
    Generator,
    Grid,
    Load,
    LoadModel,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    Storage,
    Switch,
    Transformer,
    WindingConnection,
    ZipCoefficients,
)

_ORDERS = [1, 3, 5]

# Measured worst-case relative error (matched mode, snapshot + coherent batches, this
# feeder): ~3.4e-10 (fundamental) / ~1.5e-8 (harmonics). Pinned with a comfortable
# ~65x margin.
_MATCHED_REL_TOL = 1.0e-6


def _grid():
    """A small 4-node, single-feeder radial network (no transformer/geometry/delta)."""
    return synthetic_feeder(4, n_feeders=1, total_load_w=3.0e5)


def _snapshot_config() -> ScenarioConfig:
    """~4 scenarios varying load P/Q (balanced scale) and injected harmonic magnitude/phase."""
    return ScenarioConfig(
        n_samples=4,
        seed=0,
        method="sobol",
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
                field="pq",
                mode="scale",
                per="each",
            ),
            ParameterSpec(
                name="load_hmag",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.05, high=0.3),
                field="h_mag",
                mode="absolute",
                orders=[3, 5],
                per="each",
            ),
            ParameterSpec(
                name="load_hphase",
                selector=Selector(component="load"),
                distribution=Uniform(low=-90.0, high=90.0),
                field="h_phase",
                mode="absolute",
                orders=[3, 5],
                per="each",
            ),
        ],
    )


#: Batch shape of the sequence cases: 2 scenarios of 3 steps each.
_SEQ_B, _SEQ_T = 2, 3


def _sequence_batch(grid, *, profiled: bool = False):
    """A per-step sequence batch: ``[B, T]`` injections, optionally a ``[B, T]`` fundamental.

    The magnitudes and the per-step fundamental vary over BOTH axes, so a consumer that
    indexed the step axis by scenario (or dropped it) disagrees with the reference tool
    instead of quietly averaging. Built from explicit values, which is what the engine
    offers for a step axis it does not generate itself.
    """
    b, t = _SEQ_B, _SEQ_T
    load_ids = [a.id for a in grid.appliances if isinstance(a, Load) and a.in_service]
    # a distinct ramp per (scenario, step), and a distinct phase per device
    ramp = torch.linspace(0.4, 1.0, b * t, dtype=torch.float64).reshape(b, t)
    injection = {
        cid: {
            order: (
                ramp * (0.02 + 0.01 * k),
                torch.full((b, t), 15.0 * k, dtype=torch.float64),
            )
            for order in (3, 5)
        }
        for k, cid in enumerate(load_ids)
    }
    nameplate = {a.id: float(a.p_nom_w) for a in grid.appliances if a.id in load_ids}
    p_w = {cid: ramp * nameplate[cid] for cid in load_ids} if profiled else None
    return batch_from_values(
        grid,
        n_samples=b,
        n_steps=t,
        p_w=p_w,
        harmonic_injection=injection,
        shared_samples={"time_s": torch.arange(t, dtype=torch.float64) * 3600.0},
    )


# ---------------------------------------------------------------------------
# Exporter smoke + coverage
# ---------------------------------------------------------------------------
def test_export_solves_and_load_model_maps_to_dss():
    """A CONST_IMPEDANCE load exports as OpenDSS ``Model=2`` (the reverse of to_grid's map)."""
    abc = (Phase.A,)
    nodes = [Node(id=0, u_rated_v=230.0, phases=abc)]
    src = Source(
        id=1,
        node=0,
        phases=abc,
        u_ref_v=[230.0],
        u_angle_deg=[0.0],
        resistance_ohm=[[0.05]],
        inductance_h=[[1.0e-3]],
    )
    ld = Load(
        id=2,
        node=0,
        phases=abc,
        p_nom_w=1000.0,
        q_nom_var=200.0,
        load_model=LoadModel.CONST_IMPEDANCE,
    )
    grid = Grid(nodes=[*nodes], branches=[], appliances=[src, ld])

    circuit = export_grid_to_opendss(grid)
    assert circuit.mode == "matched"
    name = circuit.loads[2].elements[Phase.A]
    dss.Loads.Name(name)
    assert dss.Loads.Model() == 2


def test_export_refuses_conductor_geometry_line():
    from pgml.schemas.grid_schema import ConductorPlacement, Line, LineGeometry

    abc = (Phase.A,)
    nodes = [
        Node(id=0, u_rated_v=230.0, phases=abc),
        Node(id=1, u_rated_v=230.0, phases=abc),
    ]
    geom = LineGeometry(
        conductors=[
            ConductorPlacement(
                phase=Phase.A,
                x_m=0.0,
                y_m=8.0,
                gmr_m=0.004,
                radius_m=0.005,
                r_dc_ohm_per_m=3.0e-4,
            )
        ]
    )
    ln = Line(
        id=1,
        from_node=0,
        to_node=1,
        from_phases=abc,
        to_phases=abc,
        length_m=100.0,
        conductor_geometry=geom,
    )
    src = Source(
        id=2,
        node=0,
        phases=abc,
        u_ref_v=[230.0],
        u_angle_deg=[0.0],
        resistance_ohm=[[0.05]],
        inductance_h=[[1.0e-3]],
    )
    grid = Grid(nodes=nodes, branches=[ln], appliances=[src])
    with pytest.raises(ConversionError, match="conductor_geometry"):
        export_grid_to_opendss(grid)


def test_export_refuses_zigzag_transformer():
    abc = (Phase.A, Phase.B, Phase.C)
    nodes = [
        Node(id=0, u_rated_v=20000.0, phases=abc),
        Node(id=1, u_rated_v=400.0, phases=abc),
    ]
    diag = lambda v: [[v if i == j else 0.0 for j in range(3)] for i in range(3)]  # noqa: E731
    src = Source(
        id=1,
        node=0,
        phases=abc,
        u_ref_v=[20000.0 / 1.7320508] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=diag(0.5),
        inductance_h=diag(5.0e-3),
    )
    t = Transformer(
        id=2,
        from_node=0,
        to_node=1,
        from_phases=abc,
        to_phases=abc,
        s_rated_va=1.0e5,
        u_rated_from_v=20000.0,
        u_rated_to_v=400.0,
        from_connection=WindingConnection.DELTA,
        to_connection=WindingConnection.ZIGZAG_GROUNDED,
        series_resistance_ohm=0.01,
        series_inductance_h=1.0e-4,
    )
    grid = Grid(nodes=nodes, branches=[t], appliances=[src])
    with pytest.raises(ConversionError, match="zigzag"):
        export_grid_to_opendss(grid)


def test_export_refuses_offdiagonal_source_impedance():
    abc = (Phase.A, Phase.B, Phase.C)
    nodes = [Node(id=0, u_rated_v=400.0, phases=abc)]
    r = [[0.1, 0.02, 0.0], [0.02, 0.1, 0.0], [0.0, 0.0, 0.1]]
    ell = [[1.0e-3 if i == j else 0.0 for j in range(3)] for i in range(3)]
    src = Source(
        id=1,
        node=0,
        phases=abc,
        u_ref_v=[230.0] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=r,
        inductance_h=ell,
    )
    ld = Load(id=2, node=0, phases=abc, p_nom_w=1000.0, q_nom_var=0.0)
    grid = Grid(nodes=nodes, branches=[], appliances=[src, ld])
    with pytest.raises(ConversionError, match="off-diagonal"):
        export_grid_to_opendss(grid)


# ---------------------------------------------------------------------------
# Dataset round-trip + provenance
# ---------------------------------------------------------------------------
def test_dataset_roundtrip_and_provenance_snapshot(tmp_path):
    grid = _grid()
    sampled = sample(grid, _snapshot_config())
    result = run_opendss_scenarios(
        grid, sampled, harmonic_orders=_ORDERS, mode="matched"
    )
    assert result.v.shape == (4, len(_ORDERS), result.index.size)
    assert result.converged

    out = write_opendss_dataset(
        grid, sampled, tmp_path / "ds_snapshot", harmonic_orders=_ORDERS, mode="matched"
    )
    meta = json.loads((out / "meta.json").read_text())
    assert meta["engine"] == "opendss"
    assert meta["oracle_mode"] == "matched"
    assert meta["opendssdirect_version"]
    assert (
        "OpenDSS" in meta["opendss_engine_version"]
        or "DSS" in meta["opendss_engine_version"]
    )
    assert meta["converged"] is True

    loaded = read_dataset(out)
    assert torch.equal(loaded.node_ids, result.index.node_ids)
    assert torch.equal(loaded.phase_codes, result.index.phase_codes)
    assert torch.allclose(loaded.v, result.v)


def test_dataset_roundtrip_and_provenance_sequence(tmp_path):
    grid = _grid()
    sampled = _sequence_batch(grid)
    result = run_opendss_scenarios(
        grid, sampled, harmonic_orders=_ORDERS, mode="matched"
    )
    assert result.v.shape == (_SEQ_B, _SEQ_T, len(_ORDERS), result.index.size)

    out = write_opendss_dataset(
        grid, sampled, tmp_path / "ds_sequence", harmonic_orders=_ORDERS, mode="matched"
    )
    meta = json.loads((out / "meta.json").read_text())
    assert meta["engine"] == "opendss"
    assert meta["oracle_mode"] == "matched"

    loaded = read_dataset(out)
    assert torch.allclose(loaded.v, result.v)


# ---------------------------------------------------------------------------
# Numeric cross-validation: matched mode (tight, measured tolerance)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_matched_mode_snapshot_agrees_with_pgml(tmp_path):
    grid = _grid()
    sampled = sample(grid, _snapshot_config())
    report = compare_to_pgml(
        grid, sampled, harmonic_orders=_ORDERS, mode="matched", out_dir=tmp_path
    )
    assert report["opendss_converged"] and report["pgml_converged"]
    assert set(report["orders"]) == set(_ORDERS)
    for h, stats in report["per_order"].items():
        assert stats["rel_max"] < _MATCHED_REL_TOL, (
            f"order {h}: matched-mode relative error {stats['rel_max']:.3e} exceeds "
            f"{_MATCHED_REL_TOL:.0e} -- matched mode should isolate near-machine-precision "
            "numeric agreement (NeglectLoadY=Yes + Rg=Xg=0 on both sides)."
        )
    assert (tmp_path / "opendss_comparison.json").is_file()
    assert (tmp_path / "opendss_comparison.csv").is_file()


@pytest.mark.slow
def test_matched_mode_sequence_agrees_with_pgml():
    """A ``[B, T]`` harmonic injection against a ``[B]`` fundamental agrees with OpenDSS."""
    grid = _grid()
    sampled = _sequence_batch(grid)
    report = compare_to_pgml(grid, sampled, harmonic_orders=_ORDERS, mode="matched")
    assert report["opendss_converged"] and report["pgml_converged"]
    for h, stats in report["per_order"].items():
        assert stats["rel_max"] < _MATCHED_REL_TOL, (
            f"order {h}: matched-mode (sequence) relative error {stats['rel_max']:.3e} "
            f"exceeds {_MATCHED_REL_TOL:.0e}."
        )


# ---------------------------------------------------------------------------
# Numeric cross-validation: default mode (report only, documented divergence)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_default_mode_report_generated_and_diverges_as_documented():
    """``mode="default"`` leaves OpenDSS's own settings: ONE divergence source is left.

    The TRIPLEN order (h=3, zero-sequence-dominated) is driven by OpenDSS's own
    imperial-unit-calibrated earth-return ``Rg``/``Xg`` line correction (left at its
    defaults in this mode) on a 3-wire, no-explicit-neutral feeder -- pgml's non-geometry
    harmonic line models carry no such term at all (``docs/pgml/modeling/conventions.md``
    §8, "the earth-return calibration gotcha"). Measured on this feeder: 2.4 relative
    (a few hundred percent) at h=3. That is expected and is NOT a bug.

    The NON-TRIPLEN order is now pinned: OpenDSS's default device model
    (``NeglectLoadY=No``, ``%SeriesRL=50``) is pgml's own default, so h=5 agrees to
    8.3e-09 relative -- against 5.0e-03 with the pure current-source model, which is what
    this test used to characterise as the second divergence source.

    The fundamental (h=1) is mode-independent -- ``NeglectLoadY``/``Rg``/``Xg`` only
    affect the ``Solve mode=harmonics`` path, never the nonlinear snapshot solve.
    """
    grid = _grid()
    sampled = sample(grid, _snapshot_config())
    report = compare_to_pgml(grid, sampled, harmonic_orders=_ORDERS, mode="default")
    assert report["opendss_converged"] and report["pgml_converged"]
    assert set(report["orders"]) == set(_ORDERS)
    for stats in report["per_order"].values():
        assert stats["ref_rms_v"] > 0.0  # the report is populated, not vacuous
    # the fundamental is mode-independent (NeglectLoadY/Rg/Xg only affect harmonics mode)
    assert report["per_order"][1]["rel_max"] < _MATCHED_REL_TOL
    # the non-triplen order: same device model on both sides, only Rg/Xg differ and they
    # barely touch the positive-sequence path.
    assert report["per_order"][5]["rel_max"] < _MATCHED_REL_TOL
    # the triplen order: the earth-return term dominates, as documented.
    assert report["per_order"][3]["rel_max"] > 0.1


# ---------------------------------------------------------------------------
# Devices: Generator/Storage/ShuntAppliance/Switch (tight) + ZIP/CONST_CURRENT
# (a documented, bounded, irreducible divergence -- see the module docstring)
# ---------------------------------------------------------------------------
_ABC = (Phase.A, Phase.B, Phase.C)


def _devices_grid() -> Grid:
    """A 3-phase feeder with a closed Switch, a Capacitor+Reactor ShuntAppliance, a
    PV Generator, and a Storage unit -- everything the campaign found clean once
    exported as a negative-kW Load with Vminpu/Vmaxpu unbounded (see the module
    docstring's "Exporter coverage")."""
    diag = lambda v: [[v if i == j else 0.0 for j in range(3)] for i in range(3)]  # noqa: E731
    nodes = [Node(id=i, u_rated_v=400.0, phases=_ABC) for i in range(3)]
    src = Source(
        id=1,
        node=0,
        phases=_ABC,
        u_ref_v=[230.94] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=diag(0.05),
        inductance_h=diag(1.0e-4),
    )
    sw = Switch(
        id=2,
        from_node=0,
        to_node=1,
        from_phases=_ABC,
        to_phases=_ABC,
        closed=True,
        resistance_ohm=1.0e-4,
        inductance_h=0.0,
    )
    ln = Switch(  # a second closed switch further downstream (exercise more than one)
        id=3,
        from_node=1,
        to_node=2,
        from_phases=_ABC,
        to_phases=_ABC,
        closed=True,
        resistance_ohm=2.0e-4,
        inductance_h=0.0,
    )
    ld = Load(id=4, node=2, phases=_ABC, p_nom_w=50.0e3, q_nom_var=15.0e3)
    gen = Generator(
        id=5,
        node=2,
        phases=_ABC,
        p_nom_w=20.0e3,
        q_nom_var=0.0,
        consumer_type="pv",
    )
    storage = Storage(id=6, node=2, phases=_ABC, p_nom_w=10.0e3, q_nom_var=2.0e3)
    shunt = ShuntAppliance(
        id=7,
        node=2,
        phases=_ABC,
        conductance_s=(1.0e-6,) * 3,
        capacitance_f=(2.0e-6,) * 3,
    )
    return Grid(
        nodes=nodes, branches=[sw, ln], appliances=[src, ld, gen, storage, shunt]
    )


@pytest.mark.slow
def test_devices_switch_generator_storage_shunt_matched_tight(tmp_path, monkeypatch):
    """Switch/Generator/Storage/ShuntAppliance all agree with pgml near machine precision.

    Run with ``appliance.harmonic_shunt.generation_model: load_style``, the policy under
    which a Generator / Storage carries the same operating-point shunt a Load does. That
    is the only policy a matched-mode export can reproduce for a generation device: it
    writes one as a negative-kW DSS ``Load``, whose shunt OpenDSS always derives from that
    (negative) power, and ``NeglectLoadY`` is a global option. Under the shipped
    ``none`` policy the export refuses the combination by name, which
    ``tests/reference/test_opendss_load_shunt.py`` covers together with the size of the
    difference.
    """
    import yaml

    from pgml import defaults

    data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
    data["appliance"]["harmonic_shunt"]["generation_model"]["value"] = "load_style"
    path = tmp_path / "generation_load_style.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PGML_DEFAULTS", str(path))
    defaults.reload(str(path))
    try:
        _devices_matched_comparison()
    finally:
        monkeypatch.delenv("PGML_DEFAULTS", raising=False)
        defaults.reload()


def _devices_matched_comparison() -> None:
    """The matched-mode comparison of the multi-device grid (see the test above)."""
    grid = _devices_grid()
    orders = [3, 5]
    cfg = ScenarioConfig(
        n_samples=3,
        seed=1,
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="gen_pq",
                selector=Selector(component="generator"),
                distribution=Uniform(low=0.5, high=1.0),
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="h_mag",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.2, high=0.6),
                field="h_mag",
                mode="absolute",
                orders=orders,
                harmonic_reference="iec61000-3-2",
            ),
            ParameterSpec(
                name="h_phase",
                selector=Selector(component="load"),
                distribution=Uniform(low=-180.0, high=180.0),
                field="h_phase",
                mode="absolute",
                orders=orders,
            ),
        ],
    )
    sampled = sample(grid, cfg)
    report = compare_to_pgml(
        grid, sampled, harmonic_orders=[1, *orders], mode="matched"
    )
    assert report["opendss_converged"] and report["pgml_converged"]
    for h, stats in report["per_order"].items():
        assert stats["rel_max"] < _MATCHED_REL_TOL, (
            f"order {h}: devices-grid matched-mode error {stats['rel_max']:.3e} exceeds "
            f"{_MATCHED_REL_TOL:.0e}."
        )


@pytest.mark.slow
def test_const_current_zip_harmonic_matched_tight():
    """ZIP-model loads with harmonic content agree at the tight matched-mode floor:
    the solver anchors each device's spectrum to its MODEL-CONSISTENT fundamental
    current (S_eff at the converged voltage), matching OpenDSS's per-model
    fundamental current.

    Solved with ``load_shunt="none"`` on both sides. The harmonic device shunt is the
    one quantity where a voltage-dependent load model does NOT agree: pgml derives the
    shunt from the power the device REALLY draws at the converged voltage, OpenDSS from
    the SPECIFIED kW/kvar whatever the load model
    (``Load.pas``'s ``Yeq`` comes from ``SetNominalLoad``). The resulting deviation is
    measured and bounded by
    ``tests/reference/test_opendss_load_shunt.py::test_zip_load_shunt_divergence_is_bounded``.
    """
    grid = _devices_grid()
    grid = grid.model_copy(deep=True)
    for i, a in enumerate(grid.appliances):
        if isinstance(a, Load):
            grid.appliances[i] = a.model_copy(
                update={
                    "load_model": LoadModel.ZIP,
                    "zip_coefficients": ZipCoefficients(
                        z_p=0.3, i_p=0.3, p_p=0.4, z_q=0.3, i_q=0.3, p_q=0.4
                    ),
                }
            )
    orders = [3, 5]
    cfg = ScenarioConfig(
        n_samples=3,
        seed=1,
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="h_mag",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.2, high=0.6),
                field="h_mag",
                mode="absolute",
                orders=orders,
                harmonic_reference="iec61000-3-2",
            ),
            ParameterSpec(
                name="h_phase",
                selector=Selector(component="load"),
                distribution=Uniform(low=-180.0, high=180.0),
                field="h_phase",
                mode="absolute",
                orders=orders,
            ),
        ],
    )
    sampled = sample(grid, cfg)
    report = compare_to_pgml(
        grid, sampled, harmonic_orders=[1, *orders], mode="matched", load_shunt="none"
    )
    assert report["per_order"][1]["rel_max"] < _MATCHED_REL_TOL
    for h in orders:
        assert report["per_order"][h]["rel_max"] < _MATCHED_REL_TOL


# ---------------------------------------------------------------------------
# DELTA ShuntAppliance bank: pgml delta bank -> DSS matches pgml (both ways)
# ---------------------------------------------------------------------------
def _delta_shunt_grid() -> Grid:
    """A 2-node feeder with a balanced 3-phase DELTA capacitor bank at the load bus."""
    nodes = [Node(id=i, u_rated_v=400.0, phases=_ABC) for i in range(2)]
    src = Source(
        id=1,
        node=0,
        phases=_ABC,
        u_ref_v=[230.94] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[1.0e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    from pgml.schemas.grid_schema import Line

    ln = Line(
        id=2,
        from_node=0,
        to_node=1,
        from_phases=_ABC,
        to_phases=_ABC,
        length_m=100.0,
        series_resistance_ohm_per_m=[
            [1.0e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        series_inductance_h_per_m=[
            [1.0e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
    )
    ld = Load(id=3, node=1, phases=_ABC, p_nom_w=40.0e3, q_nom_var=10.0e3)
    shunt = ShuntAppliance(
        id=4,
        node=1,
        phases=_ABC,
        conductance_s=(0.0,) * 3,
        capacitance_f=(2.0e-5,) * 3,  # balanced delta capacitor leg
        connection=WindingConnection.DELTA,
    )
    return Grid(nodes=nodes, branches=[ln], appliances=[src, ld, shunt])


@pytest.mark.slow
def test_delta_shunt_bank_matched_tight():
    """A pgml DELTA :class:`ShuntAppliance` exports to a ``conn=delta`` DSS Capacitor
    and the exported circuit agrees with pgml near machine precision."""
    grid = _delta_shunt_grid()
    orders = [3, 5]
    cfg = ScenarioConfig(
        n_samples=3,
        seed=2,
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="h_mag",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.2, high=0.6),
                field="h_mag",
                mode="absolute",
                orders=orders,
                harmonic_reference="iec61000-3-2",
            ),
            ParameterSpec(
                name="h_phase",
                selector=Selector(component="load"),
                distribution=Uniform(low=-180.0, high=180.0),
                field="h_phase",
                mode="absolute",
                orders=orders,
            ),
        ],
    )
    sampled = sample(grid, cfg)
    report = compare_to_pgml(
        grid, sampled, harmonic_orders=[1, *orders], mode="matched"
    )
    assert report["opendss_converged"] and report["pgml_converged"]
    for h, stats in report["per_order"].items():
        assert stats["rel_max"] < _MATCHED_REL_TOL, (
            f"order {h}: delta-shunt matched-mode error {stats['rel_max']:.3e} exceeds "
            f"{_MATCHED_REL_TOL:.0e}."
        )


def test_delta_shunt_export_emits_delta_capacitor():
    """The exporter writes a ``conn=delta`` Capacitor for a DELTA ShuntAppliance
    (a wye export would carry a different admittance and fail the parity above)."""
    export_grid_to_opendss(_delta_shunt_grid())
    # export_grid_to_opendss builds into the module-global opendssdirect engine.
    assert any("sha4" in n.lower() for n in dss.Capacitors.AllNames())
    dss.Text.Command("? Capacitor.sha4c.conn")
    assert dss.Text.Result().strip().lower() == "delta"


# ---------------------------------------------------------------------------
# Sequence batch with a per-step [B, T] operating point
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_sequence_with_per_step_fundamental_matched_tight():
    """A per-STEP ``[B, T]`` operating point agrees with pgml order by order."""
    grid = _grid()
    sampled = _sequence_batch(grid, profiled=True)
    op = sampled.operating_point
    assert any(
        hasattr(v.get("p_w"), "ndim") and v["p_w"].ndim == 2 for v in op.values()
    ), "the batch should carry a per-step [B, T] fundamental"

    report = compare_to_pgml(grid, sampled, harmonic_orders=[1, 3, 5], mode="matched")
    assert report["opendss_converged"] and report["pgml_converged"]
    for h, stats in report["per_order"].items():
        assert stats["rel_max"] < _MATCHED_REL_TOL, (
            f"order {h}: per-step fundamental matched-mode error {stats['rel_max']:.3e} "
            f"exceeds {_MATCHED_REL_TOL:.0e} -- the per-step [B, T] operating point "
            "slicing must index by step, not just by scenario."
        )


# ---------------------------------------------------------------------------
# Single-phase (positive-sequence equivalent) transformer
# ---------------------------------------------------------------------------
def _single_phase_transformer_grid(shift_deg: float) -> Grid:
    abc = (Phase.A,)
    nodes = [
        Node(id=0, u_rated_v=20000.0, phases=abc),
        Node(id=1, u_rated_v=400.0, phases=abc),
    ]
    src = Source(
        id=1,
        node=0,
        phases=abc,
        u_ref_v=[20000.0],
        u_angle_deg=[0.0],
        resistance_ohm=[[0.05]],
        inductance_h=[[1.0e-4]],
    )
    # A DELTA/WYE_GROUNDED pairing needs an ODD clock (each shifting winding contributes
    # +-30 deg); a zero shift therefore uses a matching WYE_GROUNDED/WYE_GROUNDED (Yy0)
    # pairing instead (pgml's own vector-group parity check rejects clock=0 on a Dyn
    # pairing regardless of phase count).
    from_conn = (
        WindingConnection.WYE_GROUNDED if shift_deg == 0.0 else WindingConnection.DELTA
    )
    t = Transformer(
        id=2,
        from_node=0,
        to_node=1,
        from_phases=abc,
        to_phases=abc,
        s_rated_va=5.0e5,
        u_rated_from_v=20000.0,
        u_rated_to_v=400.0,
        from_connection=from_conn,
        to_connection=WindingConnection.WYE_GROUNDED,
        series_resistance_ohm=0.0032,
        series_inductance_h=4.07e-5,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=shift_deg),
    )
    ld = Load(id=3, node=1, phases=abc, p_nom_w=1.0e5, q_nom_var=3.0e4)
    return Grid(nodes=nodes, branches=[t], appliances=[src, ld])


@pytest.mark.slow
def test_single_phase_transformer_zero_shift_matched_tight():
    """A zero-shift (plain ratio) 1-phase transformer -- the IEEE-33-style case."""
    grid = _single_phase_transformer_grid(shift_deg=0.0)
    cfg = ScenarioConfig(
        n_samples=3,
        seed=1,
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            ),
        ],
    )
    sampled = sample(grid, cfg)
    report = compare_to_pgml(grid, sampled, harmonic_orders=[1], mode="matched")
    assert report["per_order"][1]["rel_max"] < _MATCHED_REL_TOL


def test_single_phase_transformer_nonzero_shift_refuses():
    """A NONZERO-shift 1-phase transformer is refused -- OpenDSS has no delta/LeadLag
    mechanism at phases=1 (see the module docstring's refusal list)."""
    grid = _single_phase_transformer_grid(shift_deg=30.0)
    with pytest.raises(ConversionError, match="phases=1"):
        export_grid_to_opendss(grid)


def test_four_wire_return_path_matched_tight():
    """WYE loads on a 4-wire (ABCN) feeder export with their return conductor:
    an "auto"/"neutral" appliance becomes a phase-to-neutral DSS load, a
    "ground" appliance keeps DSS's implicit ground — both match pgml."""
    abcn = (Phase.A, Phase.B, Phase.C, Phase.N)
    abc = (Phase.A, Phase.B, Phase.C)
    from pgml.schemas.grid_schema import Line

    def mat(diag, off):
        return [[diag if i == j else off for j in range(4)] for i in range(4)]

    nodes = [
        Node(id=0, u_rated_v=400.0, phases=abcn),
        Node(id=1, u_rated_v=400.0, phases=abcn),
    ]
    line = Line(
        id=10,
        from_node=0,
        to_node=1,
        from_phases=abcn,
        to_phases=abcn,
        length_m=300.0,
        series_resistance_ohm_per_m=mat(4.0e-4, 5.0e-5),
        series_inductance_h_per_m=mat(9.0e-7, 3.0e-7),
        shunt_capacitance_f_per_m=mat(0.0, 0.0),
    )
    src = Source(
        id=1,
        node=0,
        phases=abc,
        u_ref_v=[400.0 / 1.7320508] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[3.0e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    ld_neutral = Load(
        id=20, node=1, phases=(Phase.A,), p_nom_w=8.0e3, q_nom_var=1.5e3
    )  # auto -> neutral return on an ABCN node
    ld_ground = Load(
        id=21,
        node=1,
        phases=(Phase.B,),
        p_nom_w=6.0e3,
        q_nom_var=1.0e3,
        return_path="ground",
    )
    # Station neutral grounding: without a ground reference the floating neutral
    # subsystem is singular (a real 4-wire feeder grounds N at the transformer).
    n_ground = ShuntAppliance(
        id=2, node=0, phases=(Phase.N,), conductance_s=(1.0e4,), capacitance_f=(0.0,)
    )
    grid = Grid(
        nodes=nodes, branches=[line], appliances=[src, ld_neutral, ld_ground, n_ground]
    )

    cfg = ScenarioConfig(
        n_samples=3,
        seed=2,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.6, high=1.4),
                field="pq",
                mode="scale",
                per="each",
            ),
            ParameterSpec(
                name="h_mag",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.1, high=0.4),
                field="h_mag",
                mode="absolute",
                orders=[3, 5],
                per="each",
            ),
            ParameterSpec(
                name="h_phase",
                selector=Selector(component="load"),
                distribution=Uniform(low=-90.0, high=90.0),
                field="h_phase",
                mode="absolute",
                orders=[3, 5],
                per="each",
            ),
        ],
    )
    sampled = sample(grid, cfg)
    report = compare_to_pgml(
        grid, sampled, harmonic_orders=[1, 3, 5], mode="matched", symmetry="asymmetric"
    )
    for h, st in report["per_order"].items():
        assert st["rel_max"] < _MATCHED_REL_TOL, (h, st)
