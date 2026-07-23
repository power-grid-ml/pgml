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
- ``mode="default"`` merely produces a valid report; NOT pinned tight (documented
  divergence: OpenDSS's imperial-calibrated earth-return zero-sequence term dominates the
  TRIPLEN order on this 3-wire, no-explicit-neutral feeder, and its load Norton shunk
  — absent from pgml's harmonic model entirely — perturbs the non-triplen order);
- the exporter raises a clear :class:`~pgml.errors.ConversionError` for the documented
  refusals (conductor-geometry lines, zigzag transformer windings, an off-diagonal/coupled
  Source Thevenin impedance) rather than silently exporting a wrong circuit.
"""

from __future__ import annotations

import json

import pytest
import torch

import opendssdirect as dss  # noqa: E402

from pgml.errors import ConversionError
from pgml.evaluation.oracles.opendss_scenario_oracle import (
    compare_to_pgml,
    export_grid_to_opendss,
    run_opendss_scenarios,
    write_opendss_dataset,
)
from pgml.grids import synthetic_feeder
from pgml.scenarios import (
    CoherentSpectrumConfig,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    read_dataset,
    sample,
    sample_coherent_spectra,
)
from pgml.schemas.grid_schema import (
    Grid,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)

pytestmark = pytest.mark.opendss

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


def _coherent_config() -> CoherentSpectrumConfig:
    """A small node-coherent batch: B=2 scenarios, T=3 steps."""
    return CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=3,
        n_scenarios=2,
        n_modes=2,
        seed=0,
        harmonic_reference=None,  # absolute pu (clamped to 1.0); no IEC/EN reference needed
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


def test_dataset_roundtrip_and_provenance_coherent(tmp_path):
    grid = _grid()
    sampled = sample_coherent_spectra(grid, _coherent_config())
    result = run_opendss_scenarios(
        grid, sampled, harmonic_orders=_ORDERS, mode="matched"
    )
    assert result.v.shape == (2, 3, len(_ORDERS), result.index.size)

    out = write_opendss_dataset(
        grid, sampled, tmp_path / "ds_coherent", harmonic_orders=_ORDERS, mode="matched"
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
def test_matched_mode_coherent_agrees_with_pgml():
    grid = _grid()
    sampled = sample_coherent_spectra(grid, _coherent_config())
    report = compare_to_pgml(grid, sampled, harmonic_orders=_ORDERS, mode="matched")
    assert report["opendss_converged"] and report["pgml_converged"]
    for h, stats in report["per_order"].items():
        assert stats["rel_max"] < _MATCHED_REL_TOL, (
            f"order {h}: matched-mode (coherent) relative error {stats['rel_max']:.3e} "
            f"exceeds {_MATCHED_REL_TOL:.0e}."
        )


# ---------------------------------------------------------------------------
# Numeric cross-validation: default mode (report only, documented divergence)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_default_mode_report_generated_and_diverges_as_documented():
    """``mode="default"`` is NOT pinned tight -- it characterizes, not validates, drift.

    Two independent divergence sources are expected and are NOT bugs:
    1. The TRIPLEN order (h=3, zero-sequence-dominated) is driven by OpenDSS's own
       imperial-unit-calibrated earth-return ``Rg``/``Xg`` line correction (left at its
       defaults in this mode) on a 3-wire, no-explicit-neutral feeder -- pgml's non-geometry
       harmonic line models carry no such term at all (``docs/pgml/modeling/conventions.md``
       §8, "the earth-return calibration gotcha"). Measured on this feeder: tens to a few
       hundred percent relative error at h=3.
    2. The non-triplen order (h=5) is driven by OpenDSS's default load Norton shunt
       (``NeglectLoadY=No``), which pgml's harmonic solver does not implement at all
       (``include_load_shunt`` is hard-`False`, see ``pgml.solver.harmonic_flow``). Measured
       on this feeder: ~0.4-0.5% relative error at h=5.
    The fundamental (h=1) is UNCHANGED vs matched mode -- ``NeglectLoadY``/``Rg``/``Xg`` only
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
