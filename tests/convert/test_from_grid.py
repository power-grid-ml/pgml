"""Public Grid exporters and their explicit reduction contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pgml.schemas.grid_schema import (
    Grid,
    Generator,
    HarmonicImpedance,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    ZipCoefficients,
)


def _grid(*, load_model: LoadModel = LoadModel.CONST_POWER) -> Grid:
    omega = 2.0 * math.pi * 50.0
    coefficients = (
        ZipCoefficients(z_p=0.2, i_p=0.3, p_p=0.5, z_q=0.1, i_q=0.2, p_q=0.7)
        if load_model == LoadModel.ZIP
        else None
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=10, u_rated_v=400.0, phases=(Phase.A,)),
            Node(id=20, u_rated_v=400.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=30,
                from_node=10,
                to_node=20,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=250.0,
                series_resistance_ohm_per_m=[[4e-4]],
                series_inductance_h_per_m=[[3e-4 / omega]],
                shunt_capacitance_f_per_m=[[2e-9]],
                harmonic_line_model="naive",
            )
        ],
        appliances=[
            Source(
                id=40,
                node=10,
                phases=(Phase.A,),
                u_ref_v=(400.0,),
                u_angle_deg=(2.0,),
                resistance_ohm=[[0.0]],
                inductance_h=[[0.0]],
            ),
            Load(
                id=50,
                node=20,
                phases=(Phase.A,),
                p_nom_w=12_000.0,
                q_nom_var=3_000.0,
                load_model=load_model,
                zip_coefficients=coefficients,
            ),
        ],
    )


def test_pandapower_from_grid_maps_units_and_round_trips() -> None:
    pp = pytest.importorskip("pandapower", exc_type=ImportError)
    from pgml.convert.pandapower import from_grid, to_grid

    grid = _grid()
    before = grid.model_dump(mode="json")
    exported = from_grid(grid)

    assert exported.bus_of_node.keys() == {10, 20}
    assert exported.node_of_bus == {
        value: key for key, value in exported.bus_of_node.items()
    }
    assert exported.line_of_branch.keys() == {30}
    assert exported.load_of_appliance.keys() == {50}
    assert exported.ext_grid_of_appliance.keys() == {40}
    line = exported.net.line.loc[exported.line_of_branch[30]]
    assert float(line.r_ohm_per_km) == pytest.approx(0.4)
    assert float(line.x_ohm_per_km) == pytest.approx(0.3)
    assert float(line.length_km) == pytest.approx(0.25)
    load = exported.net.load.loc[exported.load_of_appliance[50]]
    assert float(load.p_mw) == pytest.approx(0.012)
    assert float(load.q_mvar) == pytest.approx(0.003)
    assert any("harmonic_line_model" in note for note in exported.reductions)
    assert grid.model_dump(mode="json") == before

    pp.runpp(exported.net, numba=False)
    restored, id_map = to_grid(exported.net, harmonic_line_model="none")
    restored_line = next(
        branch for branch in restored.branches if branch.id == id_map["line"][0]
    )
    assert float(restored_line.length_m) == pytest.approx(250.0)
    assert np.asarray(restored_line.series_resistance_ohm_per_m)[0, 0] == pytest.approx(
        4e-4
    )
    assert id_map["bus"][exported.bus_of_node[10]] in {
        node.id for node in restored.nodes
    }


def test_pgm_from_grid_maps_components_and_round_trips() -> None:
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pgm import from_grid, to_grid

    grid = _grid()
    exported = from_grid(grid)

    assert exported.pgm_of_node.keys() == {10, 20}
    assert exported.pgm_of_branch.keys() == {30}
    assert exported.pgm_of_appliance.keys() == {40, 50}
    line = exported.input_data["line"][0]
    assert float(line["r1"]) == pytest.approx(0.1)
    assert float(line["x1"]) == pytest.approx(0.075)
    load = exported.input_data["sym_load"][0]
    assert float(load["p_specified"]) == pytest.approx(12_000.0)
    assert float(load["q_specified"]) == pytest.approx(3_000.0)
    assert any("finite short-circuit power" in note for note in exported.reductions)

    restored, id_map = to_grid(
        exported.input_data,
        load_model=LoadModel.CONST_POWER,
        harmonic_line_model="none",
    )
    restored_line = next(
        branch
        for branch in restored.branches
        if branch.id == id_map["line"][int(line["id"])]
    )
    assert np.asarray(restored_line.series_resistance_ohm_per_m)[0, 0] == pytest.approx(
        0.1
    )


def test_pgm_zip_requires_explicit_approximation() -> None:
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pgm import UnsupportedGridError, from_grid

    grid = _grid(load_model=LoadModel.ZIP)
    with pytest.raises(
        UnsupportedGridError, match="ZIP has no power-grid-model equivalent"
    ):
        from_grid(grid)

    exported = from_grid(grid, allow_approximation=True)
    assert int(exported.input_data["sym_load"][0]["type"]) == 0
    assert any(
        "ZIP load reduced to constant power" in note for note in exported.reductions
    )


def test_fundamental_export_records_ignored_harmonic_impedance() -> None:
    pytest.importorskip("pandapower", exc_type=ImportError)
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pandapower import from_grid as to_pp
    from pgml.convert.pgm import from_grid as to_pgm

    grid = _grid()
    generator = Generator(
        id=60,
        node=20,
        phases=(Phase.A,),
        p_nom_w=2_000.0,
        q_nom_var=0.0,
        harmonic_impedance=HarmonicImpedance(
            resistance_ohm=0.2,
            inductance_h=1e-3,
            spectrum_reference="opendss_voltage",
            frequency_model="opendss_admittance",
        ),
    )
    grid = grid.model_copy(update={"appliances": [*grid.appliances, generator]})

    for exported in (to_pp(grid), to_pgm(grid)):
        assert any(
            "appliance 60" in note and "harmonic_impedance" in note
            for note in exported.reductions
        )


def test_pandapower_source_impedance_requires_approximation() -> None:
    pytest.importorskip("pandapower", exc_type=ImportError)
    from pgml.convert.pandapower import UnsupportedGridError, from_grid

    grid = _grid()
    source = grid.appliances[0].model_copy(update={"resistance_ohm": [[0.1]]})
    grid = grid.model_copy(update={"appliances": [source, *grid.appliances[1:]]})

    with pytest.raises(UnsupportedGridError, match="source impedance omitted"):
        from_grid(grid)
    exported = from_grid(grid, allow_approximation=True)
    assert any("source impedance omitted" in note for note in exported.reductions)


def test_balanced_export_rejects_partial_phase_layout() -> None:
    pytest.importorskip("pandapower", exc_type=ImportError)
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pandapower import UnsupportedGridError, from_grid as to_pp
    from pgml.convert.pgm import from_grid as to_pgm

    grid = _grid()
    node = grid.nodes[0].model_copy(update={"phases": (Phase.B,)})
    grid = grid.model_copy(update={"nodes": [node, grid.nodes[1]]})

    for exporter in (to_pp, to_pgm):
        with pytest.raises(UnsupportedGridError, match=r"same \(A,\) or \(A,B,C\)"):
            exporter(grid)
