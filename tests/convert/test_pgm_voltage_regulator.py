"""power-grid-model ``voltage_regulator`` imports as a pgml PV terminal."""

from __future__ import annotations

import cmath

import pytest

pgm = pytest.importorskip("power_grid_model", exc_type=ImportError)

from pgml.convert.pgm import to_grid  # noqa: E402
from pgml.schemas.grid_schema import Generator, LoadModel  # noqa: E402

from ._solved import max_voltage_error, pgml_node_voltages_pu  # noqa: E402


def _case(*, q_limits=None, second_gen=False, regulate_load=False):
    node = pgm.initialize_array("input", "node", 3)
    node["id"] = [1, 2, 3]
    node["u_rated"] = [10e3] * 3
    line = pgm.initialize_array("input", "line", 2)
    line["id"] = [4, 5]
    line["from_node"] = [1, 2]
    line["to_node"] = [2, 3]
    line["from_status"] = line["to_status"] = 1
    line["r1"] = [0.8, 0.5]
    line["x1"] = [0.6, 0.4]
    line["c1"] = [0.0, 0.0]
    line["tan1"] = [0.0, 0.0]
    line["i_n"] = [1000.0, 1000.0]
    src = pgm.initialize_array("input", "source", 1)
    src["id"] = [6]
    src["node"] = [1]
    src["status"] = 1
    src["u_ref"] = [1.0]
    src["sk"] = [2e8]
    src["rx_ratio"] = [0.1]
    load = pgm.initialize_array("input", "sym_load", 1)
    load["id"] = [7]
    load["node"] = [3]
    load["status"] = 1
    load["type"] = 0
    load["p_specified"] = [1.5e6]
    load["q_specified"] = [0.5e6]
    n_gen = 2 if second_gen else 1
    gen = pgm.initialize_array("input", "sym_gen", n_gen)
    gen["id"] = [8, 9][:n_gen]
    gen["node"] = [2] * n_gen
    gen["status"] = 1
    gen["type"] = 0
    gen["p_specified"] = [0.6e6, 0.3e6][:n_gen]
    gen["q_specified"] = [0.1e6] * n_gen
    vr = pgm.initialize_array("input", "voltage_regulator", n_gen)
    vr["id"] = [10, 11][:n_gen]
    vr["regulated_object"] = [7] if regulate_load else [8, 9][:n_gen]
    vr["status"] = 1
    vr["u_ref"] = [1.01] * n_gen
    if q_limits is not None:
        vr["q_min"], vr["q_max"] = q_limits
    return {
        "node": node,
        "line": line,
        "source": src,
        "sym_load": load,
        "sym_gen": gen,
        "voltage_regulator": vr,
    }


def _pgm_voltages(data, id_map):
    out = pgm.PowerGridModel(data).calculate_power_flow(
        calculation_method=pgm.CalculationMethod.newton_raphson,
        error_tolerance=1e-12,
    )
    by_id = {
        int(i): cmath.rect(float(u), float(a))
        for i, u, a in zip(
            out["node"]["id"], out["node"]["u_pu"], out["node"]["u_angle"]
        )
    }
    return {node: by_id[pid] for pid, node in id_map["node"].items()}, out


@pytest.mark.parametrize("q_limits", [None, (-0.2e6, 0.2e6)])
def test_regulated_generator_matches_the_native_solve(q_limits):
    data = _case(q_limits=q_limits)
    grid, id_map, report = to_grid(
        data, load_model="source", harmonic_line_model="none", return_report=True
    )
    gen = next(a for a in grid.appliances if isinstance(a, Generator))
    assert gen.voltage_regulation is not None
    assert gen.voltage_regulation.v_set_pu == pytest.approx(1.01)
    assert id_map["voltage_regulator"] == {10: gen.id}
    assert "dropped.voltage_regulator" not in report
    assert "approx.load.model_override" not in report

    theirs, out = _pgm_voltages(data, id_map)
    ours = pgml_node_voltages_pu(grid, tol=1e-12, method="newton", slack="norton")
    assert max_voltage_error(ours, theirs) < 1e-7
    if q_limits is not None:
        # the limit binds in power-grid-model and pgml alike
        assert int(out["voltage_regulator"]["limit_violated"][0]) != 0
        assert abs(theirs[id_map["node"][2]]) < 1.01 - 1e-4


def test_regulated_generators_on_one_node_merge():
    data = _case(second_gen=True, q_limits=(-0.2e6, 0.2e6))
    grid, id_map, report = to_grid(
        data, load_model="source", harmonic_line_model="none", return_report=True
    )
    gens = [a for a in grid.appliances if isinstance(a, Generator)]
    assert len(gens) == 1
    assert gens[0].p_nom_w == pytest.approx(0.9e6)
    assert gens[0].voltage_regulation.q_max_var == pytest.approx(0.4e6)
    assert id_map["sym_gen"] == {8: gens[0].id, 9: gens[0].id}
    (entry,) = report.get("approx.generator.merged_per_node")
    assert entry.ids == (8, 9)
    theirs, _ = _pgm_voltages(data, id_map)
    ours = pgml_node_voltages_pu(grid, tol=1e-12, method="newton", slack="norton")
    assert max_voltage_error(ours, theirs) < 1e-7


def test_regulator_on_a_load_is_reported_as_dropped(caplog):
    data = _case(regulate_load=True)
    grid, id_map, report = to_grid(data, return_report=True)
    (entry,) = report.get("dropped.voltage_regulator")
    assert entry.ids == (10,)
    assert id_map["voltage_regulator"] == {}
    assert any("voltage_regulator" in r.getMessage() for r in caplog.records)


def test_load_model_override_is_reported():
    data = _case()
    grid, _, report = to_grid(data, return_report=True)
    (entry,) = report.get("approx.load.model_override")
    assert entry.ids == (7,)
    assert entry.values["applied"] == LoadModel.CONST_IMPEDANCE.value
    grid, _, report = to_grid(data, load_model="source", return_report=True)
    assert grid.appliances[-1].load_model is LoadModel.CONST_POWER or any(
        a.load_model is LoadModel.CONST_POWER
        for a in grid.appliances
        if hasattr(a, "load_model")
    )
