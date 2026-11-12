"""Converters carry source line ampacities as ``Grid.types.lines`` entries."""

from __future__ import annotations

import pytest

pp = pytest.importorskip("pandapower")

from pgml.convert import PhaseMode  # noqa: E402
from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.schemas.grid_schema import Line  # noqa: E402
from pgml.studies.limits import branch_ratings  # noqa: E402


def _net(**line_kwargs):
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)
    b = [pp.create_bus(net, vn_kv=0.4) for _ in range(4)]
    pp.create_ext_grid(net, b[0])
    pp.create_line_from_parameters(
        net,
        b[0],
        b[1],
        length_km=0.2,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.08,
        c_nf_per_km=0.0,
        max_i_ka=0.27,
        df=0.9,
        parallel=2,
    )
    pp.create_line_from_parameters(
        net,
        b[1],
        b[2],
        length_km=0.2,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.08,
        c_nf_per_km=0.0,
        max_i_ka=0.27,
    )
    pp.create_line_from_parameters(
        net,
        b[2],
        b[3],
        length_km=0.2,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.08,
        c_nf_per_km=0.0,
        max_i_ka=0.27,
    )
    pp.create_load(net, b[3], p_mw=0.05)
    return net


@pytest.mark.parametrize("mode", list(PhaseMode), ids=lambda m: m.value)
def test_pandapower_ampacity_is_max_i_ka_df_parallel(mode):
    grid, id_map = to_grid(_net(), phase_mode=mode)
    ratings = branch_ratings(grid)
    assert ratings[id_map["line"][0]].i_max_a == pytest.approx(
        270.0 * 0.9 * 2, rel=1e-14
    )
    assert ratings[id_map["line"][1]].i_max_a == pytest.approx(270.0, rel=1e-14)
    # identical lines share one catalogue type; the explicit matrices stay on the line
    lines = [b for b in grid.branches if isinstance(b, Line)]
    assert lines[1].type_ref == lines[2].type_ref
    assert lines[0].type_ref != lines[1].type_ref
    assert len(grid.types.lines) == 2
    assert all(ln.series_resistance_ohm_per_m is not None for ln in lines)


def test_unrated_line_has_no_type():
    net = _net()
    net.line["max_i_ka"] = float("nan")
    grid, _ = to_grid(net)
    assert not grid.types.lines
    assert all(b.type_ref is None for b in grid.branches if isinstance(b, Line))


def test_pgm_and_opendss_carry_their_ratings():
    pgm = pytest.importorskip("power_grid_model")
    from pgml.convert.pgm import to_grid as pgm_to_grid

    node = pgm.initialize_array("input", "node", 2)
    node["id"], node["u_rated"] = [1, 2], [400.0, 400.0]
    line = pgm.initialize_array("input", "line", 1)
    line["id"], line["from_node"], line["to_node"] = [3], [1], [2]
    line["from_status"], line["to_status"] = [1], [1]
    line["r1"], line["x1"], line["c1"], line["tan1"] = [0.1], [0.05], [0.0], [0.0]
    line["i_n"] = [123.0]
    grid, id_map = pgm_to_grid({"node": node, "line": line}, base_frequency_hz=50.0)
    assert branch_ratings(grid)[id_map["line"][3]].i_max_a == 123.0

    dss = pytest.importorskip("opendssdirect")
    from pgml.convert.opendss import to_grid as dss_to_grid

    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command("New Circuit.r basekv=0.4 phases=3 bus1=a.1.2.3")
    dss.Text.Command(
        "New Line.l1 bus1=a.1.2.3 bus2=b.1.2.3 phases=3 r1=0.1 x1=0.05 r0=0.3 x0=0.15 "
        "c1=0 c0=0 length=1 units=km normamps=210"
    )
    dss.Text.Command("New Load.ld bus1=b.1.2.3 phases=3 kv=0.4 kw=10 kvar=2")
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    for mode in PhaseMode:
        grid, id_map = dss_to_grid(dss, phase_mode=mode)
        assert branch_ratings(grid)[id_map["line"]["l1"]].i_max_a == 210.0


@pytest.mark.parametrize("mode", list(PhaseMode), ids=lambda m: m.value)
def test_pandapower_export_round_trips_the_rating(mode):
    """``max_i_ka * df * parallel`` returns as one ``max_i_ka``; loading is preserved."""
    from pgml.convert.pandapower import from_grid

    net = _net()
    grid, id_map = to_grid(net, phase_mode=mode)
    export = from_grid(grid, allow_approximation=True)
    out = export.net
    rows = [export.line_of_branch[id_map["line"][i]] for i in net.line.index]
    rated = net.line.max_i_ka * net.line.df * net.line.parallel
    assert list(out.line.max_i_ka.loc[rows]) == pytest.approx(list(rated), rel=1e-14)
    assert list(out.line.df.loc[rows]) == [1.0] * len(rows)
    assert list(out.line.parallel.loc[rows]) == [1] * len(rows)
    # Same physical network, same loading against the rating. The first line is two
    # parallel conductors in the source net and one equivalent conductor in the export,
    # so only the per-line loading is comparable, not the per-conductor current.
    pp.runpp(net)
    pp.runpp(out)
    assert list(out.res_line.loading_percent.loc[rows]) == pytest.approx(
        list(net.res_line.loading_percent), rel=1e-10
    )


def test_pandapower_export_of_an_unrated_line_uses_the_placeholder():
    from pgml.convert.pandapower import from_grid

    net = _net()
    net.line["max_i_ka"] = float("nan")
    grid, _ = to_grid(net)
    out = from_grid(grid, allow_approximation=True).net
    assert list(out.line.max_i_ka) == [10.0] * len(net.line)
