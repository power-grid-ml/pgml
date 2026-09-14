"""Per-terminal pandapower switches retain open-ended branch physics."""

from __future__ import annotations

import copy

import pytest
import torch

pp = pytest.importorskip("pandapower")
networks = pytest.importorskip("pandapower.networks")

from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.schemas import Line, Phase, Transformer  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402


def _line_net(*, open_side="to", in_service=True, duplicate=False, both=False):
    net = pp.create_empty_network(f_hz=50.0)
    buses = [pp.create_bus(net, vn_kv=20.0) for _ in range(2)]
    pp.create_ext_grid(net, bus=buses[0])
    line = pp.create_line_from_parameters(
        net,
        buses[0],
        buses[1],
        length_km=2.0,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.1,
        c_nf_per_km=250.0,
        max_i_ka=1.0,
        in_service=in_service,
    )
    terminal = buses[0] if open_side == "from" else buses[1]
    switches = [pp.create_switch(net, terminal, line, et="l", closed=False)]
    if duplicate:
        switches.append(pp.create_switch(net, terminal, line, et="l", closed=False))
    if both:
        other = buses[1] if open_side == "from" else buses[0]
        switches.append(pp.create_switch(net, other, line, et="l", closed=False))
    return net, buses, line, switches


def _transformer_net(*, open_side="to", in_service=True, both=False):
    net = pp.create_empty_network(f_hz=50.0)
    hv = pp.create_bus(net, vn_kv=20.0)
    lv = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, bus=hv)
    trafo = pp.create_transformer_from_parameters(
        net,
        hv,
        lv,
        sn_mva=0.5,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vk_percent=4.0,
        vkr_percent=1.0,
        pfe_kw=1.0,
        i0_percent=0.5,
        shift_degree=0.0,
        in_service=in_service,
    )
    terminal = hv if open_side == "from" else lv
    switches = [pp.create_switch(net, terminal, trafo, et="t", closed=False)]
    if both:
        other = lv if open_side == "from" else hv
        switches.append(pp.create_switch(net, other, trafo, et="t", closed=False))
    return net, (hv, lv), trafo, switches


@pytest.mark.parametrize("side", ["from", "to"])
def test_open_line_terminal_reroutes_only_that_end(side):
    net, buses, line_index, switches = _line_net(open_side=side)
    grid, id_map = to_grid(net, open_switch_model="terminal")
    line = next(branch for branch in grid.branches if isinstance(branch, Line))
    auxiliary = id_map["open_terminal"][switches[0]]

    assert id_map["line"][line_index] == line.id
    assert auxiliary not in id_map["bus"].values()
    assert (line.from_node if side == "from" else line.to_node) == auxiliary
    connected_bus = buses[1] if side == "from" else buses[0]
    assert (line.to_node if side == "from" else line.from_node) == id_map["bus"][
        connected_bus
    ]
    assert (
        next(node for node in grid.nodes if node.id == auxiliary).u_rated_v == 20_000.0
    )


@pytest.mark.parametrize("side", ["from", "to"])
def test_open_transformer_terminal_reroutes_only_that_end(side):
    net, buses, trafo_index, switches = _transformer_net(open_side=side)
    grid, id_map = to_grid(net, open_switch_model="terminal")
    trafo = next(branch for branch in grid.branches if isinstance(branch, Transformer))
    auxiliary = id_map["open_terminal"][switches[0]]

    assert id_map["trafo"][trafo_index] == trafo.id
    assert (trafo.from_node if side == "from" else trafo.to_node) == auxiliary
    connected_bus = buses[1] if side == "from" else buses[0]
    assert (trafo.to_node if side == "from" else trafo.from_node) == id_map["bus"][
        connected_bus
    ]


@pytest.mark.parametrize("kind", ["line", "transformer"])
@pytest.mark.parametrize("reason", ["both_open", "out_of_service"])
def test_branch_without_an_energized_terminal_is_omitted(kind, reason):
    factory = _line_net if kind == "line" else _transformer_net
    net, _, element, _ = factory(
        both=reason == "both_open", in_service=reason != "out_of_service"
    )
    grid, id_map = to_grid(net, open_switch_model="terminal")

    bucket = "line" if kind == "line" else "trafo"
    cls = Line if kind == "line" else Transformer
    assert element not in id_map[bucket]
    assert not any(isinstance(branch, cls) for branch in grid.branches)
    assert id_map["open_terminal"] == {}


def test_duplicate_switches_share_one_auxiliary_node_and_input_is_unchanged():
    net, _, _, switches = _line_net(duplicate=True)
    before = copy.deepcopy(net)
    grid, id_map = to_grid(net, open_switch_model="terminal")

    assert id_map["open_terminal"][switches[0]] == id_map["open_terminal"][switches[1]]
    assert len(grid.nodes) == len(net.bus) + 1
    for table in ("bus", "line", "trafo", "switch"):
        assert getattr(net, table).equals(getattr(before, table))


def test_drop_element_retains_historical_reduction():
    net, _, line, _ = _line_net()
    grid, id_map = to_grid(net, open_switch_model="drop_element")
    assert line not in id_map["line"]
    assert not any(isinstance(branch, Line) for branch in grid.branches)
    assert id_map["open_terminal"] == {}


def test_default_resolves_to_terminal_model():
    net, _, line, switches = _line_net()
    grid, id_map = to_grid(net)
    assert id_map["line"][line] in {branch.id for branch in grid.branches}
    assert switches[0] in id_map["open_terminal"]


def test_cigre_mv_terminal_model_matches_live_pandapower():
    net = networks.create_cigre_network_mv()
    grid, id_map = to_grid(net, open_switch_model="terminal")
    pp.runpp(net, numba=False, tolerance_mva=1e-10, max_iteration=100)
    result = solve_power_flow(grid, tol=1e-10, max_iter=100)

    assert result.converged
    errors = []
    node_by_id = {node.id: node for node in grid.nodes}
    for bus, node_id in id_map["bus"].items():
        row = result.index.row(node_id, Phase.A)
        vm_pu = float(torch.abs(result.v.reshape(-1)[row])) / float(
            node_by_id[node_id].u_rated_v
        )
        errors.append(abs(vm_pu - float(net.res_bus.at[bus, "vm_pu"])))
    assert max(errors) < 1e-10
