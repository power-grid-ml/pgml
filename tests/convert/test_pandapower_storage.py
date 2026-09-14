"""pandapower storage conversion preserves snapshot signs and inert state metadata."""

from __future__ import annotations

import pytest
import torch

pp = pytest.importorskip("pandapower")
networks = pytest.importorskip("pandapower.networks")

from pgml.convert.pandapower import PhaseMode, to_grid  # noqa: E402
from pgml.schemas import Phase, Storage  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402


def _net():
    net = pp.create_empty_network()
    source = pp.create_bus(net, vn_kv=20.0)
    bus = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, source)
    pp.create_line_from_parameters(net, source, bus, 1.0, 0.2, 0.1, 0.0, 1.0)
    charging = pp.create_storage(
        net,
        bus,
        p_mw=0.6,
        q_mvar=0.2,
        max_e_mwh=2.0,
        min_e_mwh=0.2,
        soc_percent=25.0,
        scaling=0.5,
    )
    discharging = pp.create_storage(
        net,
        bus,
        p_mw=-0.3,
        q_mvar=-0.1,
        max_e_mwh=1.0,
        scaling=2.0,
    )
    offline = pp.create_storage(
        net, bus, p_mw=1.0, q_mvar=1.0, max_e_mwh=1.0, in_service=False
    )
    return net, charging, discharging, offline


def test_storage_sign_scaling_state_and_service():
    net, charging, discharging, offline = _net()
    grid, id_map = to_grid(net)
    by_id = {appliance.id: appliance for appliance in grid.appliances}
    charge = by_id[id_map["storage"][charging]]
    discharge = by_id[id_map["storage"][discharging]]

    assert isinstance(charge, Storage) and isinstance(discharge, Storage)
    assert (charge.p_nom_w, charge.q_nom_var) == (-300_000.0, -100_000.0)
    assert (discharge.p_nom_w, discharge.q_nom_var) == (600_000.0, 200_000.0)
    assert charge.energy_capacity_wh == 2_000_000.0
    assert charge.soc == 0.25 and charge.soc_min == 0.1
    assert offline not in id_map["storage"]


def test_storage_three_phase_connection():
    net, charging, _, _ = _net()
    grid, id_map = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    storage = next(ap for ap in grid.appliances if ap.id == id_map["storage"][charging])
    assert storage.phases == (Phase.A, Phase.B, Phase.C)


def test_energy_and_soc_are_inert_in_snapshot_solve():
    net, charging, _, _ = _net()
    grid, id_map = to_grid(net)
    changed = grid.model_copy(deep=True)
    storage_id = id_map["storage"][charging]
    storage = next(ap for ap in changed.appliances if ap.id == storage_id)
    storage.energy_capacity_wh = 99_000_000.0
    storage.soc = 0.9
    storage.soc_min = 0.8
    original = solve_power_flow(grid, tol=1e-10)
    altered = solve_power_flow(changed, tol=1e-10)
    torch.testing.assert_close(original.v, altered.v, rtol=0.0, atol=0.0)


def test_cigre_mv_all_storage_and_terminal_model_match_live_pandapower():
    net = networks.create_cigre_network_mv(with_der="all")
    grid, id_map = to_grid(net)
    pp.runpp(net, numba=False, tolerance_mva=1e-10, max_iteration=100)
    result = solve_power_flow(grid, tol=1e-10, max_iter=100)

    assert result.converged and len(id_map["storage"]) == 2
    node_by_id = {node.id: node for node in grid.nodes}
    errors = []
    for bus, node_id in id_map["bus"].items():
        row = result.index.row(node_id, Phase.A)
        vm_pu = float(torch.abs(result.v.reshape(-1)[row])) / float(
            node_by_id[node_id].u_rated_v
        )
        errors.append(abs(vm_pu - float(net.res_bus.at[bus, "vm_pu"])))
    assert max(errors) < 1e-10
