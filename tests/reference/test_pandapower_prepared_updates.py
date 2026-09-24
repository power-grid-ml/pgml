"""Re-preparation after physical network changes agrees with live pandapower."""

import numpy as np
import pytest
import torch

from pgml.convert.pandapower import to_grid
from pgml.errors import InputError
from pgml.solver import prepare_power_flow, solve_power_flow

pp = pytest.importorskip("pandapower")


@pytest.mark.parametrize("change", ["resistance", "branch_state"])
def test_changed_prepared_network_rejects_then_matches_pandapower(change):
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, b0)
    lid = pp.create_line_from_parameters(
        net,
        b0,
        b1,
        length_km=2.0,
        r_ohm_per_km=0.4,
        x_ohm_per_km=0.3,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, b1, p_mw=1.0, q_mvar=0.3)
    grid, mapping = to_grid(net)
    line = grid.branches[0]
    old_args = (
        {
            "param_overrides": {
                ("line", line.id, "series_resistance_ohm_per_m"): torch.tensor(
                    [[0.0004]], dtype=torch.float64
                )
            }
        }
        if change == "resistance"
        else {"branch_states": {line.id: 1.0}}
    )
    system = prepare_power_flow(grid, **old_args)
    if change == "resistance":
        net.line.loc[lid, "r_ohm_per_km"] = 0.8
        changed_args = {
            "param_overrides": {
                ("line", line.id, "series_resistance_ohm_per_m"): torch.tensor(
                    [[0.0008]], dtype=torch.float64
                )
            }
        }
    else:
        # Halving the complete series stamp is exactly doubling R and X when C=0.
        net.line.loc[lid, ["r_ohm_per_km", "x_ohm_per_km"]] *= 2.0
        changed_args = {"branch_states": {line.id: 0.5}}
    with pytest.raises(InputError, match="stale"):
        solve_power_flow(grid, system=system, **changed_args)
    new_system = prepare_power_flow(grid, **changed_args)
    result = solve_power_flow(grid, system=new_system, tol=1e-11, **changed_args)
    pp.runpp(net, numba=False, calculate_voltage_angles=True, tolerance_mva=1e-10)
    rows = [result.index.row(mapping["bus"][b], "a") for b in net.bus.index]
    ref = (
        net.res_bus.vm_pu.to_numpy()
        * 20000.0
        * np.exp(1j * np.deg2rad(net.res_bus.va_degree.to_numpy()))
    )
    assert result.converged
    np.testing.assert_allclose(result.v[rows].numpy(), ref, atol=1e-7, rtol=1e-10)
