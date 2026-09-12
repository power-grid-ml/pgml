"""Oracle test: a power-grid-model ``link`` against pgml's exact bus fusion.

power-grid-model's ``link`` component is a perfect connection between two nodes, which
its solver realises with a very large stand-in admittance (1e6 per unit) rather than by
merging the nodes. pgml converts a link to an ideal closed ``Switch`` and collapses its
terminal rows exactly, so the two engines model the same element by the two available
means — and power-grid-model REPORTS the link's current, which makes it the reference for
the Kirchhoff recovery of a fused branch's current (pandapower reports ``NaN`` there).

Measured on CPU / complex128, power-grid-model 1.13: the node voltages agree to 3.5e-9 pu
and the link current to 6.0e-9 relative. Both residuals are the reference's own stand-in
error: its 1e6 pu admittance leaves 7.0e-5 V across the link, which fusion makes exactly
zero.

The single-phase-equivalent convention: power-grid-model reports a symmetric calculation
as a three-phase system (``i = S / (sqrt(3) U_LL)``) while a pgml one-phase-per-node grid
reports the conductor current against the same rated voltage, so the currents differ by
``sqrt(3)`` by construction.
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    from power_grid_model import PowerGridModel, initialize_array

    _PGM = True
except Exception:  # pragma: no cover - optional dependency / broken C core
    _PGM = False

if not _PGM:
    pytest.skip("power-grid-model not installed", allow_module_level=True)

from pgml.assembly import branch_currents, device_current_injections  # noqa: E402
from pgml.convert.pgm import to_grid  # noqa: E402
from pgml.schemas.grid_schema import LoadModel, Phase  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

CDT = torch.complex128


def _input_data():
    """Three 20 kV nodes: source -- link -- line -- load."""
    node = initialize_array("input", "node", 3)
    node["id"] = [1, 2, 3]
    node["u_rated"] = [20e3, 20e3, 20e3]

    source = initialize_array("input", "source", 1)
    source["id"] = [10]
    source["node"] = [1]
    source["status"] = [1]
    source["u_ref"] = [1.0]
    source["u_ref_angle"] = [0.0]
    source["sk"] = [1e20]
    source["rx_ratio"] = [0.1]
    source["z01_ratio"] = [1.0]

    link = initialize_array("input", "link", 1)
    link["id"] = [20]
    link["from_node"] = [1]
    link["to_node"] = [2]
    link["from_status"] = [1]
    link["to_status"] = [1]

    line = initialize_array("input", "line", 1)
    line["id"] = [21]
    line["from_node"] = [2]
    line["to_node"] = [3]
    line["from_status"] = [1]
    line["to_status"] = [1]
    line["r1"] = [0.2]
    line["x1"] = [0.08]
    line["c1"] = [0.0]
    line["tan1"] = [0.0]
    line["r0"] = [0.6]
    line["x0"] = [0.24]
    line["c0"] = [0.0]
    line["tan0"] = [0.0]
    line["i_n"] = [1000.0]

    load = initialize_array("input", "sym_load", 1)
    load["id"] = [30]
    load["node"] = [3]
    load["status"] = [1]
    load["type"] = [0]  # const power
    load["p_specified"] = [1e6]
    load["q_specified"] = [3e5]

    return {
        "node": node,
        "source": source,
        "link": link,
        "line": line,
        "sym_load": load,
    }


def _solve_pgm(data):
    model = PowerGridModel(data)
    return model.calculate_power_flow(error_tolerance=1e-12, max_iterations=100)


def _solve_pgml(data):
    grid, id_map = to_grid(data, load_model=LoadModel.CONST_POWER)
    res = solve_power_flow(
        grid, slack="ideal", tol=1e-12, tol_update_pu=1e-12, dtype=CDT
    )
    assert res.converged
    return grid, id_map, res


def test_a_link_converts_to_a_fused_ideal_switch():
    grid, id_map, res = _solve_pgml(_input_data())
    from pgml.schemas.grid_schema import Switch

    switches = [b for b in grid.branches if isinstance(b, Switch)]
    assert len(switches) == 1
    assert switches[0].resistance_ohm == 0.0 and switches[0].closed
    assert id_map["link"][20] == switches[0].id
    assert res.fusion is not None
    assert res.fusion.fused_branch_ids == (switches[0].id,)


def test_out_of_service_link_is_not_converted():
    data = _input_data()
    data["link"]["to_status"] = [0]
    grid, id_map, _ = _solve_pgml_unconnected(data)
    from pgml.schemas.grid_schema import Switch

    assert [b for b in grid.branches if isinstance(b, Switch)] == []
    assert id_map["link"] == {}


def _solve_pgml_unconnected(data):
    """Convert only (an open link leaves the load island unsupplied)."""
    grid, id_map = to_grid(data, load_model=LoadModel.CONST_POWER)
    return grid, id_map, None


def test_node_voltages_match_power_grid_model():
    data = _input_data()
    out = _solve_pgm(data)
    grid, id_map, res = _solve_pgml(data)
    v = res.v.reshape(-1)
    dv = 0.0
    for k, pgm_id in enumerate(out["node"]["id"].tolist()):
        ours = abs(complex(v[res.index.row(id_map["node"][pgm_id], Phase.A)]))
        dv = max(dv, abs(ours - float(out["node"]["u"][k])) / 20e3)
    # the residual IS power-grid-model's own 1e6 pu stand-in drop across the link
    assert dv < 1e-8, f"max |dV| = {dv:.3e} pu"


def test_the_fused_link_current_matches_power_grid_model():
    """The Kirchhoff recovery against an engine that reports the link current."""
    data = _input_data()
    out = _solve_pgm(data)
    grid, id_map, res = _solve_pgml(data)
    f0 = float(grid.base_frequency_hz)
    i_inj = -device_current_injections(grid, res.v, res.index, [f0], dtype=CDT)
    currents = {
        bc.branch_id: bc
        for bc in branch_currents(
            grid,
            res.v.unsqueeze(-2),
            [f0],
            res.index,
            dtype=CDT,
            fusion=res.fusion,
            i_inj=i_inj,
        )
    }
    bc = currents[id_map["link"][20]]
    ours = float(bc.i_from.abs().reshape(-1)[0])
    # power-grid-model reports a symmetric calculation in three-phase quantities.
    ref = float(out["link"]["i_from"][0]) * math.sqrt(3.0)
    assert abs(ours - ref) / ref < 1e-8, f"{ours} A vs {ref} A"
    # an ideal conductor has no shunt path: the two terminal currents are opposite
    assert torch.allclose(bc.i_from, -bc.i_to, atol=0.0, rtol=0.0)
