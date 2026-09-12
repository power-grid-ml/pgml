"""Oracle test: exact bus fusion against pandapower, which fuses the same buses.

pandapower solves a closed bus-bus switch with no ``z_ohm`` by MERGING its two buses
(``net._pd2ppc_lookups["bus"]`` maps them to one internal bus) and a zero-impedance line
by the same merge after ``replace_zero_branches_with_switches``. pgml now does the same
thing in its own formulation — it collapses the branch's terminal node-phase rows into one
row of the solved system — so the two engines describe the identical network and the
agreement is limited only by the two Newton solves' own tolerances.

That makes this test the sharpest available check on the fusion path: the same networks
disagree by ~1e-7 pu when the ideal switch is replaced by the near-ideal stand-in
resistance, which is a MODELLING difference from the reference rather than a numerical
one (it adds a voltage drop the reference does not have).

Measured on CPU / complex128, pandapower 3.5.4:

=============================  ===========  =====================  ==========
grid                           fused        near-ideal 1e-4 Ohm    kappa(Y_ff)
=============================  ===========  =====================  ==========
CIGRE LV (1-phase equivalent)  1.6e-14 pu   1.1e-07 pu             910 / 14380
CIGRE LV (three-phase)         1.3e-14 pu   1.1e-07 pu             2888 / 45610
132 kV bus coupler (below)     1.1e-15 pu   9.5e-07 pu             60 / 1.4e5
=============================  ===========  =====================  ==========
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.toolbox.grid_modification import (
        replace_zero_branches_with_switches,
    )

    _PP = True
except ImportError:  # pragma: no cover - optional dependency
    _PP = False

if not _PP:
    pytest.skip("pandapower not installed", allow_module_level=True)

from pgml.assembly import fusion_map  # noqa: E402
from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.schemas.grid_schema import Line, Phase, Switch  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

CDT = torch.complex128
#: Both engines solve to ~1e-12; the fused comparison reaches machine precision.
ATOL_FUSED_PU = 1e-12
#: What the near-ideal stand-in costs instead (a modelling deviation, not noise).
STAND_IN_PU = 1e-7


def _stand_in(grid, r_ohm: float):
    """The same grid with every ideal branch given a stand-in series resistance."""
    from pgml.assembly import zero_impedance_branches

    ids = {z.branch_id for z in zero_impedance_branches(grid)}
    out = []
    for b in grid.branches:
        if int(b.id) not in ids:
            out.append(b)
        elif isinstance(b, Switch):
            out.append(b.model_copy(update={"resistance_ohm": r_ohm}))
        elif isinstance(b, Line):
            n = len(b.from_phases)
            length = max(float(b.length_m), 1.0)
            out.append(
                b.model_copy(
                    update={
                        "length_m": length,
                        "series_resistance_ohm_per_m": [
                            [r_ohm / length if i == j else 0.0 for j in range(n)]
                            for i in range(n)
                        ],
                    }
                )
            )
        else:  # pragma: no cover - the cases above cover every fusable kind
            out.append(b)
    return grid.model_copy(update={"branches": out})


def _max_deviation(net, grid, id_map):
    """``(max |dV| [pu], max |dtheta| [deg])`` against pandapower's converged result."""
    res = solve_power_flow(
        grid, slack="ideal", tol=1e-12, tol_update_pu=1e-12, max_iter=200, dtype=CDT
    )
    assert res.converged, f"pgml did not converge (mismatch {float(res.residual):.2e})"
    v = res.v.reshape(-1)
    node_by_id = {int(n.id): n for n in grid.nodes}
    dv = dth = 0.0
    for pp_bus, node_id in id_map["bus"].items():
        if not bool(net.bus.at[pp_bus, "in_service"]):
            continue
        vm_ref = float(net.res_bus.at[pp_bus, "vm_pu"])
        if not math.isfinite(vm_ref):
            continue
        base = float(net.bus.at[pp_bus, "vn_kv"]) * 1e3
        if len(node_by_id[node_id].phases) >= 3:
            base /= math.sqrt(3.0)  # a >=3-phase node's rated voltage is line-to-line
        vc = complex(v[res.index.row(node_id, Phase.A)])
        dv = max(dv, abs(abs(vc) / base - vm_ref))
        ours = math.degrees(math.atan2(vc.imag, vc.real))
        d = (ours - float(net.res_bus.at[pp_bus, "va_degree"])) % 360.0
        dth = max(dth, abs(d - 360.0 if d > 180.0 else d))
    return dv, dth, res


# --------------------------------------------------------------------------- #
# CIGRE LV: three closed bus-bus switches with z_ohm = 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "phase_mode", [PhaseMode.SINGLE_PHASE_EQUIV, PhaseMode.THREE_PHASE]
)
def test_cigre_lv_fused_switches_match_pandapower(phase_mode):
    """The MV busbars pandapower merges are solved as one row, to 1e-12 pu."""
    net = pn.create_cigre_network_lv()
    pp.runpp(net, numba=False, tolerance_mva=1e-10)
    grid, id_map = to_grid(net, phase_mode=phase_mode)

    fm = fusion_map(grid)
    assert fm is not None, "the three z_ohm = 0 bus-bus switches must fuse"
    assert len(fm.fused_branch_ids) == 3
    # pandapower merges buses 0, 1, 20, 23 into one internal bus; so does pgml, per phase.
    per_phase = 1 if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV else 3
    assert sum(len(g) for g in fm.groups) == 4 * per_phase
    assert fm.size == fm.full_index.size - 3 * per_phase

    dv, dth, _ = _max_deviation(net, grid, id_map)
    assert dv < ATOL_FUSED_PU, f"max |dV| = {dv:.3e} pu"
    assert dth < 1e-9, f"max |dtheta| = {dth:.3e} deg"


def test_cigre_lv_near_ideal_stand_in_is_measurably_worse():
    """The stand-in's own voltage drop is the deviation fusion removes."""
    net = pn.create_cigre_network_lv()
    pp.runpp(net, numba=False, tolerance_mva=1e-10)
    grid, id_map = to_grid(net)
    dv_fused, _, _ = _max_deviation(net, grid, id_map)
    dv_stand_in, _, _ = _max_deviation(net, _stand_in(grid, 1e-4), id_map)
    assert dv_fused < ATOL_FUSED_PU
    assert dv_stand_in > STAND_IN_PU
    assert dv_stand_in / max(dv_fused, 1e-300) > 1e3


# --------------------------------------------------------------------------- #
# a 132 kV bus coupler: the zero-impedance LINE idiom of published HV data
# --------------------------------------------------------------------------- #
def _coupler_net():
    """Two 132 kV busbars joined by a zero-impedance line, feeding a loaded feeder.

    The idiom of the UKGDS HV workbooks: a line type with R = X = B = 0 standing for a
    bus coupler. pandapower cannot solve it as a line (its Newton divides by the zero
    series reactance); ``replace_zero_branches_with_switches`` turns it into the bus-bus
    switch pandapower then fuses, which is the reference behaviour pgml reproduces.
    """
    net = pp.create_empty_network(f_hz=50.0)
    b = [pp.create_bus(net, vn_kv=132.0, name=f"bus{i}") for i in range(4)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        b[0],
        b[1],
        length_km=12.0,
        r_ohm_per_km=0.08,
        x_ohm_per_km=0.4,
        c_nf_per_km=9.0,
        max_i_ka=1.0,
    )
    pp.create_line_from_parameters(  # the coupler
        net,
        b[1],
        b[2],
        length_km=1.0,
        r_ohm_per_km=0.0,
        x_ohm_per_km=0.0,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_line_from_parameters(
        net,
        b[2],
        b[3],
        length_km=20.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.42,
        c_nf_per_km=9.5,
        max_i_ka=1.0,
    )
    pp.create_load(net, b[2], p_mw=25.0, q_mvar=8.0)
    pp.create_load(net, b[3], p_mw=40.0, q_mvar=12.0)
    return net


def test_zero_impedance_line_matches_pandapower_after_its_own_replacement():
    net = _coupler_net()
    grid, id_map = to_grid(net)  # pgml reads the zero-impedance LINE directly
    replace_zero_branches_with_switches(
        net,
        elements=("line",),
        zero_length=False,
        zero_impedance=True,
        drop_affected=True,
    )
    pp.runpp(net, numba=False, tolerance_mva=1e-10)
    # pandapower merged the coupler's buses:
    lookup = net._pd2ppc_lookups["bus"]
    assert lookup[1] == lookup[2]

    fm = fusion_map(grid)
    assert fm.fused_branch_ids == (id_map["line"][1],)
    dv, dth, res = _max_deviation(net, grid, id_map)
    assert dv < ATOL_FUSED_PU, f"max |dV| = {dv:.3e} pu"
    assert dth < 1e-9
    v = res.v.reshape(-1)
    assert (
        v[res.index.row(id_map["bus"][1], Phase.A)]
        == (v[res.index.row(id_map["bus"][2], Phase.A)])
    )


def test_the_stand_in_costs_conditioning_the_fused_system_does_not():
    """kappa(Y_ff) grows like 1/R with the stand-in; fusion removes the row entirely."""
    from pgml.assembly import assemble_network_ybus
    from pgml.solver.power_flow import _slack_rows_and_vref

    net = _coupler_net()
    grid, _ = to_grid(net)

    def kappa(g, fusion):
        yb = assemble_network_ybus(g, [50.0], dtype=CDT, fusion=fusion)
        y = yb.Y.reshape(yb.index.size, -1)
        rows, _ = _slack_rows_and_vref(
            g, yb.index, torch.float64, CDT, torch.device("cpu")
        )
        mask = torch.ones(y.shape[-1], dtype=torch.bool)
        mask[rows] = False
        free = torch.nonzero(mask).reshape(-1)
        return float(torch.linalg.cond(y.index_select(0, free).index_select(1, free)))

    k_fused = kappa(grid, fusion_map(grid))
    k_1em4 = kappa(_stand_in(grid, 1e-4), None)
    k_1em6 = kappa(_stand_in(grid, 1e-6), None)
    assert k_1em4 > 50.0 * k_fused
    assert k_1em6 > 50.0 * k_1em4  # one decade of R costs two decades of kappa
