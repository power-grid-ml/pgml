"""Oracle tests: branch currents and line loading against pandapower and power-grid-model.

The default ``SINGLE_PHASE_EQUIV`` converter output is ONE PHASE of a balanced three-phase
system: line-to-neutral node voltages and per-phase powers. A branch current of that grid
is therefore the physical conductor current, the quantity both reference tools report:

- pandapower ``res_line.i_from_ka`` / ``i_to_ka`` / ``loading_percent`` and
  ``res_trafo.i_hv_ka`` / ``i_lv_ka`` (conductor currents, ``S = sqrt(3) U_LL I``);
- power-grid-model symmetric ``line`` / ``transformer`` / ``link`` ``i_from`` / ``i_to`` and
  ``loading`` (the same).

Both phase modes are checked at ratio 1: under ``THREE_PHASE`` the current of one phase
conductor of a balanced solve is the same number. The pgml side is
``assembly.branch_currents`` evaluated at the ideal-slack power-flow solution; the
references solve their own networks independently, so the ratio is a real check of the
converter's voltage base and power split, not a per-unit identity.

Alignment: branches are matched through the converter ``id_map`` (pandapower line/trafo
index, pgm element id); pandapower kA are multiplied by 1e3; ``THREE_PHASE`` currents are
taken per phase (all three phases carry the same magnitude in a balanced solve, asserted).
Loading is ``|I| / I_rated`` with the ampacity the converter carried into
``Grid.types.lines`` (``branch_ratings(grid)``: pandapower ``max_i_ka * df * parallel``, pgm
``i_n``), compared with the reference's reported loading.

Tolerance: relative 1e-6 on every current (the references' own solver tolerances are
1e-8..1e-12; the transformer magnetizing placement is selected with the matching modeling
preset, so no model residual is left). Measured worst relative residuals (CPU, complex128, pandapower
3.0 / power-grid-model 1.13): pandapower 6.1e-12 (both modes, with and without magnetizing
branch) and 1.0e-12 (case33bw lines); power-grid-model 6.1e-12 (both modes). Without the
converter fix the ratios are 1.7321 (single-phase equivalent) and 1.0000 (three-phase).
"""

from __future__ import annotations


import numpy as np
import pytest
import torch

from pgml import defaults
from pgml.assembly import branch_currents, device_current_injections
from pgml.convert import PhaseMode
from pgml.studies.limits import branch_ratings
from pgml.schemas.grid_schema import LoadModel
from pgml.solver import solve_power_flow

try:
    import pandapower as pp

    _PP = True
except Exception:  # pragma: no cover - optional dependency
    _PP = False

try:
    from power_grid_model import (
        CalculationMethod,
        LoadGenType,
        PowerGridModel,
        WindingType,
        initialize_array,
    )

    _PGM = True
except Exception:  # pragma: no cover - optional dependency
    _PGM = False

CDT = torch.complex128
RTOL = 1.0e-6
MODES = list(PhaseMode)


def _pgml_currents(
    grid, tol: float = 1e-12, slack: str = "ideal"
) -> dict[int, np.ndarray]:
    """``{branch_id: [|i_from| per phase, |i_to| per phase]}`` in A at the solved state."""
    res = solve_power_flow(grid, slack=slack, tol=tol, max_iter=200, dtype=CDT)
    assert res.converged
    f0 = float(grid.base_frequency_hz)
    i_inj = -device_current_injections(grid, res.v, res.index, [f0], dtype=CDT)
    out = {}
    for bc in branch_currents(
        grid,
        res.v.unsqueeze(-2),
        [f0],
        res.index,
        dtype=CDT,
        fusion=res.fusion,
        i_inj=i_inj,
    ):
        i_from = bc.i_from.abs().reshape(-1).numpy()
        i_to = bc.i_to.abs().reshape(-1).numpy()
        # a balanced solve: every phase conductor carries the same current
        np.testing.assert_allclose(i_from, i_from[0], rtol=1e-8)
        out[bc.branch_id] = np.array([i_from[0], i_to[0]])
    return out


# ---------------------------------------------------------------------------
# pandapower
# ---------------------------------------------------------------------------
def _pp_net(*, i0_percent: float, pfe_kw: float):
    """MV overhead line -> Dyn5 20/0.4 kV transformer -> LV cable feeder with two loads."""
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)
    b_src = pp.create_bus(net, vn_kv=20.0, name="src")
    b_hv = pp.create_bus(net, vn_kv=20.0, name="hv")
    b_lv = pp.create_bus(net, vn_kv=0.4, name="lv")
    b_end = pp.create_bus(net, vn_kv=0.4, name="end")
    pp.create_ext_grid(net, bus=b_src, vm_pu=1.02, va_degree=0.0)
    pp.create_line_from_parameters(
        net,
        b_src,
        b_hv,
        length_km=8.0,
        r_ohm_per_km=0.3,
        x_ohm_per_km=0.35,
        c_nf_per_km=11.0,
        max_i_ka=0.28,
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=0.63,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=1.0,
        vk_percent=4.0,
        pfe_kw=pfe_kw,
        i0_percent=i0_percent,
        shift_degree=150.0,
        vector_group="Dyn5",
    )
    pp.create_line_from_parameters(
        net,
        b_lv,
        b_end,
        length_km=0.35,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.08,
        c_nf_per_km=300.0,
        max_i_ka=0.27,
    )
    pp.create_load(net, b_lv, p_mw=0.08, q_mvar=0.03)
    pp.create_load(net, b_end, p_mw=0.15, q_mvar=0.05)
    return net


def _pp_check(net, mode: PhaseMode):
    from pgml.convert.pandapower import to_grid

    pp.runpp(
        net,
        numba=False,
        calculate_voltage_angles=True,
        trafo_model="pi",
        tolerance_mva=1e-12,
        max_iteration=50,
    )
    with defaults.use_preset("pandapower"):
        grid, id_map = to_grid(net, phase_mode=mode)
        cur = _pgml_currents(grid)
        ratings = branch_ratings(grid)
    ratios = []
    for idx in net.line.index:
        ours = cur[id_map["line"][idx]]
        ref = (
            np.array(
                [net.res_line.at[idx, "i_from_ka"], net.res_line.at[idx, "i_to_ka"]]
            )
            * 1e3
        )
        ratios.append(ours / ref)
        loading_ref = float(net.res_line.at[idx, "loading_percent"])
        rated = ratings[id_map["line"][idx]].i_max_a  # the converted ampacity
        loading_ours = 100.0 * ours.max() / rated
        assert loading_ours == pytest.approx(loading_ref, rel=RTOL)
    for idx in net.trafo.index:
        ours = cur[id_map["trafo"][idx]]
        ref = (
            np.array(
                [net.res_trafo.at[idx, "i_hv_ka"], net.res_trafo.at[idx, "i_lv_ka"]]
            )
            * 1e3
        )
        ratios.append(ours / ref)
    return np.concatenate(ratios)


@pytest.mark.skipif(not _PP, reason="pandapower not installed")
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_pandapower_currents_without_magnetizing_branch(mode):
    """Lines, transformer and line loading, ratio 1 (``i0 = pfe = 0`` isolates the
    leakage / vector-group / charging path)."""
    ratios = _pp_check(_pp_net(i0_percent=0.0, pfe_kw=0.0), mode)
    assert np.max(np.abs(ratios - 1.0)) < RTOL, ratios


@pytest.mark.skipif(not _PP, reason="pandapower not installed")
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_pandapower_currents_with_magnetizing_branch(mode):
    """With a no-load branch the transformer terminal currents carry the magnetizing
    current; pandapower's ``pi`` model splits it per terminal like pgml's default."""
    ratios = _pp_check(_pp_net(i0_percent=0.4, pfe_kw=1.1), mode)
    assert np.max(np.abs(ratios - 1.0)) < RTOL, ratios


@pytest.mark.skipif(not _PP, reason="pandapower not installed")
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_pandapower_case33bw_line_currents(mode):
    import pandapower.networks as pn

    from pgml.convert.pandapower import to_grid

    net = pn.case33bw()
    pp.runpp(net, numba=False, tolerance_mva=1e-12)
    grid, id_map = to_grid(net, phase_mode=mode)
    cur = _pgml_currents(grid)
    worst = 0.0
    for idx in net.line.index:
        if idx not in id_map["line"]:
            continue
        ours = cur[id_map["line"][idx]]
        ref = (
            np.array(
                [net.res_line.at[idx, "i_from_ka"], net.res_line.at[idx, "i_to_ka"]]
            )
            * 1e3
        )
        worst = max(worst, float(np.max(np.abs(ours / ref - 1.0))))
    assert worst < RTOL, worst


# ---------------------------------------------------------------------------
# power-grid-model
# ---------------------------------------------------------------------------
def _pgm_case():
    node = initialize_array("input", "node", 3)
    node["id"] = [1, 2, 3]
    node["u_rated"] = [20e3, 400.0, 400.0]
    src = initialize_array("input", "source", 1)
    src["id"], src["node"], src["status"] = [10], [1], [1]
    src["u_ref"], src["u_ref_angle"] = [1.01], [0.0]
    src["sk"], src["rx_ratio"], src["z01_ratio"] = [5e8], [0.1], [1.0]
    tr = initialize_array("input", "transformer", 1)
    tr["id"], tr["from_node"], tr["to_node"] = [20], [1], [2]
    tr["from_status"], tr["to_status"] = [1], [1]
    tr["u1"], tr["u2"], tr["sn"] = [20e3], [400.0], [630e3]
    tr["uk"], tr["pk"], tr["i0"], tr["p0"] = [0.06], [6300.0], [0.004], [1100.0]
    tr["winding_from"], tr["winding_to"] = [WindingType.delta], [WindingType.wye_n]
    tr["clock"] = [5]
    tr["tap_side"], tr["tap_pos"], tr["tap_nom"] = [0], [0], [0]
    tr["tap_min"], tr["tap_max"], tr["tap_size"] = [-5], [5], [0.0]
    line = initialize_array("input", "line", 1)
    line["id"], line["from_node"], line["to_node"] = [21], [2], [3]
    line["from_status"], line["to_status"] = [1], [1]
    line["r1"], line["x1"], line["c1"], line["tan1"] = [0.07], [0.028], [1.0e-7], [0.0]
    line["r0"], line["x0"], line["c0"], line["tan0"] = [0.21], [0.084], [1.0e-7], [0.0]
    line["i_n"] = [270.0]
    load = initialize_array("input", "sym_load", 1)
    load["id"], load["node"], load["status"] = [30], [3], [1]
    load["type"] = [LoadGenType.const_power]
    load["p_specified"], load["q_specified"] = [220e3], [70e3]
    return {
        "node": node,
        "source": src,
        "transformer": tr,
        "line": line,
        "sym_load": load,
    }


@pytest.mark.skipif(not _PGM, reason="power-grid-model not installed")
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_pgm_line_and_transformer_currents(mode):
    """Transformer (``i_from``/``i_to``) and line currents and line loading, ratio 1."""
    from pgml.convert.pgm import to_grid

    data = _pgm_case()
    out = PowerGridModel(data).calculate_power_flow(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1e-12,
        max_iterations=100,
    )
    with defaults.use_preset("power-grid-model"):
        grid, id_map = to_grid(
            data,
            base_frequency_hz=50.0,
            load_model=LoadModel.CONST_POWER,
            phase_mode=mode,
        )
        # the pgm source has a finite short-circuit power: a Thevenin (Norton) slack
        cur = _pgml_currents(grid, slack="norton")
    for kind, pgm_id in (("transformer", 20), ("line", 21)):
        row = out[kind][out[kind]["id"] == pgm_id][0]
        ref = np.array([row["i_from"], row["i_to"]])
        ours = cur[id_map[kind][pgm_id]]
        assert np.max(np.abs(ours / ref - 1.0)) < RTOL, (kind, ours / ref)
        if kind == "line":
            rated = branch_ratings(grid)[id_map["line"][21]].i_max_a  # converted i_n
            assert ours.max() / rated == pytest.approx(float(row["loading"]), rel=RTOL)
