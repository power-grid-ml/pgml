"""Exporting a per-phase-equivalent grid restores the system quantities.

An ``(A,)`` grid is one phase of a balanced system (line-to-neutral nodes, per-phase powers).
The pandapower and power-grid-model exporters describe the whole system, so a
convert-then-export round trip must return the original network's voltage ratings, powers and
therefore currents, not a network with line-to-neutral buses and a third of the load.
"""

from __future__ import annotations

import math

import pytest

pp = pytest.importorskip("pandapower")

from pgml.convert._export import system_view  # noqa: E402
from pgml.convert.pandapower import from_grid, to_grid  # noqa: E402


def _net():
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b0, vm_pu=1.02)
    pp.create_transformer_from_parameters(
        net,
        b0,
        b1,
        sn_mva=0.63,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=1.0,
        vk_percent=4.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=150.0,
        vector_group="Dyn5",
    )
    pp.create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=0.3,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.08,
        c_nf_per_km=100.0,
        max_i_ka=0.27,
    )
    pp.create_load(net, b2, p_mw=0.2, q_mvar=0.06)
    pp.create_sgen(net, b2, p_mw=0.05, q_mvar=0.01)
    return net


def test_round_trip_restores_ratings_and_powers():
    net = _net()
    grid, _ = to_grid(net)
    exported = from_grid(grid, allow_approximation=True).net
    assert list(exported.bus.vn_kv) == pytest.approx(list(net.bus.vn_kv), rel=1e-12)
    assert float(exported.load.p_mw.iloc[0]) == pytest.approx(0.2, rel=1e-12)
    assert float(exported.sgen.p_mw.iloc[0]) == pytest.approx(0.05, rel=1e-12)
    assert float(exported.trafo.sn_mva.iloc[0]) == pytest.approx(0.63, rel=1e-12)
    assert float(exported.trafo.vn_hv_kv.iloc[0]) == pytest.approx(20.0, rel=1e-12)
    assert float(exported.trafo.vn_lv_kv.iloc[0]) == pytest.approx(0.4, rel=1e-12)
    assert float(exported.ext_grid.vm_pu.iloc[0]) == pytest.approx(1.02, rel=1e-12)


def test_round_trip_reproduces_the_currents():
    net = _net()
    pp.runpp(net, numba=False, trafo_model="pi", tolerance_mva=1e-12)
    exported = from_grid(to_grid(net)[0], allow_approximation=True).net
    pp.runpp(exported, numba=False, trafo_model="pi", tolerance_mva=1e-12)
    assert float(exported.res_line.i_from_ka.iloc[0]) == pytest.approx(
        float(net.res_line.i_from_ka.iloc[0]), rel=1e-6
    )


def test_system_view_leaves_a_three_phase_grid_and_the_input_untouched():
    from pgml.convert import PhaseMode

    net = _net()
    three, _ = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    assert system_view(three) is three
    single, _ = to_grid(net)
    before = single.model_dump()
    view = system_view(single)
    assert single.model_dump() == before
    assert view.nodes[0].u_rated_v == pytest.approx(
        single.nodes[0].u_rated_v * math.sqrt(3.0)
    )


def test_untagged_one_phase_grid_keeps_its_established_reading():
    """A hand-built or legacy (A,) grid has no tag and is exported as written."""
    grid, _ = to_grid(_net())
    grid.metadata.tags.pop("phase_mode")
    assert system_view(grid) is grid
