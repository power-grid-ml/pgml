"""Hosting capacity: closed forms, a brute-force serial search, and every limit kind.

Tolerances: the study reports ``p_max_w`` as the largest power at which every limit was
verified, within ``tol_w`` of the limit. A closed form or an independent serial bisection
therefore agrees to ``tol_w`` plus the power-flow convergence error (1e-8 pu mismatch
on a 1 MVA base, i.e. about 0.01 W per row), stated per test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from pgml import SimulationConfig, simulate
from pgml.errors import InputError
from pgml.schemas.grid_schema import Generator, Load, Phase, WindingConnection
from pgml.studies import HostingCapacity, HostingLimits, hosting_capacity
from pgml.studies.hosting import _MARGIN_KEYS

from .grids import (
    ABC,
    four_wire_two_bus,
    lv_feeder_with_switch,
    mv_lv_transformer,
    two_bus,
)

R, X, V1 = 0.2, 0.1, 230.0


def _voltage_limit_power(u_pu: float, k: float, *, sign: float) -> float:
    """Exact power at which bus 2 of :func:`two_bus` reaches ``u_pu``.

    ``S = P (1 + jk)`` injected (``sign=+1``) or drawn (``sign=-1``) at bus 2 behind
    ``Z = R + jX`` from an ideal 230 V bus: ``|U + sign * Z conj(S) / U| = V1`` with
    ``U = u_pu * V1``, a quadratic in ``P``; the root closest to zero is the limit.
    """
    u = u_pu * V1
    a = ((R + k * X) ** 2 + (X - k * R) ** 2) / u**2
    b = -sign * 2.0 * (R + k * X)
    c = u**2 - V1**2
    roots = np.roots([a, b, c])
    roots = roots[np.isreal(roots)].real
    return float(min(r for r in roots if r > 0))


def test_voltage_rise_matches_the_two_bus_closed_form():
    res = hosting_capacity(two_bus(), limits=HostingLimits(v_max_pu=1.03), tol_w=1.0)
    exact = _voltage_limit_power(1.03, 0.0, sign=+1.0)
    assert res.binding[0] == {
        "kind": "voltage",
        "limit": "v_max_pu",
        "node_id": 2,
        "phase": "a",
    }
    assert exact - 1.0 - 1e-3 <= res.p_max_w[0] <= exact + 1e-3
    assert 0.0 <= res.margins["v_max_pu"][0] < 1e-4
    # the linearised rise dV ~ (R P + X Q) / V, V at the device bus, is within 1 %
    approx = 0.03 * V1 * (1.03 * V1) / R
    assert abs(approx - exact) / exact < 0.01


def test_under_excited_generation_and_three_phase_split():
    pf = -0.95
    k = -math.tan(math.acos(0.95))
    exact_phase = _voltage_limit_power(1.03, k, sign=+1.0)
    res1 = hosting_capacity(
        two_bus(), limits=HostingLimits(v_max_pu=1.03), power_factor=pf, tol_w=1.0
    )
    res3 = hosting_capacity(
        two_bus(phases=ABC),
        limits=HostingLimits(v_max_pu=1.03),
        power_factor=pf,
        tol_w=3.0,
    )
    assert abs(res1.p_max_w[0] - exact_phase) <= 1.0 + 1e-3
    # a balanced three-phase device at P splits P / 3 per phase
    assert abs(res3.p_max_w[0] - 3.0 * exact_phase) <= 3.0 + 1e-3
    # absorbing Q lets the bus host more than at unity power factor
    assert exact_phase > _voltage_limit_power(1.03, 0.0, sign=+1.0)


def _rise_limit(r_loop, x_loop, u_limit, v_source):
    """Injected power at which a device behind ``r_loop + j x_loop`` reaches ``u_limit``."""
    a = (r_loop**2 + x_loop**2) / u_limit**2
    return float(min(np.roots([a, -2.0 * r_loop, u_limit**2 - v_source**2]).real))


def test_four_wire_bus_uses_phase_to_neutral_voltage():
    grid = four_wire_two_bus()
    lim = HostingLimits(v_max_pu=1.03)
    u_limit = 1.03 * 400.0 / math.sqrt(3.0)  # the node's nominal line-to-neutral base
    r, x = 0.2e-3 * 200.0, 2.0 * math.pi * 50.0 * 0.25e-6 * 200.0
    single = hosting_capacity(grid, [2], phases="a", limits=lim, tol_w=10.0)
    # phase and neutral conductor in series plus the 0.01 ohm neutral grounding
    exact = _rise_limit(2.0 * r + 0.01, 2.0 * x, u_limit, 230.0)
    assert abs(single.p_max_w[0] - exact) <= 10.0 + 1e-2
    balanced = hosting_capacity(grid, [2], limits=lim, tol_w=30.0)
    # balanced: no neutral current, one phase conductor per third of the power
    exact = 3.0 * _rise_limit(r, x, u_limit, 230.0)
    assert abs(balanced.p_max_w[0] - exact) <= 30.0 + 1e-2


def test_new_demand_hits_the_lower_band_and_the_nose():
    res = hosting_capacity(two_bus(), kind="load", tol_w=1.0)
    exact = _voltage_limit_power(0.90, 0.0, sign=-1.0)
    assert res.binding[0]["limit"] == "v_min_pu"
    assert abs(res.p_max_w[0] - exact) <= 1.0 + 1e-3

    unbounded = HostingLimits(
        v_min_pu=None, v_max_pu=None, line_loading=None, transformer_loading=None
    )
    nose = V1**2 / (2.0 * (math.hypot(R, X) + R))  # max transfer to a resistive load
    res = hosting_capacity(two_bus(), kind="load", limits=unbounded, tol_w=10.0)
    assert res.binding[0] == {"kind": "convergence"}
    # beyond the nose no operating point exists; the fixed-point solve may stop earlier
    assert 0.5 * nose < res.p_max_w[0] <= nose


def test_line_rating_binds_before_voltage():
    res = hosting_capacity(two_bus(i_max_a=20.0), tol_w=1.0)
    assert res.binding[0] == {"kind": "line", "branch_id": 10}
    assert res.margins["v_max_pu"][0] > 0.0
    assert 0.0 <= res.margins["line_loading"][0] < 1e-3
    # |I| = 20 A at |V2| ~ V1 + R I: P = |V2| * 20 A
    v2 = abs(V1 + complex(R, X) * 20.0)
    assert abs(res.p_max_w[0] - v2 * 20.0) / (v2 * 20.0) < 1e-3


def test_transformer_rating_binds():
    grid = mv_lv_transformer(s_rated_va=100.0e3)
    res = hosting_capacity(grid, limits=HostingLimits(v_max_pu=1.2), tol_w=50.0)
    assert all(b == {"kind": "transformer", "branch_id": 30} for b in res.binding)
    # independent loading at the found power: |I_lv| * U_ln * 3 / S_rated
    gen = Generator(
        id=99,
        node=3,
        phases=ABC,
        connection=WindingConnection.WYE,
        p_nom_w=float(res.p_max_w[1]),
    )
    st = simulate(
        grid.model_copy(update={"appliances": [*grid.appliances, gen]}),
        SimulationConfig(calculation="power_flow"),
    )
    bc = {b.branch_id: b for b in st.branch_currents()}[30]
    u_ln = 400.0 / math.sqrt(3.0)
    loading = float(bc.i_to.abs().max()) * u_ln * 3.0 / 100.0e3
    assert abs((1.0 - loading) - res.margins["transformer_loading"][1]) < 1e-9
    assert 1.0 - 50.0 / 100.0e3 <= loading <= 1.0


def _serial_reference(grid, node_id, limits, *, cap, tol):
    """Serial bisection with ``pgml.simulate`` and independently coded limit checks."""
    zero_loads = {
        a.id: {"p_w": 0.0, "q_var": 0.0} for a in grid.appliances if isinstance(a, Load)
    }
    u_ln = 400.0 / math.sqrt(3.0)

    def feasible(p: float) -> bool:
        gen = Generator(
            id=99,
            node=node_id,
            phases=ABC,
            connection=WindingConnection.WYE,
            p_nom_w=p,
        )
        g = grid.model_copy(update={"appliances": [*grid.appliances, gen]})
        cfg = SimulationConfig(calculation="power_flow", operating_point=zero_loads)
        st = simulate(g, cfg, strict=False)
        if not st.converged:
            return False
        assert st.fusion is not None  # the ideal switch is fused, currents still read
        v_pu = [
            float(st.voltage(n.id, ph).abs().max()) / u_ln
            for n in g.nodes
            for ph in n.phases
        ]
        if max(v_pu) > limits.v_max_pu or min(v_pu) < limits.v_min_pu:
            return False
        for bc in st.branch_currents():
            if bc.branch_id in (21, 22):
                i = max(float(bc.i_from.abs().max()), float(bc.i_to.abs().max()))
                if i / 150.0 > limits.line_loading:
                    return False
        return True

    lo, hi = 0.0, cap
    if feasible(hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


def test_agrees_with_a_serial_simulate_search_on_a_switched_feeder():
    grid = lv_feeder_with_switch(i_max_a=150.0)
    limits = HostingLimits(v_max_pu=1.03)
    tol, cap = 50.0, 200.0e3
    res = hosting_capacity(grid, [3, 4], limits=limits, p_max_w=cap, tol_w=tol)
    for k, node_id in enumerate(res.node_ids):
        ref = _serial_reference(grid, node_id, limits, cap=cap, tol=tol)
        assert abs(res.p_max_w[k] - ref) <= tol, (node_id, res.p_max_w[k], ref)
    assert res.binding[0]["kind"] in ("line", "voltage")
    assert np.all(res.iterations == res.iterations[0])


def test_bus_fused_to_the_ideal_source_reaches_the_cap():
    res = hosting_capacity(lv_feeder_with_switch(), [2, 4], p_max_w=500.0e3)
    assert res.binding[0] == {"kind": "none"} and res.p_max_w[0] == 500.0e3
    assert res.binding[1]["kind"] != "none" and res.p_max_w[1] < 500.0e3
    assert res.p_cap_w == 500.0e3


def test_single_phase_device_and_unbalance_limit():
    grid = lv_feeder_with_switch()
    lim = HostingLimits(vuf_max=0.01)
    res = hosting_capacity(grid, [4], phases="a", limits=lim, tol_w=10.0)
    assert res.binding[0] == {"kind": "vuf", "node_id": 4}
    assert 0.0 <= res.margins["vuf"][0] < 1e-3
    balanced = hosting_capacity(grid, [4], limits=lim, tol_w=10.0)
    assert balanced.binding[0]["kind"] != "vuf"
    assert balanced.p_max_w[0] > res.p_max_w[0]
    with pytest.raises(InputError):
        hosting_capacity(two_bus(), phases="b")


def test_harmonic_emission_and_thd_limit():
    grid = lv_feeder_with_switch()
    cfg = SimulationConfig(calculation="harmonic", harmonic_orders=[1, 5, 7])
    emission = {5: (0.3, 0.0), 7: (0.2, 0.0)}
    lim = HostingLimits(thd_max=0.01, line_loading=None)
    res = hosting_capacity(
        grid, [4], limits=lim, config=cfg, harmonic_emission=emission, tol_w=20.0
    )
    assert res.binding[0]["kind"] == "thd" and res.binding[0]["node_id"] == 4
    zero = {
        a.id: {"p_w": 0.0, "q_var": 0.0} for a in grid.appliances if isinstance(a, Load)
    }

    def thd_at(p: float) -> float:
        gen = Generator(
            id=99, node=4, phases=ABC, connection=WindingConnection.WYE, p_nom_w=p
        )
        g = grid.model_copy(update={"appliances": [*grid.appliances, gen]})
        st = simulate(
            g,
            cfg.model_copy(update={"operating_point": zero}),
            harmonic_injection={99: emission},
        )
        return max(float(st.thd(4, ph).max()) for ph in ABC)

    assert thd_at(float(res.p_max_w[0])) <= 0.01
    assert thd_at(float(res.p_max_w[0]) + 2.0 * 20.0) > 0.01
    with pytest.raises(InputError):
        hosting_capacity(grid, [4], limits=lim)


def test_harmonic_calculation_keeps_the_fundamental_answer():
    # a zero-nameplate candidate load carries a harmonic shunt built from zero power;
    # the fundamental, and so the voltage-band answer, must not change
    grid = lv_feeder_with_switch()
    cfg = SimulationConfig(
        calculation="harmonic", harmonic_orders=[1, 5], load_shunt="opendss"
    )
    kw = dict(kind="load", p_max_w=400.0e3, tol_w=100.0)
    harmonic = hosting_capacity(grid, [3, 4], config=cfg, **kw)
    fundamental = hosting_capacity(grid, [3, 4], **kw)
    np.testing.assert_array_equal(harmonic.p_max_w, fundamental.p_max_w)
    assert harmonic.binding == fundamental.binding


def test_base_case_violation_reports_zero():
    res = hosting_capacity(two_bus(), limits=HostingLimits(v_max_pu=0.99))
    assert not res.base_case_feasible
    assert res.p_max_w[0] == 0.0
    assert res.binding[0]["limit"] == "v_max_pu"
    assert res.margins["v_max_pu"][0] < 0.0


def test_base_operating_point_options():
    grid = lv_feeder_with_switch()
    worst = hosting_capacity(grid, [4], tol_w=50.0)
    nameplate = hosting_capacity(
        grid, [4], base_operating_point="nameplate", tol_w=50.0
    )
    custom = hosting_capacity(
        grid,
        [4],
        base_operating_point={2: {"p_w": 0.0, "q_var": 0.0}, 3: {"p_w": 0.0}},
        tol_w=50.0,
    )
    assert worst.base_operating_point == "worst_case"
    assert custom.base_operating_point == "custom"
    # local demand absorbs part of the new generation
    assert nameplate.p_max_w[0] > worst.p_max_w[0]
    # with no demand the custom base is the worst case up to the 5 kvar of load 3 left
    assert abs(custom.p_max_w[0] - worst.p_max_w[0]) < 0.05 * worst.p_max_w[0]
    with pytest.raises(InputError):
        hosting_capacity(grid, [4], base_operating_point="peak")


def test_round_trip_and_input_is_not_mutated():
    grid = lv_feeder_with_switch()
    before = grid.model_dump()
    res = hosting_capacity(grid, limits=HostingLimits(v_max_pu=1.03, vuf_max=0.02))
    assert grid.model_dump() == before
    data = json.loads(json.dumps(res.to_dict()))
    back = HostingCapacity.from_dict(data)
    assert back.node_ids == res.node_ids and back.binding == res.binding
    np.testing.assert_array_equal(back.p_max_w, res.p_max_w)
    for key in _MARGIN_KEYS:
        np.testing.assert_array_equal(back.margins[key], res.margins[key])
    assert back.to_dict() == data
    entry = res.result_for(res.node_ids[-1])
    assert entry["p_max_w"] == float(res.p_max_w[-1])
    assert data["margins"]["thd"] == [None] * len(res.node_ids)


def test_default_candidates_skip_source_buses():
    res = hosting_capacity(two_bus(), p_max_w=1.0e3)
    assert res.node_ids == (2,)


def test_complex64_agrees_with_complex128():
    grid = lv_feeder_with_switch()
    kw = dict(limits=HostingLimits(v_max_pu=1.03), p_max_w=200.0e3, tol_w=100.0)
    r128 = hosting_capacity(grid, [3, 4], **kw)
    r64 = hosting_capacity(grid, [3, 4], dtype=torch.complex64, **kw)
    # single precision resolves the voltage to ~1e-6 pu; the limit then moves by far
    # less than one bisection tolerance, so the two land within one step of each other
    assert np.all(np.abs(r64.p_max_w - r128.p_max_w) <= 100.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "storage"},
        {"power_factor": 0.0},
        {"power_factor": 1.2},
        {"p_max_w": -1.0},
        {"tol_w": 0.0},
        {"harmonic_emission": {5: (0.1, 0.0)}},
        {"config": SimulationConfig(calculation="power_flow", operating_point={})},
        {"config": SimulationConfig(calculation="harmonic", harmonic_orders=[5])},
    ],
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(InputError):
        hosting_capacity(two_bus(), **kwargs)


def test_invalid_candidates_raise():
    with pytest.raises(InputError):
        hosting_capacity(two_bus(), [7])
    with pytest.raises(InputError):
        hosting_capacity(two_bus(), [2, 2])
    with pytest.raises(InputError):
        hosting_capacity(two_bus(), [])
    assert Phase.A in two_bus().nodes[1].phases
