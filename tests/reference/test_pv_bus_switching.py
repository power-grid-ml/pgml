"""PV-to-PQ switching of two interacting generators: release and the round cap.

Three-bus 400 V feeder. Unit A (node 1) has a tight reactive band and a high setpoint,
unit B (node 2) a wide band and a setpoint slightly below nominal. The first solve
needs far more reactive power from A than it has and drives B below its lower limit,
so both are pinned in round one. With A saturated the voltage at B falls below B's
setpoint, which releases B back to regulation in round two.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import pytest
import torch

import pgml.solver.power_flow as power_flow
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Load,
    Node,
    Phase,
    VoltageRegulation,
)
from pgml.solver import solve_power_flow

from ..differentiability.test_pv_bus_gradcheck import ABC, CDT, U_RATED, _line, _src

GEN_A, GEN_B = 10, 11
V_SET_B = 0.992


def _grid() -> Grid:
    def gen(gid, node, v_set, q_min, q_max):
        return Generator(
            id=gid,
            node=node,
            phases=ABC,
            p_nom_w=1000.0,
            voltage_regulation=VoltageRegulation(
                v_set_pu=v_set, q_min_var=q_min, q_max_var=q_max
            ),
        )

    lines = [
        _line().model_copy(update={"id": k, "from_node": k, "to_node": k + 1})
        for k in range(2)
    ]
    return Grid(
        nodes=[Node(id=i, u_rated_v=U_RATED, phases=ABC) for i in range(3)],
        branches=lines,
        appliances=[
            _src(),
            Load(id=3, node=2, phases=ABC, p_nom_w=9000.0, q_nom_var=3000.0),
            gen(GEN_A, 1, 1.05, -500.0, 500.0),
            gen(GEN_B, 2, V_SET_B, -3000.0, 20000.0),
        ],
    )


def _solve(grid, **kw):
    return solve_power_flow(
        grid,
        method="newton",
        tol=1e-10,
        max_iter=60,
        dtype=CDT,
        criticality="never",
        **kw,
    )


def _v_pu(res, node):
    return res.v[..., res.index.row(node, Phase.A)].abs() / (U_RATED / math.sqrt(3.0))


def test_pinned_unit_is_released_back_to_regulation():
    res = _solve(_grid())
    reg = res.regulation

    assert res.converged
    assert reg.switch_rounds == 2  # pin both, then release B
    assert not bool(reg.regulating[GEN_A])
    assert bool(reg.regulating[GEN_B])
    assert bool(reg.settled)
    assert reg.unsettled_generators == ()
    assert float(reg.q_var[GEN_A]) == pytest.approx(500.0, abs=1e-6)
    assert -3000.0 < float(reg.q_var[GEN_B]) < 20000.0
    assert float(_v_pu(res, 2)) == pytest.approx(V_SET_B, abs=1e-9)


def _cap_rounds(monkeypatch, cap: int) -> None:
    original = power_flow.collect_pv_terminals

    def capped(*args, **kwargs):
        pv = original(*args, **kwargs)
        return None if pv is None else replace(pv, max_rounds=cap)

    monkeypatch.setattr(power_flow, "collect_pv_terminals", capped)


def test_round_cap_reports_the_solve_as_not_converged(monkeypatch, caplog):
    _cap_rounds(monkeypatch, 2)
    with caplog.at_level(logging.WARNING, logger="pgml"):
        res = _solve(_grid())
    reg = res.regulation

    # The kept active set leaves B pinned at q_min below its setpoint.
    assert not bool(reg.regulating[GEN_B])
    assert float(_v_pu(res, 2)) < V_SET_B - 1e-4
    assert not res.converged
    assert not res.diagnostics.converged
    assert "switching did not settle" in res.diagnostics.likely_cause
    assert not bool(reg.settled)
    assert reg.unsettled_generators == (GEN_B,)
    msgs = [r.getMessage() for r in caplog.records if "did not settle" in r.message]
    assert msgs and f"[{GEN_B}]" in msgs[0]


def test_round_cap_marks_only_the_unsettled_scenarios(monkeypatch):
    """Scenario 0 settles within the cap (B never leaves its band), scenario 1 not."""
    _cap_rounds(monkeypatch, 2)
    v_set_a = torch.tensor([1.0, 1.05], dtype=torch.float64)
    free = _solve(_grid(), operating_point={GEN_A: {"v_set_pu": v_set_a[:1]}})
    assert free.regulation.switch_rounds <= 1 and free.converged

    res = _solve(_grid(), operating_point={GEN_A: {"v_set_pu": v_set_a}})
    assert res.regulation.settled.tolist() == [True, False]
    assert res.converged_mask.tolist() == [True, False]
    assert res.failed_states == (1,)
    assert not res.converged
