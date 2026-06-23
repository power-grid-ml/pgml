"""Storage element: snapshot injection sign + state-of-charge / dispatch resolution.

The snapshot tests confirm a :class:`Storage` injects with the generator-consistent sign
(``p_nom_w > 0`` = discharge/inject) and is honoured by the solver exactly like a
:class:`Generator`. The dispatch tests pin the off-tape SoC integrator
(:func:`pgml.scenarios.integrate_soc`): energy balance, reserve / capacity curtailment,
efficiency, rating clamp. See ``references/der_pv_storage_modeling.md`` §4.4.
"""

from __future__ import annotations

import math

import torch

from pgml.scenarios import dispatch_storage, integrate_soc, storage_operating_point
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Node,
    Phase,
    Source,
    Storage,
)
from pgml.solver import solve_power_flow

ABC = (Phase.A, Phase.B, Phase.C)


# --------------------------------------------------------------------------- #
# SoC integration
# --------------------------------------------------------------------------- #
def test_soc_energy_balance_no_clamp():
    """With ample headroom and unit efficiency, energy follows P*dt exactly."""
    res = integrate_soc(
        [1000.0, -500.0], dt_s=3600.0, energy_capacity_wh=10000.0, soc0=0.5
    )
    # discharge 1000 Wh, then charge 500 Wh -> 5000 -> 4000 -> 4500.
    assert torch.allclose(
        res.energy_wh, torch.tensor([5000.0, 4000.0, 4500.0], dtype=torch.float64)
    )
    assert torch.allclose(
        res.realized_power_w, torch.tensor([1000.0, -500.0], dtype=torch.float64)
    )


def test_soc_efficiency_applied():
    """Charging adds eff*|P|*dt; discharging removes P*dt/eff."""
    res = integrate_soc(
        [-1000.0],
        dt_s=3600.0,
        energy_capacity_wh=10000.0,
        soc0=0.5,
        efficiency_charge=0.9,
    )
    assert math.isclose(res.energy_wh[-1].item(), 5000.0 + 900.0, rel_tol=1e-12)
    res2 = integrate_soc(
        [1000.0],
        dt_s=3600.0,
        energy_capacity_wh=10000.0,
        soc0=0.5,
        efficiency_discharge=0.8,
    )
    assert math.isclose(res2.energy_wh[-1].item(), 5000.0 - 1250.0, rel_tol=1e-12)


def test_soc_reserve_and_cap_curtail():
    """Requested power is curtailed so SoC stays within [soc_min, soc_max]."""
    res = integrate_soc(
        [5000.0, 5000.0],
        dt_s=3600.0,
        energy_capacity_wh=2000.0,
        soc0=0.5,
        soc_min=0.2,
        soc_max=1.0,
    )
    # only (0.5-0.2)*2000 = 600 Wh deliverable, then 0.
    assert math.isclose(res.realized_power_w[0].item(), 600.0, rel_tol=1e-9)
    assert abs(res.realized_power_w[1].item()) < 1e-9
    assert res.soc.min().item() >= 0.2 - 1e-9


def test_soc_power_rating_clamp():
    res = integrate_soc(
        [9999.0], dt_s=3600.0, energy_capacity_wh=1e6, soc0=0.5, p_rated_w=5000.0
    )
    assert math.isclose(res.realized_power_w[0].item(), 5000.0, rel_tol=1e-12)


def test_soc_no_capacity_only_rating():
    res = integrate_soc([8000.0, -8000.0], dt_s=3600.0, p_rated_w=5000.0)
    assert res.soc is None and res.energy_wh is None
    assert torch.allclose(
        res.realized_power_w, torch.tensor([5000.0, -5000.0], dtype=torch.float64)
    )


def test_soc_batched():
    """A leading scenario batch is integrated element-wise."""
    req = torch.tensor([[1000.0, 1000.0], [-1000.0, -1000.0]], dtype=torch.float64)
    res = integrate_soc(req, dt_s=3600.0, energy_capacity_wh=10000.0, soc0=0.5)
    assert res.soc.shape == (2, 3)
    assert res.soc[0, -1].item() < 0.5  # discharging scenario
    assert res.soc[1, -1].item() > 0.5  # charging scenario


def test_dispatch_storage_reads_element():
    s = Storage(
        id=7,
        node=0,
        phases=ABC,
        p_nom_w=0.0,
        energy_capacity_wh=2000.0,
        soc=0.5,
        soc_min=0.2,
        efficiency_discharge=0.9,
    )
    res = dispatch_storage(s, [5000.0], dt_s=3600.0)
    # reserve-limited: (0.5-0.2)*2000*0.9 = 540 W deliverable in the hour.
    assert math.isclose(res.realized_power_w[0].item(), 540.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Snapshot injection in a feeder
# --------------------------------------------------------------------------- #
def _feeder_with(appliance):
    nodes = [
        Node(id=0, u_rated_v=400.0, phases=ABC),
        Node(id=1, u_rated_v=400.0, phases=ABC),
    ]
    src = Source(
        id=0,
        node=0,
        phases=ABC,
        u_ref_v=[230.94] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[2e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    line = Line(
        id=0,
        from_node=0,
        to_node=1,
        from_phases=ABC,
        to_phases=ABC,
        length_m=300.0,
        series_resistance_ohm_per_m=[
            [3e-4 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        series_inductance_h_per_m=[
            [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
    )
    return Grid(nodes=nodes, branches=[line], appliances=[src, appliance])


def _mean_vpu(res):
    return (res.v.abs()[3:] / (400.0 / math.sqrt(3.0))).mean().item()


def test_storage_discharge_matches_generator():
    """Discharging storage and a generator of equal P inject identically."""
    st = Storage(id=2, node=1, phases=ABC, p_nom_w=15000.0, energy_capacity_wh=1e5)
    gen = Generator(id=2, node=1, phases=ABC, p_nom_w=15000.0)
    vs = solve_power_flow(_feeder_with(st)).v
    vg = solve_power_flow(_feeder_with(gen)).v
    assert torch.allclose(vs, vg, atol=1e-9)


def test_storage_charge_lowers_voltage():
    discharge = _mean_vpu(
        solve_power_flow(
            _feeder_with(
                Storage(
                    id=2, node=1, phases=ABC, p_nom_w=15000.0, energy_capacity_wh=1e5
                )
            )
        )
    )
    charge = _mean_vpu(
        solve_power_flow(
            _feeder_with(
                Storage(
                    id=2, node=1, phases=ABC, p_nom_w=-15000.0, energy_capacity_wh=1e5
                )
            )
        )
    )
    assert discharge > charge


def test_storage_operating_point_override():
    """A dispatch setpoint fed via operating_point equals the directly-authored value."""
    base = Storage(id=2, node=1, phases=ABC, p_nom_w=0.0, energy_capacity_wh=1e5)
    direct = Storage(id=2, node=1, phases=ABC, p_nom_w=12000.0, energy_capacity_wh=1e5)
    op = storage_operating_point({2: 12000.0})
    v_op = solve_power_flow(_feeder_with(base), operating_point=op).v
    v_dir = solve_power_flow(_feeder_with(direct)).v
    assert torch.allclose(v_op, v_dir, atol=1e-9)
