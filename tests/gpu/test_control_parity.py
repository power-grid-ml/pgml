"""GPU gate: CPU-vs-CUDA parity for inverter control + storage injections.

A Volt-VAr / Volt-Watt controlled generator and a storage element must assemble and solve
identically on CPU and CUDA (skips cleanly without CUDA). Pins device/dtype honoring of
the control evaluation (``_control``) and the storage injection on the differentiable path.
"""

from __future__ import annotations


import pytest
import torch

from pgml.schemas.grid_schema import (
    Characteristic,
    Generator,
    Grid,
    Line,
    Node,
    Phase,
    Source,
    Storage,
    VoltVarControl,
    VoltWattControl,
)
from pgml.solver import solve_power_flow

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

ABC = (Phase.A, Phase.B, Phase.C)


def _feeder(appliance):
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


def _parity(appliance, method="newton"):
    cpu = solve_power_flow(
        _feeder(appliance), method=method, device=torch.device("cpu")
    ).v
    cuda = solve_power_flow(
        _feeder(appliance), method=method, device=torch.device("cuda")
    ).v
    assert torch.allclose(cpu, cuda.cpu(), atol=1e-8, rtol=1e-6)


def test_volt_var_parity():
    ctrl = VoltVarControl(
        s_rated_va=35000.0,
        smoothing=0.02,
        characteristic=Characteristic(
            x_values=[0.95, 1.0, 1.05], y_values=[1.0, 0.0, -1.0]
        ),
    )
    _parity(Generator(id=2, node=1, phases=ABC, p_nom_w=30000.0, control=ctrl))


def test_volt_watt_parity():
    ctrl = VoltWattControl(
        s_rated_va=35000.0,
        characteristic=Characteristic(x_values=[1.0, 1.05], y_values=[1.0, 0.0]),
    )
    _parity(Generator(id=2, node=1, phases=ABC, p_nom_w=30000.0, control=ctrl))


def test_storage_parity():
    _parity(
        Storage(id=2, node=1, phases=ABC, p_nom_w=15000.0, energy_capacity_wh=1e5),
        method="current_injection",
    )
