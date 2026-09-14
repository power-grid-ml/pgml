"""GPU gate: CPU-vs-CUDA parity for the voltage-regulating generator (PV terminal).

The regulated row pair, the reactive-power readout and the PV-to-PQ switching decision
must all behave identically on CPU and CUDA (the test skips cleanly without CUDA). It
pins the device handling of the tensors built in ``pgml.solver._pv_bus``: the row
gather / scatter, the positive-sequence rotation, the setpoint and limit tensors, the
int8 active set, and the batched (per-scenario) setpoint.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    RegulatedQuantity,
    Source,
    VoltageRegulation,
)
from pgml.solver import solve_power_flow

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

ABC = (Phase.A, Phase.B, Phase.C)
U_RATED = 400.0
GEN_ID = 2


def _feeder(regulation, *, load_w=9000.0):
    nodes = [
        Node(id=0, u_rated_v=U_RATED, phases=ABC),
        Node(id=1, u_rated_v=U_RATED, phases=ABC),
    ]
    src = Source(
        id=0,
        node=0,
        phases=ABC,
        u_ref_v=[U_RATED / math.sqrt(3.0)] * 3,
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
    gen = Generator(
        id=GEN_ID, node=1, phases=ABC, p_nom_w=6000.0, voltage_regulation=regulation
    )
    load = Load(id=3, node=1, phases=ABC, p_nom_w=load_w, q_nom_var=3000.0)
    return Grid(nodes=nodes, branches=[line], appliances=[src, load, gen])


def _solve(regulation, device, *, operating_point=None, load_w=9000.0):
    return solve_power_flow(
        _feeder(regulation, load_w=load_w),
        method="newton",
        device=torch.device(device),
        tol=1e-10,
        max_iter=60,
        criticality="never",
        operating_point=operating_point,
    )


def _parity(regulation, **kw):
    cpu = _solve(regulation, "cpu", **kw)
    cuda = _solve(regulation, "cuda", **kw)
    assert cuda.v.device.type == "cuda"
    assert torch.allclose(cpu.v, cuda.v.cpu(), atol=1e-8, rtol=1e-6)
    q_cpu = cpu.regulation.q_var[GEN_ID]
    q_cuda = cuda.regulation.q_var[GEN_ID]
    assert q_cuda.device.type == "cuda"
    assert torch.allclose(q_cpu, q_cuda.cpu(), atol=1e-6, rtol=1e-9)
    assert torch.equal(
        cpu.regulation.regulating[GEN_ID], cuda.regulation.regulating[GEN_ID].cpu()
    )
    return cpu, cuda


def test_positive_sequence_setpoint_parity():
    cpu, _ = _parity(VoltageRegulation(v_set_pu=1.02))
    assert bool(cpu.regulation.regulating[GEN_ID])


def test_per_phase_setpoint_parity():
    _parity(VoltageRegulation(v_set_pu=1.02, regulated=RegulatedQuantity.PER_PHASE))


def test_pinned_at_a_reactive_limit_parity():
    """The switching decision (an int8 active set) must match on both devices."""
    cpu, _ = _parity(
        VoltageRegulation(v_set_pu=1.08, q_min_var=-1500.0, q_max_var=1500.0)
    )
    assert not bool(cpu.regulation.regulating[GEN_ID])
    assert cpu.regulation.switch_rounds >= 1


def test_batched_setpoint_parity():
    v_set = torch.tensor([0.99, 1.02, 1.05], dtype=torch.float64)
    cpu = _solve(
        VoltageRegulation(v_set_pu=1.0),
        "cpu",
        operating_point={GEN_ID: {"v_set_pu": v_set}},
    )
    cuda = _solve(
        VoltageRegulation(v_set_pu=1.0),
        "cuda",
        operating_point={GEN_ID: {"v_set_pu": v_set.cuda()}},
    )
    assert cpu.v.shape == cuda.v.shape == (3, 6)
    assert torch.allclose(cpu.v, cuda.v.cpu(), atol=1e-8, rtol=1e-6)
    assert torch.allclose(
        cpu.regulation.q_var[GEN_ID],
        cuda.regulation.q_var[GEN_ID].cpu(),
        atol=1e-6,
        rtol=1e-9,
    )


def test_complex64_setpoint_is_honored_on_cuda():
    """The single-precision path keeps the setpoint to its own resolution."""
    res = solve_power_flow(
        _feeder(VoltageRegulation(v_set_pu=1.02)),
        method="newton",
        device=torch.device("cuda"),
        dtype=torch.complex64,
        tol=1e-3,
        max_iter=60,
        criticality="never",
    )
    assert res.v.dtype == torch.complex64
    row = res.index.row(1, Phase.A)
    v_pu = float(res.v[row].abs()) / (U_RATED / math.sqrt(3.0))
    assert abs(v_pu - 1.02) < 1e-5


def test_gradient_parity_through_the_setpoint():
    """``dV*/dv_set`` agrees on both devices (the IFT backward with a PV row)."""
    grads = {}
    for device in ("cpu", "cuda"):
        v_set = torch.tensor(
            1.02, dtype=torch.float64, device=device, requires_grad=True
        )
        res = _solve(VoltageRegulation(v_set_pu=v_set), device)
        res.v.abs().sum().backward()
        grads[device] = float(v_set.grad)
    assert abs(grads["cpu"] - grads["cuda"]) < 1e-6 * abs(grads["cpu"])
