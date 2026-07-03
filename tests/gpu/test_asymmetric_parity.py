"""GPU gate: CPU-vs-CUDA parity of the connection-aware load models.

A DELTA load and a WYE-with-neutral load grid must assemble + solve identically on
CPU and CUDA (skips cleanly without CUDA). Pins device/dtype honoring of the
incidence stamp + current injection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_ybus, device_current_injections, node_phase_index
from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)


def _source(node_phases):
    n = len(node_phases)
    angles = [0.0, -120.0, 120.0, 0.0][:n]
    mags = [231.0, 231.0, 231.0, 0.0][:n]
    r = [[0.05 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[1e-4 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Source(
        id=10,
        node=1,
        phases=node_phases,
        u_ref_v=tuple(mags),
        u_angle_deg=tuple(angles),
        resistance_ohm=r,
        inductance_h=ll,
    )


def _line(node_phases):
    n = len(node_phases)
    r = [[2e-3 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[2e-6 if i == j else 0.0 for j in range(n)] for i in range(n)]
    c = [[0.0] * n for _ in range(n)]
    return Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=node_phases,
        to_phases=node_phases,
        length_m=50.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=ll,
        shunt_capacitance_f_per_m=c,
    )


def _grid(load, node_phases):
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=node_phases),
            Node(id=2, u_rated_v=400.0, phases=node_phases),
        ],
        branches=[_line(node_phases)],
        appliances=[_source(node_phases), load],
    )


DELTA_LOAD = Load(
    id=30,
    node=2,
    phases=ABC,
    p_nom_w=4500.0,
    q_nom_var=900.0,
    p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
    q_nom_per_phase_var=(400.0, 300.0, 200.0),
    connection=WindingConnection.DELTA,
)
WYE_N_LOAD = Load(
    id=30,
    node=2,
    phases=ABC,
    p_nom_w=4500.0,
    q_nom_var=900.0,
    p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
    q_nom_per_phase_var=(400.0, 300.0, 200.0),
)

CASES = [("delta", DELTA_LOAD, ABC), ("wye_neutral", WYE_N_LOAD, ABCN)]


@pytest.mark.parametrize("name,load,nph", CASES, ids=[c[0] for c in CASES])
def test_assemble_and_solve_cpu_cuda_parity(name, load, nph):
    grid = _grid(load, nph)
    dev = torch.device("cuda")

    yb_cpu = assemble_ybus(
        grid, 50.0, dtype=torch.complex128, device=torch.device("cpu")
    )
    yb_cuda = assemble_ybus(grid, 50.0, dtype=torch.complex128, device=dev)
    assert yb_cuda.Y.device.type == "cuda"
    torch.testing.assert_close(yb_cuda.Y.cpu(), yb_cpu.Y, rtol=1e-12, atol=1e-12)

    res_cpu = solve_power_flow(
        grid,
        slack="ideal",
        dtype=torch.complex128,
        device=torch.device("cpu"),
        symmetry="asymmetric",
    )
    res_cuda = solve_power_flow(
        grid, slack="ideal", dtype=torch.complex128, device=dev, symmetry="asymmetric"
    )
    assert res_cuda.v.device.type == "cuda"
    torch.testing.assert_close(res_cuda.v.cpu(), res_cpu.v, rtol=1e-9, atol=1e-9)


def test_device_current_injection_cpu_cuda_parity():
    grid = _grid(WYE_N_LOAD, ABCN)
    grid_z = Grid(
        base_frequency_hz=50.0,
        nodes=grid.nodes,
        branches=grid.branches,
        appliances=[
            grid.appliances[0],
            Load(
                id=30,
                node=2,
                phases=ABC,
                p_nom_w=4500.0,
                q_nom_var=900.0,
                p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
                q_nom_per_phase_var=(400.0, 300.0, 200.0),
                load_model=LoadModel.CONST_IMPEDANCE,
            ),
        ],
    )
    index = node_phase_index(grid_z)
    u_ln = 400.0 / np.sqrt(3.0)
    v = torch.tensor(
        [
            231.0,
            231 * np.exp(-2j * np.pi / 3),
            231 * np.exp(2j * np.pi / 3),
            0.0,
            u_ln,
            u_ln * np.exp(-2j * np.pi / 3),
            u_ln * np.exp(2j * np.pi / 3),
            1.0 + 1j,
        ],
        dtype=torch.complex128,
    )
    dev = torch.device("cuda")
    i_cpu = device_current_injections(grid_z, v, index, [50.0], dtype=torch.complex128)
    i_cuda = device_current_injections(
        grid_z, v.to(dev), index, [50.0], dtype=torch.complex128, device=dev
    )
    assert i_cuda.device.type == "cuda"
    torch.testing.assert_close(i_cuda.cpu(), i_cpu, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Connection-aware per-phase harmonic injection CPU-vs-CUDA parity.
# ---------------------------------------------------------------------------
def _spec(comps):
    return StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a)
                for o, m, a in comps
            ]
        )
    )


DELTA_HARM_LOAD = Load(
    id=30,
    node=2,
    phases=ABC,
    p_nom_w=4500.0,
    q_nom_var=900.0,
    p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
    q_nom_per_phase_var=(400.0, 300.0, 200.0),
    connection=WindingConnection.DELTA,
    spectrum=_spec([(1, 1.0, 0.0), (5, 0.2, 0.0), (7, 0.14, 30.0)]),
)
WYE_PER_PHASE_LOAD = Load(
    id=30,
    node=2,
    phases=ABC,
    p_nom_w=4500.0,
    q_nom_var=900.0,
    p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
    q_nom_per_phase_var=(400.0, 300.0, 200.0),
    spectrum_per_phase={
        Phase.A: _spec([(1, 1.0, 0.0), (5, 0.3, 10.0)]),
        Phase.C: _spec([(1, 1.0, 0.0), (7, 0.1, -20.0)]),
    },
)

HARM_CASES = [
    ("delta_spectrum", DELTA_HARM_LOAD, ABC),
    ("wye_per_phase", WYE_PER_PHASE_LOAD, ABCN),
]


@pytest.mark.parametrize("name,load,nph", HARM_CASES, ids=[c[0] for c in HARM_CASES])
def test_harmonic_per_phase_cpu_cuda_parity(name, load, nph):
    grid = _grid(load, nph)
    dev = torch.device("cuda")
    res_cpu = solve_harmonic_flow(
        grid,
        [1, 5, 7],
        slack="norton",
        dtype=torch.complex128,
        device=torch.device("cpu"),
        symmetry="asymmetric",
    )
    res_cuda = solve_harmonic_flow(
        grid,
        [1, 5, 7],
        slack="norton",
        dtype=torch.complex128,
        device=dev,
        symmetry="asymmetric",
    )
    assert res_cuda.v.device.type == "cuda"
    torch.testing.assert_close(res_cuda.v.cpu(), res_cpu.v, rtol=1e-9, atol=1e-9)
