"""GPU gate: CPU-vs-CUDA parity of the harmonic device shunt (skips without CUDA).

The shunt adds a device-grouped stamp to ``Y(h)`` built from the operating point, so it
introduces new tensor work on the harmonic path: a complex element admittance per order,
the ``M^T diag(y) M`` incidence contraction and an ``index_add`` scatter. All of it must
honour the caller's device and dtype, for a WYE and a DELTA device, for the motor
variant, and with a BATCHED operating point (which promotes ``Y(h)`` to
``[*batch, H, N, N]`` and makes every scenario factor its own matrix).
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

F0 = 50.0
W0 = 2.0 * math.pi * F0
CDT = torch.complex128
ABC = (Phase.A, Phase.B, Phase.C)

_SPECTRUM = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
            HarmonicComponent(order=7, magnitude_pu=0.1, phase_deg=0.0),
        ]
    )
)


def _grid(connection: WindingConnection) -> Grid:
    def diag(value: float):
        return [[value if i == j else 0.0 for j in range(3)] for i in range(3)]

    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABC),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                length_m=1.0,
                series_resistance_ohm_per_m=diag(0.05),
                series_inductance_h_per_m=diag(0.05 / W0),
                shunt_capacitance_f_per_m=diag(0.0),
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=(400.0 / math.sqrt(3.0),) * 3,
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=diag(0.05),
                inductance_h=diag(0.05 / W0),
            ),
            Load(
                id=2,
                node=2,
                phases=ABC,
                p_nom_w=9000.0,
                q_nom_var=3000.0,
                connection=connection,
                spectrum=_SPECTRUM,
            ),
        ],
    )


def _sparse_shunt_grid() -> Grid:
    """Five three-phase nodes with one device select the low-rank shunt path."""
    grid = _grid(WindingConnection.DELTA)

    def diag(value: float):
        return [[value if i == j else 0.0 for j in range(3)] for i in range(3)]

    nodes = list(grid.nodes) + [
        Node(id=i, u_rated_v=400.0, phases=ABC) for i in range(3, 6)
    ]
    branches = list(grid.branches) + [
        Line(
            id=i,
            from_node=i - 1,
            to_node=i,
            from_phases=ABC,
            to_phases=ABC,
            length_m=1.0,
            series_resistance_ohm_per_m=diag(0.05),
            series_inductance_h_per_m=diag(0.05 / W0),
            shunt_capacitance_f_per_m=diag(0.0),
        )
        for i in range(3, 6)
    ]
    return grid.model_copy(update={"nodes": nodes, "branches": branches})


def _run(
    device, *, connection=WindingConnection.WYE, shunt="opendss", dtype=CDT, op=None
):
    return solve_harmonic_flow(
        _grid(connection),
        [1, 5, 7],
        slack="norton",
        dtype=dtype,
        device=device,
        operating_point=op,
        load_shunt=shunt,
    ).v


@pytest.mark.parametrize("connection", [WindingConnection.WYE, WindingConnection.DELTA])
@pytest.mark.parametrize("shunt", ["opendss", "motor"])
def test_cpu_cuda_parity(connection, shunt):
    cpu = _run(torch.device("cpu"), connection=connection, shunt=shunt)
    cuda = _run(torch.device("cuda"), connection=connection, shunt=shunt)
    assert cuda.device.type == "cuda"
    assert torch.allclose(cpu, cuda.cpu(), rtol=1e-9, atol=1e-9)


def test_cpu_cuda_parity_batched_operating_point():
    """A per-scenario operating point makes ``Y(h)`` batched; parity must hold there too."""

    def run(device):
        p = torch.tensor([6000.0, 9000.0, 12000.0], dtype=torch.float64, device=device)
        return _run(device, op={2: {"p_w": p}})

    cpu = run(torch.device("cpu"))
    cuda = run(torch.device("cuda"))
    assert cpu.shape == (3, 3, 6)
    assert torch.allclose(cpu, cuda.cpu(), rtol=1e-9, atol=1e-9)


def test_cpu_cuda_parity_sparse_shunt_woodbury():
    def run(device):
        p = torch.tensor([6000.0, 9000.0, 12000.0], dtype=torch.float64, device=device)
        return solve_harmonic_flow(
            _sparse_shunt_grid(),
            [1, 5, 7],
            slack="norton",
            dtype=CDT,
            device=device,
            operating_point={2: {"p_w": p}},
        ).v

    cpu = run(torch.device("cpu"))
    cuda = run(torch.device("cuda"))
    assert cuda.device.type == "cuda"
    assert torch.allclose(cpu, cuda.cpu(), rtol=1e-9, atol=1e-9)


def test_complex64_runs_on_both_devices():
    """The stamp honours the working dtype (no hard-coded complex128)."""
    cpu = _run(torch.device("cpu"), dtype=torch.complex64)
    cuda = _run(torch.device("cuda"), dtype=torch.complex64)
    assert cpu.dtype == torch.complex64 and cuda.dtype == torch.complex64
    assert torch.allclose(cpu, cuda.cpu(), rtol=1e-4, atol=1e-4)
