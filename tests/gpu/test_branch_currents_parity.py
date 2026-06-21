"""GPU gate: CPU-vs-CUDA parity of ``branch_currents`` (skips cleanly without CUDA)."""

from __future__ import annotations

import math

import pytest
import torch

from pgml.assembly import (
    assemble_network_ybus,
    branch_currents,
    node_phase_index,
)
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Line,
    Node,
    Phase,
    ShuntReactor,
    Source,
    Switch,
    Transformer,
    WindingConnection,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

ABC = (Phase.A, Phase.B, Phase.C)


def _sym(diag: float, off: float) -> list[list[float]]:
    return [[diag if i == j else off for j in range(3)] for i in range(3)]


def _grid() -> Grid:
    nodes = [
        Node(id=1, u_rated_v=20_000.0, phases=ABC),
        Node(id=2, u_rated_v=400.0, phases=ABC),
        Node(id=3, u_rated_v=400.0, phases=ABC),
        Node(id=4, u_rated_v=400.0, phases=ABC),
    ]
    xfmr = Transformer(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=ABC,
        to_phases=ABC,
        s_rated_va=0.4e6,
        u_rated_from_v=20_000.0,
        u_rated_to_v=400.0,
        from_connection=WindingConnection.DELTA,
        to_connection=WindingConnection.WYE_GROUNDED,
        series_resistance_ohm=0.01,
        series_inductance_h=1.0e-4,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=30.0),
    )
    line = Line(
        id=21,
        from_node=2,
        to_node=3,
        from_phases=ABC,
        to_phases=ABC,
        length_m=100.0,
        series_resistance_ohm_per_m=_sym(1.0e-3, 1.0e-4),
        series_inductance_h_per_m=_sym(1.0e-6, 1.0e-7),
        shunt_capacitance_f_per_m=_sym(1.0e-9, 1.0e-10),
    )
    sw = Switch(
        id=22,
        from_node=3,
        to_node=4,
        from_phases=ABC,
        to_phases=ABC,
        closed=True,
        resistance_ohm=1.0e-3,
        inductance_h=1.0e-7,
    )
    reactor = ShuntReactor(
        id=23,
        from_node=4,
        to_node=4,
        from_phases=ABC,
        to_phases=ABC,
        conductance_s=_sym(1.0e-4, 0.0),
        capacitance_f=_sym(1.0e-7, 0.0),
    )
    src = Source(
        id=10,
        node=1,
        phases=ABC,
        u_ref_v=(20_000.0 / math.sqrt(3),) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=_sym(0.5, 0.0),
        inductance_h=_sym(5.0e-3, 0.0),
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[xfmr, line, sw, reactor],
        appliances=[src],
    )


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_cpu_cuda_branch_currents_parity(dtype):
    grid = _grid()
    freqs = [50.0, 250.0]
    index = node_phase_index(grid)
    n = index.size

    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    torch.manual_seed(0)
    v_cpu = torch.randn(len(freqs), n, dtype=dtype)

    bc_cpu = branch_currents(grid, v_cpu, freqs, index, dtype=dtype)

    dev = torch.device("cuda")
    f_cuda = torch.tensor(freqs, dtype=rdt, device=dev)
    bc_cuda = branch_currents(grid, v_cpu.to(dev), f_cuda, index, dtype=dtype)

    tol = 1e-9 if dtype == torch.complex128 else 1e-3
    for a, b in zip(bc_cpu, bc_cuda):
        assert b.i_from.device.type == "cuda"
        assert b.i_to.device.type == "cuda"
        torch.testing.assert_close(b.i_from.cpu(), a.i_from, rtol=tol, atol=tol)
        torch.testing.assert_close(b.i_to.cpu(), a.i_to, rtol=tol, atol=tol)


def test_cpu_cuda_branch_currents_kcl_on_cuda():
    """KCL holds on CUDA: scattered terminal currents == Y_net @ V."""
    grid = _grid()
    freqs = [50.0, 250.0]
    index = node_phase_index(grid)
    n = index.size
    dev = torch.device("cuda")

    f = torch.tensor(freqs, dtype=torch.float64, device=dev)
    yb = assemble_network_ybus(grid, f, dtype=torch.complex128, device=dev)
    v = torch.randn(len(freqs), n, dtype=torch.complex128, device=dev)
    yv = torch.einsum("hij,hj->hi", yb.Y, v)

    bcs = branch_currents(grid, v, f, index, dtype=torch.complex128, device=dev)
    acc = torch.zeros(len(freqs), n, dtype=torch.complex128, device=dev)
    for bc in bcs:
        from_rows = [index.row(bc.from_node, ph) for ph in bc.from_phases]
        acc[..., from_rows] += bc.i_from
        if bc.to_node is not None:
            to_rows = [index.row(bc.to_node, ph) for ph in bc.to_phases]
            acc[..., to_rows] += bc.i_to
    assert (acc - yv).abs().max().item() < 1e-9
