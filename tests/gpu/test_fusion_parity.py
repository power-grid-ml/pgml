"""GPU gate: CPU-vs-CUDA parity of a solve that fuses zero-impedance branches.

The fusion map is structural (int64 row indices and small constant real matrices), so the
only device-sensitive parts are the tensors it INDEXES: the reduced assembly's scatter,
the prolongation gather, and the constant map of the Kirchhoff current recovery. Each is
exercised here on both devices, including the batched switch-state path that coexists with
fusion. Skips cleanly without CUDA.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import branch_currents, device_current_injections, fusion_map
from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source, Switch
from pgml.solver import solve_harmonic_flow, solve_power_flow

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

ABC = (Phase.A, Phase.B, Phase.C)
CDT = torch.complex128


def _grid() -> Grid:
    """Four 20 kV nodes: source -- ideal 3-phase switch -- bus -- two feeders."""
    eye = [[(1.0e-6 if i == j else 0.0) for j in range(3)] for i in range(3)]
    ind = [[(1.0e-12 if i == j else 0.0) for j in range(3)] for i in range(3)]

    def line(bid, u, v, r, ell):
        return Line(
            id=bid,
            from_node=u,
            to_node=v,
            from_phases=ABC,
            to_phases=ABC,
            length_m=1_000.0,
            series_resistance_ohm_per_m=[
                [(r if i == j else 0.0) for j in range(3)] for i in range(3)
            ],
            series_inductance_h_per_m=[
                [(ell if i == j else 0.0) for j in range(3)] for i in range(3)
            ],
            shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            harmonic_line_model="naive",
        )

    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=20_000.0, phases=ABC) for i in (1, 2, 3, 4)],
        branches=[
            Switch(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                closed=True,
            ),
            line(11, 2, 3, 2.0e-4, 8.0e-7),
            line(12, 3, 4, 3.0e-4, 9.0e-7),
            # a finite-impedance tie a switch-state sweep can toggle
            Switch(
                id=13,
                from_node=2,
                to_node=4,
                from_phases=ABC,
                to_phases=ABC,
                closed=True,
                resistance_ohm=1.0e-4,
            ),
        ],
        appliances=[
            Source(
                id=20,
                node=1,
                phases=ABC,
                u_ref_v=(11_547.0,) * 3,
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=eye,
                inductance_h=ind,
            ),
            Load(id=21, node=2, phases=ABC, p_nom_w=3.0e5, q_nom_var=1.0e5),
            Load(id=22, node=4, phases=ABC, p_nom_w=9.0e5, q_nom_var=3.0e5),
        ],
    )


def _solve(device, **kwargs):
    return solve_power_flow(
        _grid(), dtype=CDT, device=device, tol=1e-12, tol_update_pu=1e-12, **kwargs
    )


def test_fused_power_flow_cpu_cuda_parity():
    cpu = _solve(torch.device("cpu"))
    cuda = _solve(torch.device("cuda"))
    assert cpu.fusion is not None and cuda.fusion is not None
    assert cpu.fusion.size == cuda.fusion.size == cpu.index.size - 3
    assert cuda.v.device.type == "cuda"
    assert torch.allclose(cpu.v, cuda.v.cpu(), atol=1e-9, rtol=1e-12)


def test_fused_harmonic_flow_cpu_cuda_parity():
    def run(device):
        return solve_harmonic_flow(
            _grid(), [1, 5], dtype=CDT, device=device, tol=1e-12, tol_update_pu=1e-12
        )

    cpu, cuda = run(torch.device("cpu")), run(torch.device("cuda"))
    assert cuda.v.device.type == "cuda"
    assert torch.allclose(cpu.v, cuda.v.cpu(), atol=1e-9, rtol=1e-12)


def test_fused_branch_current_cpu_cuda_parity():
    """The Kirchhoff recovery's constant map follows the defect's device."""

    def run(device):
        grid = _grid()
        res = _solve(device)
        i_inj = -device_current_injections(
            grid, res.v, res.index, [50.0], dtype=CDT, device=device
        )
        return next(
            bc
            for bc in branch_currents(
                grid,
                res.v.unsqueeze(-2),
                [50.0],
                res.index,
                dtype=CDT,
                device=device,
                fusion=res.fusion,
                i_inj=i_inj,
            )
            if bc.branch_id == 10
        )

    cpu, cuda = run(torch.device("cpu")), run(torch.device("cuda"))
    assert cuda.i_from.device.type == "cuda"
    assert torch.allclose(cpu.i_from, cuda.i_from.cpu(), atol=1e-9, rtol=1e-9)


def test_fusion_and_a_batched_switch_state_coexist_on_cuda():
    """A swept FINITE-impedance tie stays stamped while the ideal switch stays fused."""
    states = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def run(device):
        return _solve(device, branch_states={13: states.to(device)})

    cpu, cuda = run(torch.device("cpu")), run(torch.device("cuda"))
    assert cpu.v.shape == (2, cpu.index.size)
    assert cpu.fusion is not None and cpu.fusion.fused_branch_ids == (10,)
    assert torch.allclose(cpu.v, cuda.v.cpu(), atol=1e-9, rtol=1e-12)


def test_the_fusion_map_moves_to_cuda():
    fm = fusion_map(_grid())
    moved = fm.to(torch.device("cuda"))
    assert moved.row_to_reduced.device.type == "cuda"
    assert moved.representative.device.type == "cuda"
    assert moved.size == fm.size
