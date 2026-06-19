"""GPU gate: CPU-vs-CUDA parity of the per-node harmonic "error" source.

A grid with a Thevenin voltage source AND a Norton current source must solve
identically on CPU and CUDA (skips cleanly without CUDA). Pins device/dtype honoring
of the diagonal ``Y_s`` add and the ``I_N`` current add (incl. a batched
``source_power_va`` that promotes ``Y(h)`` to ``[*batch, H, N, N]``).
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
)
from pgml.solver import NodeHarmonicSource, solve_harmonic_flow

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

F0 = 50.0
W0 = 2.0 * math.pi * F0
CDT = torch.complex128


def _grid() -> Grid:
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1.0,
                series_resistance_ohm_per_m=[[0.5]],
                series_inductance_h_per_m=[[0.5 / W0]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W0]],
            ),
            Load(
                id=2,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=500.0,
                load_model=LoadModel.CONST_POWER,
            ),
        ],
    )


def _sources():
    return [
        NodeHarmonicSource(
            node_id=1,
            spectrum={1: (1.0, 0.0), 5: (0.1, 15.0)},
            source_power_va=2.0e5,
            kind="current",
        ),
        NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 0.0), 7: (0.1, -30.0)},
            source_power_va=5.0e5,
            kind="voltage",
        ),
    ]


def test_cpu_cuda_parity_scalar():
    cpu = solve_harmonic_flow(
        _grid(),
        [1, 5, 7],
        slack="norton",
        dtype=CDT,
        device=torch.device("cpu"),
        node_sources=_sources(),
    ).v
    cuda = solve_harmonic_flow(
        _grid(),
        [1, 5, 7],
        slack="norton",
        dtype=CDT,
        device=torch.device("cuda"),
        node_sources=_sources(),
    ).v
    assert cuda.device.type == "cuda"
    assert torch.allclose(cpu, cuda.cpu(), rtol=1e-9, atol=1e-9)


def test_cpu_cuda_parity_batched_source_power():
    """Batched ``source_power_va`` (promotes Y(h) to [S,H,N,N]) is CPU/CUDA-identical."""

    def run(device):
        s_cpu = torch.tensor([1.0e5, 5.0e5, 1.0e6], dtype=torch.float64)
        s = s_cpu.to(device)
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
            source_power_va=s,
            kind="voltage",
        )
        return solve_harmonic_flow(
            _grid(),
            [1, 5],
            slack="norton",
            dtype=CDT,
            device=device,
            node_sources=[src],
        ).v

    cpu = run(torch.device("cpu"))
    cuda = run(torch.device("cuda"))
    assert cpu.shape == (3, 2, 2)
    assert torch.allclose(cpu, cuda.cpu(), rtol=1e-9, atol=1e-9)
