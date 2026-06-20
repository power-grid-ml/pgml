"""Differentiability gate for the vector-group transformer stamp.

Gradients of the solved node voltages w.r.t. the transformer leakage R / L and the
off-nominal tap magnitude must pass ``gradcheck`` (float64), through BOTH the
3-phase winding-incidence path (``Y = Nᵀ Y_winding N``) and the single-phase /
positive-sequence scalar-tap path. Leaf tensors are injected via the
``param_overrides`` hook so the frozen schema is not mutated.
"""

from __future__ import annotations

import math

import torch

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Load,
    Node,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)
from pgml.solver import solve_harmonic

ABC = (Phase.A, Phase.B, Phase.C)


def _dyn_grid(phases) -> Grid:
    """HV source -> Dyn transformer -> LV node with a const-Z load."""
    n = len(phases)
    src = Source(
        id=10,
        node=1,
        phases=phases,
        u_ref_v=(20_000.0 / math.sqrt(3),) * n,
        u_angle_deg=(0.0, -120.0, 120.0)[:n],
        resistance_ohm=[[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)],
        inductance_h=[[5.0e-3 if i == j else 0.0 for j in range(n)] for i in range(n)],
    )
    xfmr = Transformer(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=phases,
        to_phases=phases,
        s_rated_va=0.4e6,
        u_rated_from_v=20_000.0,
        u_rated_to_v=400.0,
        from_connection=WindingConnection.DELTA,
        to_connection=WindingConnection.WYE_GROUNDED,
        series_resistance_ohm=0.01,
        series_inductance_h=1.0e-4,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=30.0),
    )
    load = Load(id=30, node=2, phases=phases, p_nom_w=9.0e3, q_nom_var=2.0e3)
    nodes = [
        Node(id=1, u_rated_v=20_000.0, phases=phases),
        Node(id=2, u_rated_v=400.0, phases=phases),
    ]
    return Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=[xfmr], appliances=[src, load]
    )


def _voltages(grid, overrides):
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, [50.0], dtype=torch.complex128, param_overrides=overrides)
    i = build_injections(
        grid, [50.0], idx, dtype=torch.complex128, param_overrides=overrides
    )
    return solve_harmonic(yb.Y, i).reshape(-1)


def _run(phases):
    grid = _dyn_grid(phases)
    r = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    ell = torch.tensor(1.0e-4, dtype=torch.float64, requires_grad=True)
    tap = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    def fn(r, ell, tap):
        overrides = {
            ("transformer", 20, "series_resistance_ohm"): r,
            ("transformer", 20, "series_inductance_h"): ell,
            ("transformer", 20, "tap_magnitude"): tap,
        }
        return _voltages(grid, overrides)

    assert torch.autograd.gradcheck(fn, (r, ell, tap), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_transformer_three_phase_incidence():
    """3-phase Dyn winding-incidence path: grad w.r.t. R, L, tap magnitude."""
    _run(ABC)


def test_gradcheck_transformer_single_phase_scalar():
    """Single-phase / positive-sequence scalar-tap path: grad w.r.t. R, L, tap."""
    _run((Phase.A,))
