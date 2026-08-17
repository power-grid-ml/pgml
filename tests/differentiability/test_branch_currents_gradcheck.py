"""Differentiability gate for ``branch_currents``.

Gradients must flow from physical grid parameters (a line R/L and a transformer
R/L) through the solved node voltages AND the per-branch primitive into the
terminal currents ``i_from``. Leaf tensors are injected via the ``param_overrides``
hook so the frozen schema is not mutated.
"""

from __future__ import annotations

import math

import torch

from pgml.assembly import (
    assemble_ybus,
    branch_currents,
    build_injections,
    node_phase_index,
)
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)
from pgml.solver import solve_harmonic

ABC = (Phase.A, Phase.B, Phase.C)


def _sym(diag: float, off: float) -> list[list[float]]:
    return [[diag if i == j else off for j in range(3)] for i in range(3)]


def _grid() -> Grid:
    """HV source -> Dyn transformer -> line -> const-Z load (3-phase)."""
    nodes = [
        Node(id=1, u_rated_v=20_000.0, phases=ABC),
        Node(id=2, u_rated_v=400.0, phases=ABC),
        Node(id=3, u_rated_v=400.0, phases=ABC),
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
    src = Source(
        id=10,
        node=1,
        phases=ABC,
        u_ref_v=(20_000.0 / math.sqrt(3),) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=_sym(0.5, 0.0),
        inductance_h=_sym(5.0e-3, 0.0),
    )
    load = Load(id=30, node=3, phases=ABC, p_nom_w=9.0e3, q_nom_var=2.0e3)
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[xfmr, line],
        appliances=[src, load],
    )


def test_gradcheck_branch_currents_through_solve():
    """grad of a real scalar of i_from w.r.t. a line R/L and a transformer R/L."""
    grid = _grid()
    index = node_phase_index(grid)
    freqs = [50.0]

    line_r = torch.tensor(_sym(1.0e-3, 1.0e-4), dtype=torch.float64, requires_grad=True)
    line_l = torch.tensor(_sym(1.0e-6, 1.0e-7), dtype=torch.float64, requires_grad=True)
    xf_r = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    xf_l = torch.tensor(1.0e-4, dtype=torch.float64, requires_grad=True)

    def fn(line_r, line_l, xf_r, xf_l):
        overrides = {
            ("line", 21, "series_resistance_ohm_per_m"): line_r,
            ("line", 21, "series_inductance_h_per_m"): line_l,
            ("transformer", 20, "series_resistance_ohm"): xf_r,
            ("transformer", 20, "series_inductance_h"): xf_l,
        }
        yb = assemble_ybus(
            grid, freqs, dtype=torch.complex128, param_overrides=overrides
        )
        i = build_injections(
            grid, freqs, index, dtype=torch.complex128, param_overrides=overrides
        )
        v = solve_harmonic(yb.Y, i)  # [H, N]
        bcs = branch_currents(
            grid, v, freqs, index, dtype=torch.complex128, param_overrides=overrides
        )
        # A real scalar mixing the line and transformer from-terminal currents.
        line_bc = next(bc for bc in bcs if bc.branch_id == 21)
        xf_bc = next(bc for bc in bcs if bc.branch_id == 20)
        s = line_bc.i_from.sum() + xf_bc.i_from.sum()
        return torch.stack([s.real, s.imag])

    assert torch.autograd.gradcheck(
        fn, (line_r, line_l, xf_r, xf_l), eps=1e-6, atol=1e-5, rtol=1e-3
    )
