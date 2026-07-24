"""Differentiability gate for the DELTA :class:`ShuntAppliance` stamp.

float64 gradcheck of the assembled nodal admittance (real + imaginary parts of the
shunt node block) w.r.t. a DELTA shunt bank's per-leg conductance ``G`` and
capacitance ``C``. The leaves are passed straight into the schema via the
float/tensor duality (physical fields accept an array-like untouched), so autograd
flows ``G/C -> M^T diag(G + jB) M -> Y``. The cyclic-incidence einsum stamp is the
new tape territory this pins.
"""

from __future__ import annotations

import torch

from pgml.assembly import assemble_network_ybus
from pgml.schemas.grid_schema import (
    Grid,
    Node,
    Phase,
    ShuntAppliance,
    WindingConnection,
)

CDT = torch.complex128
ABC = (Phase.A, Phase.B, Phase.C)


def _node_block(y, index, node_id):
    rows = torch.as_tensor(index.rows(node_id), dtype=torch.int64)
    return y.index_select(-2, rows).index_select(-1, rows)


def test_gradcheck_delta_shunt_g_c():
    node = Node(id=1, u_rated_v=400.0, phases=ABC)

    g = torch.tensor([1.0e-6, 2.0e-6, 0.5e-6], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([3.0e-6, 1.0e-6, 2.0e-6], dtype=torch.float64, requires_grad=True)

    def fn(g, c):
        grid = Grid(
            nodes=[node],
            appliances=[
                ShuntAppliance(
                    id=1,
                    node=1,
                    phases=ABC,
                    conductance_s=g,
                    capacitance_f=c,
                    connection=WindingConnection.DELTA,
                )
            ],
        )
        # Two harmonic orders so the frequency scaling (B = 2*pi*f*C) is exercised.
        yb = assemble_network_ybus(grid, [50.0, 150.0], dtype=CDT)
        block = _node_block(yb.Y, yb.index, 1).reshape(-1)
        return torch.cat([block.real, block.imag])

    assert torch.autograd.gradcheck(fn, (g, c), eps=1e-9, atol=1e-6, rtol=1e-4)
