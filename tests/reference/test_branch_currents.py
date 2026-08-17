"""KCL consistency of per-branch terminal currents vs the assembled Y-bus.

``branch_currents`` derives each branch's terminal current from the SAME primitive
admittance block the Y-bus stamp scatters. The hard, oracle-free invariant: scatter
every branch's ``(i_from, i_to)`` back to its node rows and sum; the result must
equal ``Y_net @ V`` at every node row to machine precision. This proves the
primitives used here are byte-for-byte the ones assembled into the network Y, and
covers all branch kinds (transformer, line, switch, shunt reactor) at once.
"""

from __future__ import annotations

import math

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

ABC = (Phase.A, Phase.B, Phase.C)


def _sym(diag: float, off: float) -> list[list[float]]:
    return [[diag if i == j else off for j in range(3)] for i in range(3)]


def _multi_branch_grid() -> Grid:
    """HV source -> Dyn transformer -> line -> switch -> shunt-reactor node.

    Exercises every :class:`~pgml.schemas.grid_schema.BranchBase` kind that carries
    a primitive admittance block: the vector-group transformer, an explicit-R/L/C
    line, a closed switch, and a single-terminal shunt reactor.
    """
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


def _scatter_back(bcs, index, shape, dtype) -> torch.Tensor:
    acc = torch.zeros(shape, dtype=dtype)
    for bc in bcs:
        from_rows = [index.row(bc.from_node, ph) for ph in bc.from_phases]
        acc[..., from_rows] += bc.i_from
        if bc.to_node is not None:
            to_rows = [index.row(bc.to_node, ph) for ph in bc.to_phases]
            acc[..., to_rows] += bc.i_to
    return acc


def test_branch_currents_satisfy_kcl_against_network_ybus():
    """Sum of scattered terminal currents == Y_net @ V at every node row."""
    grid = _multi_branch_grid()
    freqs = [50.0, 250.0, 350.0]
    index = node_phase_index(grid)
    n = index.size

    yb = assemble_network_ybus(grid, freqs, dtype=torch.complex128)
    torch.manual_seed(0)
    v = torch.randn(len(freqs), n, dtype=torch.complex128)
    yv = torch.einsum("hij,hj->hi", yb.Y, v)  # [H, N]

    bcs = branch_currents(grid, v, freqs, index, dtype=torch.complex128)

    # One BranchCurrent per in-service branch, in grid.branches order.
    assert [bc.branch_id for bc in bcs] == [b.id for b in grid.branches]

    acc = _scatter_back(bcs, index, (len(freqs), n), torch.complex128)
    err = (acc - yv).abs().max().item()
    assert err < 1e-9, f"KCL residual {err:.2e} too large"


def test_branch_currents_shunt_reactor_single_terminal():
    """A ShuntReactor reports ``to_node=None`` and an empty ``i_to``."""
    grid = _multi_branch_grid()
    freqs = [50.0]
    index = node_phase_index(grid)
    v = torch.randn(index.size, dtype=torch.complex128)
    bcs = branch_currents(grid, v, freqs, index, dtype=torch.complex128)

    reactor_bc = next(bc for bc in bcs if bc.branch_id == 23)
    assert reactor_bc.to_node is None
    assert reactor_bc.to_phases == ()
    assert reactor_bc.i_to.numel() == 0
    assert torch.count_nonzero(reactor_bc.i_to) == 0


def test_branch_currents_batched_and_squeeze():
    """A leading batch dim and a missing-H axis broadcast like the rest of assembly."""
    grid = _multi_branch_grid()
    freqs = [50.0, 250.0]
    index = node_phase_index(grid)
    n = index.size
    yb = assemble_network_ybus(grid, freqs, dtype=torch.complex128)

    # Batched voltages [B, H, N].
    v = torch.randn(5, len(freqs), n, dtype=torch.complex128)
    bcs = branch_currents(grid, v, freqs, index, dtype=torch.complex128)
    assert bcs[0].i_from.shape == (5, len(freqs), 3)

    yv = torch.einsum("hij,bhj->bhi", yb.Y, v)
    acc = _scatter_back(bcs, index, (5, len(freqs), n), torch.complex128)
    assert (acc - yv).abs().max().item() < 1e-9

    # Missing H axis [N] broadcasts over H.
    v1 = torch.randn(n, dtype=torch.complex128)
    bcs1 = branch_currents(grid, v1, freqs, index, dtype=torch.complex128)
    assert bcs1[0].i_from.shape == (len(freqs), 3)
