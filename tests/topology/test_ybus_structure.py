"""The topology-derived sparsity pattern of ``Y`` covers every stamp.

:func:`pgml.assembly.ybus_structure` lists the positions the assembler can write to,
read off the grid's incidence rather than off an assembled matrix. The sparse solver
backend builds each system's compressed-column form from it, so a position the pattern
misses would be silently dropped from the factorized matrix. These tests pin the
superset property on the structures that change which rows a stamp touches — four-wire
and delta terminals, a transformer relating two different phase sets, bus fusion,
batched branch states — and pin it against the stamp registry itself, so a future
builder that writes outside its components' nodes fails here.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import (
    assemble_network_ybus,
    assemble_ybus,
    node_phase_index,
    ybus_structure,
)
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _BRANCH_STAMPS, _unfused_view
from pgml.grids import synthetic_feeder
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    Switch,
    Transformer,
    WindingConnection,
)

A = (Phase.A,)
ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
FREQS = [50.0, 250.0, 550.0]


def _matrix(n, value):
    return [[(value if i == j else 0.0) for j in range(n)] for i in range(n)]


def _line(bid, u, v, phases=ABC):
    n = len(phases)
    return Line(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=phases,
        to_phases=phases,
        length_m=250.0,
        series_resistance_ohm_per_m=_matrix(n, 2.0e-4),
        series_inductance_h_per_m=_matrix(n, 8.0e-7),
        shunt_capacitance_f_per_m=_matrix(n, 1.0e-11),
    )


def _source(aid, node, phases=ABC):
    return Source(
        id=aid,
        node=node,
        phases=phases,
        u_ref_v=tuple(20_000.0 for _ in phases),
        u_angle_deg=tuple(0.0 for _ in phases),
        resistance_ohm=_matrix(len(phases), 1.0e-3),
        inductance_h=_matrix(len(phases), 1.0e-6),
    )


def _four_wire_grid() -> Grid:
    """A four-wire terminal, a delta load and a fixed shunt bank on one feeder."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABCN),
            Node(id=3, u_rated_v=400.0, phases=ABC),
        ],
        branches=[_line(10, 1, 2), _line(11, 2, 3)],
        appliances=[
            _source(100, 1),
            Load(id=200, node=2, phases=ABC, p_nom_w=4.0e3, q_nom_var=1.0e3),
            Load(
                id=201,
                node=3,
                phases=ABC,
                p_nom_w=6.0e3,
                q_nom_var=2.0e3,
                connection=WindingConnection.DELTA,
            ),
            ShuntAppliance(
                id=202,
                node=3,
                phases=ABC,
                conductance_s=(0.0,) * 3,
                capacitance_f=(1.0e-6,) * 3,
                connection=WindingConnection.DELTA,
            ),
        ],
    )


def _transformer_grid() -> Grid:
    """A Dyn transformer between a three-wire MV bus and a four-wire LV bus."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=20_000.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABCN),
            Node(id=3, u_rated_v=400.0, phases=ABCN),
        ],
        branches=[
            Transformer(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                u_rated_from_v=20_000.0,
                u_rated_to_v=400.0,
                s_rated_va=630_000.0,
                series_resistance_ohm=1.0e-3,
                series_inductance_h=1.0e-4,
                from_connection=WindingConnection.DELTA,
                to_connection=WindingConnection.WYE_GROUNDED,
                tap=ComplexTap(ratio_magnitude=1.0, shift_deg=330.0),
            ),
            _line(11, 2, 3, phases=ABCN),
        ],
        appliances=[
            _source(100, 1),
            Load(id=200, node=3, phases=ABC, p_nom_w=5.0e3, q_nom_var=1.0e3),
        ],
    )


def _fused_grid() -> Grid:
    """An ideal switch collapses two bus rows into one reduced row."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=400.0, phases=ABC) for i in (1, 2, 3, 4)],
        branches=[
            _line(10, 1, 2),
            Switch(
                id=11,
                from_node=2,
                to_node=3,
                from_phases=ABC,
                to_phases=ABC,
                closed=True,
                resistance_ohm=0.0,
            ),
            _line(12, 3, 4),
        ],
        appliances=[
            _source(100, 1),
            Load(id=200, node=4, phases=ABC, p_nom_w=5.0e3, q_nom_var=1.0e3),
        ],
    )


GRIDS = {
    "feeder": lambda: synthetic_feeder(30),
    "four_wire": _four_wire_grid,
    "transformer": _transformer_grid,
    "fused": _fused_grid,
}


def _covered(pattern: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.zeros(n * n, dtype=torch.bool)
    mask[pattern] = True
    return mask


def _assert_covers(y: torch.Tensor, index, grid: Grid) -> None:
    n = index.size
    pattern = ybus_structure(grid, index)
    assert pattern.dtype == torch.int64
    assert bool((pattern[1:] > pattern[:-1]).all()), "pattern must be sorted and unique"
    nonzero = (y.reshape(-1, n * n) != 0).any(0)
    missing = nonzero & ~_covered(pattern, n)
    assert int(missing.sum()) == 0


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_pattern_covers_every_assembled_nonzero(name):
    """Both assemblers, over several frequencies, stay inside the pattern."""
    grid = GRIDS[name]()
    for yb in (
        assemble_network_ybus(grid, FREQS),
        assemble_ybus(grid, FREQS),
    ):
        _assert_covers(yb.Y, yb.index, grid)


def test_pattern_covers_a_batched_branch_state_assembly():
    """A swept branch is stamped whatever its static flags say."""
    grid = _fused_grid()
    states = {12: torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)}
    yb = assemble_network_ybus(grid, FREQS, branch_states=states)
    _assert_covers(yb.Y, yb.index, grid)


def test_pattern_covers_an_out_of_service_branch_reopened_by_a_state():
    """An open switch carries no admittance until a state stamps it."""
    grid = _four_wire_grid()
    grid = grid.model_copy(
        update={
            "branches": [
                *grid.branches,
                Switch(
                    id=12,
                    from_node=1,
                    to_node=3,
                    from_phases=ABC,
                    to_phases=ABC,
                    closed=False,
                    resistance_ohm=1.0e-4,
                ),
            ]
        }
    )
    yb = assemble_network_ybus(grid, FREQS, branch_states={12: 1.0})
    _assert_covers(yb.Y, yb.index, grid)


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_pattern_covers_every_registered_stamp(name):
    """Straight against the stamp registry, not against one assembled matrix."""
    grid = GRIDS[name]()
    yb = assemble_network_ybus(grid, FREQS)
    index, n = yb.index, yb.index.size
    covered = _covered(ybus_structure(grid, index), n)
    view = _unfused_view(grid, yb.fusion)
    f = torch.as_tensor(FREQS, dtype=torch.float64)
    for stamp in _BRANCH_STAMPS:
        for _, _, rows, cols in stamp.builder(
            view,
            f,
            index,
            _cdtype(torch.complex128),
            _rdtype(torch.complex128),
            None,
            None,
            None,
        ):
            lin = (rows[:, :, None] * n + cols[:, None, :]).reshape(-1)
            assert bool(covered[lin].all()), (name, stamp.kind)


def test_pattern_of_a_fused_grid_is_the_reduced_one():
    """With fusion the pattern describes the REDUCED matrix, not the full layout."""
    grid = _fused_grid()
    yb = assemble_network_ybus(grid, FREQS)
    assert yb.fusion is not None
    assert yb.index.size < node_phase_index(grid).size
    pattern = ybus_structure(grid, yb.index)
    assert int(pattern.max()) < yb.index.size**2
