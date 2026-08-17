"""Connection-aware terminal incidence ``M`` construction.

Verifies the WYE-ground / WYE-neutral / DELTA-3 incidence matrices and the
identity-reduction guarantee ``M^T diag(y) M == diag(y)`` for WYE-to-ground.
"""

from __future__ import annotations

import pytest
import torch

import pgml.errors as errors
from pgml.assembly._incidence import (
    build_incidence,
    cyclic_delta_incidence,
    group_appliances,
    used_rows,
)
from pgml.assembly.index import node_phase_index
from pgml.schemas.grid_schema import (
    Grid,
    Load,
    Node,
    Phase,
    WindingConnection,
)

ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
RDT = torch.float64
DEV = torch.device("cpu")


def _grid(node_phases, appliance):
    return Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=node_phases)], appliances=[appliance]
    )


def _grp(grid):
    node_map = {nd.id: nd for nd in grid.nodes}
    loads = [a for a in grid.appliances if isinstance(a, Load)]
    groups = group_appliances(loads, node_map)
    assert len(groups) == 1
    return groups[0], grid


# --- WYE to ground: M = I ----------------------------------------------------
def test_wye_ground_incidence_is_identity():
    grp, _ = _grp(_grid(ABC, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)))
    m = build_incidence(grp, RDT, DEV)
    assert grp.has_neutral_return is False
    assert m.shape == (3, 3)
    assert torch.allclose(m, torch.eye(3, dtype=RDT))


def test_wye_ground_mtdm_equals_diag():
    grp, _ = _grp(_grid(ABC, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)))
    m = build_incidence(grp, RDT, DEV).to(torch.complex128)
    y = torch.tensor([1.0 + 2.0j, 3.0 - 1.0j, 0.5 + 0.5j], dtype=torch.complex128)
    block = torch.einsum("ei,e,ej->ij", m, y, m)
    assert torch.allclose(block, torch.diag(y))


# --- WYE with neutral: M = [I | -1] -----------------------------------------
def test_wye_neutral_incidence():
    grp, grid = _grp(_grid(ABCN, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)))
    assert grp.has_neutral_return is True
    assert grp.n_used == 4
    m = build_incidence(grp, RDT, DEV)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=RDT,
    )
    assert torch.allclose(m, expected)

    # used_rows: the 3 phase rows + the Phase.N row (row 3 for an A,B,C,N node).
    index = node_phase_index(grid)
    rows = used_rows(grp, index, DEV)
    assert rows.shape == (1, 4)
    assert rows[0].tolist() == [0, 1, 2, 3]


def test_wye_neutral_kirchhoff_in_block():
    # M^T diag(y) M: the N row/col must carry sum y (Kirchhoff coupling).
    grp, _ = _grp(_grid(ABCN, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)))
    m = build_incidence(grp, RDT, DEV).to(torch.complex128)
    y = torch.tensor([1.0j, 2.0j, 3.0j], dtype=torch.complex128)
    block = torch.einsum("ei,e,ej->ij", m, y, m)  # [4,4]
    # Phase-phase diagonal = y; N-N entry = sum(y); phase-N = -y_phase.
    assert torch.allclose(block[:3, :3], torch.diag(y))
    assert torch.allclose(block[3, 3], y.sum())
    assert torch.allclose(block[:3, 3], -y)
    assert torch.allclose(block[3, :3], -y)


# --- DELTA, n==3: circulant difference ---------------------------------------
def test_delta3_incidence_is_circulant():
    grp, _ = _grp(
        _grid(
            ABC,
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                connection=WindingConnection.DELTA,
            ),
        )
    )
    m = build_incidence(grp, RDT, DEV)
    expected = torch.tensor(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]], dtype=RDT
    )
    assert torch.allclose(m, expected)


def test_delta_n2_raises():
    grid = _grid(
        (Phase.A, Phase.B),
        Load(
            id=1,
            node=1,
            phases=(Phase.A, Phase.B),
            p_nom_w=2000.0,
            connection=WindingConnection.DELTA,
        ),
    )
    node_map = {nd.id: nd for nd in grid.nodes}
    with pytest.raises(NotImplementedError, match="open/2-phase delta"):
        group_appliances([grid.appliances[0]], node_map)


# --- return_path: WYE return-conductor override ------------------------------
def test_return_path_auto_matches_historical_neutral_on_4wire():
    """'auto' (default) on a 4-wire node keeps the historical neutral-return rule."""
    grp, _ = _grp(_grid(ABCN, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0)))
    assert grp.has_neutral_return is True
    assert grp.n_used == 4


def test_return_path_ground_on_4wire_is_identity():
    """'ground' pins the return to ground even when the node carries Phase.N."""
    grp, _ = _grp(
        _grid(
            ABCN, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0, return_path="ground")
        )
    )
    assert grp.has_neutral_return is False
    assert grp.n_used == 3
    m = build_incidence(grp, RDT, DEV)
    assert m.shape == (3, 3)
    assert torch.allclose(m, torch.eye(3, dtype=RDT))


def test_return_path_neutral_on_4wire_uses_neutral():
    grp, _ = _grp(
        _grid(
            ABCN, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0, return_path="neutral")
        )
    )
    assert grp.has_neutral_return is True
    assert grp.n_used == 4


def test_return_path_neutral_without_node_neutral_raises():
    """'neutral' on a 3-wire node (no Phase.N) is a modeling error."""
    grid = _grid(
        ABC, Load(id=1, node=1, phases=ABC, p_nom_w=3000.0, return_path="neutral")
    )
    node_map = {nd.id: nd for nd in grid.nodes}
    with pytest.raises(errors.ModelingError, match="requires the host node"):
        group_appliances([grid.appliances[0]], node_map)


def test_return_path_nonauto_on_delta_raises():
    grid = _grid(
        ABC,
        Load(
            id=1,
            node=1,
            phases=ABC,
            p_nom_w=3000.0,
            connection=WindingConnection.DELTA,
            return_path="ground",
        ),
    )
    node_map = {nd.id: nd for nd in grid.nodes}
    with pytest.raises(errors.ModelingError, match="WYE connection only"):
        group_appliances([grid.appliances[0]], node_map)


def test_grounded_and_neutral_wye_split_into_two_groups():
    """A grounded and a neutral-returning WYE load on the SAME 4-wire node need
    different incidence matrices -> distinct groups."""
    grid = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABCN)],
        appliances=[
            Load(id=1, node=1, phases=ABC, p_nom_w=3000.0, return_path="ground"),
            Load(id=2, node=1, phases=ABC, p_nom_w=2000.0, return_path="neutral"),
        ],
    )
    node_map = {nd.id: nd for nd in grid.nodes}
    groups = group_appliances(list(grid.appliances), node_map)
    assert len(groups) == 2
    assert {g.has_neutral_return for g in groups} == {True, False}


# --- cyclic delta incidence (shunt bank generalisation) ----------------------
def test_cyclic_delta_incidence_n3_matches_build_incidence():
    grp, _ = _grp(
        _grid(
            ABC,
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                connection=WindingConnection.DELTA,
            ),
        )
    )
    assert torch.allclose(
        cyclic_delta_incidence(3, RDT, DEV), build_incidence(grp, RDT, DEV)
    )


def test_cyclic_delta_incidence_n2_is_phase_to_phase():
    """A 2-leg delta collapses to one phase-to-phase branch: M^T diag(y) M is the
    2x2 [[y0+y1, -(y0+y1)], [-(y0+y1), y0+y1]] stamp."""
    m = cyclic_delta_incidence(2, RDT, DEV).to(torch.complex128)
    y = torch.tensor([1.0j, 3.0j], dtype=torch.complex128)
    block = torch.einsum("ei,e,ej->ij", m, y, m)
    ytot = y.sum()
    expected = torch.tensor([[ytot, -ytot], [-ytot, ytot]], dtype=torch.complex128)
    assert torch.allclose(block, expected)
