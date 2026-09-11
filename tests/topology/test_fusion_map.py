"""Exact bus fusion of zero-impedance branches: the map and its bookkeeping.

:func:`pgml.assembly.fusion_map` collapses the terminal node-phase rows of every ideal
conductor (a closed switch with no impedance data, a bus coupler or jumper modelled as a
zero-impedance line, a zero-length line) into ONE row of the solved system. These tests
pin the structural contract — which rows are merged, how a state prolongs back to the
grid's own layout, which row sets stay sets — and the cases the engine must refuse
instead of fusing.

The physics (voltages and currents against a reference tool) is covered by
``tests/reference/test_switch_fusion_pandapower.py`` and
``tests/reference/test_switch_fusion_pgm.py``.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import (
    assemble_network_ybus,
    fusion_map,
    node_phase_index,
    zero_impedance_branches,
)
from pgml.errors import ModelingError
from pgml.schemas.grid_schema import (
    GenericBranch,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
    Switch,
    Transformer,
    WindingConnection,
)
from pgml.solver import solve_power_flow

A = (Phase.A,)
ABC = (Phase.A, Phase.B, Phase.C)
CDT = torch.complex128


def _line(bid, u, v, *, phases=A, r=2.0e-4, ell=8.0e-7, length=1000.0):
    n = len(phases)
    eye = [[(r if i == j else 0.0) for j in range(n)] for i in range(n)]
    ind = [[(ell if i == j else 0.0) for j in range(n)] for i in range(n)]
    zero = [[0.0] * n for _ in range(n)]
    return Line(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=phases,
        to_phases=phases,
        length_m=length,
        series_resistance_ohm_per_m=eye,
        series_inductance_h_per_m=ind,
        shunt_capacitance_f_per_m=zero,
    )


def _switch(bid, u, v, *, phases=A, closed=True, r=0.0):
    return Switch(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=phases,
        to_phases=phases,
        closed=closed,
        resistance_ohm=r,
    )


def _grid(branches, *, n_nodes=3, phases=A, sources=(1,), loads=(3,)):
    n = len(phases)
    eye = [[(1.0e-6 if i == j else 0.0) for j in range(n)] for i in range(n)]
    ind = [[(1.0e-12 if i == j else 0.0) for j in range(n)] for i in range(n)]
    appliances: list = [
        Source(
            id=100 + k,
            node=node,
            phases=phases,
            u_ref_v=tuple(20_000.0 for _ in phases),
            u_angle_deg=tuple(0.0 for _ in phases),
            resistance_ohm=eye,
            inductance_h=ind,
        )
        for k, node in enumerate(sources)
    ]
    appliances += [
        Load(id=200 + k, node=node, phases=phases, p_nom_w=1.0e6, q_nom_var=3.0e5)
        for k, node in enumerate(loads)
    ]
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=i, u_rated_v=20_000.0, phases=phases) for i in range(1, n_nodes + 1)
        ],
        branches=branches,
        appliances=appliances,
    )


# --------------------------------------------------------------------------- #
# what fuses
# --------------------------------------------------------------------------- #
def test_no_zero_impedance_branch_yields_no_map():
    """Nothing fuses on an ordinary grid, and the solve takes the unreduced path."""
    grid = _grid([_line(10, 1, 2), _line(11, 2, 3)])
    assert fusion_map(grid) is None
    assert solve_power_flow(grid, dtype=CDT).fusion is None


def test_closed_ideal_switch_merges_its_two_rows():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    fm = fusion_map(grid)
    assert fm is not None
    assert fm.fused_branch_ids == (10,)
    assert (fm.full_index.size, fm.size) == (3, 2)
    assert fm.node_groups() == (((1, "a"), (2, "a")),)
    # the two fused rows index the same reduced row, the third its own
    assert fm.row_to_reduced.tolist() == [0, 0, 1]


def test_open_ideal_switch_does_not_fuse():
    """An open switch conducts nothing; its impedance is irrelevant."""
    grid = _grid([_switch(10, 1, 2, closed=False), _line(11, 1, 3), _line(12, 2, 3)])
    assert fusion_map(grid) is None


def test_out_of_service_zero_impedance_line_does_not_fuse():
    grid = _grid(
        [
            _line(10, 1, 2, r=0.0, ell=0.0).model_copy(update={"in_service": False}),
            _line(11, 1, 2),
            _line(12, 2, 3),
        ]
    )
    assert fusion_map(grid) is None


def test_a_chain_of_ideal_branches_forms_one_group():
    """Fusion is transitive: three nodes joined by two ideal branches are one row."""
    grid = _grid(
        [_switch(10, 1, 2), _line(11, 2, 3, r=0.0, ell=0.0), _line(12, 3, 4)],
        n_nodes=4,
        loads=(4,),
    )
    fm = fusion_map(grid)
    assert fm.size == 2
    assert fm.node_groups() == (((1, "a"), (2, "a"), (3, "a")),)
    assert sorted(fm.fused_branch_ids) == [10, 11]


def test_zero_length_line_fuses():
    """The schema's positivity validator covers plain floats, so a tensor length reaches
    the assembly — where a zero length scales the whole stamp (series AND shunt) away."""
    zero = torch.zeros((), dtype=torch.float64)
    grid = _grid([_line(10, 1, 2, length=zero), _line(11, 2, 3)])
    assert fusion_map(grid).fused_branch_ids == (10,)


def test_three_phase_switch_fuses_per_phase():
    """Each conductor is shorted to the conductor at the SAME terminal position."""
    grid = _grid(
        [_switch(10, 1, 2, phases=ABC), _line(11, 2, 3, phases=ABC)], phases=ABC
    )
    fm = fusion_map(grid)
    assert (fm.full_index.size, fm.size) == (9, 6)
    assert len(fm.groups) == 3  # one per phase, never across phases
    for grp in fm.node_groups():
        assert {ph for _nid, ph in grp} == {grp[0][1]}


def test_phase_rotating_jumper_pairs_by_position():
    """``from_phases=(a,b,c)`` against ``to_phases=(b,c,a)`` shorts a-b, b-c, c-a."""
    sw = Switch(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=ABC,
        to_phases=(Phase.B, Phase.C, Phase.A),
        closed=True,
    )
    grid = _grid([sw, _line(11, 2, 3, phases=ABC)], phases=ABC)
    fm = fusion_map(grid)
    groups = {frozenset(g) for g in fm.node_groups()}
    assert frozenset({(1, "a"), (2, "b")}) in groups
    assert frozenset({(1, "b"), (2, "c")}) in groups
    assert frozenset({(1, "c"), (2, "a")}) in groups


# --------------------------------------------------------------------------- #
# what is refused
# --------------------------------------------------------------------------- #
def test_zero_impedance_transformer_is_never_fused():
    """A ratio and a vector group relate the terminals by more than equality."""
    xf = Transformer(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=A,
        to_phases=A,
        s_rated_va=1.0e6,
        u_rated_from_v=20_000.0,
        u_rated_to_v=20_000.0,
        from_connection=WindingConnection.WYE_GROUNDED,
        to_connection=WindingConnection.WYE_GROUNDED,
        series_resistance_ohm=0.0,
        series_inductance_h=0.0,
    )
    with pytest.raises(ModelingError, match="transformer 10"):
        fusion_map(_grid([xf, _line(11, 2, 3)]))


def test_zero_series_branch_with_a_shunt_is_refused():
    """Collapsing it would drop the shunt admittance it still carries."""
    gb = GenericBranch(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=A,
        to_phases=A,
        series_resistance_ohm=[[0.0]],
        series_inductance_h=[[0.0]],
        shunt_capacitance_from_f=[[1.0e-9]],
    )
    with pytest.raises(ModelingError, match="generic_branch 10"):
        fusion_map(_grid([gb, _line(11, 2, 3)]))


def test_a_swept_branch_is_never_fused():
    """``branch_states`` scales a STAMPED admittance, which an ideal branch has not."""
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    with pytest.raises(ModelingError, match="branch_states"):
        fusion_map(grid, branch_states={10: 1.0})
    with pytest.raises(ModelingError, match="branch_states"):
        solve_power_flow(grid, dtype=CDT, branch_states={10: 1.0})


def test_a_swept_branch_with_a_finite_impedance_still_sweeps():
    """The fix the error names: give the swept switch the near-ideal resistance."""
    grid = _grid([_switch(10, 1, 2, r=1.0e-4), _line(11, 2, 3)])
    assert fusion_map(grid, branch_states={10: 1.0}) is None
    states = torch.tensor([1.0, 1.0], dtype=torch.float64)
    res = solve_power_flow(grid, dtype=CDT, branch_states={10: states})
    assert res.converged and res.v.shape[0] == 2


def test_an_ideal_branch_across_two_voltage_levels_is_refused():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    grid = grid.model_copy(
        update={
            "nodes": [
                n.model_copy(update={"u_rated_v": 400.0}) if n.id == 2 else n
                for n in grid.nodes
            ]
        }
    )
    with pytest.raises(ModelingError, match="voltage level"):
        fusion_map(grid)


def test_two_slack_terminals_with_different_references_are_refused():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)], sources=(1, 2))
    grid = grid.model_copy(
        update={
            "appliances": [
                a.model_copy(update={"u_ref_v": (19_000.0,)})
                if getattr(a, "component", "") == "source" and a.node == 2
                else a
                for a in grid.appliances
            ]
        }
    )
    with pytest.raises(ModelingError, match="reference different voltages"):
        solve_power_flow(grid, dtype=CDT)


def test_two_slack_terminals_with_the_same_reference_are_pinned_once():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)], sources=(1, 2))
    res = solve_power_flow(grid, dtype=CDT)
    assert res.converged
    v = res.v.reshape(-1)
    assert abs(complex(v[res.index.row(1, Phase.A)]) - 20_000.0) < 1e-9
    assert abs(complex(v[res.index.row(2, Phase.A)]) - 20_000.0) < 1e-9


# --------------------------------------------------------------------------- #
# the layout transforms
# --------------------------------------------------------------------------- #
def test_prolong_and_restrict_are_adjoint():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    fm = fusion_map(grid)
    x = torch.randn(fm.size, dtype=torch.float64)
    y = torch.randn(fm.full_index.size, dtype=torch.float64)
    # <P x, y> == <x, P^T y>
    assert torch.allclose(fm.prolong(x) @ y, x @ fm.restrict(y))


def test_prolong_repeats_the_group_value_and_sample_inverts_it():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    fm = fusion_map(grid)
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert fm.prolong(x).tolist() == [1.0, 1.0, 2.0]
    assert fm.sample(fm.prolong(x)).tolist() == x.tolist()


def test_prolong_keeps_the_batch_and_harmonic_axes():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    fm = fusion_map(grid)
    x = torch.zeros((4, 7, fm.size), dtype=CDT)
    assert fm.prolong(x).shape == (4, 7, fm.full_index.size)


def test_reduce_rows_keeps_a_row_set_a_set():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    fm = fusion_map(grid)
    rows = torch.tensor([0, 1, 2], dtype=torch.int64)
    assert fm.reduce_rows(rows).tolist() == [0, 1]
    assert fm.duplicate_rows(rows) == [(0, 1)]


def test_the_reduced_y_is_the_restricted_full_y():
    """``Y_red == P^T Y P`` for the SAME network without the fused branch."""
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    fm = fusion_map(grid)
    y_red = assemble_network_ybus(grid, [50.0], dtype=CDT, fusion=fm).Y.reshape(
        fm.size, fm.size
    )
    unfused = grid.model_copy(
        update={"branches": [b for b in grid.branches if b.id != 10]}
    )
    full = node_phase_index(grid)
    y_full = assemble_network_ybus(unfused, [50.0], dtype=CDT).Y.reshape(
        full.size, full.size
    )
    p = torch.zeros((full.size, fm.size), dtype=CDT)
    p[torch.arange(full.size), fm.row_to_reduced] = 1.0
    assert torch.allclose(y_red, p.mT @ y_full @ p, atol=0.0, rtol=0.0)


def test_the_fused_nodes_share_one_voltage_exactly():
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    res = solve_power_flow(grid, dtype=CDT)
    v = res.v.reshape(-1)
    assert v[res.index.row(1, Phase.A)] == v[res.index.row(2, Phase.A)]
    assert res.index.size == 3  # reported on the grid's own rows


def test_the_fusion_is_logged_once_naming_the_nodes(caplog):
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    with caplog.at_level("INFO"):
        solve_power_flow(grid, dtype=CDT)
    lines = [r.message for r in caplog.records if "fused" in r.message]
    assert len(lines) == 1
    assert "1a" in lines[0] and "2a" in lines[0]


# --------------------------------------------------------------------------- #
# parameter substitution decides what fuses
# --------------------------------------------------------------------------- #
def test_a_param_override_with_a_finite_value_keeps_the_branch_stamped():
    """Fusion reads the EFFECTIVE impedance, so a substituted leaf stays a parameter."""
    grid = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    leaf = torch.tensor(1.0e-4, dtype=torch.float64, requires_grad=True)
    overrides = {("switch", 10, "resistance_ohm"): leaf}
    assert fusion_map(grid, param_overrides=overrides) is None
    assert zero_impedance_branches(grid, param_overrides=overrides) == []


def test_a_param_override_of_zero_fuses_an_ordinary_branch():
    grid = _grid([_switch(10, 1, 2, r=1.0e-4), _line(11, 2, 3)])
    overrides = {
        ("switch", 10, "resistance_ohm"): torch.zeros((), dtype=torch.float64),
        ("switch", 10, "inductance_h"): torch.zeros((), dtype=torch.float64),
    }
    fm = fusion_map(grid, param_overrides=overrides)
    assert fm is not None and fm.fused_branch_ids == (10,)


# --------------------------------------------------------------------------- #
# merged grids
# --------------------------------------------------------------------------- #
def test_a_fused_group_never_spans_two_merged_member_grids():
    from pgml.multigrid import merge_grids

    member = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    merged = merge_grids([member, member, member])
    fm = fusion_map(merged.grid)
    assert fm.size == 3 * 2
    for grp in fm.groups:
        starts = {r // member_rows for r in grp} if (member_rows := 3) else set()
        assert len(starts) == 1  # inside one member's contiguous row block


def test_block_rows_stay_a_partition_after_fusion():
    from pgml.multigrid import merge_grids

    member = _grid([_switch(10, 1, 2), _line(11, 2, 3)])
    merged = merge_grids([member, member])
    fm = fusion_map(merged.grid)
    blocks = [fm.reduce_rows(rows) for rows in merged.block_rows()]
    flat = torch.cat(blocks).tolist()
    assert sorted(flat) == list(range(fm.size))
    res = solve_power_flow(
        merged.grid,
        dtype=CDT,
        linear_solver="block",
        block_rows=merged.block_rows(),
    )
    assert res.converged
    single = solve_power_flow(member, dtype=CDT)
    for part in merged.split(res.v):
        assert torch.allclose(part, single.v, atol=1e-9)
