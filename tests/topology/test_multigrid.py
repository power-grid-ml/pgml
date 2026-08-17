"""Multi-grid batching: merged ensemble solve == per-member solves.

A disjoint union of grids assembles to the block-diagonal Y, so one merged solve
must reproduce every member's individual solution exactly — for the power flow,
the harmonic flow, batched operating points, branch states, gradients (which
must reach the ORIGINAL member grids' tensor leaves), and the prepared-system /
sparse pipeline.
"""

from __future__ import annotations

import pytest
import torch

from pgml.errors import ConnectivityError, InputError
from pgml.grids import synthetic_feeder
from pgml.multigrid import merge_grids
from pgml.solver import prepare_power_flow, solve_harmonic_flow, solve_power_flow


@pytest.fixture(scope="module")
def members():
    # Deliberately DIFFERENT sizes and colliding ids (every synthetic_feeder
    # reuses the same id ranges).
    return [
        synthetic_feeder(8, n_feeders=2),
        synthetic_feeder(15, n_feeders=3, tie_switches=1),
        synthetic_feeder(5, n_feeders=1),
    ]


def test_merge_bookkeeping(members):
    merged = merge_grids(members)
    assert len(merged.grid.nodes) == sum(len(g.nodes) for g in members)
    node_ids = [n.id for n in merged.grid.nodes]
    assert len(set(node_ids)) == len(node_ids)  # ids disjoint after remap
    assert merged.n_rows == sum(3 * len(g.nodes) for g in members)
    # first member keeps its ids; every map hits the merged grid
    assert merged.members[0].node_ids == {n.id: n.id for n in members[0].nodes}
    all_appliance_ids = {a.id for a in merged.grid.appliances}
    for m, g in zip(merged.members, members):
        assert set(m.appliance_ids.values()) <= all_appliance_ids
        assert m.n_rows == 3 * len(g.nodes)


def test_merged_power_flow_matches_member_solves(members):
    merged = merge_grids(members)
    res = solve_power_flow(merged.grid)
    assert res.converged
    parts = merged.split(res.v)
    for g, v_part in zip(members, parts):
        ref = solve_power_flow(g)
        assert torch.allclose(v_part, ref.v, atol=1e-6)


def test_block_rows_partition_the_merged_state(members):
    """The accessor names the same rows ``split`` slices — a full row partition."""
    merged = merge_grids(members)
    rows = merged.block_rows()
    assert [int(r.numel()) for r in rows] == [3 * len(g.nodes) for g in members]
    flat = torch.cat(rows)
    assert flat.dtype == torch.int64
    assert torch.equal(flat.sort().values, torch.arange(merged.n_rows))
    res = solve_power_flow(merged.grid, linear_solver="block", block_rows=rows)
    for part, gathered in zip(merged.split(res.v), rows):
        assert torch.equal(part, res.v.index_select(-1, gathered))


def test_merged_solve_with_per_member_operating_points(members):
    merged = merge_grids(members)
    g_rand = torch.Generator().manual_seed(3)
    ops = []
    for g in members:
        ops.append(
            {
                a.id: {
                    "p_w": float(a.p_nom_w) * (0.5 + torch.rand(4, generator=g_rand))
                }
                for a in g.appliances
                if a.id >= 20000
            }
        )
    res = solve_power_flow(merged.grid, operating_point=merged.operating_point(ops))
    assert res.converged and res.v.shape == (4, merged.n_rows)
    for g, op, v_part in zip(members, ops, merged.split(res.v)):
        ref = solve_power_flow(g, operating_point=op)
        assert torch.allclose(v_part, ref.v, atol=1e-6)


def test_merged_harmonic_flow_matches_members(members):
    from pgml.schemas.grid_schema import (
        HarmonicComponent,
        SpectrumPoint,
        StaticSpectrum,
    )

    spec = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=30.0),
            ]
        )
    )
    members = [g.model_copy(deep=True) for g in members]
    for g in members:
        next(a for a in g.appliances if a.id >= 20000).spectrum = spec
    merged = merge_grids(members)
    res = solve_harmonic_flow(merged.grid, [1, 5])
    for g, v_part in zip(members, merged.split(res.v)):
        ref = solve_harmonic_flow(g, [1, 5])
        assert torch.allclose(v_part, ref.v, atol=1e-6)


def test_gradients_reach_member_leaves(members):
    """Tensor leaves of the ORIGINAL member grids receive gradients (sharing)."""
    g0 = members[0].model_copy(deep=True)
    r_leaf = torch.tensor(
        [[0.3e-3 if i == j else 0.05e-3 for j in range(3)] for i in range(3)],
        dtype=torch.float64,
        requires_grad=True,
    )
    g0.branches[0].series_resistance_ohm_per_m = r_leaf
    merged = merge_grids([g0, members[2]])
    res = solve_power_flow(merged.grid)
    res.v.abs().sum().backward()
    assert r_leaf.grad is not None
    assert torch.isfinite(r_leaf.grad).all()
    assert float(r_leaf.grad.abs().sum()) > 0.0


def test_branch_states_remap_and_solve(members):
    merged = merge_grids(members)
    # member 1 owns the tie switch (local id 30000)
    states = merged.branch_states([None, {30000: torch.tensor([0.0, 1.0])}, None])
    res = solve_power_flow(merged.grid, branch_states=states)
    assert res.converged and res.v.shape == (2, merged.n_rows)
    ref_open = solve_power_flow(members[1], branch_states={30000: 0.0})
    ref_closed = solve_power_flow(members[1], branch_states={30000: 1.0})
    part = merged.split(res.v)[1]
    assert torch.allclose(part[0], ref_open.v, atol=1e-6)
    assert torch.allclose(part[1], ref_closed.v, atol=1e-6)


def test_prepared_system_on_merged_grid(members):
    merged = merge_grids(members)
    system = prepare_power_flow(merged.grid)
    ref = solve_power_flow(merged.grid)
    res = solve_power_flow(merged.grid, system=system)
    assert torch.allclose(ref.v, res.v)


def test_member_without_source_raises_connectivity(members):
    dead = members[0].model_copy(
        update={
            "appliances": [a for a in members[0].appliances if a.id != 1],
        }
    )
    merged = merge_grids([members[1], dead])
    with pytest.raises(ConnectivityError):
        solve_power_flow(merged.grid)


def test_frequency_mismatch_raises(members):
    g60 = members[0].model_copy(update={"base_frequency_hz": 60.0})
    with pytest.raises(InputError, match="base_frequency_hz"):
        merge_grids([members[1], g60])


def test_remap_rejects_unknown_ids(members):
    merged = merge_grids(members)
    with pytest.raises(InputError, match="member 1"):
        merged.operating_point([None, {999999: {"p_w": 1.0}}, None])
    with pytest.raises(InputError, match="one entry per member"):
        merged.operating_point([None])


def test_large_ensemble_uses_sparse_backend():
    """Many small grids merge past the sparse threshold and still solve exactly."""
    members = [synthetic_feeder(20, n_feeders=2) for _ in range(12)]  # 720 rows
    merged = merge_grids(members)
    system = prepare_power_flow(merged.grid)
    assert system.factorization.backend == "sparse"
    res = solve_power_flow(merged.grid, system=system)
    assert res.converged
    ref = solve_power_flow(members[0])
    assert torch.allclose(merged.split(res.v)[0], ref.v, atol=1e-6)
