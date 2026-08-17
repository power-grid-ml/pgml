"""Block-diagonal factorization backend: parity with dense/sparse + error paths.

An ensemble of independent grids merged with :func:`pgml.multigrid.merge_grids`
assembles to ``Y = diag(Y_1, …, Y_G)``. The block backend
(:func:`pgml.solver.harmonic.lu_factor_system` with ``backend="block"``) factors
each member's diagonal block instead of the union, so it must reproduce — exactly —
the dense and sparse union solves AND every member's individual solve, in both
slack modes, for scenario batches and for a batch holding a member that does not
converge. It must also hold ONLY the per-block factors: a union-sized dense factor
would defeat the purpose.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_network_ybus, assemble_ybus
from pgml.errors import InputError
from pgml.grids import synthetic_feeder
from pgml.multigrid import merge_grids
from pgml.solver import prepare_power_flow, solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored

RTOL = 1.0e-9
ATOL_V = 1.0e-6  # 1e-10 relative at the ~11 kV line-to-neutral scale


@pytest.fixture(scope="module")
def members():
    """Five heterogeneous feeders; two share a size (one bucket holds B=2)."""
    return [
        synthetic_feeder(6, n_feeders=2),
        synthetic_feeder(9, n_feeders=3),
        synthetic_feeder(6, n_feeders=2),
        synthetic_feeder(4, n_feeders=1),
        synthetic_feeder(12, n_feeders=4, tie_switches=1),
    ]


@pytest.fixture(scope="module")
def merged(members):
    return merge_grids(members)


def _op(grids, b, seed=0):
    """A per-member operating point with a leading scenario batch of ``b``."""
    g = torch.Generator().manual_seed(seed)
    return [
        {
            a.id: {"p_w": float(a.p_nom_w) * (0.5 + torch.rand(b, generator=g))}
            for a in grid.appliances
            if a.id >= 20000
        }
        for grid in grids
    ]


# --- linear-system level ---------------------------------------------------- #
def test_block_norton_parity_multi_frequency(merged):
    """A frequency stack folds into the bucket batch: same result as dense."""
    y = assemble_ybus(merged.grid, [50.0, 250.0, 350.0]).Y  # [3, N, N]
    rhs = torch.randn(5, 3, y.shape[-1], dtype=y.dtype)  # 5 scenarios x 3 orders
    vd = solve_factored(lu_factor_system(y, backend="dense"), rhs)
    vb = solve_factored(
        lu_factor_system(y, backend="block", block_rows=merged.block_rows()), rhs
    )
    assert torch.allclose(vd, vb, rtol=RTOL, atol=RTOL * float(vd.abs().max()))


def test_block_ideal_slack_parity(merged):
    y = assemble_network_ybus(merged.grid, [50.0]).Y  # [1, N, N]
    fixed_rows = torch.tensor(
        [merged.members[k].row_start + p for k in range(5) for p in range(3)]
    )
    v_fixed = 11547.0 * torch.exp(
        1j * torch.tensor([0.0, -2.0943951, 2.0943951] * 5, dtype=torch.float64)
    )
    rhs = torch.randn(4, 1, y.shape[-1], dtype=y.dtype)
    vd = solve_factored(
        lu_factor_system(y, fixed_rows=fixed_rows, backend="dense"),
        rhs,
        v_fixed=v_fixed,
    )
    vb = solve_factored(
        lu_factor_system(
            y, fixed_rows=fixed_rows, backend="block", block_rows=merged.block_rows()
        ),
        rhs,
        v_fixed=v_fixed,
    )
    assert torch.allclose(vd, vb, rtol=RTOL, atol=RTOL * float(vd.abs().max()))


@pytest.mark.parametrize("fixed_rows", [None, torch.tensor([2, 5])])
def test_block_rows_need_not_be_contiguous(fixed_rows):
    """Interleaved blocks: the backend gathers/scatters, it does not slice."""
    torch.manual_seed(1)
    rows = [torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5])]
    y = torch.zeros(6, 6, dtype=torch.complex128)
    for r in rows:
        blk = torch.randn(3, 3, dtype=torch.complex128) + 4.0 * torch.eye(
            3, dtype=torch.complex128
        )
        y[r.unsqueeze(-1), r.unsqueeze(-2)] = blk
    rhs = torch.randn(6, dtype=torch.complex128)
    v_fixed = None if fixed_rows is None else torch.randn(2, dtype=torch.complex128)
    vd = solve_factored(
        lu_factor_system(y, fixed_rows=fixed_rows, backend="dense"),
        rhs,
        v_fixed=v_fixed,
    )
    vb = solve_factored(
        lu_factor_system(y, fixed_rows=fixed_rows, backend="block", block_rows=rows),
        rhs,
        v_fixed=v_fixed,
    )
    assert vb.shape == vd.shape
    assert torch.allclose(vd, vb, rtol=RTOL, atol=RTOL * float(vd.abs().max()))


# --- power flow ------------------------------------------------------------- #
@pytest.mark.parametrize("slack", ["ideal", "norton"])
def test_block_matches_union_and_member_solves(members, merged, slack):
    rows = merged.block_rows()
    rd = solve_power_flow(merged.grid, slack=slack, linear_solver="dense")
    rs = solve_power_flow(merged.grid, slack=slack, linear_solver="sparse")
    rb = solve_power_flow(
        merged.grid, slack=slack, linear_solver="block", block_rows=rows
    )
    assert rd.converged and rs.converged and rb.converged
    assert torch.allclose(rb.v, rd.v, rtol=RTOL, atol=ATOL_V)
    assert torch.allclose(rb.v, rs.v, rtol=RTOL, atol=ATOL_V)
    for grid, part in zip(members, merged.split(rb.v)):
        ref = solve_power_flow(grid, slack=slack)
        assert torch.allclose(part, ref.v, rtol=RTOL, atol=ATOL_V)


def test_block_batched_operating_point(members, merged):
    ops = _op(members, 4, seed=3)
    op = merged.operating_point(ops)
    rd = solve_power_flow(merged.grid, operating_point=op, linear_solver="dense")
    rb = solve_power_flow(
        merged.grid,
        operating_point=op,
        linear_solver="block",
        block_rows=merged.block_rows(),
    )
    assert rd.converged and rb.converged
    assert rb.v.shape == (4, merged.n_rows)
    assert torch.allclose(rb.v, rd.v, rtol=RTOL, atol=ATOL_V)
    for grid, member_op, part in zip(members, ops, merged.split(rb.v)):
        ref = solve_power_flow(grid, operating_point=member_op)
        assert torch.allclose(part, ref.v, rtol=RTOL, atol=ATOL_V)


def test_block_with_batched_branch_states(merged, members):
    """A per-scenario ``Y`` gives every block its own factorization per scenario."""
    states = merged.branch_states(
        [None, None, None, None, {30000: torch.tensor([0.0, 1.0])}]
    )
    rd = solve_power_flow(merged.grid, branch_states=states, linear_solver="dense")
    rb = solve_power_flow(
        merged.grid,
        branch_states=states,
        linear_solver="block",
        block_rows=merged.block_rows(),
    )
    assert rd.converged and rb.converged
    assert rb.v.shape == (2, merged.n_rows)
    assert torch.allclose(rb.v, rd.v, rtol=RTOL, atol=ATOL_V)


def test_block_preserves_converged_mask_of_failing_member(members, merged):
    """One diverging member must not poison the batch (mask semantics unchanged)."""
    load = next(a for a in members[1].appliances if a.id >= 20000)
    op = merged.operating_point(
        [None, {load.id: {"p_w": torch.tensor([1.0e6, 5.0e9])}}, None, None, None]
    )
    rd = solve_power_flow(merged.grid, operating_point=op, linear_solver="dense")
    rb = solve_power_flow(
        merged.grid,
        operating_point=op,
        linear_solver="block",
        block_rows=merged.block_rows(),
    )
    assert not rd.converged and not rb.converged  # no exception on either backend
    assert torch.equal(rb.converged_mask, rd.converged_mask)
    assert bool(rb.converged_mask[0]) and not bool(rb.converged_mask[1])
    assert rb.failed_states == rd.failed_states == (1,)
    # the healthy scenario still carries the same voltages as the dense backend
    assert torch.allclose(rb.v[0], rd.v[0], rtol=RTOL, atol=ATOL_V)


def test_block_prepared_system_reuse(merged, members):
    rows = merged.block_rows()
    system = prepare_power_flow(merged.grid, linear_solver="block", block_rows=rows)
    assert system.factorization.backend == "block"
    op = merged.operating_point(_op(members, 3, seed=5))
    ref = solve_power_flow(merged.grid, operating_point=op, linear_solver="dense")
    res = solve_power_flow(merged.grid, operating_point=op, system=system)
    assert torch.allclose(res.v, ref.v, rtol=RTOL, atol=ATOL_V)


# --- structure -------------------------------------------------------------- #
def test_factor_holds_only_per_block_tensors(merged):
    """No union-sized factor: only the per-bucket blocks are stored."""
    rows = merged.block_rows()
    fac = prepare_power_flow(
        merged.grid, linear_solver="block", block_rows=rows
    ).factorization
    n = merged.n_rows
    assert fac.lu is None and fac.piv is None and fac.y_mat is None
    block = fac.block
    assert block.n_blocks == len(rows)
    assert len(block.buckets) == 4  # distinct free-row sizes among the 5 members
    entries = 0
    for pos, lu, piv in block.buckets:
        assert lu.shape[-1] == lu.shape[-2] == pos.shape[-1] < n
        assert piv.shape[-1] == pos.shape[-1]
        entries += lu.numel()
    # Sum of squared block sizes, far below the union's N^2.
    assert entries == sum((int(r.numel()) - 3) ** 2 for r in rows)
    assert entries < n * n


# --- opt-in / error paths --------------------------------------------------- #
def test_auto_never_selects_block(merged):
    y = assemble_ybus(merged.grid, [50.0]).Y
    assert lu_factor_system(y, backend="auto").backend != "block"


def test_block_without_block_rows_raises(merged):
    y = assemble_ybus(merged.grid, [50.0]).Y
    with pytest.raises(InputError, match="block_rows"):
        lu_factor_system(y, backend="block")


def test_block_rows_without_block_backend_raises(merged):
    y = assemble_ybus(merged.grid, [50.0]).Y
    with pytest.raises(InputError, match="only by backend='block'"):
        lu_factor_system(y, backend="dense", block_rows=merged.block_rows())


@pytest.mark.parametrize(
    "bad",
    [
        [torch.tensor([0, 1, 2]), torch.tensor([2, 3])],  # overlapping
        [torch.tensor([0, 1])],  # incomplete
        [torch.arange(6), torch.tensor([6])],  # out of range / too many
    ],
)
def test_non_partition_block_rows_raises(bad):
    y = torch.eye(6, dtype=torch.complex128) * 3.0
    with pytest.raises(InputError, match="partition"):
        lu_factor_system(y, backend="block", block_rows=bad)


def test_empty_block_raises():
    y = torch.eye(4, dtype=torch.complex128) * 3.0
    with pytest.raises(InputError, match="empty"):
        lu_factor_system(
            y,
            backend="block",
            block_rows=[torch.arange(4), torch.tensor([], dtype=torch.int64)],
        )


def test_solve_power_flow_block_requires_block_rows(merged):
    with pytest.raises(InputError, match="block_rows"):
        solve_power_flow(merged.grid, linear_solver="block")


def test_solve_power_flow_block_rows_need_block_solver(merged):
    with pytest.raises(InputError, match="block_rows"):
        solve_power_flow(merged.grid, block_rows=merged.block_rows())


def test_newton_rejects_block(merged):
    with pytest.raises(InputError, match="newton"):
        solve_power_flow(
            merged.grid,
            method="newton",
            linear_solver="block",
            block_rows=merged.block_rows(),
        )


def test_block_rejects_on_disconnected_zero(merged):
    with pytest.raises(InputError, match="on_disconnected"):
        solve_power_flow(
            merged.grid,
            linear_solver="block",
            block_rows=merged.block_rows(),
            on_disconnected="zero",
        )
