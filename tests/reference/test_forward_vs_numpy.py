"""Forward-correctness: pgml assemble + solve vs an independent numpy oracle."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.solver import solve_harmonic

from tests.fixtures.tiny_grids import (
    single_phase_chain,
    single_phase_shunt_only,
    three_phase_two_bus,
)
from tests.reference.numpy_oracle import build_y_and_i, solve_norton

GRIDS = [single_phase_chain, single_phase_shunt_only, three_phase_two_bus]

# These tiny grids have no frequency-dependent harmonic branch (a single ``[f]``
# is always treated as a scalar frequency), so a second frequency adds no coverage
# to the Y/I assembly comparisons — one frequency exercises the same code path.
FREQ = 50.0


@pytest.mark.parametrize("grid_fn", GRIDS)
def test_ybus_matches_numpy(grid_fn):
    grid = grid_fn()
    yb = assemble_ybus(grid, [FREQ], dtype=torch.complex128)
    y_torch = yb.Y[0].numpy()  # [N, N]
    y_np, _, row_of, n = build_y_and_i(grid, FREQ)

    # Row layouts must agree element-by-element.
    idx = node_phase_index(grid)
    assert idx.size == n
    np.testing.assert_allclose(y_torch, y_np, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("grid_fn", GRIDS)
def test_injections_match_numpy(grid_fn):
    grid = grid_fn()
    idx = node_phase_index(grid)
    i_torch = build_injections(grid, [FREQ], idx, dtype=torch.complex128)[0].numpy()
    _, i_np, _, _ = build_y_and_i(grid, FREQ)
    np.testing.assert_allclose(i_torch, i_np, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("grid_fn", GRIDS)
def test_norton_solve_matches_numpy(grid_fn):
    grid = grid_fn()
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, [FREQ], dtype=torch.complex128)
    i = build_injections(grid, [FREQ], idx, dtype=torch.complex128)
    v_torch = solve_harmonic(yb.Y, i)[0].numpy()

    v_np, _, _, _ = solve_norton(grid, FREQ)
    np.testing.assert_allclose(v_torch, v_np, rtol=1e-8, atol=1e-10)


def test_batched_multi_frequency_shapes():
    """Batched assembly/solve over multiple frequencies has the right shapes.

    The per-frequency values are already validated against the numpy oracle by
    ``test_norton_solve_matches_numpy``; this test guards only the batch dimension.
    """
    grid = single_phase_chain()
    idx = node_phase_index(grid)
    freqs = [50.0, 150.0, 250.0]
    yb = assemble_ybus(grid, freqs, dtype=torch.complex128)
    i = build_injections(grid, freqs, idx, dtype=torch.complex128)
    assert yb.Y.shape == (3, idx.size, idx.size)
    assert i.shape == (3, idx.size)
    v = solve_harmonic(yb.Y, i)
    assert v.shape == (3, idx.size)


def test_ideal_slack_holds_fixed_voltage():
    """Ideal-slack mode: the source node voltage is held exactly at v_fixed."""
    grid = single_phase_shunt_only()
    idx = node_phase_index(grid)
    f = 50.0
    yb = assemble_ybus(grid, [f], dtype=torch.complex128)
    i = build_injections(grid, [f], idx, dtype=torch.complex128)

    slack_row = idx.row(1, grid.nodes[0].phases[0])
    v_slack = torch.tensor([1.05 * 230.0 + 0.0j], dtype=torch.complex128)
    fixed_rows = torch.tensor([slack_row], dtype=torch.int64)

    v = solve_harmonic(yb.Y, i, fixed_rows=fixed_rows, v_fixed=v_slack)
    np.testing.assert_allclose(
        v[0, slack_row].numpy(), v_slack[0].numpy(), rtol=1e-10, atol=1e-12
    )

    # Independent check of the free node via the Schur complement in numpy.
    y_np, i_np, _, n = build_y_and_i(grid, f)
    free = [r for r in range(n) if r != slack_row]
    y_ff = y_np[np.ix_(free, free)]
    y_fs = y_np[np.ix_(free, [slack_row])]
    rhs = i_np[free] - (y_fs @ v_slack.numpy())
    v_free = np.linalg.solve(y_ff, rhs)
    np.testing.assert_allclose(v[0, free].numpy(), v_free, rtol=1e-8, atol=1e-10)
