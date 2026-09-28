"""The values accumulator holds exactly the structural entries of the dense one.

:class:`pgml.assembly._scatter.PatternAccumulator` is the form of ``Y`` the harmonic
orders assemble on the sparse backend: the same scatter, but into ``[*batch, H, nnz]``
over the topology's sparsity pattern instead of ``[*batch, H, N, N]``. It must agree
with the dense accumulator BIT FOR BIT at every pattern position (the additions reach
each entry in the same order), and a stamp outside the pattern must raise instead of
being dropped. Pinned on the structures that change which rows a stamp touches.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import node_phase_index, ybus_structure
from pgml.assembly._fusion import resolve_fusion
from pgml.assembly._scatter import PatternAccumulator, scatter_blocks_into
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_network, _stamp_sources
from pgml.errors import InputError

from tests.topology.test_ybus_structure import FREQS, GRIDS, _fused_grid

CDT = torch.complex128


def _both(grid, branch_states=None):
    """The passive network plus the source stamp, dense and structural."""
    fused = resolve_fusion(grid, None, branch_states=branch_states)
    index = fused.index if fused is not None else node_phase_index(grid)
    n = index.size
    f = torch.as_tensor(FREQS, dtype=torch.float64)
    cdt, rdt = _cdtype(CDT), _rdtype(CDT)
    pattern = ybus_structure(grid, index)
    dense = torch.zeros((len(FREQS), n, n), dtype=cdt)
    acc = PatternAccumulator.zeros(pattern, n, len(FREQS), cdt, None)
    out = []
    for y in (dense, acc):
        y = _stamp_network(
            grid, f, y, index, cdt, rdt, None, None, branch_states, fused
        )
        out.append(_stamp_sources(grid, f, y, index, cdt, rdt, None, None))
    return out[0], out[1], n


def _assert_same_entries(dense, acc, n):
    flat = dense.reshape(*dense.shape[:-2], n * n)
    assert torch.equal(flat.index_select(-1, acc.pattern), acc.values)
    outside = torch.ones(n * n, dtype=torch.bool)
    outside[acc.pattern] = False
    assert not bool((flat[..., outside] != 0).any())
    assert torch.equal(acc.to_dense(), dense)


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_structural_entries_equal_the_dense_matrix(name):
    dense, acc, n = _both(GRIDS[name]())
    _assert_same_entries(dense, acc, n)


def test_batched_branch_states_widen_both_accumulators_alike():
    states = {12: torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)}
    dense, acc, n = _both(_fused_grid(), branch_states=states)
    assert acc.values.shape == (3, len(FREQS), acc.pattern.numel())
    _assert_same_entries(dense, acc, n)


def test_a_stamp_outside_the_pattern_raises():
    """A pattern that misses an entry fails loudly instead of losing the coupling."""
    n = 3
    pattern = torch.tensor([0, 4, 8], dtype=torch.int64)  # the diagonal only
    acc = PatternAccumulator.zeros(pattern, n, 1, CDT, None)
    block = torch.ones((1, 1, 2, 2), dtype=CDT)
    rows = torch.tensor([[0, 1]])
    with pytest.raises(InputError, match=r"Y\[0, 1\]"):
        scatter_blocks_into(acc, block, rows, rows)


def test_every_lookup_hits_and_duplicates_accumulate():
    n = 3
    pattern = torch.tensor([0, 1, 3, 4, 8], dtype=torch.int64)
    acc = PatternAccumulator.zeros(pattern, n, 2, CDT, None)
    # Two elements on the same rows: their blocks land on the same entries.
    block = torch.arange(1, 17, dtype=torch.float64).to(CDT).reshape(2, 2, 2, 2)
    rows = torch.tensor([[0, 1], [0, 1]])
    out = scatter_blocks_into(acc, block, rows, rows)
    dense = scatter_blocks_into(torch.zeros((2, n, n), dtype=CDT), block, rows, rows)
    assert torch.equal(out.to_dense(), dense)
    assert torch.equal(acc.positions(pattern), torch.arange(pattern.numel()))
    assert torch.equal(acc.diagonal_positions(), torch.tensor([0, 3, 4]))


def test_the_scatter_is_differentiable_in_the_values():
    n = 2
    pattern = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    block = torch.randn((1, 1, 2, 2), dtype=CDT, requires_grad=True)
    rows = torch.tensor([[0, 1]])

    def entries(b):
        acc = PatternAccumulator.zeros(pattern, n, 1, CDT, None)
        return scatter_blocks_into(acc, b, rows, rows).values

    assert torch.autograd.gradcheck(entries, (block,))
