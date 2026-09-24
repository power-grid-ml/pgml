"""Structural sparsity pattern of the nodal admittance, derived from the topology.

Every contribution the assembler scatters into ``Y`` is a COMPONENT-LOCAL block: a
branch stamps the outer product of its two terminals' node-phase rows, a
single-terminal branch and every appliance stamp (source Norton, const-Z device
shunt, ShuntAppliance bank, harmonic device shunt) the outer product of its host
node's rows, and a node harmonic source adds to the diagonal. The union of those
blocks over the grid is therefore a SUPERSET of the nonzero positions of ``Y`` at
any frequency, operating point, branch state and scenario -- it depends on the
topology alone, so it is built once and reused for every system a study assembles.

The sparse solver backend consumes it to build each system's compressed-column form
by gathering ``nnz`` values instead of scanning ``N^2`` dense entries
(:class:`pgml.solver.harmonic._SciPySparseLU`). Pure indexing bookkeeping: int64
tensors and Python loops over components, never a differentiable quantity.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Grid

from .index import NodePhaseIndex


def ybus_structure(
    grid: Grid,
    index: NodePhaseIndex,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Sorted int64 linear indices ``row * N + col`` of every stampable ``Y`` entry.

    Parameters
    ----------
    grid:
        The grid whose topology defines the pattern. Out-of-service and open
        components are INCLUDED: ``branch_states`` may stamp them anyway, and an
        entry that stays zero costs one explicit zero, never a wrong answer.
    index:
        The row layout of the matrix the pattern describes -- the grid's own
        :class:`~pgml.assembly.index.NodePhaseIndex`, or the REDUCED one of a
        :class:`~pgml.assembly._fusion.FusionMap`. The reduced layout is a
        many-to-one ``(node, phase) -> row`` map, so reading each node's rows out of
        it accumulates the fused pattern of ``P^T Y P`` exactly as the stamps do.
    device:
        Device of the returned tensor (default: ``index``'s).

    Returns
    -------
    Tensor
        int64 ``[nnz]``, strictly increasing. ``row = lin // N``, ``col = lin % N``
        with ``N = index.size``.
    """
    n = int(index.size)
    rows_of_node = index._rows_of_node
    # Group the touched row sets by their SIZE so each group is one vectorized
    # outer product; a grid has a handful of distinct sizes (phase counts), never
    # one per component.
    by_size: dict[int, list[list[int]]] = {}

    def add(touched: list[int]) -> None:
        if touched:
            by_size.setdefault(len(touched), []).append(touched)

    for branch in grid.branches:
        touched = list(rows_of_node.get(branch.from_node, ()))
        to_rows = rows_of_node.get(branch.to_node, ())
        if branch.to_node != branch.from_node:
            touched += list(to_rows)
        add(touched)
    for appliance in grid.appliances:
        add(list(rows_of_node.get(appliance.node, ())))

    dev = device if device is not None else index.node_ids.device

    # The diagonal is always structurally present: it carries every shunt term, the
    # equilibration reads it, and a node harmonic source adds to it directly.
    diag = torch.arange(n, dtype=torch.int64, device=dev)
    parts = [diag * n + diag]
    for sets in by_size.values():
        r = torch.as_tensor(sets, dtype=torch.int64, device=dev)  # [K, size]
        parts.append((r[:, :, None] * n + r[:, None, :]).reshape(-1))
    return torch.unique(torch.cat(parts))


__all__ = ["ybus_structure"]
