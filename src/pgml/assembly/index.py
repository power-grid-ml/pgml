"""Compact node-phase indexing — THE canonical layout for Y-bus / solve.

One matrix row per *existing* ``(node, phase)`` slot (NOT a padded A,B,C,N grid).
Rows are assigned by iterating ``grid.nodes`` in list order and, within a node, by
that node's ``phases`` tuple order; row indices are consecutive from 0.

``N = sum(len(node.phases) for node in grid.nodes)``.

This module is pure indexing bookkeeping (Python ints + int64 tensors). It does
NOT touch differentiable quantities, so plain Python loops here are fine and do
not violate the no-loop rule (which applies to the differentiable tensor path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Grid, Phase


@dataclass(frozen=True)
class NodePhaseIndex:
    """Compact (node, phase) -> row map for the nodal admittance system.

    Attributes
    ----------
    size:
        Number of rows ``N``.
    node_ids:
        int64 tensor ``[N]`` — node id of each row.
    phase_codes:
        int64 tensor ``[N]`` — :class:`Phase` ordinal of each row (a=0,b=1,c=2,n=3).
    """

    size: int
    node_ids: Tensor
    phase_codes: Tensor
    # Internal lookup maps (not part of the public tensor surface).
    _row_of: dict[tuple[int, Phase], int] = field(default_factory=dict, repr=False)
    _rows_of_node: dict[int, list[int]] = field(default_factory=dict, repr=False)
    _phase_order: tuple[Phase, ...] = field(
        default=(Phase.A, Phase.B, Phase.C, Phase.N), repr=False
    )

    # -- scalar lookups -------------------------------------------------------
    def row(self, node_id: int, phase: Phase) -> int:
        """Row index of ``(node_id, phase)``."""
        return self._row_of[(node_id, phase)]

    def rows(self, node_id: int) -> list[int]:
        """Row indices of all phases of ``node_id`` (aligned to its ``phases`` order)."""
        return list(self._rows_of_node[node_id])

    def has(self, node_id: int, phase: Phase) -> bool:
        """Does the layout carry a row for ``(node_id, phase)``?"""
        return (node_id, phase) in self._row_of

    def node_id_of(self, row: int) -> int:
        """Node id of ``row``."""
        return int(self.node_ids[row])

    def phase_of(self, row: int) -> Phase:
        """:class:`Phase` of ``row``."""
        return self._phase_order[int(self.phase_codes[row])]

    # -- vectorized helpers ---------------------------------------------------
    def rows_for_terminal(
        self,
        node_id: int,
        phases: tuple[Phase, ...],
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """int64 row-index tensor ``[len(phases)]`` for one terminal's phase set.

        Used to build the scatter indices for a branch terminal or appliance.
        """
        idx = [self._row_of[(node_id, ph)] for ph in phases]
        return torch.as_tensor(idx, dtype=torch.int64, device=device)


_PHASE_CODE = {Phase.A: 0, Phase.B: 1, Phase.C: 2, Phase.N: 3}
_PHASE_ORDER = (Phase.A, Phase.B, Phase.C, Phase.N)


def node_phase_index(grid: Grid) -> NodePhaseIndex:
    """Build the compact :class:`NodePhaseIndex` for ``grid``.

    Deterministic ordering: nodes in ``grid.nodes`` list order, phases in each
    node's ``phases`` tuple order, rows consecutive from 0.
    """
    row_of: dict[tuple[int, Phase], int] = {}
    rows_of_node: dict[int, list[int]] = {}
    node_ids: list[int] = []
    phase_codes: list[int] = []

    r = 0
    for node in grid.nodes:
        node_rows: list[int] = []
        for ph in node.phases:
            row_of[(node.id, ph)] = r
            node_rows.append(r)
            node_ids.append(node.id)
            phase_codes.append(_PHASE_CODE[ph])
            r += 1
        rows_of_node[node.id] = node_rows

    return NodePhaseIndex(
        size=r,
        node_ids=torch.as_tensor(node_ids, dtype=torch.int64),
        phase_codes=torch.as_tensor(phase_codes, dtype=torch.int64),
        _row_of=row_of,
        _rows_of_node=rows_of_node,
        _phase_order=_PHASE_ORDER,
    )


def base_voltage_per_row(
    grid: Grid,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Per-row line-to-neutral base voltage ``[N]``, aligned to :func:`node_phase_index`.

    The per-unit voltage reference for each node-phase row: dividing a state or residual
    ``[..., N]`` by this puts it in per-unit. Uses the same connection-aware convention as
    the solver (:func:`pgml.assembly._params.phase_voltage_magnitude`): ``Node.u_rated_v`` is
    line-to-line for nodes with >= 3 phases (so the L-N base is ``u_rated / sqrt(3)``) and
    already line-to-neutral for 1-phase nodes.

    This is a fixed reference constant for normalization / per-unit reporting, NOT a value on
    the differentiable path; if ``u_rated_v`` is a tracked tensor its scalar value is read
    out. Row order matches :func:`node_phase_index`.
    """
    from ._params import phase_voltage_magnitude

    idx = node_phase_index(grid)
    base = torch.empty(idx.size, dtype=dtype)
    for node in grid.nodes:
        u = node.u_rated_v
        u = float(u.detach().reshape(-1)[0]) if hasattr(u, "detach") else float(u)
        b = phase_voltage_magnitude(u, len(node.phases))
        for row in idx.rows(node.id):
            base[row] = b
    return base.to(device) if device is not None else base


__all__ = ["NodePhaseIndex", "node_phase_index", "base_voltage_per_row"]
