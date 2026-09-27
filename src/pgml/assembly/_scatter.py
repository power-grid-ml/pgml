"""Vectorized differentiable scatter-add of primitive blocks into the Y-bus.

The Y-bus is built by accumulating, for each component group, a batched stack of
dense primitive blocks ``blocks[*batch, H, K, M, M]`` at global ``(row, col)``
positions given by per-element index tensors ``rows[K, M]`` / ``cols[K, M]``.

Implementation: flatten the last two axes of the accumulator to a single ``N*N``
axis and the ``(K, M, M)`` element/local axes of ``blocks`` to a single list,
then ``Tensor.index_add_`` along that flat axis with the linear index
``row*N + col``. ``index_add_`` accumulates duplicate targets (essential for
mutual / overlapping stamps), broadcasts the SAME (row, col) targets across all
leading batch/H dims, and has a defined autograd-safe backward. The index tensor
is plain int64 (non-differentiable); the VALUES carry gradients. No in-place op on
a tracked LEAF tensor and no python loop over elements.

Two accumulators share that one operation. A dense tensor ``[*batch, H, N, N]`` is
the matrix itself. A :class:`PatternAccumulator` holds only the STRUCTURAL entries
``[*batch, H, nnz]`` of a sorted sparsity pattern (:func:`pgml.assembly.ybus_structure`):
each linear index is mapped to its position in the pattern by ``searchsorted`` and
the same ``index_add_`` runs along the ``nnz`` axis. The additions reach every entry
in the same order either way, so the structural entries of the two accumulators are
equal bit for bit, while the pattern form holds ``nnz`` numbers per system instead
of ``N^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from pgml.errors import InputError


@dataclass(frozen=True)
class PatternAccumulator:
    """The structural entries of a batch of ``N x N`` matrices over one sparsity pattern.

    Attributes
    ----------
    values:
        Complex ``[*batch, H, nnz]``; entry ``k`` is the matrix entry at linear index
        ``pattern[k]``. Carries the autograd graph of everything scattered into it.
    pattern:
        int64 ``[nnz]``, strictly increasing linear indices ``row * n + col`` on the
        device of ``values`` (:func:`pgml.assembly.ybus_structure`).
    n:
        The row count ``N`` of the matrices described.

    Every position outside ``pattern`` is an exact zero by construction: a scatter to
    an index the pattern does not list raises instead of being dropped.
    """

    values: Tensor
    pattern: Tensor
    n: int

    @classmethod
    def zeros(
        cls,
        pattern: Tensor,
        n: int,
        n_orders: int,
        dtype: torch.dtype,
        device,
    ) -> "PatternAccumulator":
        """An all-zero accumulator ``[n_orders, nnz]`` over ``pattern``."""
        pattern = pattern.to(device=device, dtype=torch.int64)
        values = torch.zeros(
            (n_orders, pattern.numel()), dtype=dtype, device=pattern.device
        )
        return cls(values=values, pattern=pattern, n=int(n))

    def with_values(self, values: Tensor) -> "PatternAccumulator":
        """The same pattern holding ``values``."""
        return PatternAccumulator(values=values, pattern=self.pattern, n=self.n)

    def clone(self) -> "PatternAccumulator":
        """An accumulator owning a copy of the values (the pattern is shared)."""
        return self.with_values(self.values.clone())

    def positions(self, linear: Tensor) -> Tensor:
        """Position in :attr:`pattern` of every linear index in ``linear``.

        Raises :class:`~pgml.errors.InputError` if any index is not in the pattern:
        a contribution outside the pattern would otherwise be lost without a trace.
        Reads one boolean back from the device (the check), so it is bookkeeping for
        the host-side sparse path rather than for a CUDA hot loop.
        """
        nnz = self.pattern.numel()
        pos = torch.searchsorted(self.pattern, linear)
        found = pos.clamp(max=max(nnz - 1, 0))
        hit = (pos < nnz) & (self.pattern[found] == linear) if nnz else pos < 0
        if not bool(hit.all()):
            missing = linear[~hit][0]
            row, col = divmod(int(missing), self.n)
            raise InputError(
                f"A stamp writes to Y[{row}, {col}], which the sparsity pattern of "
                f"this {self.n}-row system does not list; the pattern has to be a "
                "superset of every stamped position (pgml.assembly.ybus_structure)."
            )
        return pos

    def diagonal_positions(self) -> Tensor:
        """Positions of the ``n`` diagonal entries in :attr:`pattern`, in row order."""
        diag = torch.arange(self.n, dtype=torch.int64, device=self.pattern.device)
        return self.positions(diag * (self.n + 1))

    def to_dense(self) -> Tensor:
        """The matrices as a dense ``[*batch, H, N, N]`` tensor (differentiable)."""
        lead = self.values.shape[:-1]
        dense = torch.zeros(
            (*lead, self.n * self.n), dtype=self.values.dtype, device=self.values.device
        )
        dense = dense.index_add(-1, self.pattern, self.values)
        return dense.reshape(*lead, self.n, self.n)


Accumulator = Union[Tensor, PatternAccumulator]


def scatter_blocks_into(
    y: Accumulator,
    blocks: Tensor,
    rows: Tensor,
    cols: Tensor,
) -> Accumulator:
    """Return ``y`` with ``blocks`` accumulated at the (rows, cols) targets.

    Parameters
    ----------
    y:
        The accumulator: a dense complex tensor ``[*batch, H, N, N]``, or a
        :class:`PatternAccumulator` holding the structural entries ``[*batch, H,
        nnz]`` of such a matrix.
    blocks:
        Complex primitive blocks ``[*batch_b, H, K, M, M]`` whose leading
        (batch, H) dims broadcast against ``y``'s leading dims.
    rows, cols:
        int64 tensors ``[K, M]`` mapping each block's local row/col axis to a
        global Y index. The (i, j) entry of element k scatters to
        ``(rows[k, i], cols[k, j])``.

    Returns
    -------
    Tensor or PatternAccumulator
        A new accumulator of the same kind as ``y`` (never ``y`` itself) with the
        blocks added; a batched ``blocks`` widens its leading dims by broadcasting.

    Raises
    ------
    InputError
        For a :class:`PatternAccumulator`, when a target is not in its pattern.
    """
    if blocks.shape[-3] == 0:
        # No elements of this kind; nothing to scatter.
        return y

    k = blocks.shape[-3]
    m = blocks.shape[-1]
    n = y.n if isinstance(y, PatternAccumulator) else y.shape[-1]

    # Linear (row*N + col) index for every (k, i, j) local position -> [K, M, M].
    gr = rows[:, :, None].expand(k, m, m)  # row index varies along local-i
    gc = cols[:, None, :].expand(k, m, m)  # col index varies along local-j
    lin = (gr * n + gc).reshape(-1)  # [K*M*M] int64

    # Flatten block element+local axes: [*batch_b, H, K*M*M].
    lead_b = blocks.shape[:-3]
    flat_blocks = blocks.reshape(*lead_b, k * m * m)

    if isinstance(y, PatternAccumulator):
        target_lead = torch.broadcast_shapes(y.values.shape[:-1], lead_b)
        nnz = y.values.shape[-1]
        acc = y.values.broadcast_to(*target_lead, nnz).clone()
        acc.index_add_(
            -1, y.positions(lin), flat_blocks.broadcast_to(*target_lead, k * m * m)
        )
        return y.with_values(acc)

    # Broadcast accumulator and blocks to a common leading shape, then flatten the
    # NxN tail to a single N*N axis for index_add_ along that axis.
    target_lead = torch.broadcast_shapes(y.shape[:-2], lead_b)
    y_full = y.broadcast_to(*target_lead, n, n).reshape(*target_lead, n * n).clone()
    flat_blocks = flat_blocks.broadcast_to(*target_lead, k * m * m)

    y_full.index_add_(-1, lin, flat_blocks)
    return y_full.reshape(*target_lead, n, n)


__all__ = ["PatternAccumulator", "scatter_blocks_into"]
