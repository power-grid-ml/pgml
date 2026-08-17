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
"""

from __future__ import annotations

import torch
from torch import Tensor


def scatter_blocks_into(
    y: Tensor,
    blocks: Tensor,
    rows: Tensor,
    cols: Tensor,
) -> Tensor:
    """Return ``y`` with ``blocks`` accumulated at the (rows, cols) targets.

    Parameters
    ----------
    y:
        Complex accumulator ``[*batch, H, N, N]``.
    blocks:
        Complex primitive blocks ``[*batch_b, H, K, M, M]`` whose leading
        (batch, H) dims broadcast against ``y``'s leading dims.
    rows, cols:
        int64 tensors ``[K, M]`` mapping each block's local row/col axis to a
        global Y index. The (i, j) entry of element k scatters to
        ``(rows[k, i], cols[k, j])``.

    Returns
    -------
    Tensor
        A new accumulator (not the input ``y``) with the blocks added.
    """
    if blocks.shape[-3] == 0:
        # No elements of this kind; nothing to scatter.
        return y

    k = blocks.shape[-3]
    m = blocks.shape[-1]
    n = y.shape[-1]

    # Linear (row*N + col) index for every (k, i, j) local position -> [K, M, M].
    gr = rows[:, :, None].expand(k, m, m)  # row index varies along local-i
    gc = cols[:, None, :].expand(k, m, m)  # col index varies along local-j
    lin = (gr * n + gc).reshape(-1)  # [K*M*M] int64

    # Flatten block element+local axes: [*batch_b, H, K*M*M].
    lead_b = blocks.shape[:-3]
    flat_blocks = blocks.reshape(*lead_b, k * m * m)

    # Broadcast accumulator and blocks to a common leading shape, then flatten the
    # NxN tail to a single N*N axis for index_add_ along that axis.
    target_lead = torch.broadcast_shapes(y.shape[:-2], lead_b)
    y_full = y.broadcast_to(*target_lead, n, n).reshape(*target_lead, n * n).clone()
    flat_blocks = flat_blocks.broadcast_to(*target_lead, k * m * m)

    y_full.index_add_(-1, lin, flat_blocks)
    return y_full.reshape(*target_lead, n, n)


__all__ = ["scatter_blocks_into"]
