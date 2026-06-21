"""Complex batched linear solve of the per-frequency nodal system Y(f) V(f) = I(f).

Public API
----------
- ``solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v``

Two slack / reference modes, both differentiable:

1. **Norton (default, ``fixed_rows=None``)**: sources are already stamped as a
   shunt ``Y_s`` plus a Norton current ``I_s`` by ``assembly/``, so ``Y`` is
   non-singular and ``v = torch.linalg.solve(Y, I)``. The slack voltage equals
   ``u_ref`` only up to the drop across ``Z_s`` (matches OpenDSS Vsource).

2. **Ideal slack (``fixed_rows`` + ``v_fixed``)**: hold
   ``v[..., fixed_rows] = v_fixed`` exactly via a partitioned (Schur) solve
   ``v_free = Y_ff^-1 (I_free - Y_fs v_fixed)`` and reassemble the full ``v`` with
   gather/scatter (no in-place on tracked tensors). Matches pandapower / pgm.

Differentiability + GPU (CLAUDE.md): gradients flow w.r.t. ``Y``, ``I`` and
``v_fixed``. No ``.item()/.detach()/.numpy()``, no in-place op on tracked tensors,
no python control flow on tensor values, no hard-coded device. Runs unchanged on
CPU and CUDA; honors the input complex dtype (complex128 for gradcheck).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from pgml.errors import InputError


def solve_harmonic(
    y_bus: Tensor,
    i_inj: Tensor,
    *,
    fixed_rows: Optional[Tensor] = None,
    v_fixed: Optional[Tensor] = None,
) -> Tensor:
    """Solve ``Y V = I`` per frequency, batched, complex, differentiable.

    Parameters
    ----------
    y_bus:
        Complex tensor ``[*batch, H, N, N]`` (or ``[N, N]`` unbatched).
    i_inj:
        Complex tensor ``[*batch, H, N]`` (or ``[N]`` unbatched). Broadcasts
        against ``y_bus`` over all leading/batch dims and over ``H``.
    fixed_rows:
        Optional int64 1-D tensor of row indices to hold fixed (ideal slack). When
        given, ``v_fixed`` must also be given. ``None`` -> Norton mode.
    v_fixed:
        Complex tensor of the fixed voltages, broadcastable to
        ``[*batch, H, len(fixed_rows)]`` (e.g. ``[len(fixed_rows)]`` for a constant
        slack across all frequencies/batches).

    Returns
    -------
    Tensor
        Complex node voltages ``[*batch, H, N]`` (``[N]`` if both inputs were
        unbatched 2-D/1-D).
    """
    unbatched = y_bus.ndim == 2
    if unbatched:
        y = y_bus.unsqueeze(0)  # [1, N, N]
        i = i_inj.reshape(1, -1)  # [1, N]
    else:
        y = y_bus
        i = i_inj

    if fixed_rows is None:
        v = _solve_norton(y, i)
    else:
        if v_fixed is None:
            raise InputError(
                "Ideal-slack mode requires `v_fixed` when `fixed_rows` is given."
            )
        v = _solve_ideal_slack(y, i, fixed_rows, v_fixed)

    if unbatched:
        v = v.reshape(-1)
    return v


def _solve_norton(y: Tensor, i: Tensor) -> Tensor:
    """Dense solve ``v = Y^-1 I`` broadcasting over leading dims and H.

    ``torch.linalg.solve`` wants the RHS as a column; we expand to ``[..., N, 1]``
    and squeeze back to ``[..., N]``. Broadcasting between ``y`` and ``i`` is done
    explicitly so a 1-D-per-frequency RHS lines up with a batched matrix.
    """
    # Broadcast leading (all but last 2 of y) against i's leading (all but last).
    batch = torch.broadcast_shapes(y.shape[:-2], i.shape[:-1])
    n = y.shape[-1]
    y_b = y.broadcast_to(*batch, n, n)
    i_b = i.broadcast_to(*batch, n)
    v = torch.linalg.solve(y_b, i_b.unsqueeze(-1)).squeeze(-1)
    return v


def _solve_ideal_slack(
    y: Tensor, i: Tensor, fixed_rows: Tensor, v_fixed: Tensor
) -> Tensor:
    """Partitioned (Schur) solve holding ``v[..., fixed_rows] = v_fixed`` exactly.

    Free rows are the complement of ``fixed_rows``. We solve
    ``Y_ff v_free = I_free - Y_fs v_fixed`` then scatter ``v_free`` and ``v_fixed``
    back into the full vector. All gathers use ``index_select`` (autograd-safe);
    the reassembly uses an out-of-place scatter so no tracked tensor is mutated
    in place.
    """
    n = y.shape[-1]
    fixed_rows = fixed_rows.to(device=y.device, dtype=torch.int64)
    # Free-row complement via a boolean mask (index math, not on tensor values of
    # the differentiable path).
    all_rows = torch.arange(n, device=y.device)
    mask = torch.ones(n, dtype=torch.bool, device=y.device)
    mask = mask.index_fill(0, fixed_rows, False)
    free_rows = all_rows[mask]

    batch = torch.broadcast_shapes(y.shape[:-2], i.shape[:-1])
    y_b = y.broadcast_to(*batch, n, n)
    i_b = i.broadcast_to(*batch, n)

    # Partition Y and I.
    y_ff = _index_2d(y_b, free_rows, free_rows)  # [..., F, F]
    y_fs = _index_2d(y_b, free_rows, fixed_rows)  # [..., F, S]
    i_free = i_b.index_select(-1, free_rows)  # [..., F]

    # Broadcast v_fixed to [..., S].
    s = fixed_rows.shape[0]
    vf = v_fixed.to(dtype=y.dtype, device=y.device)
    vf_b = vf.broadcast_to(*batch, s)

    rhs = i_free - torch.matmul(y_fs, vf_b.unsqueeze(-1)).squeeze(-1)  # [..., F]
    v_free = torch.linalg.solve(y_ff, rhs.unsqueeze(-1)).squeeze(-1)  # [..., F]

    # Reassemble: scatter v_free at free_rows and vf_b at fixed_rows.
    v_full = torch.zeros(*batch, n, dtype=y.dtype, device=y.device)
    free_idx = free_rows.expand(*batch, free_rows.shape[0])
    fixed_idx = fixed_rows.expand(*batch, s)
    v_full = v_full.scatter(-1, free_idx, v_free)
    v_full = v_full.scatter(-1, fixed_idx, vf_b)
    return v_full


def _index_2d(y: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
    """Select a sub-block ``y[..., rows, cols]`` (outer product of index sets)."""
    return y.index_select(-2, rows).index_select(-1, cols)


__all__ = ["solve_harmonic"]
