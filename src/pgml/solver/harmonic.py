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

from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# factor-once-solve-many (LU reuse when Y is constant across many RHS)
# ---------------------------------------------------------------------------
@dataclass
class FactoredSystem:
    """An LU factorization of the per-frequency system, reusable across many RHS.

    ``Y`` is constant across the current-injection fixed-point iterations (it is the
    network admittance — const-P/ZIP loads live on the RHS as ``I_device(V)``) AND across
    a scenario batch (which varies injections, not the network), so factoring once and
    back-substituting is far cheaper than re-factoring every solve. For ideal slack the
    factored matrix is the free-row block ``Y_ff`` and ``y_fs`` is kept for the RHS
    correction. Differentiable through ``torch.linalg.lu_factor`` / ``lu_solve`` (so it is
    safe in the differentiable per-harmonic path as well as the ``no_grad`` fixed point).
    """

    mode: str  # "norton" | "ideal"
    lu: Tensor
    piv: Tensor
    n: int
    free_rows: Optional[Tensor] = None
    fixed_rows: Optional[Tensor] = None
    y_fs: Optional[Tensor] = None  # [*, F, S] for the ideal-slack RHS correction


def lu_factor_system(
    y_bus: Tensor, *, fixed_rows: Optional[Tensor] = None
) -> FactoredSystem:
    """Factor ``Y`` (Norton) or the free block ``Y_ff`` (ideal slack) for repeated solves.

    Pair with :func:`solve_factored`, which back-substitutes a new RHS against the cached
    factorization. ``y_bus`` is ``[*batch, H, N, N]`` (or ``[N, N]``).
    """
    n = y_bus.shape[-1]
    if fixed_rows is None:
        lu, piv = torch.linalg.lu_factor(y_bus)
        return FactoredSystem("norton", lu, piv, n)
    fixed_rows = fixed_rows.to(device=y_bus.device, dtype=torch.int64)
    all_rows = torch.arange(n, device=y_bus.device)
    mask = torch.ones(n, dtype=torch.bool, device=y_bus.device).index_fill(
        0, fixed_rows, False
    )
    free_rows = all_rows[mask]
    y_ff = _index_2d(y_bus, free_rows, free_rows)
    y_fs = _index_2d(y_bus, free_rows, fixed_rows)
    lu, piv = torch.linalg.lu_factor(y_ff)
    return FactoredSystem("ideal", lu, piv, n, free_rows, fixed_rows, y_fs)


def _lu_solve_shared(lu: Tensor, piv: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``LU x = rhs`` reusing ONE factorization across a whole scenario batch.

    ``lu`` / ``piv`` carry the factorization's own batch ``*fb`` (``lu`` is
    ``[*fb, m, m]``, ``piv`` is ``[*fb, m]``); ``rhs`` is ``[*scenario, *fb, m]`` where
    the leading ``*scenario`` dims index independent right-hand sides that SHARE the
    factorization (the network is constant across the batch — only the injections vary).
    Those scenario dims are folded into the trailing multiple-RHS axis of
    :func:`torch.linalg.lu_solve`, so the factorization is solved against all
    ``prod(scenario)`` columns at once and is NEVER broadcast/replicated across the batch.
    Memory is ``O(prod(fb)*m^2 + prod(batch)*m)`` instead of the ``O(prod(batch)*m^2)`` a
    per-scenario LU broadcast would cost — the difference between fitting and OOMing for a
    large batch (a ``[B, H, N, N]`` LU tile dwarfs the ``[B, H, N]`` solution). Fully
    differentiable; returns ``[*scenario, *fb, m]`` (same shape ``solve`` would give).
    """
    m = lu.shape[-1]
    fb = tuple(lu.shape[:-2])
    nfb = len(fb)
    fb_numel = 1
    for sz in fb:
        fb_numel *= sz
    batch = torch.broadcast_shapes(fb, rhs.shape[:-1])  # full leading batch
    rhs_b = rhs.broadcast_to(*batch, m)  # [*batch, m]

    if fb_numel == 1:
        # Exactly one factorization (no per-harmonic axis, or a singleton one): EVERY
        # leading dim is just another right-hand side. Collapse them into the columns.
        k = 1
        for sz in batch:
            k *= sz
        lu_k = lu.reshape(m, m)
        piv_k = piv.reshape(m)
        cols = rhs_b.reshape(k, m).transpose(0, 1).contiguous()  # [m, k]
        sol = torch.linalg.lu_solve(lu_k, piv_k, cols)  # [m, k]
        return sol.transpose(0, 1).reshape(*batch, m)

    # Distinct factorizations along the trailing ``nfb`` dims of ``batch`` (== ``fb``);
    # the leading dims are the scenario batch -> fold them into the column axis.
    nb = len(batch)
    n_sb = nb - nfb
    sb = batch[:n_sb]
    k = 1
    for sz in sb:
        k *= sz
    perm = list(range(n_sb, nb)) + [nb] + list(range(n_sb))  # [*fb, m, *sb]
    cols = rhs_b.permute(*perm).reshape(
        *fb, m, k
    )  # [*fb, m, K] (contiguous after reshape)
    sol = torch.linalg.lu_solve(lu, piv, cols)  # [*fb, m, K]
    sol = sol.reshape(*fb, m, *sb)
    inv = list(range(nfb + 1, nfb + 1 + n_sb)) + list(range(nfb)) + [nfb]
    return sol.permute(*inv)  # [*scenario, *fb, m]


def solve_factored(
    fac: FactoredSystem, i_inj: Tensor, *, v_fixed: Optional[Tensor] = None
) -> Tensor:
    """Solve ``Y V = I`` for a new RHS against a cached factorization (see
    :func:`lu_factor_system`). Identical result to :func:`solve_harmonic` with the same
    ``Y`` / slack mode; only the factorization is reused. Returns ``[*batch, N]`` (the
    leading dims broadcast ``i_inj`` against the factorization). The scenario batch is
    solved as MULTIPLE right-hand sides of the one shared factorization
    (:func:`_lu_solve_shared`), so the dense ``Y`` is never tiled across the batch."""
    n = fac.n
    if fac.mode == "norton":
        return _lu_solve_shared(fac.lu, fac.piv, i_inj)

    if v_fixed is None:
        raise InputError("Ideal-slack factored solve requires `v_fixed`.")
    free_rows, fixed_rows, y_fs = fac.free_rows, fac.fixed_rows, fac.y_fs
    f, s = free_rows.shape[0], fixed_rows.shape[0]
    vf = v_fixed.to(dtype=fac.lu.dtype, device=fac.lu.device)
    i_free = i_inj.index_select(-1, free_rows)  # [*ib, F]
    # Build the corrected RHS ``I_free - Y_fs v_fixed`` (cheap: ``S`` slack rows), then
    # back-substitute it against the shared free-block factorization as multiple RHS.
    batch = torch.broadcast_shapes(
        fac.lu.shape[:-2], i_free.shape[:-1], y_fs.shape[:-2], vf.shape[:-1]
    )
    y_fs_b = y_fs.broadcast_to(*batch, f, s)
    vf_b = vf.broadcast_to(*batch, s)
    rhs = i_free.broadcast_to(*batch, f) - torch.matmul(
        y_fs_b, vf_b.unsqueeze(-1)
    ).squeeze(-1)  # [*batch, F]
    v_free = _lu_solve_shared(fac.lu, fac.piv, rhs)  # [*batch, F]
    v_full = torch.zeros(*batch, n, dtype=fac.lu.dtype, device=fac.lu.device)
    v_full = v_full.scatter(-1, free_rows.expand(*batch, f), v_free)
    v_full = v_full.scatter(-1, fixed_rows.expand(*batch, s), vf_b)
    return v_full


__all__ = ["solve_harmonic", "lu_factor_system", "solve_factored", "FactoredSystem"]
