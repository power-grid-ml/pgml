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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from pgml.errors import ComputationError, InputError


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


def solve_anchored(
    y_bus: Tensor,
    i_inj: Tensor,
    *,
    row_weight: Optional[Tensor] = None,
    row_target: Optional[Tensor] = None,
    op: Optional[Tensor] = None,
    op_weight: Optional[Tensor] = None,
    op_target: Optional[Tensor] = None,
    fixed_rows: Optional[Tensor] = None,
    v_fixed: Optional[Tensor] = None,
) -> Tensor:
    r"""Measurement-anchored (over-determined) network solve, batched and differentiable.

    Solves, per right-hand side, the weighted least squares

    .. math::
        \min_V \; \lVert Y V - I \rVert^2
            + \sum_r w^{\text{row}}_r \, \lvert V_r - t^{\text{row}}_r \rvert^2
            + \sum_k w^{\text{op}}_k \, \lvert (\mathrm{op}\,V)_k - t^{\text{op}}_k \rvert^2

    subject to ``V[fixed_rows] = v_fixed`` (optional hard Dirichlet slack). The primary term is
    the ordinary nodal law ``Y V = I``; the two anchor terms softly pull node values
    (``row_*`` — e.g. measured bus voltages) and a linear functional of the state (``op_*`` —
    e.g. ``op`` = the branch-current map, anchoring measured branch currents) toward
    measurements. With no anchors this is exactly :func:`solve_harmonic` (Norton, or ideal
    slack when ``fixed_rows`` is given).

    The solve is a REDUCED-correction solve that inverts the (shared) operator ``Y`` ONCE for
    the whole batch: with ``V = V_0 + Y^{-1} r`` and ``V_0 = Y^{-1} I`` the physics term
    becomes ``\lVert r \rVert^2`` and the anchors a correction system in ``r`` whose matrix
    ``G = I + \sum w\,(A Y^{-1})^H (A Y^{-1})`` is Hermitian positive-definite with
    eigenvalues ``\ge 1``, Cholesky-factored per right-hand side. The identity floor keeps
    that factorization stable, and the physics block never passes through normal equations
    (no ``\kappa(Y)^2`` squaring there) — but ``\kappa(G)`` itself grows with
    ``w \cdot \sigma_{\max}(Y^{-1})^2``, so callers should scale anchor weights relative to
    ``Y`` (e.g. by a typical singular value, as the pgl consumer does). Anchor weights are
    cast to the real dtype paired with ``y_bus``'s complex dtype (complex64 and complex128
    both supported).

    Parameters
    ----------
    y_bus:
        Complex ``[N, N]`` — the SHARED network operator (one system; loop externally over
        harmonic orders / topologies, each with its own anchors).
    i_inj:
        Complex ``[*batch, N]`` (or ``[N]``) right-hand side.
    row_weight, row_target:
        ``[*batch, N]`` real weights ``\ge 0`` (``0`` = row not anchored) and ``[*batch, N]``
        complex targets for the node anchors. ``row_target`` defaults to ``0``.
    op, op_weight, op_target:
        ``op`` is a complex ``[K, N]`` linear operator (grid-constant, e.g. the branch-current
        map); ``op_weight`` ``[*batch, K]`` real weights and ``op_target`` ``[*batch, K]``
        complex targets anchor ``op·V`` toward the target. All three must be given together.
    fixed_rows, v_fixed:
        Optional hard Dirichlet slack, identical contract to :func:`solve_harmonic`.

    Returns
    -------
    Tensor
        Complex node voltages ``[*batch, N]`` (``[N]`` if ``i_inj`` was 1-D).
    """
    if y_bus.ndim != 2:
        raise InputError(
            "solve_anchored takes a single shared operator y_bus [N, N] (the reduced solve "
            "reuses one factorization); loop externally over any harmonic/topology axis."
        )
    has_row = row_weight is not None
    has_op = op is not None and op_weight is not None
    if not has_row and not has_op:
        # no anchors -> the plain solve; add an explicit system axis so a batched RHS against
        # the single [N, N] operator broadcasts correctly (solve_harmonic's 2-D path folds a
        # batched RHS into one vector otherwise).
        v = solve_harmonic(
            y_bus.unsqueeze(0), i_inj, fixed_rows=fixed_rows, v_fixed=v_fixed
        )
        return v.squeeze(0) if i_inj.ndim == 1 else v

    dtype, dev, n = y_bus.dtype, y_bus.device, y_bus.shape[-1]
    rdt = y_bus.real.dtype if y_bus.is_complex() else y_bus.dtype
    unbatched = i_inj.ndim == 1
    i = i_inj.reshape(1, -1) if unbatched else i_inj  # [*b, N]

    # free / fixed partition (slack held exactly, anchored softly among the free rows)
    if fixed_rows is not None:
        fixed_rows = fixed_rows.to(device=dev, dtype=torch.int64)
        keep = torch.ones(n, dtype=torch.bool, device=dev).index_fill(
            0, fixed_rows, False
        )
        free = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    else:
        free = torch.arange(n, device=dev)
    f = int(free.numel())

    # Z = Y_ff^{-1} — the single shared factorization the whole batch reuses.
    y_ff = _index_2d(y_bus, free, free)  # [F, F]
    z = torch.linalg.solve(y_ff, torch.eye(f, dtype=dtype, device=dev))  # [F, F]

    i_free = i.index_select(-1, free)  # [*b, F]
    if fixed_rows is not None:
        y_fs = _index_2d(y_bus, free, fixed_rows)  # [F, S]
        vf = v_fixed.to(dtype=dtype, device=dev)
        vf = vf.broadcast_to(*i_free.shape[:-1], fixed_rows.numel())  # [*b, S]
        rhs_phys = i_free - torch.matmul(y_fs, vf.unsqueeze(-1)).squeeze(-1)
    else:
        rhs_phys = i_free
    v0 = torch.matmul(z, rhs_phys.unsqueeze(-1)).squeeze(-1)  # [*b, F] = Y_ff^{-1} rhs

    lead = rhs_phys.shape[:-1]
    zh = z.conj().mT  # [F, F]
    g = torch.eye(f, dtype=dtype, device=dev).expand(*lead, f, f).clone()
    rhs_r = torch.zeros(*lead, f, dtype=dtype, device=dev)

    if has_row:
        wr = row_weight.index_select(-1, free).to(rdt)  # [*b, F]
        wz = wr.unsqueeze(-1) * z  # diag(w_row) Z  [*b, F, F]
        g = g + torch.matmul(zh, wz)  # Z^H diag(w_row) Z
        tgt = (
            row_target.index_select(-1, free)
            if row_target is not None
            else torch.zeros((), dtype=dtype, device=dev)
        )
        dv = wr * (v0 - tgt)  # [*b, F]
        rhs_r = rhs_r - torch.matmul(zh, dv.unsqueeze(-1)).squeeze(-1)

    if has_op:
        op_free = op.index_select(-1, free)  # [K, F]
        oz = torch.matmul(op_free, z)  # op_free Y_ff^{-1}  [K, F]
        ozh = oz.conj().mT  # [F, K]
        wo = op_weight.to(rdt)  # [*b, K]
        woz = wo.unsqueeze(-1) * oz  # [*b, K, F]
        g = g + torch.matmul(ozh, woz)
        op_v0 = torch.matmul(op_free, v0.unsqueeze(-1)).squeeze(-1)  # [*b, K]
        ot = op_target if op_target is not None else 0.0
        if fixed_rows is not None:
            op_fx = op.index_select(-1, fixed_rows)  # [K, S]
            ot = ot - torch.matmul(op_fx, vf.unsqueeze(-1)).squeeze(-1)  # [*b, K]
        di = wo * (op_v0 - ot)  # [*b, K]
        rhs_r = rhs_r - torch.matmul(ozh, di.unsqueeze(-1)).squeeze(-1)

    # G is Hermitian PD by construction (eigenvalues >= 1), so a batched Cholesky is the
    # cheapest stable factorization for the per-right-hand-side correction system.
    r = torch.cholesky_solve(rhs_r.unsqueeze(-1), torch.linalg.cholesky(g)).squeeze(-1)
    v_free = v0 + torch.matmul(z, r.unsqueeze(-1)).squeeze(-1)  # [*b, F]

    v_full = torch.zeros(*lead, n, dtype=dtype, device=dev)
    v_full = v_full.scatter(-1, free.expand(*lead, f), v_free)
    if fixed_rows is not None:
        v_full = v_full.scatter(-1, fixed_rows.expand(*lead, fixed_rows.numel()), vf)
    return v_full.reshape(-1) if unbatched else v_full


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
# A power-grid ``Y`` has O(N) nonzeros (each branch stamps a fixed-size block), so a
# sparse direct factorization scales ~O(N) on radial/meshed feeders where the dense
# LU is O(N^3). Below this row count the dense factorization's lower constant wins;
# from ~600 rows the sparse backend wins every metric on CPU (measured with
# ``run/examples/pgml/benchmark_sparse.py`` on an i7-12700: at 600 rows factor 3.7x /
# single-RHS 7x faster, batched back-substitution equal; at 4800 rows factor >4x,
# single-RHS 50x, 256-RHS 4x, end-to-end nonlinear solve 3x). CUDA stays dense:
# torch has no batched sparse direct solve, and dense batched LU is what GPUs are
# built for — revisit only against a measured cuDSS/cuSOLVER baseline.
_SPARSE_MIN_ROWS = 512


# SuperLU's back-substitution releases the GIL and is deterministic under
# concurrent solves against one factorization (each call owns its output/work
# arrays), so a large multi-RHS batch is split across threads — measured 4.5x
# with 8 workers on a 4800-row / 256-RHS system, the large-N fixed-point
# bottleneck. Chunked solves differ from the single multi-RHS call only at
# machine epsilon (a different internal blocking), like any BLAS reordering;
# for a fixed column count the chunk layout — and thus the result — is
# deterministic.
_SPARSE_SOLVE_MAX_THREADS = max(1, min(8, os.cpu_count() or 1))
_SPARSE_SOLVE_MIN_WORK = 100_000  # m * k below this solves sequentially

_sparse_pool: Optional[ThreadPoolExecutor] = None


def _sparse_executor() -> ThreadPoolExecutor:
    global _sparse_pool
    if _sparse_pool is None:
        _sparse_pool = ThreadPoolExecutor(
            max_workers=_SPARSE_SOLVE_MAX_THREADS, thread_name_prefix="pgml-sparse"
        )
    return _sparse_pool


class _SciPySparseLU:
    """SuperLU factorizations of ``[*fb, m, m]`` (CPU; one factorization per fb index).

    The sparse counterpart of ``torch.linalg.lu_factor`` for :class:`FactoredSystem`:
    factors each leading-batch matrix once (COLAMD ordering, scipy ``splu``) and
    back-substitutes arbitrarily-batched right-hand sides with the SAME
    scenario-dims-as-columns folding as :func:`_lu_solve_shared`. ``trans="H"``
    solves with the conjugate-transposed factors — the adjoint of the linear solve —
    so the backward of :class:`_SparseSolveFn` needs no second factorization.
    """

    def __init__(self, y: Tensor) -> None:
        import scipy.sparse as sp

        if y.device.type != "cpu":
            raise InputError(
                "The sparse solver backend runs on CPU only (scipy SuperLU); "
                'move the system to CPU or use backend="dense" on CUDA.'
            )
        self.fb = tuple(y.shape[:-2])
        self.m = int(y.shape[-1])
        self.dtype = y.dtype
        self.device = y.device
        mats = y.detach().reshape(-1, self.m, self.m)
        try:
            self.lus = [sp.linalg.splu(sp.csc_matrix(mat.numpy())) for mat in mats]
        except RuntimeError as e:
            raise ComputationError(
                f"Sparse factorization failed: {e}. A singular Y usually means "
                "disconnected (node, phase) rows — run pgml.solver.check_connectivity "
                "on the grid, or fix zero-impedance / degenerate branch parameters."
            ) from e

    @staticmethod
    def _solve_cols(lu, cols, trans: str):
        """Back-substitute ``cols`` ``[m, k]``, splitting large ``k`` across threads."""
        import numpy as np

        m, k = cols.shape
        n_threads = _SPARSE_SOLVE_MAX_THREADS
        if n_threads <= 1 or k < 4 * n_threads or m * k < _SPARSE_SOLVE_MIN_WORK:
            return lu.solve(cols, trans=trans)
        chunks = np.array_split(np.arange(k), min(n_threads, (k + 7) // 8))
        parts = list(
            _sparse_executor().map(
                lambda c: lu.solve(np.ascontiguousarray(cols[:, c]), trans=trans),
                chunks,
            )
        )
        return np.concatenate(parts, axis=1)

    def solve(self, rhs: Tensor, trans: str = "N") -> Tensor:
        """Solve against every RHS in ``rhs`` ``[*scenario, *fb, m]`` -> same shape.

        Identical broadcasting contract to :func:`_lu_solve_shared`: the trailing
        ``fb`` dims of the broadcast batch select the factorization, every leading
        scenario dim becomes an extra RHS column of that factorization.
        """
        import numpy as np

        m = self.m
        fb = self.fb
        nfb = len(fb)
        fb_numel = 1
        for sz in fb:
            fb_numel *= sz
        batch = torch.broadcast_shapes(fb, rhs.shape[:-1])
        rhs_b = rhs.detach().to(dtype=self.dtype).broadcast_to(*batch, m)

        if fb_numel == 1:
            k = 1
            for sz in batch:
                k *= sz
            cols = rhs_b.reshape(k, m).transpose(0, 1).contiguous().numpy()  # [m, k]
            sol = self._solve_cols(self.lus[0], np.ascontiguousarray(cols), trans)
            out = torch.from_numpy(np.ascontiguousarray(sol))
            return out.transpose(0, 1).reshape(*batch, m).to(self.dtype)

        nb = len(batch)
        n_sb = nb - nfb
        sb = batch[:n_sb]
        k = 1
        for sz in sb:
            k *= sz
        perm = list(range(n_sb, nb)) + [nb] + list(range(n_sb))  # [*fb, m, *sb]
        cols = rhs_b.permute(*perm).reshape(fb_numel, m, k).contiguous().numpy()
        if (
            _SPARSE_SOLVE_MAX_THREADS > 1
            and fb_numel > 1
            and fb_numel * m * k >= _SPARSE_SOLVE_MIN_WORK
        ):
            # Independent factorizations (per frequency / per scenario topology):
            # solve them concurrently, one factorization per task.
            sols = list(
                _sparse_executor().map(
                    lambda i: self.lus[i].solve(
                        np.ascontiguousarray(cols[i]), trans=trans
                    ),
                    range(fb_numel),
                )
            )
        else:
            sols = [
                self.lus[i].solve(np.ascontiguousarray(cols[i]), trans=trans)
                for i in range(fb_numel)
            ]
        sol = torch.from_numpy(np.ascontiguousarray(np.stack(sols, 0)))
        sol = sol.reshape(*fb, m, *sb).to(self.dtype)
        inv = list(range(nfb + 1, nfb + 1 + n_sb)) + list(range(nfb)) + [nfb]
        return sol.permute(*inv)  # [*scenario, *fb, m]


def _sum_to_shape(t: Tensor, shape: tuple) -> Tensor:
    """Sum-reduce broadcast dims of ``t`` down to ``shape`` (autograd convention)."""
    while t.ndim > len(shape):
        t = t.sum(0)
    for d, sz in enumerate(shape):
        if t.shape[d] != sz:  # broadcast singleton
            t = t.sum(d, keepdim=True)
    return t


class _SparseSolveFn(torch.autograd.Function):
    """``V = Y⁻¹ RHS`` through a sparse factorization, with the linear-solve adjoint.

    Forward back-substitutes against the prebuilt :class:`_SciPySparseLU`.
    Backward is the standard adjoint of a linear solve — one solve with the
    conjugate-transposed factors, no re-factorization::

        λ = Y⁻ᴴ grad_V            (grad_RHS = λ)
        grad_Y[i, j] = -λ[i] · conj(V[j])   (summed over the scenario batch)

    ``grad_Y`` is accumulated per factorization with a matmul over the folded
    scenario axis (never materialising a per-scenario ``[m, m]`` outer product), so
    the backward memory is ``O(prod(fb)·m²)`` — the size of ``Y`` itself.
    """

    @staticmethod
    def forward(ctx, y: Tensor, rhs: Tensor, handle: _SciPySparseLU) -> Tensor:
        v = handle.solve(rhs)
        ctx.handle = handle
        ctx.save_for_backward(v)
        ctx.rhs_shape = tuple(rhs.shape)
        ctx.y_shape = tuple(y.shape)
        return v

    @staticmethod
    def backward(ctx, grad_v: Tensor):
        (v,) = ctx.saved_tensors
        handle = ctx.handle
        m = handle.m
        fb = handle.fb
        nfb = len(fb)
        lam = handle.solve(grad_v, trans="H")  # [*batch, m]
        grad_rhs = (
            _sum_to_shape(lam, ctx.rhs_shape) if ctx.needs_input_grad[1] else None
        )
        grad_y = None
        if ctx.needs_input_grad[0]:
            batch = torch.broadcast_shapes(fb, v.shape[:-1], lam.shape[:-1])
            fb_numel = 1
            for sz in fb:
                fb_numel *= sz
            nb = len(batch)
            n_sb = nb - nfb
            perm = list(range(n_sb, nb)) + list(range(n_sb)) + [nb]  # [*fb, *sb, m]
            lam_g = lam.broadcast_to(*batch, m).permute(*perm).reshape(fb_numel, -1, m)
            v_g = v.broadcast_to(*batch, m).permute(*perm).reshape(fb_numel, -1, m)
            gy = -torch.einsum("fki,fkj->fij", lam_g, v_g.conj())  # [fb_numel, m, m]
            grad_y = _sum_to_shape(gy.reshape(*fb, m, m), ctx.y_shape)
        return grad_y, grad_rhs, None


# ---------------------------------------------------------------------------
# block-diagonal factorization (an ensemble of independent grids in one system)
# ---------------------------------------------------------------------------
def _validate_block_rows(
    block_rows: Sequence[Tensor], n: int, device: torch.device
) -> list[Tensor]:
    """Normalize ``block_rows`` to int64 row-index tensors and check the partition.

    Every row of the ``n``-row system must appear in EXACTLY one block: the
    block-diagonal inverse is only the inverse of the whole system when the blocks
    are a disjoint, complete cover of the rows.
    """
    if block_rows is None:
        raise InputError(
            'backend="block" needs block_rows=[rows_of_block_0, ...]: the row '
            "indices of each independent diagonal block. For a merged ensemble "
            "use pgml.multigrid.MergedGrid.block_rows()."
        )
    blocks: list[Tensor] = []
    for k, rows in enumerate(block_rows):
        t = torch.as_tensor(rows, dtype=torch.int64, device=device).reshape(-1)
        if t.numel() == 0:
            raise InputError(
                f"block_rows[{k}] is empty; every block needs at least one row."
            )
        blocks.append(t)
    if not blocks:
        raise InputError("block_rows is empty; give at least one block.")
    flat = torch.cat(blocks)
    covers = flat.numel() == n and bool(
        torch.equal(flat.sort().values, torch.arange(n, device=device))
    )
    if not covers:
        raise InputError(
            f"block_rows must partition the {n} rows of the system exactly once: "
            f"{len(blocks)} block(s) covering {int(flat.numel())} index/indices in "
            f"[{int(flat.min())}, {int(flat.max())}] were given. A block-diagonal "
            "factorization is defined only for a disjoint, complete row partition."
        )
    return blocks


def _size_buckets(
    rows: Sequence[Tensor], positions: Sequence[Tensor]
) -> list[tuple[Tensor, Tensor]]:
    """Stack equal-size blocks into ``[B, n]`` index matrices (input order kept).

    One entry per DISTINCT block size (ascending): its ``rows`` matrix indexes the
    matrix, its ``positions`` matrix the solved vector, and ``B`` becomes the batch
    axis of that bucket's ``lu_factor`` / ``lu_solve``.
    """
    by_size: dict[int, tuple[list[Tensor], list[Tensor]]] = {}
    for r, p in zip(rows, positions):
        entry = by_size.setdefault(int(r.numel()), ([], []))
        entry[0].append(r)
        entry[1].append(p)
    return [
        (torch.stack(rs), torch.stack(ps)) for _, (rs, ps) in sorted(by_size.items())
    ]


class _BlockLU:
    """Batched dense LU of each diagonal block of a BLOCK-DIAGONAL system.

    A disjoint union of independent grids (:func:`pgml.multigrid.merge_grids`)
    assembles to ``Y = diag(Y_1, …, Y_G)``, whose inverse is the block-diagonal
    inverse of the members: factoring them separately costs ``O(Σ n_k³)`` where a
    dense LU of the union costs ``O((Σ n_k)³)``. Blocks of EQUAL size are stacked
    into one ``[*fb, B, n, n]`` tensor and factored by a SINGLE batched
    ``torch.linalg.lu_factor``, so the number of LU calls is the number of distinct
    member sizes — not the number of members — which is the shape a GPU wants.

    Only the per-bucket factors are stored; the ``[N, N]`` factor of the union is
    never formed. Pure torch (advanced indexing + ``lu_factor`` / ``lu_solve`` +
    ``scatter``), so gradients flow to the factored matrix and the right-hand side,
    device/dtype follow the input, and the ``lu_solve`` backward answers the adjoint
    system with the SAME factors (no re-factorization).

    ``blocks`` index the rows of ``y``; ``solve_blocks`` (default: ``blocks``) index
    the same entries in the vector space the right-hand sides live in. They differ
    on the ideal-slack path, where the blocks are gathered from the full admittance
    but the solved vector holds only the free rows.
    """

    def __init__(
        self,
        y: Tensor,
        blocks: Sequence[Tensor],
        *,
        solve_blocks: Optional[Sequence[Tensor]] = None,
        m: Optional[int] = None,
    ) -> None:
        self.fb = tuple(y.shape[:-2])
        self.m = int(y.shape[-1]) if m is None else int(m)
        self.device = y.device
        # An empty tensor carrying the factorization's leading batch / dtype /
        # device (the block factors carry an extra bucket axis, so they cannot
        # stand in for it).
        self.batch_ref = y.new_empty((*self.fb, 0, 0))
        self.buckets: list[tuple[Tensor, Tensor, Tensor]] = []
        pairs = _size_buckets(blocks, blocks if solve_blocks is None else solve_blocks)
        for rows, pos in pairs:
            # [*fb, B, n, n]: gathered straight from ``y`` — no [N, N] intermediate.
            sub = y[..., rows.unsqueeze(-1), rows.unsqueeze(-2)]
            lu, piv = torch.linalg.lu_factor(sub)
            self.buckets.append((pos, lu, piv))
        self.n_blocks = sum(int(pos.shape[0]) for pos, _, _ in self.buckets)

    def solve(self, rhs: Tensor) -> Tensor:
        """Back-substitute ``rhs`` ``[*batch, m]`` against the per-block factors.

        Each bucket gathers its blocks' entries out of the right-hand side, folds
        the whole scenario batch into the multiple-RHS axis of its factorization
        (:func:`_lu_solve_shared`, so no factor is tiled across the batch) and
        scatters the block solutions back into the full vector. Loops over BUCKETS
        (one iteration per distinct block size), never over blocks. Returns
        ``[*batch, m]`` with ``batch = broadcast(fb, rhs batch)``, the same shape
        the dense backend gives.
        """
        batch = torch.broadcast_shapes(self.fb, rhs.shape[:-1])
        rhs_b = rhs.broadcast_to(*batch, self.m)
        fb_numel = 1
        for sz in self.fb:
            fb_numel *= sz
        out = torch.zeros(*batch, self.m, dtype=rhs_b.dtype, device=self.device)
        for pos, lu, piv in self.buckets:
            nb, blk = int(pos.shape[0]), int(pos.shape[1])
            flat = pos.reshape(-1)
            cols = rhs_b.index_select(-1, flat).reshape(*batch, nb, blk)
            if fb_numel == 1:
                # One factorization per block: every leading dim of the RHS is
                # just another column of it.
                sol = _lu_solve_shared(
                    lu.reshape(nb, blk, blk), piv.reshape(nb, blk), cols
                )
            else:
                # Distinct factorizations per frequency / scenario topology: the
                # bucket axis extends the factorization's own batch.
                sol = _lu_solve_shared(lu, piv, cols)
            out = out.scatter(
                -1,
                flat.expand(*batch, nb * blk),
                sol.reshape(*batch, nb * blk),
            )
        return out

    @classmethod
    def for_free_rows(
        cls, y: Tensor, blocks: Sequence[Tensor], free_mask: Tensor, free_rows: Tensor
    ) -> "_BlockLU":
        """Factor the FREE-row sub-block of every block (ideal slack).

        A block's free rows are its own rows minus the fixed (slack) rows it holds,
        and the solved vector holds the free rows only — so each block's solve-space
        index is its position within ``free_rows``. Blocks that are entirely fixed
        contribute no equations and are dropped.
        """
        dev = y.device
        n_blocks = len(blocks)
        sizes = torch.tensor([int(b.numel()) for b in blocks], device=dev)
        owner = torch.repeat_interleave(torch.arange(n_blocks, device=dev), sizes)
        flat = torch.cat(blocks)
        keep = free_mask.index_select(0, flat)
        counts = torch.bincount(owner[keep], minlength=n_blocks).tolist()
        kept_rows = flat[keep]
        # Free-space position of every row (fixed rows are never looked up).
        pos = torch.zeros(int(free_mask.numel()), dtype=torch.int64, device=dev)
        pos = pos.index_copy(
            0, free_rows, torch.arange(int(free_rows.numel()), device=dev)
        )
        rows_split = torch.split(kept_rows, counts)
        pos_split = torch.split(pos.index_select(0, kept_rows), counts)
        free_blocks = [r for r in rows_split if r.numel()]
        free_pos = [p for p in pos_split if p.numel()]
        return cls(
            y,
            free_blocks,
            solve_blocks=free_pos,
            m=int(free_rows.numel()),
        )


@dataclass
class FactoredSystem:
    """A factorization of the per-frequency system, reusable across many RHS.

    ``Y`` is constant across the current-injection fixed-point iterations (it is the
    network admittance — const-P/ZIP loads live on the RHS as ``I_device(V)``) AND across
    a scenario batch (which varies injections, not the network), so factoring once and
    back-substituting is far cheaper than re-factoring every solve. For ideal slack the
    factored matrix is the free-row block ``Y_ff`` and ``y_fs`` is kept for the RHS
    correction.

    Three backends, selected in :func:`lu_factor_system`:

    - ``"dense"`` — batched ``torch.linalg.lu_factor`` / ``lu_solve`` (CPU + CUDA);
      differentiable through the torch ops.
    - ``"sparse"`` — scipy SuperLU (:class:`_SciPySparseLU`, CPU only), ~O(N) for the
      O(N)-nnz power-grid ``Y`` where dense LU is O(N³); differentiable through the
      adjoint :class:`_SparseSolveFn` (``y_mat`` carries the autograd graph of the
      factored matrix).
    - ``"block"`` — one batched dense LU per diagonal block of a BLOCK-DIAGONAL
      system (:class:`_BlockLU`, ``block`` holds the per-bucket factors), the CUDA
      path for an ensemble of independent grids: O(Σ n_k³) instead of O((Σ n_k)³).
    """

    mode: str  # "norton" | "ideal"
    lu: Optional[Tensor]
    piv: Optional[Tensor]
    n: int
    free_rows: Optional[Tensor] = None
    fixed_rows: Optional[Tensor] = None
    y_fs: Optional[Tensor] = None  # [*, F, S] for the ideal-slack RHS correction
    backend: str = "dense"  # "dense" | "sparse" | "block"
    sparse: Optional[_SciPySparseLU] = None
    y_mat: Optional[Tensor] = None  # sparse backend: the factored matrix (autograd)
    block: Optional[_BlockLU] = None  # block backend: the per-bucket factors

    @property
    def _fb_tensor(self) -> Tensor:
        """The tensor carrying the factorization's leading batch / dtype / device."""
        if self.backend == "dense":
            return self.lu
        if self.backend == "block":
            return self.block.batch_ref
        return self.y_mat


def _resolve_backend(
    backend: str, y_bus: Tensor, block_rows: Optional[Sequence[Tensor]] = None
) -> str:
    """Resolve ``"auto"`` by size and device (see ``_SPARSE_MIN_ROWS``).

    ``"block"`` is never auto-selected: it needs the caller's row partition and is
    only correct for a system that IS block diagonal, so it stays an explicit opt-in.
    """
    if backend not in ("auto", "dense", "sparse", "block"):
        raise InputError(
            f"Unsupported factorization backend {backend!r} "
            "(use 'auto'/'dense'/'sparse'/'block')."
        )
    if block_rows is not None and backend != "block":
        raise InputError(
            f"block_rows is used only by backend='block'; got {backend!r}. The "
            "block-diagonal factorization is an explicit opt-in — 'auto' never "
            "selects it."
        )
    if backend != "auto":
        return backend
    if y_bus.device.type == "cpu" and y_bus.shape[-1] >= _SPARSE_MIN_ROWS:
        return "sparse"
    return "dense"


def lu_factor_system(
    y_bus: Tensor,
    *,
    fixed_rows: Optional[Tensor] = None,
    backend: str = "auto",
    block_rows: Optional[Sequence[Tensor]] = None,
) -> FactoredSystem:
    """Factor ``Y`` (Norton) or the free block ``Y_ff`` (ideal slack) for repeated solves.

    Pair with :func:`solve_factored`, which back-substitutes a new RHS against the cached
    factorization. ``y_bus`` is ``[*batch, H, N, N]`` (or ``[N, N]``). ``backend``:
    ``"auto"`` (default) picks the scipy SuperLU sparse factorization on CPU systems of
    at least ``_SPARSE_MIN_ROWS`` rows (a power-grid ``Y`` has O(N) nonzeros, so sparse
    is ~O(N) where dense LU is O(N³)) and the batched dense ``torch.linalg.lu_factor``
    everywhere else (CUDA is ALWAYS dense — torch has no batched sparse direct solve);
    ``"dense"`` / ``"sparse"`` / ``"block"`` force the choice. Every backend is
    differentiable (dense and block through the torch ops, sparse through the adjoint
    :class:`_SparseSolveFn`).

    ``backend="block"`` factors a BLOCK-DIAGONAL system one diagonal block at a time
    and needs ``block_rows``: one int64 row-index tensor per block, together
    partitioning the ``N`` rows exactly once (``pgml.multigrid.MergedGrid.block_rows()``
    for a merged ensemble). Blocks of equal size share one batched LU, so an ensemble
    of ``G`` grids costs ``O(Σ n_k³)`` instead of the union's ``O((Σ n_k)³)`` and the
    factor holds ``O(Σ n_k²)`` numbers instead of ``O((Σ n_k)²)``. This is the CUDA
    path for a many-grid ensemble; on CPU the sparse union backend exploits the same
    structure (plus the sparsity WITHIN each block) and remains the better choice.
    The row partition is taken on trust — any admittance OUTSIDE the listed blocks is
    ignored by the factorization, so only pass blocks that are galvanically
    independent. ``"auto"`` never resolves to ``"block"``.
    """
    n = y_bus.shape[-1]
    resolved = _resolve_backend(backend, y_bus, block_rows)
    blocks = (
        _validate_block_rows(block_rows, n, y_bus.device) if resolved == "block" else []
    )
    if fixed_rows is None:
        if resolved == "sparse":
            return FactoredSystem(
                "norton",
                None,
                None,
                n,
                backend="sparse",
                sparse=_SciPySparseLU(y_bus),
                y_mat=y_bus,
            )
        if resolved == "block":
            return FactoredSystem(
                "norton", None, None, n, backend="block", block=_BlockLU(y_bus, blocks)
            )
        lu, piv = torch.linalg.lu_factor(y_bus)
        return FactoredSystem("norton", lu, piv, n)
    fixed_rows = fixed_rows.to(device=y_bus.device, dtype=torch.int64)
    all_rows = torch.arange(n, device=y_bus.device)
    mask = torch.ones(n, dtype=torch.bool, device=y_bus.device).index_fill(
        0, fixed_rows, False
    )
    free_rows = all_rows[mask]
    y_fs = _index_2d(y_bus, free_rows, fixed_rows)
    if resolved == "block":
        # The free-row sub-block of each block, gathered straight from ``y_bus``:
        # the dense free-free block [F, F] is never materialised.
        return FactoredSystem(
            "ideal",
            None,
            None,
            n,
            free_rows,
            fixed_rows,
            y_fs,
            backend="block",
            block=_BlockLU.for_free_rows(y_bus, blocks, mask, free_rows),
        )
    y_ff = _index_2d(y_bus, free_rows, free_rows)
    if resolved == "sparse":
        return FactoredSystem(
            "ideal",
            None,
            None,
            n,
            free_rows,
            fixed_rows,
            y_fs,
            backend="sparse",
            sparse=_SciPySparseLU(y_ff),
            y_mat=y_ff,
        )
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


def back_substitute(fac: FactoredSystem, rhs: Tensor) -> Tensor:
    """Back-substitute ``rhs`` against the factors, whichever backend holds them.

    The RHS lives in the factored matrix's own space: the FULL ``N`` rows in Norton
    mode, the FREE rows only in ideal-slack mode. ``rhs`` is ``[*scenario, *fb, m]``
    and the result has the broadcast shape (see :func:`_lu_solve_shared`). This is
    the single dispatch point every factored solve — plain
    (:func:`solve_factored`) or low-rank-updated
    (:func:`pgml.solver.lowrank.solve_factored_updated`) — goes through.
    """
    if fac.backend == "sparse":
        return _SparseSolveFn.apply(fac.y_mat, rhs, fac.sparse)
    if fac.backend == "block":
        return fac.block.solve(rhs)
    return _lu_solve_shared(fac.lu, fac.piv, rhs)


def ideal_slack_rhs(
    fac: FactoredSystem, i_inj: Tensor, v_fixed: Optional[Tensor]
) -> tuple[Tensor, Tensor]:
    """Free-row right-hand side ``I_free − Y_fs v_fixed`` and the broadcast ``v_fixed``.

    The ideal-slack half of a factored solve that does not depend on the factors:
    gather the free rows of the injection and subtract the slack coupling (cheap —
    ``S`` slack columns). Returns ``(rhs [*batch, F], v_fixed [*batch, S])`` with
    ``batch`` the broadcast of the factorization's own batch and every input's.
    """
    if v_fixed is None:
        raise InputError("Ideal-slack factored solve requires `v_fixed`.")
    sys_t = fac._fb_tensor
    free_rows, fixed_rows, y_fs = fac.free_rows, fac.fixed_rows, fac.y_fs
    f, s = free_rows.shape[0], fixed_rows.shape[0]
    vf = v_fixed.to(dtype=sys_t.dtype, device=sys_t.device)
    i_free = i_inj.index_select(-1, free_rows)  # [*ib, F]
    batch = torch.broadcast_shapes(
        sys_t.shape[:-2], i_free.shape[:-1], y_fs.shape[:-2], vf.shape[:-1]
    )
    y_fs_b = y_fs.broadcast_to(*batch, f, s)
    vf_b = vf.broadcast_to(*batch, s)
    rhs = i_free.broadcast_to(*batch, f) - torch.matmul(
        y_fs_b, vf_b.unsqueeze(-1)
    ).squeeze(-1)  # [*batch, F]
    return rhs, vf_b


def scatter_slack_solution(fac: FactoredSystem, v_free: Tensor, vf: Tensor) -> Tensor:
    """Reassemble the full ``[*batch, N]`` voltage from the free solution + slack.

    Out-of-place scatter (no in-place op on a tracked tensor); ``batch`` follows
    ``v_free`` so a correction that widened the batch (e.g. a per-state low-rank
    update over a single-scenario right-hand side) is carried through.
    """
    free_rows, fixed_rows = fac.free_rows, fac.fixed_rows
    f, s = free_rows.shape[0], fixed_rows.shape[0]
    batch = v_free.shape[:-1]
    vf_b = vf.broadcast_to(*batch, s)
    v_full = torch.zeros(*batch, fac.n, dtype=v_free.dtype, device=v_free.device)
    v_full = v_full.scatter(-1, free_rows.expand(*batch, f), v_free)
    return v_full.scatter(-1, fixed_rows.expand(*batch, s), vf_b)


def solve_factored(
    fac: FactoredSystem, i_inj: Tensor, *, v_fixed: Optional[Tensor] = None
) -> Tensor:
    """Solve ``Y V = I`` for a new RHS against a cached factorization (see
    :func:`lu_factor_system`). Identical result to :func:`solve_harmonic` with the same
    ``Y`` / slack mode; only the factorization is reused. Returns ``[*batch, N]`` (the
    leading dims broadcast ``i_inj`` against the factorization). The scenario batch is
    solved as MULTIPLE right-hand sides of the one shared factorization
    (:func:`_lu_solve_shared`), so the dense ``Y`` is never tiled across the batch. The
    ``"block"`` backend does the same per diagonal block, gathering / scattering each
    block's entries of the right-hand side around its own batched back-substitution."""
    if fac.mode == "norton":
        return back_substitute(fac, i_inj)
    rhs, vf_b = ideal_slack_rhs(fac, i_inj, v_fixed)
    v_free = back_substitute(fac, rhs)  # [*batch, F]
    return scatter_slack_solution(fac, v_free, vf_b)


__all__ = [
    "solve_harmonic",
    "solve_anchored",
    "lu_factor_system",
    "solve_factored",
    "FactoredSystem",
    # shared building blocks of a factored solve (reused by pgml.solver.lowrank)
    "back_substitute",
    "ideal_slack_rhs",
    "scatter_slack_solution",
]
