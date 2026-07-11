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
from typing import Optional

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
# ``examples/pgml/benchmark_sparse.py`` on an i7-12700: at 600 rows factor 3.7x /
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


@dataclass
class FactoredSystem:
    """A factorization of the per-frequency system, reusable across many RHS.

    ``Y`` is constant across the current-injection fixed-point iterations (it is the
    network admittance — const-P/ZIP loads live on the RHS as ``I_device(V)``) AND across
    a scenario batch (which varies injections, not the network), so factoring once and
    back-substituting is far cheaper than re-factoring every solve. For ideal slack the
    factored matrix is the free-row block ``Y_ff`` and ``y_fs`` is kept for the RHS
    correction.

    Two backends, selected in :func:`lu_factor_system`:

    - ``"dense"`` — batched ``torch.linalg.lu_factor`` / ``lu_solve`` (CPU + CUDA);
      differentiable through the torch ops.
    - ``"sparse"`` — scipy SuperLU (:class:`_SciPySparseLU`, CPU only), ~O(N) for the
      O(N)-nnz power-grid ``Y`` where dense LU is O(N³); differentiable through the
      adjoint :class:`_SparseSolveFn` (``y_mat`` carries the autograd graph of the
      factored matrix).
    """

    mode: str  # "norton" | "ideal"
    lu: Optional[Tensor]
    piv: Optional[Tensor]
    n: int
    free_rows: Optional[Tensor] = None
    fixed_rows: Optional[Tensor] = None
    y_fs: Optional[Tensor] = None  # [*, F, S] for the ideal-slack RHS correction
    backend: str = "dense"  # "dense" | "sparse"
    sparse: Optional[_SciPySparseLU] = None
    y_mat: Optional[Tensor] = None  # sparse backend: the factored matrix (autograd)

    @property
    def _fb_tensor(self) -> Tensor:
        """The tensor carrying the factorization's leading batch / dtype / device."""
        return self.lu if self.backend == "dense" else self.y_mat


def _resolve_backend(backend: str, y_bus: Tensor) -> str:
    """Resolve ``"auto"`` by size and device (see ``_SPARSE_MIN_ROWS``)."""
    if backend not in ("auto", "dense", "sparse"):
        raise InputError(
            f"Unsupported factorization backend {backend!r} "
            "(use 'auto'/'dense'/'sparse')."
        )
    if backend != "auto":
        return backend
    if y_bus.device.type == "cpu" and y_bus.shape[-1] >= _SPARSE_MIN_ROWS:
        return "sparse"
    return "dense"


def lu_factor_system(
    y_bus: Tensor, *, fixed_rows: Optional[Tensor] = None, backend: str = "auto"
) -> FactoredSystem:
    """Factor ``Y`` (Norton) or the free block ``Y_ff`` (ideal slack) for repeated solves.

    Pair with :func:`solve_factored`, which back-substitutes a new RHS against the cached
    factorization. ``y_bus`` is ``[*batch, H, N, N]`` (or ``[N, N]``). ``backend``:
    ``"auto"`` (default) picks the scipy SuperLU sparse factorization on CPU systems of
    at least ``_SPARSE_MIN_ROWS`` rows (a power-grid ``Y`` has O(N) nonzeros, so sparse
    is ~O(N) where dense LU is O(N³)) and the batched dense ``torch.linalg.lu_factor``
    everywhere else (CUDA is ALWAYS dense — torch has no batched sparse direct solve);
    ``"dense"`` / ``"sparse"`` force the choice. Both backends are differentiable
    (dense through the torch ops, sparse through the adjoint :class:`_SparseSolveFn`).
    """
    n = y_bus.shape[-1]
    resolved = _resolve_backend(backend, y_bus)
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
    sys_t = fac._fb_tensor
    if fac.mode == "norton":
        if fac.backend == "sparse":
            return _SparseSolveFn.apply(fac.y_mat, i_inj, fac.sparse)
        return _lu_solve_shared(fac.lu, fac.piv, i_inj)

    if v_fixed is None:
        raise InputError("Ideal-slack factored solve requires `v_fixed`.")
    free_rows, fixed_rows, y_fs = fac.free_rows, fac.fixed_rows, fac.y_fs
    f, s = free_rows.shape[0], fixed_rows.shape[0]
    vf = v_fixed.to(dtype=sys_t.dtype, device=sys_t.device)
    i_free = i_inj.index_select(-1, free_rows)  # [*ib, F]
    # Build the corrected RHS ``I_free - Y_fs v_fixed`` (cheap: ``S`` slack rows), then
    # back-substitute it against the shared free-block factorization as multiple RHS.
    batch = torch.broadcast_shapes(
        sys_t.shape[:-2], i_free.shape[:-1], y_fs.shape[:-2], vf.shape[:-1]
    )
    y_fs_b = y_fs.broadcast_to(*batch, f, s)
    vf_b = vf.broadcast_to(*batch, s)
    rhs = i_free.broadcast_to(*batch, f) - torch.matmul(
        y_fs_b, vf_b.unsqueeze(-1)
    ).squeeze(-1)  # [*batch, F]
    if fac.backend == "sparse":
        v_free = _SparseSolveFn.apply(fac.y_mat, rhs, fac.sparse)  # [*batch, F]
    else:
        v_free = _lu_solve_shared(fac.lu, fac.piv, rhs)  # [*batch, F]
    v_full = torch.zeros(*batch, n, dtype=sys_t.dtype, device=sys_t.device)
    v_full = v_full.scatter(-1, free_rows.expand(*batch, f), v_free)
    v_full = v_full.scatter(-1, fixed_rows.expand(*batch, s), vf_b)
    return v_full


__all__ = ["solve_harmonic", "lu_factor_system", "solve_factored", "FactoredSystem"]
