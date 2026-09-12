"""Diagonal equilibration of a linear system, applied around its factorization.

An SI-unit nodal admittance matrix spans many decades: a stiff source row carries
``|Y_ii| ~ 1e5 S`` while a low-voltage cable row carries ``~1e-2 S``, and a harmonic
order multiplies the spread again. The matrix is therefore badly SCALED long before it
is badly conditioned in any intrinsic sense, and a diagonal rescaling removes most of
the condition number without touching the physics:

.. math::
    \\hat{A} = D_r A D_c , \\qquad
    A x = b \\iff \\hat{A}\\,(D_c^{-1} x) = D_r b

so a solve factors ``Â``, scales the right-hand side by ``D_r`` and the solution by
``D_c``. Nothing outside the factorization sees it: the public API keeps SI units in
and out, and the composite map is exactly ``A^{-1} b``, so gradients w.r.t. ``A`` and
``b`` are unchanged.

Two scalings, both measured in the solver's reports:

``"symmetric"`` (default)
    van der Sluis scaling ``d_i = |A_ii|^{-1/2}``, ``D_r = D_c = D``. It is a
    congruence, so it keeps symmetry and the sparsity pattern, reads only the
    diagonal (no extra ``[N, N]`` temporary), and is within a factor ``sqrt(n)`` of
    the best possible condition number over all diagonal scalings for a matrix with
    a nonzero diagonal.
``"row_column"``
    the two-sided LAPACK-style variant (``xGEEQU``): ``r_i = 1/max_j |a_ij|``, then
    ``c_j = 1/max_i |r_i a_ij|``, which makes every row and column of ``Â`` have
    max-norm 1. It needs the magnitude of every entry, which on a CPU matrix is read
    from the nonzeros (a structural zero is never a maximum) and on an accelerator or a
    batched matrix from one real ``[N, N]`` temporary. It is unavailable on the
    block-diagonal backend, whose free-row matrix is never materialised.

The scale factors are rounded to POWERS OF TWO by default, so ``D_r A D_c`` is exact
in binary floating point: the equilibration introduces no rounding error of its own,
and it cannot change a solution it was only meant to condition. The rounding also
makes the scale's own derivative zero, which is correct — the composite solve does not
depend on it — while the scaling multiplications stay on the autograd tape, so
gradients flow to the matrix and the right-hand side exactly as without equilibration.

Reference libraries for comparison (none of them scales more than one side):
OpenDSS factors its SI-unit system with KLU, whose default ``Common.scale = 2`` is a
max-norm ROW scaling; MATLAB's sparse backslash (MATPOWER's default linear solver)
applies UMFPACK's default row-sum scaling to the per-unit Jacobian; pandapower's
Newton-Raphson calls ``scipy.sparse.linalg.spsolve``, i.e. SuperLU's simple driver,
which does not equilibrate; power-grid-model factors its per-unit system with a
block-sparse LU that uses full pivoting inside each block plus optional pivot
perturbation with iterative refinement, and no scaling.

Public API
----------
- ``EQUILIBRATION_MODES`` — the accepted modes.
- ``resolve_equilibration(equilibrate) -> str`` — resolve ``None`` / bool / name.
- ``equilibration_scales(a, *, mode, power_of_two=True) -> (d_row, d_col)``
- ``equilibrate_matrix(a, *, mode, power_of_two=True) -> (a_hat, d_row, d_col)``
- ``equilibrated_lu_factor(a, *, mode, power_of_two=True, factor_dtype=None)
  -> EquilibratedLU`` — factor once, solve many (``EquilibratedLU.solve``).

Differentiability + GPU: pure torch elementwise ops and ``torch.linalg.lu_factor`` /
``lu_solve``; no ``.item()``, no data-dependent python branch, device and dtype follow
the input, every function is batched over leading dims.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from pgml import defaults
from pgml.errors import InputError

#: Accepted equilibration modes (``"off"`` disables it).
EQUILIBRATION_MODES = ("off", "symmetric", "row_column")


def resolve_equilibration(equilibrate: Union[None, bool, str]) -> str:
    """Resolve the ``equilibrate`` argument of a solver entry point to a mode name.

    ``None`` takes the documented default ``solver.equilibration.mode``; ``True`` /
    ``False`` are shorthands for that default / ``"off"``; a string must be one of
    :data:`EQUILIBRATION_MODES`.
    """
    if equilibrate is None:
        equilibrate = str(defaults.get("solver.equilibration.mode"))
    elif isinstance(equilibrate, bool):
        equilibrate = (
            str(defaults.get("solver.equilibration.mode")) if equilibrate else "off"
        )
    if equilibrate not in EQUILIBRATION_MODES:
        raise InputError(
            f"Unsupported equilibrate {equilibrate!r} (use one of "
            f"{', '.join(repr(m) for m in EQUILIBRATION_MODES)}, True for the "
            "documented default, or False/'off' to solve the unscaled system)."
        )
    return equilibrate


def _power_of_two_default() -> bool:
    return bool(defaults.get("solver.equilibration.power_of_two"))


def _nonzero_magnitudes(a: Tensor) -> Optional[tuple[Tensor, Tensor, Tensor]]:
    """``(rows, cols, |a_ij|)`` over the NONZEROS of ``a``, or ``None`` where that form
    does not apply.

    A nodal admittance matrix has O(N) nonzeros, and every magnitude-based scale of it —
    the row and column maxima of the two-sided equilibration, the row sums behind the
    solver's per-row precision floor — reads those entries only: the magnitude of a
    structural zero is neither a maximum nor a contribution to a sum. One read of the
    matrix into CSR therefore turns an ``[m, m]`` magnitude pass (a square root per entry,
    plus a full real temporary) into O(nnz) work on the nonzeros.

    ``None`` is returned for a matrix this form does not cover: a non-CPU tensor (the
    scatter reductions that consume it are order-deterministic on CPU only, and an
    accelerator prefers the dense elementwise pass anyway), a batched one (no single
    sparsity pattern), and one that is being differentiated (the sparse conversion carries
    no gradient for a complex matrix, and the dense pass keeps whatever the scale's own
    derivative contributes). Shapes: ``rows`` / ``cols`` int64 ``[nnz]``, magnitudes real
    ``[nnz]`` in the real dtype paired with ``a``'s, all on ``a``'s device.
    """
    if (
        a.device.type != "cpu"
        or (torch.is_grad_enabled() and a.requires_grad)
        or a.reshape(-1, *a.shape[-2:]).shape[0] != 1
    ):
        return None
    m = a.shape[-1]
    with warnings.catch_warnings():
        # The backend's beta-status notice for CSR tensors is not a caller's concern.
        warnings.simplefilter("ignore", UserWarning)
        csr = a.reshape(m, m).to_sparse_csr()
    rows = torch.repeat_interleave(csr.crow_indices().diff())  # [nnz]
    return rows, csr.col_indices(), csr.values().abs()


def _scatter_max(index: Tensor, values: Tensor, m: int) -> Tensor:
    """``max`` of ``values`` per ``index``, over ``m`` slots, 0 where an index is absent.

    The magnitudes are non-negative, so the zero an absent index keeps is the maximum over
    an empty set of entries — exactly what an all-zero row of the dense pass reports.
    """
    out = torch.zeros(m, dtype=values.dtype, device=values.device)
    return out.scatter_reduce_(0, index, values, reduce="amax")


def _reciprocal(scale: Tensor, *, sqrt: bool, power_of_two: bool) -> Tensor:
    """``1/scale`` (or ``scale^{-1/2}``), 1 where ``scale`` is not finite and positive.

    A row whose magnitude measure is zero or not finite (an empty row of a
    structurally singular system) keeps scale 1, so the equilibration never turns a
    finite matrix into one holding infinities — the factorization then reports the
    singularity itself, with its own message.
    """
    ok = torch.isfinite(scale) & (scale > 0)
    safe = torch.where(ok, scale, torch.ones_like(scale))
    out = safe.rsqrt() if sqrt else safe.reciprocal()
    if power_of_two:
        # Exact in binary floating point: D A D then introduces no rounding error,
        # and round() has a zero derivative, so the scale itself is not differentiated
        # (the composite solve does not depend on it).
        out = torch.exp2(torch.round(torch.log2(out)))
    return torch.where(ok, out, torch.ones_like(out))


def equilibration_scales(
    a: Tensor, *, mode: str = "symmetric", power_of_two: Optional[bool] = None
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Row / column scales of ``Â = diag(d_row) A diag(d_col)``.

    ``a`` is ``[*batch, m, m]``, real or complex. Returns ``(d_row, d_col)``, both
    ``[*batch, m]`` in the REAL dtype paired with ``a``'s dtype (so scaling a
    complex64 matrix stays complex64), or ``(None, None)`` for ``mode="off"``.
    ``"symmetric"`` returns the same tensor twice.
    """
    if mode == "off":
        return None, None
    if mode not in EQUILIBRATION_MODES:
        raise InputError(
            f"Unsupported equilibration mode {mode!r} "
            f"(use one of {', '.join(repr(m) for m in EQUILIBRATION_MODES)})."
        )
    if power_of_two is None:
        power_of_two = _power_of_two_default()
    rdt = a.real.dtype if a.is_complex() else a.dtype
    if mode == "symmetric":
        diag = a.diagonal(dim1=-2, dim2=-1).abs().to(rdt)  # [*batch, m]
        d = _reciprocal(diag, sqrt=True, power_of_two=power_of_two)
        return d, d
    nz = _nonzero_magnitudes(a)
    if nz is not None:
        # The maxima read the nonzeros only, and a maximum does not depend on the order it
        # is taken in, so the scales are the dense pass's to the bit at O(nnz) instead of
        # O(m²) (complex128, CPU: 9 ms against 67 on a 2469-row feeder, 50 against 267 at
        # 5505 rows).
        rows, cols, mag = nz
        m = a.shape[-1]
        d_row = _reciprocal(
            _scatter_max(rows, mag, m), sqrt=False, power_of_two=power_of_two
        )  # [m]
        d_col = _reciprocal(
            _scatter_max(cols, mag * d_row.index_select(0, rows), m),
            sqrt=False,
            power_of_two=power_of_two,
        )
        return d_row.reshape(*a.shape[:-1]), d_col.reshape(*a.shape[:-1])
    mag = a.abs().to(rdt)  # [*batch, m, m]
    d_row = _reciprocal(
        mag.amax(dim=-1), sqrt=False, power_of_two=power_of_two
    )  # [*batch, m]
    d_col = _reciprocal(
        (mag * d_row.unsqueeze(-1)).amax(dim=-2), sqrt=False, power_of_two=power_of_two
    )
    return d_row, d_col


def scale_matrix(a: Tensor, d_row: Optional[Tensor], d_col: Optional[Tensor]) -> Tensor:
    """``diag(d_row) A diag(d_col)`` for ``a`` ``[*batch, m, m]`` (identity on ``None``).

    Two-sided scaling writes a second full matrix when both multiplications are
    out-of-place, and on a large system that second pass costs as much as a sparse
    factorization of the same matrix (measured on a 1176-row feeder, complex128, CPU:
    13.9 ms for two passes against 6.5 ms for one, where the SuperLU factorization of
    that matrix is 6.8 ms). Where no gradient is being recorded — every forward solve,
    which is where the cost shows — the column scaling therefore runs IN PLACE on the
    fresh tensor the row scaling just produced, which nothing else references. With
    autograd active both multiplications stay out-of-place and on the tape.
    """
    if d_row is None and d_col is None:
        return a
    if d_row is None:
        return a * d_col.unsqueeze(-2)
    out = a * d_row.unsqueeze(-1)
    if d_col is None:
        return out
    if torch.is_grad_enabled() and (a.requires_grad or out.requires_grad):
        return out * d_col.unsqueeze(-2)
    return out.mul_(d_col.unsqueeze(-2))


def equilibrate_matrix(
    a: Tensor, *, mode: str = "symmetric", power_of_two: Optional[bool] = None
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """``(Â, d_row, d_col)`` with ``Â = diag(d_row) A diag(d_col)``; see
    :func:`equilibration_scales`. ``mode="off"`` returns ``a`` itself and two
    ``None`` scales, so a caller needs no special case."""
    d_row, d_col = equilibration_scales(a, mode=mode, power_of_two=power_of_two)
    return scale_matrix(a, d_row, d_col), d_row, d_col


@dataclass(frozen=True)
class EquilibratedLU:
    """LU factors of an equilibrated matrix, plus the scales that undo the scaling.

    Built by :func:`equilibrated_lu_factor` for a factor-once-solve-many system that
    is NOT a network admittance (the Newton state Jacobian and the implicit-function
    adjoint), where :class:`~pgml.solver.harmonic.FactoredSystem` — which carries a
    slack partition and three backends — would be the wrong tool.

    Attributes
    ----------
    lu, piv:
        ``torch.linalg.lu_factor`` output of ``Â``, ``[*batch, m, m]`` / ``[*batch, m]``.
    d_row, d_col:
        The scales, ``[*batch, m]`` real, or ``None`` when equilibration is off.
    mode:
        The resolved equilibration mode, for reporting.
    out_dtype:
        Dtype the solutions are returned in (the input matrix's dtype; the factors may
        be single precision).
    """

    lu: Tensor
    piv: Tensor
    d_row: Optional[Tensor]
    d_col: Optional[Tensor]
    mode: str
    out_dtype: torch.dtype

    def solve(self, rhs: Tensor, *, adjoint: bool = False) -> Tensor:
        """Solve ``A x = rhs`` (``A^H x = rhs`` when ``adjoint``), ``rhs`` ``[*batch, m]``.

        The scaling is undone around the back-substitution, so the result is the
        solution of the UNSCALED system: ``x = D_c Â^{-1} D_r b`` and, for the adjoint
        system, ``x = D_r Â^{-H} D_c b`` (the scales are real, so the adjoint only
        swaps their roles). Returns ``[*batch, m]`` in :attr:`out_dtype`.
        """
        d_in, d_out = (self.d_col, self.d_row) if adjoint else (self.d_row, self.d_col)
        b = rhs if d_in is None else rhs * d_in
        x = torch.linalg.lu_solve(
            self.lu, self.piv, b.to(self.lu.dtype).unsqueeze(-1), adjoint=adjoint
        ).squeeze(-1)
        if d_out is not None:
            x = x * d_out.to(x.dtype)
        return x.to(self.out_dtype)


def equilibrated_lu_factor(
    a: Tensor,
    *,
    mode: str = "symmetric",
    power_of_two: Optional[bool] = None,
    factor_dtype: Optional[torch.dtype] = None,
) -> EquilibratedLU:
    """Equilibrate ``a`` ``[*batch, m, m]`` and LU-factor it for repeated solves.

    ``factor_dtype`` factors a lower-precision copy (the inexact-Newton direction);
    solutions always come back in ``a``'s dtype. The returned object answers
    :meth:`EquilibratedLU.solve` for any number of right-hand sides without
    re-factoring, which is what makes a repeated vector-Jacobian product cheap.
    """
    a_hat, d_row, d_col = equilibrate_matrix(a, mode=mode, power_of_two=power_of_two)
    lu, piv = torch.linalg.lu_factor(
        a_hat if factor_dtype is None else a_hat.to(factor_dtype)
    )
    return EquilibratedLU(
        lu=lu, piv=piv, d_row=d_row, d_col=d_col, mode=mode, out_dtype=a.dtype
    )


__all__ = [
    "EQUILIBRATION_MODES",
    "EquilibratedLU",
    "equilibrate_matrix",
    "equilibrated_lu_factor",
    "equilibration_scales",
    "resolve_equilibration",
    "scale_matrix",
]
