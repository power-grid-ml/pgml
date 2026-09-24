"""Dense LAPACK calls on a STACK of matrices, guarded against a threading defect.

Some CPU LAPACK builds get the row interchange of a MULTI-THREADED batched LU wrong.
Factoring or solving a stack of more than one matrix then fails with ``Intel oneMKL
ERROR: Parameter 6 was incorrect on entry to ZLASWP`` followed by a rejected
factorization ("Pivots given to lu_solve must all be greater or equal to 1"). The same
call is exact in one thread, at every size and dtype measured. A batched matrix is what
a per-scenario admittance, a switch-state sweep and the implicit-function adjoint all
hand to LAPACK, so an affected install would otherwise lose those paths on any
multi-core machine.

The guard applies only on CPU and only to a stack of MORE THAN ONE matrix, so a single
system, the CUDA path, the sparse backend and the per-iteration back-substitutions of
the nonlinear solvers are untouched. For those calls it:

1. runs the call as the caller asked, then measures the residual of the result against
   one probe right-hand side per matrix: one triangular solve and one matrix-vector
   product per matrix, ``O(n²)`` against the factorization's ``O(n³)``: measured at
   11 % of a batched factorization of sixteen 300-row systems, 6 % at 450 rows and
   within run-to-run noise at 900;
2. on a raised call or a residual above round-off, repeats the call in ONE thread and
   remembers for the rest of the process that this build needs it, so the cost of the
   discovery is paid once rather than per call;
3. raises :class:`~pgml.errors.ComputationError` if even the single-threaded call is
   wrong, rather than returning a wrong voltage.

A build without the defect therefore keeps its parallel factorization and pays only the
check, and a build with it keeps working. The thread count is a process-global setting,
so a caller that solves from several threads at once must serialize its solves.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import math

import torch
from torch import Tensor

from pgml.errors import ComputationError

_log = logging.getLogger("pgml")

#: Set once a batched dense call on this build has needed the single-threaded retry.
_single_thread_required = False

_ADVICE = (
    "The multi-threaded batched LU of this CPU LAPACK build is wrong and the "
    "single-threaded one did not repair it. Solve with linear_solver='sparse', or on "
    "one matrix at a time."
)


def is_batched_cpu(a: Tensor) -> bool:
    """Whether ``a`` is a CPU stack of MORE THAN ONE matrix (the affected case)."""
    return a.device.type == "cpu" and a.ndim > 2 and math.prod(a.shape[:-2]) > 1


def single_thread_required() -> bool:
    """Whether a batched dense CPU call on this build has already needed one thread."""
    return _single_thread_required


@contextmanager
def single_thread(active: bool):
    """Run the block in one thread when ``active``, restoring the previous count."""
    threads = torch.get_num_threads()
    if not active or threads <= 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _ratio(residual: Tensor, scale: Tensor) -> float:
    safe = scale if bool(scale > 0) else torch.ones_like(scale)
    value = residual / safe
    return float(value) if bool(torch.isfinite(value)) else float("inf")


def _relative_residual(a: Tensor, x: Tensor, b: Tensor, limit: float) -> float:
    """The normwise backward error of ``A x = b``, computed in two stages.

    The sound scale is ``‖A‖∞‖x‖∞ + ‖b‖∞``, but ``‖A‖∞`` is a second pass over the
    matrix and costs more than everything else here. The cheaper scale
    ``max(‖b‖∞, ‖A x‖∞)`` reuses the product the residual needs anyway and agrees with
    it whenever the matrix is reasonably scaled, which is the common case; it is only
    too small when forming ``A x`` cancels badly, as it does for a network holding a
    milliohm switch next to a line. So the cheap ratio decides on its own when it is
    already below ``limit``, and the full one is formed only for the few systems where
    it is not.
    """
    with torch.no_grad():
        ax = torch.matmul(a, x)
        residual = (ax - b).abs().amax()
        cheap = _ratio(residual, torch.maximum(b.abs().amax(), ax.abs().amax()))
        if cheap <= limit:
            return cheap
        scale = a.abs().sum(-1).amax() * x.abs().amax() + b.abs().amax()
        return _ratio(residual, scale)


#: Floor and round-off multiple of the residual a correct factorization may show. A
#: corrupted row interchange gives a residual of ORDER ONE, so the threshold only has to
#: sit comfortably between the two; the multiple leaves room for a badly scaled matrix
#: and for the single-precision factors of a mixed-precision solve, whose residual
#: against their own (already single-precision) matrix is around 1e-7.
_RESIDUAL_FLOOR = 1.0e-8
_RESIDUAL_EPS_MULTIPLE = 1.0e4


def _limit(dtype: torch.dtype) -> float:
    """How large a residual the precision that was factored can legitimately give."""
    return max(_RESIDUAL_FLOOR, _RESIDUAL_EPS_MULTIPLE * torch.finfo(dtype).eps)


def _probe_rhs(a: Tensor) -> Tensor:
    """A right-hand side ``[*batch, n, 1]`` that no row interchange leaves unchanged.

    The entries must be DISTINCT. A corrupted permutation ``P'`` turns the solve into
    ``A x = P⁻¹P' b``, so a constant right-hand side would have a zero residual for
    every permutation and hide exactly the failure this check exists for.
    """
    n = a.shape[-1]
    ramp = torch.arange(1, n + 1, device=a.device).reshape(n, 1).to(a.dtype)
    return ramp.expand(*a.shape[:-1], 1)


def _note_single_thread(where: str) -> None:
    global _single_thread_required
    if not _single_thread_required:
        _single_thread_required = True
        _log.warning(
            "%s: this CPU LAPACK build factors a stack of matrices incorrectly on "
            "several threads, so batched dense factorizations now run in one thread. "
            "Results are unaffected; a batched factorization loses its parallelism.",
            where,
        )


def _attempt(run, *, where: str, limit: float):
    """Run a batched call, and repeat it single-threaded if it is not right.

    ``run()`` performs the LAPACK call AND measures the residual of its result,
    returning both. It runs inside the thread context under test, because the defect
    reaches the triangular solve the measurement itself needs: a raised measurement is
    the same evidence as a raised factorization and counts as an infinite error.
    """
    for single in (_single_thread_required, True):
        try:
            with single_thread(single):
                result, error = run()
        except RuntimeError as exc:
            if single:
                raise ComputationError(f"{where}: {exc} {_ADVICE}") from exc
            _note_single_thread(where)
            continue
        if error <= limit:
            return result
        if single:
            raise ComputationError(
                f"{where}: the batched dense factorization has a relative "
                f"residual of {error:.3e}, so its solutions would be wrong. {_ADVICE}"
            )
        _note_single_thread(where)
    raise AssertionError("unreachable")  # pragma: no cover


def lu_factor(a: Tensor, *, where: str) -> tuple[Tensor, Tensor]:
    """``torch.linalg.lu_factor``, verified and repaired when ``a`` is a CPU stack."""
    if not is_batched_cpu(a):
        return torch.linalg.lu_factor(a)
    a_d = a.detach()
    b = _probe_rhs(a_d)
    limit = _limit(a.dtype)

    def run():
        lu, piv = torch.linalg.lu_factor(a)
        with torch.no_grad():
            x = torch.linalg.lu_solve(lu.detach(), piv, b)
        return (lu, piv), _relative_residual(a_d, x, b, limit)

    return _attempt(run, where=where, limit=limit)


def lu_solve(lu: Tensor, piv: Tensor, rhs: Tensor, *, adjoint: bool = False) -> Tensor:
    """``torch.linalg.lu_solve``, in one thread when this build needs it.

    No residual check: this is the per-right-hand-side call of the nonlinear solvers,
    and the factorization it uses was verified when it was built.
    """
    if not (_single_thread_required and is_batched_cpu(lu)):
        return torch.linalg.lu_solve(lu, piv, rhs, adjoint=adjoint)
    with single_thread(True):
        return torch.linalg.lu_solve(lu, piv, rhs, adjoint=adjoint)


def solve(a: Tensor, b: Tensor, *, where: str) -> Tensor:
    """``torch.linalg.solve``, verified and repaired when ``a`` is a CPU stack."""
    if not is_batched_cpu(a):
        return torch.linalg.solve(a, b)
    a_d, b_d = a.detach(), b.detach()
    limit = _limit(a.dtype)

    def run():
        x = torch.linalg.solve(a, b)
        return x, _relative_residual(a_d, x.detach(), b_d, limit)

    return _attempt(run, where=where, limit=limit)


__all__ = [
    "is_batched_cpu",
    "lu_factor",
    "lu_solve",
    "single_thread",
    "single_thread_required",
    "solve",
]
