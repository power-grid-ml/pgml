"""Dense LAPACK calls on a STACK of matrices, guarded against a threading defect.

Some CPU LAPACK builds get the row interchange of a MULTI-THREADED batched LU wrong.
Factoring or solving a stack of more than one matrix then fails with ``Intel oneMKL
ERROR: Parameter 6 was incorrect on entry to ZLASWP`` followed by a rejected
factorization ("Pivots given to lu_solve must all be greater or equal to 1"). The same
call is exact in one thread, at every size and dtype measured. A batched matrix is what
a per-scenario admittance, a switch-state sweep and the implicit-function adjoint all
hand to LAPACK, so an affected install would otherwise lose those paths on any
multi-core machine.

The mechanism decides the repair. Setting the thread count also disables the math
library's dynamic thread adjustment, which is what would otherwise make it fall back to
one thread when it is entered from inside someone else's parallel region; the batched
factorization then calls the per-matrix routine from several threads at once, each
insisting on its full thread count, and above roughly 145 rows the blocked kernel
rejects the pivot array it is handed, skips the interchange and still reports success.
Factoring the same stack ONE MATRIX AT A TIME therefore repairs it without giving up a
single thread, because a single-matrix factorization is exactly the case the math
library threads correctly.

The guard applies only on CPU and only to a stack of MORE THAN ONE matrix, so a single
system, the CUDA path, the sparse backend and the per-iteration back-substitutions of
the nonlinear solvers are untouched. For those calls it:

1. runs the call as the caller asked, then measures the residual of the result against
   one probe right-hand side per matrix: one triangular solve and one matrix-vector
   product per matrix, ``O(n²)`` against the factorization's ``O(n³)``;
2. on a raised call or a residual above round-off, repeats the factorization one matrix
   at a time at the caller's thread count and remembers for the rest of the process that
   this build needs it, so the cost of the discovery is paid once rather than per call;
3. falls back to the batched call in ONE thread if even that is wrong, since the repair
   leans on the same math library;
4. raises :class:`~pgml.errors.ComputationError` if the single-threaded call is wrong
   too, rather than returning a wrong voltage.

A build without the defect therefore keeps its parallel batched factorization and pays
only the check. A build with it keeps working and keeps its threads.

<sub>Sixteen systems, complex128, eight threads, 12-core machine. The check alone costs
1.6 ms at 200 rows, 6.7 ms at 450 and 31.0 ms at 900, which is 12 % to 18 % of a correct
threaded factorization of the same stack. The whole guarded call with the per-matrix
repair takes 10.0 / 69.1 / 274.2 ms against 10.0 / 102.8 / 707.5 ms for the
single-threaded batched repair: even at 200 rows, 1.5x at 450 and 2.6x at 900, at the
same backward error.</sub>

The thread count is a process-global setting, so a caller that reaches the last resort
and solves from several threads at once must serialize its solves.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import math

import torch
from torch import Tensor

from pgml.errors import ComputationError

_log = logging.getLogger("pgml")

#: The batched call as the caller asked for it.
AS_ASKED = "none"
#: One matrix at a time, at the caller's thread count.
PER_MATRIX = "per_matrix"
#: The batched call in one thread.
ONE_THREAD = "one_thread"

#: Escalation order. A stage is entered only once the previous one has been rejected,
#: and the stage a batched dense CPU call ends up needing is latched for the process.
_STAGES = (AS_ASKED, PER_MATRIX, ONE_THREAD)

_repair = AS_ASKED

_ADVICE = (
    "The multi-threaded batched LU of this CPU LAPACK build is wrong and neither "
    "factoring one matrix at a time nor a single thread repaired it. Solve with "
    "linear_solver='sparse', or on one matrix at a time."
)

_ESCALATION_NOTE = {
    PER_MATRIX: (
        "this CPU LAPACK build factors a stack of matrices incorrectly on several "
        "threads, so batched dense factorizations now factor one matrix at a time. "
        "Results are unaffected and the caller's threads are kept."
    ),
    ONE_THREAD: (
        "factoring one matrix at a time did not repair this CPU LAPACK build either, "
        "so batched dense factorizations now run in one thread. Results are "
        "unaffected; a batched factorization loses its parallelism."
    ),
}


def is_batched_cpu(a: Tensor) -> bool:
    """Whether ``a`` is a CPU stack of MORE THAN ONE matrix (the affected case)."""
    return a.device.type == "cpu" and a.ndim > 2 and math.prod(a.shape[:-2]) > 1


def repair_in_use() -> str:
    """Which repair a batched dense CPU call on this build has been found to need.

    One of :data:`AS_ASKED` (none), :data:`PER_MATRIX` or :data:`ONE_THREAD`.
    """
    return _repair


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


def _per_matrix_lu_factor(a: Tensor) -> tuple[Tensor, Tensor]:
    """``torch.linalg.lu_factor`` matrix by matrix, keeping the caller's threads.

    The loop over the batch is what the repair IS: it is the batched call that this
    build gets wrong, and each single-matrix factorization inside the loop still uses
    every thread the caller allocated. Gradients flow through ``stack``, so the loop
    costs the tape one node per matrix and changes nothing else.

    LAPACK returns its factors COLUMN-major, and every later back-substitution expects
    them that way. Stacking the factors directly would hand back a row-major tensor
    that each ``lu_solve`` then has to transpose, which costs more than the repair
    saves (a factor of three to four on a 450-row stack), so the transposed views are
    stacked and transposed back instead: one copy either way, the right layout.
    """
    flat = a.reshape(-1, *a.shape[-2:])
    factored = [torch.linalg.lu_factor(flat[k]) for k in range(flat.shape[0])]
    lu = torch.stack([f.mT for f, _ in factored]).mT
    piv = torch.stack([p for _, p in factored])
    batch = a.shape[:-2]
    return lu.unflatten(0, batch), piv.unflatten(0, batch)


def _per_matrix_solve(a: Tensor, b: Tensor) -> Tensor:
    """``torch.linalg.solve`` matrix by matrix, keeping the caller's threads.

    Reproduces the batched call's broadcasting and its vector / matrix right-hand-side
    convention, so the result has the shape the caller would have got.
    """
    vector_rhs = b.ndim == a.ndim - 1
    rhs = b.unsqueeze(-1) if vector_rhs else b
    batch = torch.broadcast_shapes(a.shape[:-2], rhs.shape[:-2])
    n, k = a.shape[-1], rhs.shape[-1]
    a_flat = a.expand(*batch, n, n).reshape(-1, n, n)
    b_flat = rhs.expand(*batch, n, k).reshape(-1, n, k)
    x = torch.stack(
        [torch.linalg.solve(a_flat[j], b_flat[j]) for j in range(a_flat.shape[0])]
    ).reshape(*batch, n, k)
    return x.squeeze(-1) if vector_rhs else x


def _escalate(stage: str, where: str) -> None:
    """Latch the next repair stage for the process and say so once."""
    global _repair
    following = _STAGES[_STAGES.index(stage) + 1]
    if _STAGES.index(following) > _STAGES.index(_repair):
        _repair = following
        _log.warning("%s: %s", where, _ESCALATION_NOTE[following])


def _attempt(run, *, where: str, limit: float):
    """Run a batched call, escalating through the repairs until one is right.

    ``run(per_matrix)`` performs the LAPACK call AND measures the residual of its
    result, returning both. It runs inside the thread context under test, because the
    defect reaches the triangular solve the measurement itself needs: a raised
    measurement is the same evidence as a raised factorization and counts as an
    infinite error.
    """
    for stage in _STAGES[_STAGES.index(_repair) :]:
        last = stage == _STAGES[-1]
        try:
            with single_thread(stage == ONE_THREAD):
                result, error = run(stage == PER_MATRIX)
        except RuntimeError as exc:
            if last:
                raise ComputationError(f"{where}: {exc} {_ADVICE}") from exc
            _escalate(stage, where)
            continue
        if error <= limit:
            return result
        if last:
            raise ComputationError(
                f"{where}: the batched dense factorization has a relative "
                f"residual of {error:.3e}, so its solutions would be wrong. {_ADVICE}"
            )
        _escalate(stage, where)
    raise AssertionError("unreachable")  # pragma: no cover


def lu_factor(a: Tensor, *, where: str) -> tuple[Tensor, Tensor]:
    """``torch.linalg.lu_factor``, verified and repaired when ``a`` is a CPU stack."""
    if not is_batched_cpu(a):
        return torch.linalg.lu_factor(a)
    a_d = a.detach()
    b = _probe_rhs(a_d)
    limit = _limit(a.dtype)

    def run(per_matrix: bool):
        lu, piv = _per_matrix_lu_factor(a) if per_matrix else torch.linalg.lu_factor(a)
        with torch.no_grad():
            x = torch.linalg.lu_solve(lu.detach(), piv, b)
        return (lu, piv), _relative_residual(a_d, x, b, limit)

    return _attempt(run, where=where, limit=limit)


def lu_solve(lu: Tensor, piv: Tensor, rhs: Tensor, *, adjoint: bool = False) -> Tensor:
    """``torch.linalg.lu_solve``, in one thread when this build needs that.

    No residual check: this is the per-right-hand-side call of the nonlinear solvers,
    and the factorization it uses was verified when it was built — by a BATCHED
    ``lu_solve`` against the probe right-hand side, so a build that also got the
    back-substitution wrong would have been caught there. The defect itself is in the
    factorization's row interchange, so the per-matrix repair leaves this call batched
    and multi-threaded; only the last resort, where nothing about the build is trusted,
    serializes it too.
    """
    if _repair != ONE_THREAD or not is_batched_cpu(lu):
        return torch.linalg.lu_solve(lu, piv, rhs, adjoint=adjoint)
    with single_thread(True):
        return torch.linalg.lu_solve(lu, piv, rhs, adjoint=adjoint)


def solve(a: Tensor, b: Tensor, *, where: str) -> Tensor:
    """``torch.linalg.solve``, verified and repaired when ``a`` is a CPU stack."""
    if not is_batched_cpu(a):
        return torch.linalg.solve(a, b)
    a_d, b_d = a.detach(), b.detach()
    limit = _limit(a.dtype)

    def run(per_matrix: bool):
        x = _per_matrix_solve(a, b) if per_matrix else torch.linalg.solve(a, b)
        return x, _relative_residual(a_d, x.detach(), b_d, limit)

    return _attempt(run, where=where, limit=limit)


__all__ = [
    "AS_ASKED",
    "ONE_THREAD",
    "PER_MATRIX",
    "is_batched_cpu",
    "lu_factor",
    "lu_solve",
    "repair_in_use",
    "single_thread",
    "solve",
]
