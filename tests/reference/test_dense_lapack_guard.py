"""A batch of dense systems must solve on a multi-threaded CPU, or say it cannot.

Some CPU LAPACK builds get the row interchange of a multi-threaded batched LU wrong
above a few hundred rows, which without a guard removes every path that hands a STACK
of matrices to the solver: the operating-point device shunt, a batched voltage node
source, and a switch-state sweep that assembles each state.
"""

from __future__ import annotations

import os

import pytest
import torch

from pgml.errors import ComputationError
from pgml.solver import lu_factor_system, solve_factored, solve_harmonic
from pgml.solver import _dense_lapack

#: Large enough that the defect appears on an affected build; small enough to be quick.
ROWS = 600
BATCH = 16


@pytest.fixture
def threads():
    """Run the test on several threads, whatever the session default is."""
    previous = torch.get_num_threads()
    torch.set_num_threads(max(2, min(8, os.cpu_count() or 2)))
    yield torch.get_num_threads()
    torch.set_num_threads(previous)


def _stack(rows: int = ROWS, batch: int = BATCH):
    """A well-conditioned complex stack whose LU needs row interchanges."""
    generator = torch.Generator().manual_seed(0)
    y = torch.randn(
        batch, rows, rows, dtype=torch.complex128, generator=generator
    ) / rows + torch.eye(rows, dtype=torch.complex128)
    i = torch.randn(batch, rows, dtype=torch.complex128, generator=generator)
    return y, i


def _backward_error(y, v, i):
    return float((torch.einsum("bij,bj->bi", y, v) - i).abs().amax() / i.abs().amax())


@pytest.mark.slow
def test_batched_solve_and_factorization_survive_several_threads(threads):
    y, i = _stack()
    assert threads > 1

    v = solve_harmonic(y, i)
    assert _backward_error(y, v, i) < 1e-10

    v_factored = solve_factored(lu_factor_system(y, backend="dense"), i)
    assert _backward_error(y, v_factored, i) < 1e-10

    # The guard must not leak its thread count into the caller's process.
    assert torch.get_num_threads() == threads


def test_a_single_system_is_never_guarded():
    """The per-iteration path of the nonlinear solvers must pay nothing for this."""
    one = torch.eye(4, dtype=torch.complex128)
    assert not _dense_lapack.is_batched_cpu(one)
    assert not _dense_lapack.is_batched_cpu(one.unsqueeze(0))
    assert _dense_lapack.is_batched_cpu(one.expand(2, 4, 4))
    if torch.cuda.is_available():
        assert not _dense_lapack.is_batched_cpu(one.expand(2, 4, 4).cuda())


def test_a_corrupted_factorization_is_rejected_instead_of_returned(monkeypatch):
    y, _ = _stack(rows=12, batch=3)
    truth = torch.linalg.lu_factor

    def wrong(a, *args, **kwargs):
        lu, piv = truth(a, *args, **kwargs)
        # One extra row interchange, still in range, so nothing below the
        # backward-error check would notice that the factors are not this matrix's.
        corrupted = piv.clone()
        corrupted[..., 0] = a.shape[-1]
        return lu, corrupted

    monkeypatch.setattr(torch.linalg, "lu_factor", wrong)
    with pytest.raises(ComputationError, match="residual"):
        lu_factor_system(y, backend="dense")


def test_a_raising_batched_call_is_retried_before_it_is_reported(monkeypatch):
    """A build that fails only on several threads must be repaired, not reported."""
    y, i = _stack(rows=12, batch=3)
    truth = torch.linalg.solve
    calls = []

    def fails_when_threaded(a, b, *args, **kwargs):
        calls.append(torch.get_num_threads())
        if torch.get_num_threads() > 1:
            raise RuntimeError(
                "Pivots given to lu_solve must all be greater or equal to 1."
            )
        return truth(a, b, *args, **kwargs)

    monkeypatch.setattr(_dense_lapack, "_single_thread_required", False)
    monkeypatch.setattr(torch.linalg, "solve", fails_when_threaded)
    torch.set_num_threads(max(2, torch.get_num_threads()))
    v = solve_harmonic(y, i)
    assert _backward_error(y, v, i) < 1e-10
    assert calls[0] > 1 and calls[-1] == 1
    assert _dense_lapack.single_thread_required()


def test_an_ill_conditioned_system_still_factors(threads):
    """The check measures the backward error, which conditioning does not inflate."""
    n = 40
    scale = torch.logspace(0, 9, n, dtype=torch.float64)
    y = (torch.eye(n, dtype=torch.complex128) * scale).expand(4, n, n).contiguous()
    i = torch.ones(4, n, dtype=torch.complex128)
    v = solve_factored(lu_factor_system(y, backend="dense", equilibrate="off"), i)
    assert _backward_error(y, v, i) < 1e-12


def test_a_badly_scaled_network_still_factors(threads):
    """A milliohm switch beside a line makes ``A x`` cancel, not the factors wrong.

    The residual then looks large against the right-hand side alone, which is why the
    check falls back to the matrix norm before it rejects anything.
    """
    n = 60
    generator = torch.Generator().manual_seed(1)
    y = torch.randn(4, n, n, dtype=torch.complex128, generator=generator)
    # One node tied to the next through a near-zero impedance: a 1e6 S admittance
    # beside entries of order one, which is what an ideal switch stamps.
    y[:, 0, 0] += 1.0e6
    y[:, 0, 1] -= 1.0e6
    y[:, 1, 0] -= 1.0e6
    y[:, 1, 1] += 1.0e6
    y += n * torch.eye(n, dtype=torch.complex128)
    i = torch.randn(4, n, dtype=torch.complex128, generator=generator)
    v = solve_factored(lu_factor_system(y, backend="dense", equilibrate="off"), i)
    assert _backward_error(y, v, i) < 1e-10
