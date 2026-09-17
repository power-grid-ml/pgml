"""Condition estimate of batched factorizations and the complex64 conditioning check."""

from __future__ import annotations

import logging

import pytest
import torch

import pgml.solver.power_flow as power_flow
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import estimate_condition, lu_factor_system

from .test_harmonic_shunt_woodbury import _sparse_load_grid


def _systems() -> torch.Tensor:
    """Three 6x6 systems; the third is badly row-scaled."""
    gen = torch.Generator().manual_seed(0)
    n = 6
    a = torch.randn(3, n, n, dtype=torch.complex128, generator=gen)
    a = a + 5.0 * torch.eye(n, dtype=torch.complex128)
    scale = torch.logspace(0, 6, n, dtype=torch.float64).unsqueeze(-1)
    return torch.cat([a[:2], (a[2] * scale).unsqueeze(0)])


@pytest.mark.parametrize("equilibrate", ["off", "symmetric"])
def test_batched_estimate_equals_the_per_matrix_estimates(equilibrate):
    a = _systems()
    single = torch.tensor(
        [
            estimate_condition(lu_factor_system(a[k], equilibrate=equilibrate))
            for k in range(a.shape[0])
        ],
        dtype=torch.float64,
    )
    fac = lu_factor_system(a, equilibrate=equilibrate)
    per_matrix = estimate_condition(fac, per_matrix=True)

    assert per_matrix.shape == (3,)
    assert torch.allclose(per_matrix, single, rtol=1e-10)
    assert estimate_condition(fac) == pytest.approx(float(single.max()), rel=1e-10)
    # A lower bound of the exact 1-norm condition number, and a useful one.
    exact = torch.linalg.cond(fac.y_mat, p=1)
    assert (per_matrix <= exact * (1 + 1e-10)).all()
    assert (per_matrix >= 0.3 * exact).all()


def test_zero_pivot_in_a_batch_reports_inf_for_that_matrix_only():
    """A factorization with a zero pivot back-substitutes to inf / nan."""
    fac = lu_factor_system(_systems(), equilibrate="off")
    fac.lu[1].zero_()
    per_matrix = estimate_condition(fac, per_matrix=True)
    assert torch.isinf(per_matrix[1])
    assert torch.isfinite(per_matrix[[0, 2]]).all()
    assert estimate_condition(fac) == float("inf")


def _looped_feeder():
    grid = _sparse_load_grid()
    parallel = grid.branches[3].model_copy(update={"id": 99})
    return grid.model_copy(update={"branches": [*grid.branches, parallel]})


@pytest.mark.parametrize("method", ["assemble", "woodbury"])
def test_first_complex64_solve_with_batched_branch_states(method):
    """The once-per-process conditioning check runs on a batched factorization."""
    states = {99: torch.tensor([1.0, 0.0, 1.0])}
    power_flow._COMPLEX64_COND_CHECKED = False
    try:
        result = solve_power_flow(
            _looped_feeder(),
            branch_states=states,
            branch_states_method=method,
            dtype=torch.complex64,
        )
    finally:
        power_flow._COMPLEX64_COND_CHECKED = False
    assert result.v.shape[0] == 3
    assert result.converged


def test_failing_condition_estimate_does_not_fail_the_solve(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("estimate unavailable")

    monkeypatch.setattr(power_flow, "estimate_condition", broken)
    power_flow._COMPLEX64_COND_CHECKED = False
    try:
        with caplog.at_level(logging.DEBUG, logger="pgml"):
            result = solve_power_flow(_sparse_load_grid(), dtype=torch.complex64)
    finally:
        power_flow._COMPLEX64_COND_CHECKED = False
    assert result.converged
    assert "conditioning check skipped" in caplog.text
