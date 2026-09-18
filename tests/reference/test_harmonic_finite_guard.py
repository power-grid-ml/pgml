"""A non-finite harmonic solution is reported, not returned silently."""

from __future__ import annotations

import logging

import torch

import pgml.solver.harmonic_flow as harmonic_flow
from pgml.solver import solve_harmonic_flow

from .test_harmonic_shunt_woodbury import _operating_point, _sparse_load_grid

POWER = torch.tensor([1_500.0, 2_000.0, 2_500.0], dtype=torch.float64)


def _poison(monkeypatch, scenario):
    original = harmonic_flow._solve_harmonic_orders

    def poisoned(*args, **kwargs):
        vh = original(*args, **kwargs).clone()
        if scenario is None:
            vh[..., 0, 0] = float("nan")
        else:
            vh[scenario, 0, 0] = float("inf")
        return vh

    monkeypatch.setattr(harmonic_flow, "_solve_harmonic_orders", poisoned)


def test_finite_solution_reports_every_scenario_as_converged():
    result = solve_harmonic_flow(
        _sparse_load_grid(), [1, 5, 7], operating_point=_operating_point(POWER)
    )
    assert result.converged
    assert result.harmonic_finite.tolist() == [True, True, True]
    assert result.converged_mask.tolist() == [True, True, True]
    assert result.failed_states == ()


def test_non_finite_scenario_is_reported_as_failed(monkeypatch, caplog):
    _poison(monkeypatch, 1)
    with caplog.at_level(logging.ERROR, logger="pgml"):
        result = solve_harmonic_flow(
            _sparse_load_grid(), [1, 5, 7], operating_point=_operating_point(POWER)
        )
    assert result.pf.converged  # the fundamental itself is fine
    assert not result.converged
    assert result.harmonic_finite.tolist() == [True, False, True]
    assert result.converged_mask.tolist() == [True, False, True]
    assert result.failed_states == (1,)
    assert "NON-FINITE harmonic solution" in caplog.text
    assert "[1]" in caplog.text


def test_non_finite_unbatched_solution_is_not_converged(monkeypatch, caplog):
    _poison(monkeypatch, None)
    with caplog.at_level(logging.ERROR, logger="pgml"):
        result = solve_harmonic_flow(_sparse_load_grid(), [1, 5, 7])
    assert result.pf.converged
    assert not result.converged
    assert result.converged_mask is None
    assert "NON-FINITE harmonic solution" in caplog.text
