"""Section 6: per-target structured perturbation sweep (one error per node)."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import Perturbation, Selector, perturbation_sweep, run_scenarios


def test_diagonal_structure(grid3):
    # 2 loads -> B=2; scenario j perturbs target j only (off-diagonal nominal).
    s = perturbation_sweep(
        grid3,
        Selector(component="load"),
        Perturbation(field="p", mode="scale", value=1.5),
    )
    assert s.n_samples == 2
    assert s.samples["perturbation_target_id"].tolist() == [10, 11]
    # load 10 (nominal 2000): perturbed only in scenario 0
    torch.testing.assert_close(
        s.operating_point[10]["p_w"],
        torch.tensor([3000.0, 2000.0], dtype=torch.float64),
    )
    # load 11 (nominal 3000): perturbed only in scenario 1
    torch.testing.assert_close(
        s.operating_point[11]["p_w"],
        torch.tensor([3000.0, 4500.0], dtype=torch.float64),
    )


def test_modes(grid3):
    sel = Selector(component="load", ids=[10])  # single target, B=1
    scale = perturbation_sweep(
        grid3, sel, Perturbation(field="p", mode="scale", value=2.0)
    )
    delta = perturbation_sweep(
        grid3, sel, Perturbation(field="p", mode="delta", value=500.0)
    )
    setp = perturbation_sweep(
        grid3, sel, Perturbation(field="p", mode="set", value=100.0)
    )
    assert float(scale.operating_point[10]["p_w"][0]) == 4000.0  # 2000 * 2
    assert float(delta.operating_point[10]["p_w"][0]) == 2500.0  # 2000 + 500
    assert float(setp.operating_point[10]["p_w"][0]) == 100.0  # = 100


def test_field_q_leaves_p(grid3):
    s = perturbation_sweep(
        grid3,
        Selector(component="load", ids=[10]),
        Perturbation(field="q", mode="delta", value=100.0),
    )
    e = s.operating_point[10]
    assert "q_var" in e and "p_w" not in e  # p untouched -> solver keeps nominal
    assert float(e["q_var"][0]) == 600.0  # 500 + 100


def test_parameter_perturbation_records(grid3):
    s = perturbation_sweep(
        grid3,
        Selector(component="load"),
        Perturbation(field="pq", mode="scale", value=1.2),
    )
    # pq -> 2 fields x 2 scenarios = 4 ground-truth rows
    assert len(s.perturbations) == 4
    p0 = next(
        p
        for p in s.perturbations
        if p.scenario_id == 0 and p.parameter_path == "p_nom_w"
    )
    assert p0.component_kind == "load" and p0.component_id == 10
    assert p0.nominal_value == 2000.0 and abs(p0.perturbed_value - 2400.0) < 1e-9
    assert p0.unit_short == "W"
    q1 = next(
        p
        for p in s.perturbations
        if p.scenario_id == 1 and p.parameter_path == "q_nom_var"
    )
    assert q1.component_id == 11 and abs(q1.perturbed_value - 960.0) < 1e-9  # 800 * 1.2


def test_selector_filters_targets(grid3):
    s = perturbation_sweep(
        grid3,
        Selector(component="load", consumer_type="ev_charging"),
        Perturbation(field="p", mode="scale", value=2.0),
    )
    assert s.n_samples == 1 and set(s.operating_point) == {11}


def test_validators_and_empty():
    with pytest.raises(ValidationError):  # pq requires scale
        Perturbation(field="pq", mode="delta", value=1.0)


def test_empty_selector_raises(grid3):
    with pytest.raises(ValueError, match="matched no in-service"):
        perturbation_sweep(
            grid3,
            Selector(component="generator"),  # grid3 has no generators
            Perturbation(field="p", mode="scale", value=2.0),
        )


def test_run_end_to_end_spread(grid3):
    s = perturbation_sweep(
        grid3,
        Selector(component="load"),
        Perturbation(field="p", mode="scale", value=3.0),
    )
    res = run_scenarios(grid3, s)
    assert res.v.shape == (2, 3)  # [B=2 targets, N=3 nodes]
    # perturbing different nodes yields different voltage profiles
    assert not torch.allclose(res.v[0], res.v[1])
