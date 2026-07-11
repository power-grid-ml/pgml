"""Sampler: reproducibility, distribution correctness, selectors, modes."""

from __future__ import annotations

import math

import pytest
import torch

from pgml.errors import InputError
from pgml.scenarios import (
    Constant,
    LogNormal,
    LogUniform,
    Normal,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    sample,
)


def _cfg(method="sobol", n=128, seed=0, **spec_kw):
    spec_kw.setdefault("name", "load_pq")
    spec_kw.setdefault("selector", Selector(component="load"))
    spec_kw.setdefault("distribution", Uniform(low=0.5, high=1.5))
    return ScenarioConfig(
        n_samples=n, seed=seed, method=method, parameters=[ParameterSpec(**spec_kw)]
    )


# --- distribution icdf closed forms ----------------------------------------
def test_icdf_closed_forms():
    u = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    torch.testing.assert_close(
        Uniform(low=2.0, high=6.0).icdf(u),
        torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64),
    )
    # Normal icdf(0.5) == loc (0 and 1 are clamped, so only check the median).
    assert abs(float(Normal(loc=3.0, scale=2.0).icdf(torch.tensor([0.5]))) - 3.0) < 1e-9
    torch.testing.assert_close(
        Constant(value=7.0).icdf(u), torch.full((3,), 7.0, dtype=torch.float64)
    )


# --- reproducibility --------------------------------------------------------
def test_same_config_same_samples(grid3):
    a = sample(grid3, _cfg(seed=42))
    b = sample(grid3, _cfg(seed=42))
    torch.testing.assert_close(a.samples["load_pq"], b.samples["load_pq"])
    for cid in (10, 11):
        torch.testing.assert_close(
            a.operating_point[cid]["p_w"], b.operating_point[cid]["p_w"]
        )


def test_different_seed_differs(grid3):
    a = sample(grid3, _cfg(seed=1))
    b = sample(grid3, _cfg(seed=2))
    assert not torch.allclose(a.samples["load_pq"], b.samples["load_pq"])


# --- shapes / dimensionality ------------------------------------------------
def test_per_each_dimension_and_range(grid3):
    s = sample(grid3, _cfg())  # per="each" default, 2 loads -> d=2
    assert s.samples["load_pq"].shape == (128, 2)
    factors = s.samples["load_pq"]
    assert factors.min() >= 0.5 and factors.max() <= 1.5
    # scale mode: p_w = factor * nominal
    torch.testing.assert_close(s.operating_point[10]["p_w"], factors[:, 0] * 2000.0)
    torch.testing.assert_close(s.operating_point[11]["q_var"], factors[:, 1] * 800.0)


def test_per_shared_applies_one_sample_to_all(grid3):
    s = sample(grid3, _cfg(per="shared"))
    assert s.samples["load_pq"].shape == (128, 1)
    # both loads scaled by the SAME factor each scenario
    torch.testing.assert_close(
        s.operating_point[10]["p_w"] / 2000.0, s.operating_point[11]["p_w"] / 3000.0
    )


# --- selectors --------------------------------------------------------------
def test_selector_consumer_type(grid3):
    s = sample(
        grid3, _cfg(selector=Selector(component="load", consumer_type="ev_charging"))
    )
    assert set(s.operating_point) == {11}


def test_selector_ids(grid3):
    s = sample(grid3, _cfg(selector=Selector(component="load", ids=[10])))
    assert set(s.operating_point) == {10}


# --- modes / fields ---------------------------------------------------------
def test_absolute_p_only_leaves_q_nominal(grid3):
    s = sample(
        grid3,
        _cfg(field="p", mode="absolute", distribution=Uniform(low=0.0, high=5000.0)),
    )
    e = s.operating_point[10]
    assert "p_w" in e and "q_var" not in e  # q untouched -> solver keeps nominal
    assert e["p_w"].min() >= 0.0 and e["p_w"].max() <= 5000.0


# --- methods ----------------------------------------------------------------
def test_all_methods_produce_valid_batches(grid3):
    for method in ("sobol", "lhs", "independent"):
        s = sample(grid3, _cfg(method=method, n=64))
        assert s.operating_point[10]["p_w"].shape == (64,)
        f = s.samples["load_pq"]
        assert f.min() >= 0.5 and f.max() <= 1.5


def test_sobol_uniform_mean_near_midpoint(grid3):
    # QMC low-discrepancy: empirical mean of a uniform(0.5,1.5) ~ 1.0 for large B.
    s = sample(grid3, _cfg(method="sobol", n=1024))
    assert abs(float(s.samples["load_pq"].mean()) - 1.0) < 1e-6


def test_lognormal_loguniform_icdf_closed_forms():
    u_mid = torch.tensor([0.5], dtype=torch.float64)
    # LogNormal median = exp(loc).
    assert abs(float(LogNormal(loc=1.0, scale=0.5).icdf(u_mid)) - math.exp(1.0)) < 1e-9
    # LogUniform median = geometric mean; endpoints map to low/high.
    lu = LogUniform(low=2.0, high=8.0)
    assert abs(float(lu.icdf(u_mid)) - 4.0) < 1e-9
    ends = lu.icdf(torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(
        ends, torch.tensor([2.0, 8.0], dtype=torch.float64), atol=1e-9, rtol=0.0
    )


# --- overlapping writers -----------------------------------------------------
def test_overlapping_specs_raise(grid3):
    """Two specs varying the same field of the same component are rejected
    (last-writer-wins would desync the recorded samples from the solve)."""
    cfg = ScenarioConfig(
        n_samples=8,
        seed=0,
        method="sobol",
        parameters=[
            ParameterSpec(
                name="all_loads",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
            ),
            ParameterSpec(
                name="one_load",
                selector=Selector(component="load", ids=[10]),
                distribution=Uniform(low=0.5, high=1.5),
            ),
        ],
    )
    with pytest.raises(InputError, match="both vary"):
        sample(grid3, cfg)


def test_disjoint_specs_do_not_raise(grid3):
    """Different fields on the same component are legitimate parallel specs."""
    cfg = ScenarioConfig(
        n_samples=8,
        seed=0,
        method="sobol",
        parameters=[
            ParameterSpec(
                name="p_spec",
                field="p",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
            ),
            ParameterSpec(
                name="q_spec",
                field="q",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
            ),
        ],
    )
    s = sample(grid3, cfg)
    assert "p_w" in s.operating_point[10] and "q_var" in s.operating_point[10]


# --- float/tensor duality through the sampler --------------------------------
def test_tensor_nominal_gradient_flows(grid3):
    """A tensor-valued nameplate P stays on the autograd tape through a
    mode='scale' operating point (the schema's float/tensor duality)."""
    load = next(a for a in grid3.appliances if getattr(a, "id", None) == 10)
    p_leaf = torch.tensor(1234.0, dtype=torch.float64, requires_grad=True)
    load.p_nom_w = p_leaf
    s = sample(
        grid3,
        _cfg(field="p", mode="scale", selector=Selector(component="load", ids=[10])),
    )
    p_col = s.operating_point[10]["p_w"]
    assert p_col.requires_grad, "scale-mode p_w lost the nameplate gradient"
    p_col.sum().backward()
    assert p_leaf.grad is not None
    assert float(p_leaf.grad.abs()) > 0.0
