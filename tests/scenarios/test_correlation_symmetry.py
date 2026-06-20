"""Section 3: latent-factor correlation (Gaussian copula) + per-phase symmetry."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import (
    Correlation,
    LatentFactor,
    Normal,
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    run_scenarios,
    sample,
)


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def _corr_cfg(rho: float, n=8192, method="independent") -> ScenarioConfig:
    """Two loads coupled to one factor with a NORMAL marginal (so the realized values
    are jointly normal with correlation exactly ``rho``)."""
    return ScenarioConfig(
        n_samples=n,
        seed=0,
        method=method,
        factors=[LatentFactor(name="f")],
        parameters=[
            ParameterSpec(
                name="load_pq",
                selector=Selector(component="load"),
                distribution=Normal(loc=1.0, scale=0.2),
                field="p",
                correlation=Correlation(factor="f", rho=rho),
            )
        ],
    )


# --- correlation (single-factor Gaussian copula) ----------------------------
def test_correlation_matches_rho(grid3):
    for rho in (0.0, 0.5, 0.9):
        s = sample(grid3, _corr_cfg(rho))
        c = _corr(s.samples["load_pq"][:, 0], s.samples["load_pq"][:, 1])
        assert abs(c - rho) < 0.015, f"rho={rho} got corr={c}"


def test_correlation_rho1_is_identical(grid3):
    # rho=1 -> the idiosyncratic weight is zero, so every component sees the factor
    # exactly: realized values are identical across components (== per='shared').
    s = sample(grid3, _corr_cfg(1.0, n=256, method="sobol"))
    torch.testing.assert_close(s.samples["load_pq"][:, 0], s.samples["load_pq"][:, 1])


def test_correlation_preserves_marginal(grid3):
    # The copula must not distort the marginal: each component stays ~ Normal(1, 0.2).
    s = sample(grid3, _corr_cfg(0.7))
    col = s.samples["load_pq"][:, 0]
    assert abs(float(col.mean()) - 1.0) < 0.01
    assert abs(float(col.std()) - 0.2) < 0.005


def test_undeclared_factor_raises(grid3):
    cfg = ScenarioConfig(
        n_samples=8,
        parameters=[
            ParameterSpec(
                name="l",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.5),
                correlation=Correlation(factor="missing", rho=0.5),
            )
        ],
    )
    with pytest.raises(ValueError, match="undeclared factor"):
        sample(grid3, cfg)


# --- per-phase symmetry -----------------------------------------------------
def _sym_cfg(symmetry, imbalance=0.0, n=64) -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=n,
        seed=1,
        parameters=[
            ParameterSpec(
                name="l",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.8, high=1.2),
                symmetry=symmetry,
                imbalance=imbalance,
            )
        ],
    )


def test_balanced_writes_totals_not_per_phase(grid_3ph):
    s = sample(grid_3ph, _sym_cfg("balanced"))
    e = s.operating_point[20]
    assert "p_w" in e and "p_per_phase_w" not in e
    assert s.samples["l"].shape == (64, 2)


def test_independent_writes_distinct_per_phase(grid_3ph):
    s = sample(grid_3ph, _sym_cfg("independent"))
    e = s.operating_point[20]
    assert "p_per_phase_w" in e and len(e["p_per_phase_w"]) == 3
    pp = e["p_per_phase_w"]
    # phases drawn independently -> not identical
    assert not torch.allclose(pp[0], pp[1])
    # samples carry the full [B, n_comp, n_phase] per-phase record
    assert s.samples["l"].shape == (64, 2, 3)
    # scale mode: per-phase value = drawn factor * per-phase nominal (3000/3 = 1000)
    torch.testing.assert_close(pp[0], s.samples["l"][:, 0, 0] * 1000.0)


def test_small_imbalance_is_close_but_distinct(grid_3ph):
    s = sample(grid_3ph, _sym_cfg("small_imbalance", imbalance=0.05))
    pp = torch.stack(s.operating_point[21]["p_per_phase_w"], dim=-1)  # [B, 3]
    # per-phase values differ (imbalance) but only slightly: relative spread is small
    rel_spread = (pp.std(dim=-1) / pp.mean(dim=-1)).mean()
    assert 0.0 < float(rel_spread) < 0.15
    assert s.samples["l"].shape == (64, 2)  # records the component base


# --- validators -------------------------------------------------------------
def test_spec_validators():
    base = dict(name="l", selector=Selector(component="load"))
    with pytest.raises(ValidationError):  # pq requires scale
        ParameterSpec(
            **base, distribution=Uniform(low=0.0, high=1.0), field="pq", mode="absolute"
        )
    with pytest.raises(ValidationError):  # correlation + independent
        ParameterSpec(
            **base,
            distribution=Uniform(low=0.0, high=1.0),
            symmetry="independent",
            correlation=Correlation(factor="f", rho=0.5),
        )
    with pytest.raises(ValidationError):  # small_imbalance needs imbalance > 0
        ParameterSpec(
            **base, distribution=Uniform(low=0.0, high=1.0), symmetry="small_imbalance"
        )


# --- end-to-end: per-phase auto-promotes the solve to asymmetric ------------
def test_run_per_phase_promotes_asymmetric(grid_3ph):
    cfg = _sym_cfg("independent", n=8)
    res = run_scenarios(grid_3ph, cfg)  # auto -> asymmetric (per-phase present)
    assert res.v.shape[0] == 8
    # forcing a symmetric calc ignores the per-phase split -> a different solution
    res_sym = run_scenarios(grid_3ph, cfg, symmetry="symmetric")
    assert not torch.allclose(res.v, res_sym.v)


def test_reproducible_with_factors(grid3):
    a = sample(grid3, _corr_cfg(0.6, n=256, method="sobol"))
    b = sample(grid3, _corr_cfg(0.6, n=256, method="sobol"))
    torch.testing.assert_close(a.samples["load_pq"], b.samples["load_pq"])
