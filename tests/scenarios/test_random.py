"""Tests for public scenario random processes."""

from __future__ import annotations

import pytest
import torch

from pgml.scenarios import ar1_noise


def test_ar1_reproducible_and_stationary() -> None:
    first = ar1_noise((20_000, 4), 0.75, torch.Generator().manual_seed(41))
    second = ar1_noise((20_000, 4), 0.75, torch.Generator().manual_seed(41))

    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    assert float(first.var(dim=0).mean()) == pytest.approx(1.0, abs=0.04)
    correlation = torch.corrcoef(torch.stack((first[:, 1], first[:, 2])))[0, 1]
    assert float(correlation) == pytest.approx(0.75, abs=0.03)


def test_ar1_tensor_rho_broadcast_and_gradient() -> None:
    rho = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    draw = ar1_noise((2, 5), rho, torch.Generator().manual_seed(7))
    draw.square().sum().backward()

    assert draw.shape == (2, 5)
    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()


def test_ar1_rejects_invalid_shape_and_device_mismatch() -> None:
    with pytest.raises(ValueError, match="positive last dimension"):
        ar1_noise((2, 0), 0.5, torch.Generator())
    with pytest.raises(ValueError, match="generator is on cpu"):
        ar1_noise((2, 3), 0.5, torch.Generator(), device="meta")
    with pytest.raises(ValueError, match="rho is on meta"):
        ar1_noise((2, 3), torch.empty(2, device="meta"), torch.Generator())


def test_removed_recipe_names_explain_themselves() -> None:
    import pgml.scenarios as scenarios

    for name in ("CoherentSpectrumConfig", "affine_emission_correction"):
        with pytest.raises(ImportError, match="scenario recipe"):
            getattr(scenarios, name)
    with pytest.raises(ImportError, match=r"pgml\.dispatch"):
        scenarios.dispatch_storage
