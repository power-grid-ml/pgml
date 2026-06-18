"""Differentiability of the positive-sequence harmonic line model (float64 gradcheck).

Gradients must flow from the line's R1/X1 through the positive-sequence impedance
``Z1(h)`` (skin-effect resistance + geometric reactance, no earth floor), and through a
full harmonic Y-bus assembly using the ``carson_skin_multiplier`` resistance law — so
the corrected model supports gradient-based parameter recovery just like the Carson
geometry path.
"""

from __future__ import annotations

import torch

from pgml.assembly import assemble_network_ybus
from pgml.geometry.carson import series_impedance
from pgml.geometry.sequence import (
    positive_sequence_z,
    skin_resistance_multiplier,
    two_conductor_geometry,
)
from pgml.geometry.synthesis import apply_positive_sequence_harmonic_model

from tests.fixtures.tiny_grids import single_phase_chain

RDT = torch.float64
CDT = torch.complex128


def test_gradcheck_positive_sequence_z():
    """Z1(h) is differentiable w.r.t. R1 and X1."""
    freqs = torch.tensor([50.0, 250.0, 650.0], dtype=RDT)

    def fn(r1, x1):
        return positive_sequence_z(r1, x1, 50.0, freqs).abs().sum()

    args = (
        torch.tensor(3.6e-4, dtype=RDT, requires_grad=True),
        torch.tensor(3.0e-4, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_skin_multiplier():
    """The skin-effect multiplier m(h) is differentiable w.r.t. R1 (the fit Rdc)."""
    freqs = torch.tensor([50.0, 350.0, 750.0], dtype=RDT)

    def fn(r1):
        return skin_resistance_multiplier(r1, 50.0, freqs).sum()

    args = (torch.tensor(4.0e-4, dtype=RDT, requires_grad=True),)
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_two_conductor_loop_batched():
    """The physical go/return Carson loop is differentiable w.r.t. geometry."""
    geom = two_conductor_geometry(3.6e-4, 3.0e-4, 50.0, radius_m=0.0102)
    freqs = torch.tensor([50.0, 350.0], dtype=RDT)

    def fn(gmr, rdc, spacing):
        x = torch.stack([torch.zeros((), dtype=RDT), spacing])
        y = torch.full((2,), geom["height_m"], dtype=RDT)
        gmrv = torch.stack([gmr, gmr])
        rdcv = torch.stack([rdc, rdc])
        z = series_impedance(x, y, gmrv, rdcv, 100.0, freqs)
        zl = z[..., 0, 0] - z[..., 0, 1] - z[..., 1, 0] + z[..., 1, 1]
        return zl.abs().sum()

    args = (
        torch.tensor(geom["gmr_m"], dtype=RDT, requires_grad=True),
        torch.tensor(geom["rdc_ohm_per_m"], dtype=RDT, requires_grad=True),
        torch.tensor(geom["spacing_m"], dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradient_through_assemble_with_skin_law():
    """R-gradient flows through assemble at a harmonic with the skin multiplier active."""
    grid = single_phase_chain()
    apply_positive_sequence_harmonic_model(grid)
    line_id = next(
        b.id for b in grid.branches if hasattr(b, "series_resistance_ohm_per_m")
    )

    r_leaf = torch.tensor([[1.0e-3]], dtype=RDT, requires_grad=True)
    overrides = {("line", line_id, "series_resistance_ohm_per_m"): r_leaf}

    yb = assemble_network_ybus(
        grid, torch.tensor([550.0], dtype=RDT), dtype=CDT, param_overrides=overrides
    ).Y  # 11th harmonic -> skin multiplier > 1
    yb.abs().sum().backward()
    assert (
        r_leaf.grad is not None
        and torch.isfinite(r_leaf.grad).all()
        and r_leaf.grad.abs().sum() > 0
    )
