"""Positive-sequence-aware harmonic line model (no earth-return floor).

Why this exists
---------------
A single overhead conductor *with earth return* has a self-reactance FLOOR (the Deri
earth term, ~0.4 Ω/km at 50 Hz) that EXCEEDS the positive-sequence reactance ``X1`` of
cables and low-X feeders. Reverse-synthesising a single-conductor geometry from ``R1/X1``
(``synthesis.synthesize_line_geometry``) therefore blows the GMR up past the conductor
radius (non-physical) and can drive the reactance negative at high harmonics.

The physics: a line's phase impedance splits into conductor INTERNAL (skin) + GEOMETRIC
(Maxwell, ∝ ``ln(D/GMR)``) + EARTH-RETURN (Carson/Deri ground path). For a BALANCED
positive-sequence current there is no net ground current, so the earth-return terms
CANCEL — ``Z1`` carries only internal + geometric. Earth return shows up only in ``Z0``
(zero-sequence / ground-return loops). So a positive-sequence line's harmonic impedance
is ``X(h) = X1·h`` (geometric ∝ frequency) with skin effect on ``R1``, and NO earth
floor. This module implements that model two equivalent ways:

* :func:`positive_sequence_z` — direct scaling ``Z1(h) = R1·m_skin(h) + j·X1·(f/f0)``.
* :func:`two_conductor_loop_z` — a PHYSICAL go/return Carson loop (reuses
  :func:`pgml.geometry.carson.series_impedance`) in which the earth term cancels
  analytically; it yields a *physical* GMR/spacing for any ``X1`` and agrees with the
  direct model to the (negligible) residual earth coupling.

:func:`phase_to_sequence` decomposes a full Carson phase-impedance matrix into
``Z0/Z1/Z2`` (Fortescue), used to SHOW that genuine 3-phase geometry keeps earth return
only in ``Z0``.

Everything is torch / autograd-safe / GPU-ready / batched over a leading set of lines
``*B`` and over ``H`` frequencies, so gradients flow ``R1, X1 -> Z1(h) -> Y-bus``.
See ``references/positive_sequence_harmonic_line_model.md`` for the decision record.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .carson import MU0, _cdtype, internal_impedance, series_impedance

# Spacing-reactance coefficient: Im(Lfactor·ln(1/G)) = (f·mu0)·ln(1/G), so
# X_geom = (f·mu0)·ln(D/GMR).  k(f0) = f0·mu0 is the per-conductor coefficient.
_DEFAULT_GMR_OVER_RADIUS = 0.7788  # solid round conductor GMR = r·e^{-1/4}


def _to(v, dtype: torch.dtype, device) -> Tensor:
    """Coerce a python float / array-like / tensor to a tensor (autograd-safe)."""
    if isinstance(v, Tensor):
        return v.to(dtype=dtype, device=device)
    return torch.as_tensor(v, dtype=dtype, device=device)


def _rdtype(freqs: Tensor) -> torch.dtype:
    return freqs.dtype if freqs.is_floating_point() else torch.float64


def fit_equivalent_rdc(r1, f0, freqs_ref: Tensor, *, iters: int = 12) -> Tensor:
    """Equivalent DC resistance ``Rdc`` whose skin model gives ``R1`` at ``f0`` (Ω/m).

    Solves ``Re(Zint(Rdc, f0)) = R1`` by a contraction fixed point
    ``Rdc <- Rdc + (R1 - Re(Zint(Rdc, f0)))`` (``d Re(Zint)/d Rdc -> 1`` as skin
    vanishes, so it converges in a few steps for distribution-line resistances).
    Differentiable w.r.t. ``R1``; ``freqs_ref`` only fixes dtype/device.
    """
    rdt = _rdtype(freqs_ref)
    dev = freqs_ref.device
    r1t = _to(r1, rdt, dev)
    f0t = _to(f0, rdt, dev).reshape(())
    rdc = r1t  # skin is negligible at f0 for these R, so R1 is a good seed
    for _ in range(iters):
        re = internal_impedance(rdc, f0t.reshape(1))[..., 0].real
        rdc = torch.clamp(rdc + (r1t - re), min=1e-9)
    return rdc


def skin_resistance_multiplier(r1, f0, freqs: Tensor) -> Tensor:
    """Skin-effect resistance multiplier ``m(h)`` ``[*B, H]`` (``m(f0) = 1`` exactly).

    ``m(f) = Re(Zint(Rdc, f)) / Re(Zint(Rdc, f0))`` with ``Rdc`` fit from ``R1`` at
    ``f0`` (:func:`fit_equivalent_rdc`). This is the SAME Bessel ``I0/I1`` internal
    resistance the Carson geometry path uses, with the earth-return term DROPPED — the
    positive-sequence resistance growth without the earth floor. NO earth return.
    """
    rdt = _rdtype(freqs)
    dev = freqs.device
    f = freqs.to(rdt).reshape(-1)  # [H]
    f0t = _to(f0, rdt, dev).reshape(1)
    rdc = fit_equivalent_rdc(r1, f0, freqs)  # [*B]
    r_f = internal_impedance(rdc, f).real  # [*B, H]
    r_0 = internal_impedance(rdc, f0t).real  # [*B, 1]
    return r_f / r_0


def positive_sequence_z(r1, x1, f0, freqs: Tensor, *, skin: bool = True) -> Tensor:
    """Positive-sequence harmonic series impedance ``Z1(h)`` ``[*B, H]`` (Ω/m or Ω).

    ``Z1(h) = R1·m_skin(h) + j·X1·(f/f0)`` — geometric reactance scales ∝ frequency
    and the resistance grows with the skin-effect multiplier (``skin=False`` keeps
    ``R`` constant, the naive model). There is NO earth-return floor, so the model is
    well defined for every ``X1`` (cables included) and never produces a non-physical
    GMR. Differentiable w.r.t. ``R1`` and ``X1``; batched over lines and ``H``.
    """
    rdt = _rdtype(freqs)
    dev = freqs.device
    f = freqs.to(rdt).reshape(-1)  # [H]
    f0t = _to(f0, rdt, dev).reshape(())
    r1t = _to(r1, rdt, dev)
    x1t = _to(x1, rdt, dev)
    hb = f / f0t  # [H] harmonic order f/f0
    x = x1t.unsqueeze(-1) * hb  # [*B, H]
    if skin:
        r = r1t.unsqueeze(-1) * skin_resistance_multiplier(r1, f0, f)  # [*B, H]
    else:
        r = r1t.unsqueeze(-1) * torch.ones_like(hb)
    return torch.complex(r, x).to(_cdtype(rdt))


def two_conductor_geometry(
    r1, x1, f0, *, radius_m: float, height_m: float = 10.0, gmr_m=None
) -> dict:
    """Physical go/return conductor pair reproducing ``R1 + jX1`` at ``f0`` (no floor).

    Models the positive-sequence loop as two identical conductors (a ``+I`` go and a
    ``-I`` return) a distance ``D`` apart. For the ``+I/-I`` loop the earth-return
    terms cancel, leaving ``X1 = 2·(f0·mu0)·ln(D/GMR)`` and ``R1 = 2·Re(Zint(Rdc,f0))``.
    GMR is FIXED to a physical value (``0.7788·radius`` by default) and the SPACING is
    solved as ``D = GMR·exp(X1 / (2·f0·mu0))`` — finite and physical for ANY ``X1``,
    unlike the single-conductor earth-return synthesis whose GMR blows past the radius.

    Returns a dict of python floats: ``gmr_m``, ``rdc_ohm_per_m``, ``spacing_m``,
    ``radius_m``, ``height_m`` (the conductor data for :func:`two_conductor_loop_z`).
    """
    f0 = float(f0)
    r1 = float(r1)
    x1 = float(x1)
    gmr = float(gmr_m) if gmr_m is not None else _DEFAULT_GMR_OVER_RADIUS * radius_m
    coef = 2.0 * f0 * MU0  # X1 = coef·ln(D/GMR)
    spacing = gmr * math.exp(x1 / coef)
    # Each conductor carries R1/2; fit its Rdc from the skin model at f0.
    rdc = float(
        fit_equivalent_rdc(0.5 * r1, f0, torch.tensor([f0], dtype=torch.float64))
    )
    return {
        "gmr_m": gmr,
        "rdc_ohm_per_m": rdc,
        "spacing_m": spacing,
        "radius_m": float(radius_m),
        "height_m": float(height_m),
    }


def two_conductor_loop_z(geom: dict, freqs: Tensor, *, rho=100.0) -> Tensor:
    """Carson go/return loop impedance ``Z(h)`` ``[H]`` (Ω/m) for a :func:`two_conductor_geometry`.

    Builds the full 2x2 Carson/Deri impedance (earth return included) for the go/return
    pair and applies the loop transform ``[1, -1]·Z·[1, -1]^T`` — for the ``+I/-I``
    current the large earth penetration-depth term is common to all four entries and
    CANCELS, so the result is the earth-floor-free positive-sequence impedance. Verifies,
    via the bit-exact Carson code, that the direct :func:`positive_sequence_z` model is
    physically grounded.
    """
    rdt = _rdtype(freqs)
    dev = freqs.device
    x = torch.tensor([0.0, geom["spacing_m"]], dtype=rdt, device=dev)
    y = torch.tensor([geom["height_m"], geom["height_m"]], dtype=rdt, device=dev)
    gmr = torch.tensor([geom["gmr_m"], geom["gmr_m"]], dtype=rdt, device=dev)
    rdc = torch.tensor(
        [geom["rdc_ohm_per_m"], geom["rdc_ohm_per_m"]], dtype=rdt, device=dev
    )
    z = series_impedance(x, y, gmr, rdc, rho, freqs.to(rdt))  # [H, 2, 2]
    # Loop transform t = [1, -1]: Z_loop = z00 - z01 - z10 + z11.
    return z[..., 0, 0] - z[..., 0, 1] - z[..., 1, 0] + z[..., 1, 1]


def fortescue_matrix(dtype: torch.dtype, device) -> Tensor:
    """Symmetrical-component transform ``A`` (3x3 complex), columns = 0/+/- sequences."""
    a = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    rows = [
        [1.0 + 0j, 1.0 + 0j, 1.0 + 0j],
        [1.0 + 0j, a * a, a],
        [1.0 + 0j, a, a * a],
    ]
    return torch.tensor(rows, dtype=dtype, device=device)


def phase_to_sequence(z_phase: Tensor) -> Tensor:
    """Sequence impedance matrix ``A^{-1} Z_phase A`` ``[*, 3, 3]`` (0/+/- on the diagonal).

    For a 3-phase phase-domain impedance ``[*, 3, 3]`` (e.g. Carson + Kron), returns the
    symmetrical-component matrix; ``[..., 0, 0]=Z0``, ``[..., 1, 1]=Z1``, ``[..., 2,
    2]=Z2``. A balanced/transposed line is diagonal. Use :func:`sequence_impedances`
    for just the diagonal.
    """
    if z_phase.shape[-1] != 3 or z_phase.shape[-2] != 3:
        raise ValueError(
            f"phase_to_sequence expects a 3x3 phase matrix, got {z_phase.shape}"
        )
    a = fortescue_matrix(z_phase.dtype, z_phase.device)
    a_inv = torch.linalg.inv(a)
    return a_inv @ z_phase.to(a.dtype) @ a


def sequence_impedances(z_phase: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(Z0, Z1, Z2)`` diagonal sequence impedances ``[*]`` from a 3x3 phase matrix."""
    zseq = phase_to_sequence(z_phase)
    return zseq[..., 0, 0], zseq[..., 1, 1], zseq[..., 2, 2]


__all__ = [
    "fit_equivalent_rdc",
    "skin_resistance_multiplier",
    "positive_sequence_z",
    "two_conductor_geometry",
    "two_conductor_loop_z",
    "fortescue_matrix",
    "phase_to_sequence",
    "sequence_impedances",
]
