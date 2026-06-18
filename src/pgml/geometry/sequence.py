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

from pgml.config import get as _cfg

from .carson import MU0, _cdtype, internal_impedance, series_impedance

# Spacing-reactance coefficient: Im(Lfactor·ln(1/G)) = (f·mu0)·ln(1/G), so
# X_geom = (f·mu0)·ln(D/GMR).  k(f0) = f0·mu0 is the per-conductor coefficient.
# Defaults are sourced from the top-level config (`pgml.config` / defaults.yaml) so every
# modeling constant is documented in one place; these module names are the config values.
_DEFAULT_GMR_OVER_RADIUS = _cfg("line.conductor.gmr_over_radius")  # e^{-1/4}


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


# ---------------------------------------------------------------------------
# Sequence-aware harmonic model (UNBALANCED / 4-wire studies)
# ---------------------------------------------------------------------------
# Carson's earth-return resistance is geometry-INDEPENDENT: Re(f) = mu0*omega/8 =
# pi^2 * f * 1e-7 (Ohm/m), growing linearly with frequency. It is the dominant
# frequency-dependent DAMPING of zero-sequence (ground-loop) harmonics, and it does
# NOT appear in the positive sequence. (OpenDSS exposes the same physics through its
# per-LineCode Rg/Xg parameters; its defaults are calibrated for imperial length units,
# so on a metric line they are ~3.28x smaller than this physical per-metre value — hence
# the coefficient is configurable here and in `pgml.config`.)
CARSON_EARTH_R_PER_HZ = _cfg(
    "line.earth_return.resistance_coeff_ohm_per_m_per_hz"
)  # Ohm/m per Hz (= pi^2 * 1e-7)


def carson_earth_resistance(freqs: Tensor, *, coeff: float = CARSON_EARTH_R_PER_HZ):
    """Carson earth-return resistance ``Re(f) = coeff*f`` (Ω/m), one phase (∝ frequency)."""
    return coeff * freqs.to(_rdtype(freqs))


def zero_sequence_harmonic_z(
    r0,
    x0,
    f0,
    freqs: Tensor,
    *,
    skin: bool = True,
    earth_resistance_coeff: float = CARSON_EARTH_R_PER_HZ,
) -> Tensor:
    """Zero-sequence harmonic impedance ``Z0(h)`` ``[*B, H]`` (Ω/m or Ω).

    The conductor part scales like the positive sequence (``X0 ∝ h``, optional skin on
    ``R0``); ADDED to it is the Carson earth/ground-loop resistance growth
    ``3·(Re(f) − Re(f0))`` (``Re`` per :func:`carson_earth_resistance`), the
    frequency-dependent DAMPING that a balanced positive-sequence current never sees.
    The growth is ``≥ 0`` and monotone, so ``Z0(h)`` can never become non-physical.
    Set ``earth_resistance_coeff=0`` to recover a pure conductor (no earth) zero
    sequence. (The earth-return REACTANCE sub-linearity is geometry / return-path
    dependent — deep earth vs nearby neutral/sheath — and is left to the full Carson
    geometry path; here ``X0`` scales ∝ h.) Differentiable in ``R0``/``X0``; batched.
    """
    z = positive_sequence_z(r0, x0, f0, freqs, skin=skin)  # [*B, H] conductor-like
    rdt = _rdtype(freqs)
    f = freqs.to(rdt).reshape(-1)  # [H]
    f0t = _to(f0, rdt, freqs.device).reshape(())
    d_re = (3.0 * earth_resistance_coeff) * (f - f0t)  # [H] Ω/m, >= 0
    return z + d_re.to(z.dtype)


def sequence_to_phase_z(z1: Tensor, z0: Tensor) -> Tensor:
    """Phase impedance matrix ``Z_abc`` ``[*, 3, 3]`` from sequence ``Z1`` (=Z2) and ``Z0``.

    ``Z_self = (Z0 + 2·Z1)/3`` on the diagonal, ``Z_mutual = (Z0 − Z1)/3`` off-diagonal
    — the inverse Fortescue for a balanced/transposed line (``Z2 = Z1``). The earth
    return, carried only by ``Z0``, therefore appears in BOTH the self and mutual terms,
    coupling the phases; it cancels again in any balanced (positive-sequence) current.
    ``z1``/``z0`` are complex ``[*B, H]`` -> ``[*B, H, 3, 3]``.
    """
    zs = (z0 + 2.0 * z1) / 3.0
    zm = (z0 - z1) / 3.0
    eye = torch.eye(3, dtype=z1.dtype, device=z1.device)
    off = 1.0 - eye
    return zs[..., None, None] * eye + zm[..., None, None] * off


def sequence_aware_phase_z(
    r1,
    x1,
    r0,
    x0,
    f0,
    freqs: Tensor,
    *,
    skin: bool = True,
    earth_resistance_coeff: float = CARSON_EARTH_R_PER_HZ,
) -> Tensor:
    """Full coupled phase impedance ``Z_abc(h)`` ``[*B, H, 3, 3]`` for UNBALANCED studies.

    Combines an earth-free positive sequence ``Z1(h)`` (:func:`positive_sequence_z`,
    ``X1 ∝ h`` + skin) with a damped zero sequence ``Z0(h)``
    (:func:`zero_sequence_harmonic_z`, conductor + Carson earth-return resistance) and
    recombines them (:func:`sequence_to_phase_z`). An unbalanced / zero-sequence current
    then sees the earth-return damping in ``Z0``, while a balanced positive-sequence
    current still sees only the earth-free ``Z1``. This is the model an asymmetric
    4-wire LV harmonic study needs; differentiable in ``R1/X1/R0/X0``, batched over
    lines and ``H``.
    """
    z1 = positive_sequence_z(r1, x1, f0, freqs, skin=skin)
    z0 = zero_sequence_harmonic_z(
        r0, x0, f0, freqs, skin=skin, earth_resistance_coeff=earth_resistance_coeff
    )
    return sequence_to_phase_z(z1, z0)


__all__ = [
    "fit_equivalent_rdc",
    "skin_resistance_multiplier",
    "positive_sequence_z",
    "two_conductor_geometry",
    "two_conductor_loop_z",
    "fortescue_matrix",
    "phase_to_sequence",
    "sequence_impedances",
    "carson_earth_resistance",
    "zero_sequence_harmonic_z",
    "sequence_to_phase_z",
    "sequence_aware_phase_z",
]
