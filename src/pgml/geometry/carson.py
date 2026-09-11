"""Differentiable Carson/Deri line constants — conductor geometry -> Z(f), Yc(f).

Implements the OpenDSS DERI earth model (see
``docs/pgml/modeling/references/opendss/carson.md``): series impedance with a
complex-penetration-depth earth return, GMR geometric reactance, and a skin-effect
internal resistance (Bessel ``I0/I1`` of a complex argument, via a continued fraction);
plus Maxwell potential coefficients for the shunt capacitance. Neutrals/shield wires are
Kron-reduced out.

Agreement with OpenDSS on the same geometry: ``Z(f)`` to ~5e-8 relative and ``C`` to
~2.1e-5 relative, both set by the physical constants — this module uses the SI values of
``mu0`` and ``e0`` while OpenDSS truncates them (see :data:`MU0`, :data:`E0`). The MODEL
is the same to floating point.

Scope of that agreement: BELOW 1 kHz. OpenDSS's geometry line model switches the
conductor's geometric mean radius for its physical radius at exactly 1 kHz (its
`LineGeometry` code selects ``GMR`` only while ``f < 1000 Hz``, on the argument that the
current has crowded into the conductor surface above that), and pgml always uses the
published ``GMR``. Above 1 kHz the two therefore differ by the geometric reactance
``(f*mu0)*ln(radius/GMR)`` per conductor, which is an intentional, documented difference
and not a defect on either side.

Everything is torch and autograd-safe (complex ``sqrt``/``log``, no ``.item()`` /
control flow on tensor values), batched over a leading set of lines ``*B`` and over
``H`` frequencies, and device/dtype-agnostic. Gradients flow from conductor geometry
(x, y, GMR, Rdc, radius, earth resistivity) -> Z/Yc -> Y-bus -> solve -> outputs,
which is the whole point of the geometry path (parameter recovery, geometry tuning).

Shapes: conductor arrays are ``[*B, N]`` (N conductors, phases first then neutrals),
frequencies ``[H]``. ``series_impedance`` returns ``Z[*B, H, N, N]`` (Ω/m);
reduction keeps the first ``n_phase`` conductors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: Vacuum permeability (SI, H/m). OpenDSS truncates it to ``12.56637e-7``, which is
#: 4.9e-8 smaller in relative terms; `series_impedance` is linear in ``mu0`` (and
#: square-root in it through the earth penetration depth and the skin term), so the
#: constant alone moves ``Z`` by ~5e-8 relative — the floor of the OpenDSS comparison.
MU0 = 4.0e-7 * math.pi
#: Vacuum permittivity (SI, F/m). OpenDSS truncates it to ``8.854e-12``, which is
#: 2.1e-5 smaller in relative terms, and ``C = 2*pi*e0*inv(P)`` is linear in it.
E0 = 8.8541878128e-12
_TWO_PI = 2.0 * math.pi


def _cdtype(rdt: torch.dtype) -> torch.dtype:
    return torch.complex128 if rdt == torch.float64 else torch.complex64


def i0_over_i1(z: Tensor, *, terms: int = 40) -> Tensor:
    """``I0(z)/I1(z)`` for complex ``z`` via a continued fraction (autograd-safe).

    ``I1/I0 = 1/(2/z + 1/(4/z + 1/(6/z + ...)))`` evaluated bottom-up; the result is
    its reciprocal. Overflow-free for all ``z`` and differentiable (pure arithmetic).
    Clamped to 1 for ``|z| > 35`` (matches OpenDSS's skin-effect cutoff).
    """
    # Both branches of a `torch.where` participate in the backward pass, so the
    # continued fraction must never see the huge arguments the cutoff masks out:
    # its truncated evaluation can produce non-finite intermediates there, which
    # would poison the gradient of `z` even though the forward value is clamped.
    clamped = z.abs() > 35.0
    z_safe = torch.where(clamped, torch.ones_like(z), z)
    f = torch.zeros_like(z_safe)
    for k in range(terms, 0, -1):
        f = 1.0 / ((2.0 * k) / z_safe + f)
    ratio = 1.0 / f  # I0/I1
    return torch.where(clamped, torch.ones_like(ratio), ratio)


def internal_impedance(rdc: Tensor, freqs: Tensor) -> Tensor:
    """Skin-effect internal impedance ``Zint(f)`` ``[*B, H]`` (Ω/m), Bessel I0/I1 model.

    ``alpha = (1+j)*sqrt(f*mu0/Rdc)``; ``Zint = (1+j)*(I0/I1)(alpha)*sqrt(Rdc*f*mu0)/2``.
    ``rdc`` carries a trailing batch shape ``[*B]`` (e.g. one entry per conductor) and
    ``freqs`` is ``[H]``; ``rdc`` broadcasts against an appended frequency axis so the
    result is ``[*B, H]``. Both the internal RESISTANCE (real part) and the internal
    REACTANCE (imag part) are returned; OpenDSS (and :func:`series_impedance`) keep only
    the real part in the 40-1000 Hz band (the internal inductance is carried by GMR),
    while the positive-sequence model (:mod:`pgml.geometry.sequence`) reuses the real
    part for the skin-effect resistance growth. Pure torch / autograd-safe.
    """
    rdt = rdc.dtype
    cdt = _cdtype(rdt)
    rdc_f = rdc.unsqueeze(-1)  # [*B, 1]
    f = freqs.to(rdt)  # [H]
    alpha = (1.0 + 1j) * torch.sqrt((f * MU0) / rdc_f).to(cdt)  # [*B, H]
    i0i1 = i0_over_i1(alpha)
    return (1.0 + 1j) * i0i1 * torch.sqrt((rdc_f * f * MU0).to(cdt)) / 2.0  # [*B, H]


def _pair_terms(x: Tensor, y: Tensor):
    """Pairwise distance helpers ``[*B, N, N]``: |Δ|, image dist, (yi+yj), (xi−xj).

    The diagonal of ``dist`` is forced to 1 (via ``+ eye``) so the ``sqrt(0)`` on the
    diagonal does not produce a NaN gradient; every consumer masks the diagonal out.
    """
    xi, xj = x.unsqueeze(-1), x.unsqueeze(-2)
    yi, yj = y.unsqueeze(-1), y.unsqueeze(-2)
    dx = xi - xj
    dy = yi - yj
    n = x.shape[-1]
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    dist = torch.sqrt(dx * dx + dy * dy + eye)  # diagonal -> 1 (masked downstream)
    image = torch.sqrt(
        dx * dx + (yi + yj) * (yi + yj)
    )  # to mirror image (diag = 2*y_i)
    return dist, image, yi + yj, dx


def series_impedance(
    x: Tensor, y: Tensor, gmr: Tensor, rdc: Tensor, rho, freqs: Tensor
) -> Tensor:
    """Carson/Deri series impedance ``Z[*B, H, N, N]`` in Ω/m (unreduced).

    ``Z_ii = Re(Zint(Rdc, f)) + Lfactor·ln(1/GMR_i) + Ze``,
    ``Z_ij = Lfactor·ln(1/D_ij) + Ze`` with ``Lfactor = j·2πf·mu0/(2π)`` and the Deri
    earth term ``Ze = Lfactor·ln(√((y_i+y_j+2/Fme)² + (x_i−x_j)²))``,
    ``Fme = √(j·2πf·mu0/rho)``.

    The conductor spacing term uses the published ``gmr`` at EVERY frequency. OpenDSS
    does the same below 1 kHz and changes the term at exactly 1 kHz (toward the physical
    radius, on the argument that the current has crowded into the conductor surface), so
    the parity with OpenDSS stated in the module docstring holds BELOW 1 kHz; at 1050 Hz
    the measured deviation steps to ~1e-2 relative on a typical ACSR geometry.

    Parameters
    ----------
    x, y, gmr, rdc:
        Conductor horizontal position, height, geometric-mean radius and DC
        resistance per metre — real tensors ``[*B, N]``.
    rho:
        Earth resistivity (Ω·m), scalar or ``[*B]``-broadcastable.
    freqs:
        Absolute frequencies ``[H]`` (Hz).
    """
    rdt = x.dtype
    cdt = _cdtype(rdt)
    fw = (_TWO_PI * freqs).to(rdt)  # [H]
    lfactor = (1j * MU0 / _TWO_PI) * fw.to(cdt)  # [H] complex coefficient
    rho_t = torch.as_tensor(rho, dtype=rdt, device=x.device)

    # Earth complex penetration: Fme = sqrt(j*Fw*mu0/rho) ; p2 = 2/Fme.  [*B?, H]
    fme = torch.sqrt(1j * (fw * MU0).to(cdt) / rho_t.to(cdt).unsqueeze(-1))
    p2 = 2.0 / fme  # [..., H]

    # Geometric spacing matrix G: off-diagonal = distance, diagonal = GMR.
    dist, _image, ysum, dx = _pair_terms(x, y)
    n = x.shape[-1]
    eye = torch.eye(n, dtype=rdt, device=x.device)
    g_mat = dist * (1.0 - eye) + gmr.unsqueeze(-1) * eye  # [*B, N, N]
    spacing = lfactor.reshape(*([1] * (g_mat.dim() - 2)), -1, 1, 1) * torch.log(
        (1.0 / g_mat).to(cdt)
    ).unsqueeze(-3)  # [*B, H, N, N]

    # Earth-return term Ze = Lfactor * log( sqrt(hterm^2 + xterm^2) ).
    hterm = ysum.unsqueeze(-3).to(cdt) + p2[..., :, None, None]  # [*B, H, N, N]
    xterm = dx.unsqueeze(-3).to(cdt)
    lnarg = torch.sqrt(hterm * hterm + xterm * xterm)
    ze = lfactor.reshape(*([1] * (lnarg.dim() - 3)), -1, 1, 1) * torch.log(lnarg)

    # Skin-effect internal resistance (Rdc, Bessel I0/I1); internal reactance dropped.
    zint = internal_impedance(rdc, freqs)  # [*B, N, H]
    zint_re = zint.real.movedim(-1, -2)  # [*B, H, N]
    diag_int = torch.diag_embed(zint_re).to(cdt)  # [*B, H, N, N]

    return spacing + ze + diag_int


def potential_coefficients(x: Tensor, y: Tensor, radius: Tensor) -> Tensor:
    """Maxwell potential-coefficient matrix ``P[*B, N, N]`` (real, unreduced).

    ``P[i,i] = ln(2 y_i / r_i)``, ``P[i,j] = ln(D'_ij / D_ij)`` (image method).
    ``C = 2*pi*e0 * inv(P)``.
    """
    dist, image, _ysum, _dx = _pair_terms(x, y)
    n = x.shape[-1]
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    self_p = torch.log(2.0 * y / radius)  # [*B, N]
    ratio = image / dist  # diagonal (masked) is safe: dist diagonal = 1
    p = torch.log(ratio) * (1.0 - eye) + torch.diag_embed(self_p)
    return p


def kron_reduce(m: Tensor, n_phase: int) -> Tensor:
    """Kron-reduce ``[*, N, N]`` to ``[*, n_phase, n_phase]`` (eliminate conductors >= n_phase)."""
    a = m[..., :n_phase, :n_phase]
    b = m[..., :n_phase, n_phase:]
    c = m[..., n_phase:, :n_phase]
    d = m[..., n_phase:, n_phase:]
    if d.shape[-1] == 0:
        return a
    return a - b @ torch.linalg.solve(d, c)


def line_constants(
    x: Tensor,
    y: Tensor,
    gmr: Tensor,
    rdc: Tensor,
    radius: Tensor,
    rho,
    freqs: Tensor,
    n_phase: int,
) -> tuple[Tensor, Tensor]:
    """Phase-reduced ``(Z[*B, H, P, P] Ω/m, C[*B, P, P] F/m)`` for line geometry.

    Series ``Z`` is Kron-reduced per frequency; capacitance ``C`` reduces the
    (frequency-independent) potential-coefficient matrix then inverts. Agreement with
    OpenDSS on the same geometry, measured: ``Z`` to 4.8e-8 relative BELOW 1 kHz (the
    ``mu0`` constant; see :func:`series_impedance` for the 1 kHz scope) and ``C`` to
    2.1212e-5 relative (the ``e0`` constant).
    """
    z = kron_reduce(series_impedance(x, y, gmr, rdc, rho, freqs), n_phase)  # [*B,H,P,P]
    p_red = kron_reduce(potential_coefficients(x, y, radius), n_phase)  # [*B,P,P]
    c = (_TWO_PI * E0) * torch.linalg.inv(p_red)  # [*B,P,P]
    return z, c


__all__ = [
    "i0_over_i1",
    "internal_impedance",
    "series_impedance",
    "potential_coefficients",
    "kron_reduce",
    "line_constants",
    "MU0",
    "E0",
]
