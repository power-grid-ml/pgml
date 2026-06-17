"""Differentiable Carson/Deri line constants — conductor geometry -> Z(f), Yc(f).

Implements the OpenDSS **DERI** earth model (verified bit-exact vs OpenDSS — see
``references/opendss/carson.md``): series impedance with a complex-penetration-depth
earth return, GMR geometric reactance, and a skin-effect internal resistance (Bessel
``I0/I1`` of a complex argument, via a continued fraction); plus Maxwell potential
coefficients for the shunt capacitance. Neutrals/shield wires are Kron-reduced out.

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

MU0 = 12.56637e-7  # H/m (OpenDSS constant)
E0 = 8.854e-12  # F/m (OpenDSS constant)
_TWO_PI = 2.0 * math.pi


def _cdtype(rdt: torch.dtype) -> torch.dtype:
    return torch.complex128 if rdt == torch.float64 else torch.complex64


def i0_over_i1(z: Tensor, *, terms: int = 40) -> Tensor:
    """``I0(z)/I1(z)`` for complex ``z`` via a continued fraction (autograd-safe).

    ``I1/I0 = 1/(2/z + 1/(4/z + 1/(6/z + ...)))`` evaluated bottom-up; the result is
    its reciprocal. Overflow-free for all ``z`` and differentiable (pure arithmetic).
    Clamped to 1 for ``|z| > 35`` (matches OpenDSS's skin-effect cutoff).
    """
    f = torch.zeros_like(z)
    for k in range(terms, 0, -1):
        f = 1.0 / ((2.0 * k) / z + f)
    ratio = 1.0 / f  # I0/I1
    return torch.where(z.abs() > 35.0, torch.ones_like(ratio), ratio)


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
    alpha = (1.0 + 1j) * torch.sqrt(
        (freqs.to(rdt).reshape(*([1] * rdc.dim()), -1) * MU0) / rdc.unsqueeze(-1)
    ).to(cdt)  # [*B, N, H]
    i0i1 = i0_over_i1(alpha)
    zint = (
        (1.0 + 1j)
        * i0i1
        * torch.sqrt(
            (
                rdc.unsqueeze(-1) * freqs.to(rdt).reshape(*([1] * rdc.dim()), -1) * MU0
            ).to(cdt)
        )
        / 2.0
    )  # [*B, N, H]
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
    (frequency-independent) potential-coefficient matrix then inverts.
    """
    z = kron_reduce(series_impedance(x, y, gmr, rdc, rho, freqs), n_phase)  # [*B,H,P,P]
    p_red = kron_reduce(potential_coefficients(x, y, radius), n_phase)  # [*B,P,P]
    c = (_TWO_PI * E0) * torch.linalg.inv(p_red)  # [*B,P,P]
    return z, c


__all__ = [
    "i0_over_i1",
    "series_impedance",
    "potential_coefficients",
    "kron_reduce",
    "line_constants",
    "MU0",
    "E0",
]
