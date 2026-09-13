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

Scope of that agreement: it depends on the ``internal_inductance`` option. A published
GMR is a POWER-FREQUENCY quantity — it folds the conductor's internal inductance
(``mu0/8pi`` for a solid round conductor, i.e. ``GMR = e^(-1/4)*radius``) into one
equivalent radius — and skin effect makes that internal inductance decay with frequency.
OpenDSS therefore keeps the published GMR only while ``40 Hz < f < 1 kHz`` and uses the
physical radius plus the full Bessel internal impedance outside that band. The default
here (``internal_inductance="gmr"``) keeps the published GMR at every frequency, so it
matches OpenDSS BELOW 1 kHz; ``internal_inductance="gmr_power_frequency"`` reproduces
OpenDSS at every frequency. See :func:`series_impedance` for all four options and their
scope.

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
from typing import Optional

import torch
from torch import Tensor

from pgml.defaults import get as _cfg
from pgml.errors import InputError

#: Vacuum permeability (SI, H/m). OpenDSS truncates it to ``12.56637e-7``, which is
#: 4.9e-8 smaller in relative terms; `series_impedance` is linear in ``mu0`` (and
#: square-root in it through the earth penetration depth and the skin term), so the
#: constant alone moves ``Z`` by ~5e-8 relative — the floor of the OpenDSS comparison.
MU0 = 4.0e-7 * math.pi
#: Vacuum permittivity (SI, F/m). OpenDSS truncates it to ``8.854e-12``, which is
#: 2.1e-5 smaller in relative terms, and ``C = 2*pi*e0*inv(P)`` is linear in it.
E0 = 8.8541878128e-12
_TWO_PI = 2.0 * math.pi

#: Internal-inductance models of the geometry path, see :func:`series_impedance`.
INTERNAL_INDUCTANCE_MODELS = ("gmr", "gmr_skin", "gmr_power_frequency", "bessel")
#: Model applied when a caller passes nothing (modeling default
#: ``line.geometry.internal_inductance``).
INTERNAL_INDUCTANCE = _cfg("line.geometry.internal_inductance")
#: Exclusive band ``(low, high)`` in Hz in which ``"gmr_power_frequency"`` keeps the
#: published GMR (modeling default ``line.geometry.power_frequency_band_hz``; OpenDSS
#: hard-codes 40 Hz and 1 kHz).
POWER_FREQUENCY_BAND_HZ = tuple(_cfg("line.geometry.power_frequency_band_hz"))


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


def _x_int_dc(freqs: Tensor, rdt: torch.dtype) -> Tensor:
    """Power-frequency internal reactance of a solid round conductor ``[H]`` (Ohm/m).

    ``X_int(f) = 2*pi*f * mu0/(8*pi) = f*mu0/4`` — the low-frequency (uniform current
    density) limit of ``Im(Zint)``, and exactly the reactance a published
    ``GMR = e^(-1/4)*radius`` adds through ``(f*mu0)*ln(radius/GMR)``.
    """
    return freqs.to(rdt) * (MU0 / 4.0)


def internal_reactance_ratio(rdc: Tensor, freqs: Tensor) -> Tensor:
    """Internal-inductance decay ``g(f) = Im(Zint(f)) / (f*mu0/4)`` ``[*B, H]``.

    The conductor's internal inductance divided by its uniform-current-density value:
    ``g -> 1`` as ``f -> 0`` and ``g -> 0`` as skin effect confines the current to the
    surface. ``rdc`` is ``[*B]`` (Ohm/m), ``freqs`` is ``[H]`` (Hz, strictly positive).
    Dimensionless, differentiable in ``rdc``, device/dtype from the inputs. Used by the
    ``"gmr_skin"`` model of :func:`series_impedance` and as the physical yardstick for
    how far a power-frequency GMR is off at a given harmonic.
    """
    zim = internal_impedance(rdc, freqs).imag  # [*B, H]
    return zim / _x_int_dc(freqs, rdc.dtype)


def _check_internal_inductance(name: str) -> str:
    if name not in INTERNAL_INDUCTANCE_MODELS:
        raise InputError(
            f"Unknown internal_inductance model {name!r}; expected one of "
            f"{', '.join(INTERNAL_INDUCTANCE_MODELS)}."
        )
    return name


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


def _self_spacing_and_reactance(
    gmr: Tensor,
    radius: Optional[Tensor],
    rdc: Tensor,
    freqs: Tensor,
    model: str,
    band: tuple[float, float],
) -> tuple[Tensor, Optional[Tensor]]:
    """Self-term pieces of the four internal-inductance models.

    Returns ``(g_self, x_int)``: the effective spacing radius on the impedance diagonal
    (``[*B, 1, N]`` when the model has no frequency dependence there, else ``[*B, H, N]``)
    and the internal REACTANCE to add on that diagonal (``[*B, H, N]``, or ``None`` when
    the spacing radius already carries it).
    """
    zim = internal_impedance(rdc, freqs).imag.movedim(-1, -2)  # [*B, H, N]
    if model == "gmr":
        return gmr.unsqueeze(-2), None
    if model == "bessel":
        return radius.unsqueeze(-2), zim
    if model == "gmr_skin":
        # Effective radius r*(GMR/r)^g(f): reproduces the published GMR at g=1 (power
        # frequency) and shrinks its internal-inductance surplus with the skin decay.
        ratio = zim / _x_int_dc(freqs, rdc.dtype).reshape(-1, 1)  # [*B, H, N]
        base = (gmr / radius).unsqueeze(-2)  # [*B, 1, N]
        return radius.unsqueeze(-2) * torch.pow(base, ratio), None
    low, high = band
    in_band = ((freqs > low) & (freqs < high)).reshape(-1, 1)  # [H, 1]
    g_self = torch.where(in_band, gmr.unsqueeze(-2), radius.unsqueeze(-2))
    return g_self, torch.where(in_band, torch.zeros_like(zim), zim)


def series_impedance(
    x: Tensor,
    y: Tensor,
    gmr: Tensor,
    rdc: Tensor,
    rho,
    freqs: Tensor,
    *,
    radius: Optional[Tensor] = None,
    internal_inductance: Optional[str] = None,
    power_frequency_band_hz: Optional[tuple[float, float]] = None,
) -> Tensor:
    """Carson/Deri series impedance ``Z[*B, H, N, N]`` in Ω/m (unreduced).

    ``Z_ii = Re(Zint(Rdc, f)) + j*X_int,i(f) + Lfactor·ln(1/G_i(f)) + Ze``,
    ``Z_ij = Lfactor·ln(1/D_ij) + Ze`` with ``Lfactor = j·2πf·mu0/(2π)`` and the Deri
    earth term ``Ze = Lfactor·ln(√((y_i+y_j+2/Fme)² + (x_i−x_j)²))``,
    ``Fme = √(j·2πf·mu0/rho)``. The skin-effect internal RESISTANCE
    ``Re(Zint)`` (Bessel ``I0/I1``) is in every model; ``internal_inductance`` selects
    the self-term spacing radius ``G_i(f)`` and the internal REACTANCE ``X_int,i(f)``.

    Internal-inductance models
    --------------------------
    A published GMR is a power-frequency quantity: for a solid round conductor
    ``GMR = e^(-1/4)*radius``, and ``(f*mu0)*ln(radius/GMR) = f*mu0/4 = omega*mu0/(8*pi)``
    is exactly the internal reactance at uniform current density. Skin effect makes that
    internal inductance decay (:func:`internal_reactance_ratio`), so the four options
    differ only in how the conductor interior is treated:

    ``"gmr"`` (shipped default)
        ``G_i = GMR_i`` at every frequency, ``X_int = 0``. The internal inductance stays
        at its power-frequency value, so the reactance is over-stated once the skin depth
        drops below the conductor radius. Reproduces published power-frequency data and
        is the only option that needs no conductor radius. Matches OpenDSS BELOW 1 kHz.
    ``"gmr_skin"``
        ``G_i = radius_i * (GMR_i/radius_i)^g(f)`` with ``g`` the Bessel decay
        :func:`internal_reactance_ratio`, ``X_int = 0``. Keeps the published GMR at power
        frequency (``g -> 1``) and decays the internal part toward the physical radius
        (``g -> 0``); continuous in frequency and identical to ``"bessel"`` when
        ``GMR = e^(-1/4)*radius``.
    ``"gmr_power_frequency"``
        ``"gmr"`` while ``power_frequency_band_hz[0] < f < power_frequency_band_hz[1]``
        and ``"bessel"`` outside. This is OpenDSS's rule (``LineConstants.pas``:
        ``if (f < 1000.0) and (f > 40.0)`` selects the published GMR and zeroes the
        internal reactance), so it reproduces OpenDSS at every frequency. ``Z(f)`` steps
        at the band edges unless ``GMR = e^(-1/4)*radius``.
    ``"bessel"``
        ``G_i = radius_i``, ``X_int = Im(Zint)`` at every frequency: the first-principles
        solid round conductor, exact for a solid homogeneous conductor of that radius and
        continuous in frequency. It discards the published GMR, so a stranded conductor
        (whose GMR is not ``e^(-1/4)*radius``) loses its power-frequency value.

    Parameters
    ----------
    x, y, gmr, rdc:
        Conductor horizontal position, height, geometric-mean radius and DC
        resistance per metre — real tensors ``[*B, N]``.
    rho:
        Earth resistivity (Ω·m), scalar or ``[*B]``-broadcastable.
    freqs:
        Absolute frequencies ``[H]`` (Hz), strictly positive.
    radius:
        Conductor outer radius ``[*B, N]`` (m). Required by every model except
        ``"gmr"``.
    internal_inductance:
        One of :data:`INTERNAL_INDUCTANCE_MODELS`; ``None`` resolves the active modeling
        default ``line.geometry.internal_inductance`` at call time.
    power_frequency_band_hz:
        Exclusive band used by ``"gmr_power_frequency"``; ``None`` resolves
        ``line.geometry.power_frequency_band_hz`` at call time.
    """
    if internal_inductance is None:
        internal_inductance = _cfg("line.geometry.internal_inductance")
    if power_frequency_band_hz is None:
        power_frequency_band_hz = tuple(_cfg("line.geometry.power_frequency_band_hz"))
    model = _check_internal_inductance(internal_inductance)
    if model != "gmr" and radius is None:
        raise InputError(
            f"series_impedance(internal_inductance={model!r}) needs the conductor "
            "radius; pass radius=... (only 'gmr' works without it)."
        )
    rdt = x.dtype
    cdt = _cdtype(rdt)
    fw = (_TWO_PI * freqs).to(rdt)  # [H]
    lfactor = (1j * MU0 / _TWO_PI) * fw.to(cdt)  # [H] complex coefficient
    rho_t = torch.as_tensor(rho, dtype=rdt, device=x.device)

    # Earth complex penetration: Fme = sqrt(j*Fw*mu0/rho) ; p2 = 2/Fme.  [*B?, H]
    fme = torch.sqrt(1j * (fw * MU0).to(cdt) / rho_t.to(cdt).unsqueeze(-1))
    p2 = 2.0 / fme  # [..., H]

    # Geometric spacing matrix G: off-diagonal = distance, diagonal = the model's
    # effective self radius (GMR, physical radius, or a blend of the two).
    g_self, x_int = _self_spacing_and_reactance(
        gmr, radius, rdc, freqs, model, power_frequency_band_hz
    )  # [*B, (1|H), N]
    dist, _image, ysum, dx = _pair_terms(x, y)
    n = x.shape[-1]
    eye = torch.eye(n, dtype=rdt, device=x.device)
    g_mat = (dist * (1.0 - eye)).unsqueeze(-3) + torch.diag_embed(g_self)
    spacing = lfactor.reshape(*([1] * (g_mat.dim() - 3)), -1, 1, 1) * torch.log(
        (1.0 / g_mat).to(cdt)
    )  # [*B, H, N, N]

    # Earth-return term Ze = Lfactor * log( sqrt(hterm^2 + xterm^2) ).
    hterm = ysum.unsqueeze(-3).to(cdt) + p2[..., :, None, None]  # [*B, H, N, N]
    xterm = dx.unsqueeze(-3).to(cdt)
    lnarg = torch.sqrt(hterm * hterm + xterm * xterm)
    ze = lfactor.reshape(*([1] * (lnarg.dim() - 3)), -1, 1, 1) * torch.log(lnarg)

    # Skin-effect internal impedance (Rdc, Bessel I0/I1) on the diagonal. The resistance
    # is always present; the reactance only where the spacing radius does not carry it.
    zint_re = internal_impedance(rdc, freqs).real.movedim(-1, -2)  # [*B, H, N]
    z_int = zint_re if x_int is None else torch.complex(zint_re, x_int)
    diag_int = torch.diag_embed(z_int).to(cdt)  # [*B, H, N, N]

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
    *,
    internal_inductance: Optional[str] = None,
    power_frequency_band_hz: Optional[tuple[float, float]] = None,
) -> tuple[Tensor, Tensor]:
    """Phase-reduced ``(Z[*B, H, P, P] Ω/m, C[*B, P, P] F/m)`` for line geometry.

    Series ``Z`` is Kron-reduced per frequency; capacitance ``C`` reduces the
    (frequency-independent) potential-coefficient matrix then inverts. ``C`` always uses
    the physical ``radius`` (no GMR, no frequency dependence); ``internal_inductance``
    and ``power_frequency_band_hz`` select the conductor internal model of
    :func:`series_impedance`; ``None`` resolves each active modeling default at call time.

    Agreement with OpenDSS on the same geometry, measured: ``Z`` to 4.8e-8 relative (the
    ``mu0`` constant) below 1 kHz with the shipped
    ``internal_inductance="gmr"`` default and at every frequency with
    ``"gmr_power_frequency"``; ``C`` to 2.1212e-5 relative (the ``e0`` constant).
    """
    z = kron_reduce(
        series_impedance(
            x,
            y,
            gmr,
            rdc,
            rho,
            freqs,
            radius=radius,
            internal_inductance=internal_inductance,
            power_frequency_band_hz=power_frequency_band_hz,
        ),
        n_phase,
    )  # [*B,H,P,P]
    p_red = kron_reduce(potential_coefficients(x, y, radius), n_phase)  # [*B,P,P]
    c = (_TWO_PI * E0) * torch.linalg.inv(p_red)  # [*B,P,P]
    return z, c


__all__ = [
    "i0_over_i1",
    "internal_impedance",
    "internal_reactance_ratio",
    "series_impedance",
    "potential_coefficients",
    "kron_reduce",
    "line_constants",
    "INTERNAL_INDUCTANCE",
    "INTERNAL_INDUCTANCE_MODELS",
    "POWER_FREQUENCY_BAND_HZ",
    "MU0",
    "E0",
]
