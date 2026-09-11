"""The four conductor internal-inductance models of the Carson/Deri geometry path.

A published GMR is a POWER-FREQUENCY quantity. For a solid round conductor carrying a
uniform current density, ``GMR = e^(-1/4)*radius`` and the reactance it adds through
``(f*mu0)*ln(radius/GMR) = f*mu0/4 = omega*mu0/(8*pi)`` is exactly the internal reactance
of that conductor. Skin effect confines the current to the surface, so the internal
inductance decays with frequency and a fixed GMR over-states the reactance at harmonics.

Three references are used here, each pinning a different claim:

- OpenDSS (live, through ``opendssdirect``) for ``"gmr_power_frequency"``: OpenDSS keeps
  the published GMR only while ``40 Hz < f < 1 kHz`` and uses the physical radius plus
  the full Bessel internal impedance outside that band (``LineConstants.pas``,
  ``TLineConstants.Calc`` / ``Get_Zint``). The option must match it on BOTH sides of
  both band edges, which the default ``"gmr"`` does not.
- scipy's modified Bessel functions for ``"bessel"``: the analytic internal impedance of
  a solid round conductor, ``Zint = (k*rho_c/(2*pi*a))*I0(k*a)/I1(k*a)`` with
  ``k = sqrt(j*omega*mu0/rho_c)``, is an independent formulation of the continued
  fraction :func:`pgml.geometry.carson.internal_impedance` evaluates.
- the analytic low-frequency limit ``f*mu0/4`` for the consistency of all four models at
  power frequency.

Tolerances: the OpenDSS comparisons are bounded by the physical constants (pgml uses the
SI ``mu0``, OpenDSS truncates it to ``12.56637e-7``, 4.9e-8 relative), so 2e-7 with
headroom; the scipy comparison is at floating point, measured at 4.6e-16 relative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pgml.errors import InputError
from pgml.geometry.carson import (
    INTERNAL_INDUCTANCE_MODELS,
    MU0,
    POWER_FREQUENCY_BAND_HZ,
    internal_impedance,
    internal_reactance_ratio,
    line_constants,
)

CDT = torch.float64
#: Straddles both OpenDSS band edges (40 Hz, 1 kHz) and reaches the 50th harmonic of 50 Hz.
FREQS = [20.0, 40.0, 45.0, 50.0, 250.0, 950.0, 999.0, 1000.0, 1050.0, 1250.0, 2500.0]
#: ACSR-like single conductor: GMR/radius = 0.7647, i.e. NOT a solid round conductor.
ACSR = dict(x=0.0, y=10.0, gmr=0.0078, radius=0.0102, rdc=0.12e-3)
#: Solid round aluminium conductor: GMR = e^(-1/4)*radius exactly.
SOLID = dict(x=0.0, y=10.0, radius=0.0102, gmr=0.0102 * math.exp(-0.25), rdc=0.12e-3)


def _z_single(conductor, freqs, **kw):
    """``Z(f)`` in Ω/km of a one-conductor geometry, for one internal-inductance model."""
    f = torch.tensor(freqs, dtype=CDT)
    z, _ = line_constants(
        torch.tensor([conductor["x"]], dtype=CDT),
        torch.tensor([conductor["y"]], dtype=CDT),
        torch.tensor([conductor["gmr"]], dtype=CDT),
        torch.tensor([conductor["rdc"]], dtype=CDT),
        torch.tensor([conductor["radius"]], dtype=CDT),
        100.0,
        f,
        1,
        **kw,
    )
    return (z * 1000.0).squeeze(-1).squeeze(-1).numpy()


# ---------------------------------------------------------------------------
# 1. OpenDSS reference for "gmr_power_frequency"
# ---------------------------------------------------------------------------
def _dss_series_z(cmds, nph, freqs):
    import opendssdirect as dss

    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(
        f"New Circuit.t basekv=12.47 phases={nph} bus1=s frequency=50 r1=1e-6 x1=1e-6"
    )
    for c in cmds:
        dss.Text.Command(c)
    dss.Text.Command("Set voltagebases=[12.47]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    out = []
    for f in freqs:
        dss.Text.Command(f"set frequency={f}")
        dss.Solution.BuildYMatrix(2, 1)
        dss.Circuit.SetActiveElement("Line.l1")
        yp = np.array(dss.CktElement.YPrim())
        n = int(round((len(yp) / 2) ** 0.5))
        yy = (yp[0::2] + 1j * yp[1::2]).reshape(n, n)
        half = n // 2
        out.append(np.linalg.inv(-yy[:half, half:]))  # series Z (Ω for length 1 km)
    return np.array(out)


_SINGLE_CMDS = [
    "New WireData.w1 Rdc=0.12 GMRac=0.0078 radius=0.0102 GMRunits=m radunits=m Runits=km",
    "New LineGeometry.g1 nconds=1 nphases=1 cond=1 wire=w1 x=0 h=10 units=m",
    "New Line.l1 phases=1 bus1=a bus2=b geometry=g1 length=1 units=km",
]
_THREE_PHASE_CMDS = [
    "New WireData.ph Rdc=0.1 GMRac=0.0078 radius=0.0102 GMRunits=m radunits=m Runits=km",
    "New WireData.nt Rdc=0.3 GMRac=0.0050 radius=0.0070 GMRunits=m radunits=m Runits=km",
    "New LineGeometry.g3 nconds=4 nphases=3 reduce=y "
    "cond=1 wire=ph x=-1.0 h=10 units=m cond=2 wire=ph x=0 h=10 units=m "
    "cond=3 wire=ph x=1 h=10 units=m cond=4 wire=nt x=0 h=9 units=m",
    "New Line.l1 phases=3 bus1=a.1.2.3 bus2=b.1.2.3 geometry=g3 length=1 units=km",
]


@pytest.mark.opendss
def test_gmr_power_frequency_matches_opendss_at_every_frequency():
    """``"gmr_power_frequency"`` reproduces OpenDSS across both band edges.

    The default ``"gmr"`` matches only inside 40 Hz-1 kHz (where OpenDSS also uses the
    published GMR) and departs by ~1e-2 relative outside it; measured 1.18e-2 at
    1050 Hz and 1.75e-2 at 2500 Hz on this ACSR-like conductor.
    """
    pytest.importorskip("opendssdirect", exc_type=ImportError)
    z_dss = _dss_series_z(_SINGLE_CMDS, 1, FREQS)[:, 0, 0]
    z_pf = _z_single(ACSR, FREQS, internal_inductance="gmr_power_frequency")
    z_gmr = _z_single(ACSR, FREQS, internal_inductance="gmr")
    rel_pf = np.abs(z_pf - z_dss) / np.abs(z_dss)
    rel_gmr = np.abs(z_gmr - z_dss) / np.abs(z_dss)
    assert rel_pf.max() < 2e-7, f"max rel {rel_pf.max():.2e} at f={FREQS}"
    # The default model is the one that steps; 40 Hz and 1000 Hz belong to OpenDSS's
    # radius branch (its band is exclusive: `(f < 1000.0) and (f > 40.0)`).
    outside = [i for i, f in enumerate(FREQS) if not (40.0 < f < 1000.0)]
    assert min(rel_gmr[i] for i in outside) > 1e-3


@pytest.mark.opendss
def test_gmr_power_frequency_matches_opendss_three_phase_kron():
    """Same parity on a 3-phase + neutral geometry, Kron-reduced, above 1 kHz."""
    pytest.importorskip("opendssdirect", exc_type=ImportError)
    freqs = [50.0, 950.0, 1050.0, 2500.0]
    z_dss = _dss_series_z(_THREE_PHASE_CMDS, 3, freqs)
    f = torch.tensor(freqs, dtype=CDT)
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=CDT)
    y = torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=CDT)
    gmr = torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=CDT)
    rdc = torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=CDT) * 1e-3
    rad = torch.tensor([0.0102, 0.0102, 0.0102, 0.0070], dtype=CDT)
    z, _ = line_constants(
        x, y, gmr, rdc, rad, 100.0, f, 3, internal_inductance="gmr_power_frequency"
    )
    np.testing.assert_allclose((z * 1000.0).numpy(), z_dss, rtol=2e-7, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. First-principles Bessel reference for "bessel"
# ---------------------------------------------------------------------------
def test_internal_impedance_matches_analytic_solid_round_conductor():
    """``internal_impedance`` IS the textbook solid-round-conductor internal impedance.

    For a solid round conductor of radius ``a`` and resistivity ``rho_c``,
    ``Zint = (k*rho_c/(2*pi*a)) * I0(k*a)/I1(k*a)`` with ``k = sqrt(j*omega*mu0/rho_c)``.
    With ``Rdc = rho_c/(pi*a^2)`` this is algebraically the form pgml evaluates,
    ``(1+j)*(I0/I1)(alpha)*sqrt(Rdc*f*mu0)/2`` with ``alpha = (1+j)*sqrt(f*mu0/Rdc)``.
    scipy's ``iv`` is an independent evaluation of the Bessel functions, so this test
    validates both the algebra and the continued fraction. Measured 4.6e-16 relative
    (floating point) over 1 Hz to 10 kHz.
    """
    iv = pytest.importorskip("scipy.special").iv
    a = 0.0102
    rdc = 0.12e-3
    rho_c = rdc * math.pi * a * a
    freqs = np.array([1.0, 50.0, 250.0, 1000.0, 2500.0, 1e4])
    k = np.sqrt(1j * 2.0 * np.pi * freqs * MU0 / rho_c)
    z_ref = (k * rho_c / (2.0 * np.pi * a)) * iv(0, k * a) / iv(1, k * a)
    z_pgml = internal_impedance(
        torch.tensor([rdc], dtype=CDT), torch.tensor(freqs, dtype=CDT)
    )[0].numpy()
    rel = np.abs(z_pgml - z_ref) / np.abs(z_ref)
    assert rel.max() < 1e-13, f"max rel {rel.max():.2e}"


def test_internal_reactance_ratio_starts_at_one_and_decays():
    """``g(f) = Im(Zint)/(f*mu0/4)`` is 1 at DC, monotone decreasing, positive.

    ``f*mu0/4 = omega*mu0/(8*pi)`` is the uniform-current-density internal reactance, so
    ``g`` is the conductor's internal inductance normalised by its power-frequency value
    and measures directly how far a published GMR is off at a given frequency.
    """
    rdc = torch.tensor([0.12e-3], dtype=CDT)
    freqs = torch.tensor([0.01, 1.0, 50.0, 250.0, 1000.0, 2500.0, 1e4], dtype=CDT)
    g = internal_reactance_ratio(rdc, freqs)[0]
    assert abs(float(g[0]) - 1.0) < 1e-6
    assert torch.all(g[1:] < g[:-1]) and torch.all(g > 0.0)
    # Identical to the explicit definition.
    zim = internal_impedance(rdc, freqs).imag[0]
    torch.testing.assert_close(g, zim / (freqs * (MU0 / 4.0)))


def test_gmr_skin_equals_bessel_for_a_solid_round_conductor():
    """With ``GMR = e^(-1/4)*radius`` the ``"gmr_skin"`` blend IS the Bessel model.

    ``"gmr_skin"`` uses ``radius*(GMR/radius)^g(f)``, which adds
    ``g(f)*f*mu0*ln(radius/GMR) = g(f)*f*mu0/4 = Im(Zint)`` for a solid round conductor —
    exactly what ``"bessel"`` adds. Measured agreement 4e-18 relative.
    """
    za = _z_single(SOLID, FREQS, internal_inductance="gmr_skin")
    zb = _z_single(SOLID, FREQS, internal_inductance="bessel")
    rel = np.abs(za - zb).max() / np.abs(zb).max()
    assert rel < 1e-15, f"rel {rel:.2e}"


def test_all_models_agree_at_power_frequency_for_a_solid_round_conductor():
    """At 50 Hz the four models coincide for a solid round conductor (g(50 Hz) = 0.997).

    This is the statement that the default ``"gmr"`` is the power-frequency LIMIT of the
    other three, not a different model: the residual is the 0.3 % of internal reactance
    skin effect has already removed at 50 Hz, which is 6.3e-5 of |Z|.
    """
    ref = _z_single(SOLID, [50.0], internal_inductance="gmr")[0]
    for model in INTERNAL_INDUCTANCE_MODELS:
        z = _z_single(SOLID, [50.0], internal_inductance=model)[0]
        rel = abs(z - ref) / abs(ref)
        assert rel < 1e-4, f"{model}: rel {rel:.2e}"
    # Toward DC the agreement becomes exact (g -> 1).
    for model in INTERNAL_INDUCTANCE_MODELS:
        z = _z_single(SOLID, [0.01], internal_inductance=model)[0]
        rel = abs(z - _z_single(SOLID, [0.01], internal_inductance="gmr")[0]) / abs(z)
        assert rel < 1e-8, f"{model} at 0.01 Hz: rel {rel:.2e}"


def test_gmr_overstates_reactance_above_1_khz():
    """The default model's reactance exceeds the first-principles one above 1 kHz.

    The physical claim behind the option: at 2.5 kHz 61 % of the conductor's internal
    inductance is gone (``g = 0.387``), so holding the power-frequency GMR adds reactance
    that is not there. On the solid round conductor the difference is the full
    ``(1-g)*f*mu0/4``.
    """
    z_gmr = _z_single(SOLID, [2500.0], internal_inductance="gmr")[0]
    z_bes = _z_single(SOLID, [2500.0], internal_inductance="bessel")[0]
    g = float(
        internal_reactance_ratio(
            torch.tensor([SOLID["rdc"]], dtype=CDT), torch.tensor([2500.0], dtype=CDT)
        )
    )
    assert z_gmr.imag > z_bes.imag
    expected = (1.0 - g) * 2500.0 * MU0 / 4.0 * 1000.0  # Ω/km
    assert abs((z_gmr.imag - z_bes.imag) - expected) < 1e-9 * abs(z_gmr.imag)
    # The resistance is identical: only the internal reactance model changes.
    assert abs(z_gmr.real - z_bes.real) < 1e-12 * z_gmr.real


# ---------------------------------------------------------------------------
# 3. Option plumbing
# ---------------------------------------------------------------------------
def test_power_frequency_band_is_exclusive_and_configurable():
    """The band bounds are exclusive, and a custom band moves the switch."""
    low, high = POWER_FREQUENCY_BAND_HZ
    assert (low, high) == (40.0, 1000.0)
    freqs = [low, low + 1e-9, high - 1e-9, high]
    z_pf = _z_single(ACSR, freqs, internal_inductance="gmr_power_frequency")
    z_gmr = _z_single(ACSR, freqs, internal_inductance="gmr")
    z_bes = _z_single(ACSR, freqs, internal_inductance="bessel")
    np.testing.assert_allclose(z_pf[[1, 2]], z_gmr[[1, 2]], rtol=0, atol=0)
    np.testing.assert_allclose(z_pf[[0, 3]], z_bes[[0, 3]], rtol=0, atol=0)
    # A custom band: 250 Hz now falls outside it.
    z_custom = _z_single(
        ACSR,
        [250.0],
        internal_inductance="gmr_power_frequency",
        power_frequency_band_hz=(40.0, 200.0),
    )
    np.testing.assert_allclose(
        z_custom, _z_single(ACSR, [250.0], internal_inductance="bessel")
    )


@pytest.mark.parametrize("model", INTERNAL_INDUCTANCE_MODELS)
def test_batched_equals_per_line(model):
    """Stacking lines into the leading batch gives the per-line result, every model."""
    freqs = torch.tensor(FREQS, dtype=CDT)
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=CDT)
    y = torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=CDT)
    gmr = torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=CDT)
    rdc = torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=CDT) * 1e-3
    rad = torch.tensor([0.0102, 0.0102, 0.0102, 0.0070], dtype=CDT)
    kw = dict(internal_inductance=model)
    z1, _ = line_constants(x, y, gmr, rdc, rad, 100.0, freqs, 3, **kw)
    zb, _ = line_constants(
        torch.stack([x, 2.0 * x]),
        torch.stack([y, y]),
        torch.stack([gmr, gmr]),
        torch.stack([rdc, rdc]),
        torch.stack([rad, rad]),
        100.0,
        freqs,
        3,
        **kw,
    )
    assert torch.allclose(zb[0], z1)


@pytest.mark.parametrize("model", INTERNAL_INDUCTANCE_MODELS)
def test_float32_runs_and_tracks_float64(model):
    """Every model honours the input dtype (float32 in -> complex64 out)."""
    f64 = torch.tensor([50.0, 1050.0, 2500.0], dtype=torch.float64)
    args64 = [
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([0.0078], dtype=torch.float64),
        torch.tensor([0.12e-3], dtype=torch.float64),
        torch.tensor([0.0102], dtype=torch.float64),
    ]
    z64, _ = line_constants(*args64, 100.0, f64, 1, internal_inductance=model)
    args32 = [a.to(torch.float32) for a in args64]
    z32, _ = line_constants(
        *args32, 100.0, f64.to(torch.float32), 1, internal_inductance=model
    )
    assert z32.dtype == torch.complex64 and z64.dtype == torch.complex128
    torch.testing.assert_close(z32.to(torch.complex128), z64, rtol=1e-5, atol=1e-9)


def test_unknown_model_raises():
    with pytest.raises(InputError, match="Unknown internal_inductance model"):
        _z_single(ACSR, [50.0], internal_inductance="radius_above_1khz")


@pytest.mark.parametrize("model", ["gmr_skin", "gmr_power_frequency", "bessel"])
def test_radius_is_required_away_from_the_default_model(model):
    """``series_impedance`` refuses a radius-based model without a radius."""
    from pgml.geometry.carson import series_impedance

    with pytest.raises(InputError, match="needs the conductor radius"):
        series_impedance(
            torch.tensor([0.0], dtype=CDT),
            torch.tensor([10.0], dtype=CDT),
            torch.tensor([0.0078], dtype=CDT),
            torch.tensor([0.12e-3], dtype=CDT),
            100.0,
            torch.tensor([50.0], dtype=CDT),
            internal_inductance=model,
        )
