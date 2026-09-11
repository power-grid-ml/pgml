"""Oracle test: pgml Carson/Deri line constants vs OpenDSS geometry lines.

OpenDSS (default DERI earth model) is the ground truth for frequency-dependent line
impedance. We build geometry lines in OpenDSS, extract the series Z from the element
Yprim at several frequencies, and compare to ``pgml.geometry.carson`` — single
conductor and 3-phase+neutral (Kron-reduced). The MODEL agrees to floating point
(``docs/pgml/modeling/references/opendss/carson.md``); what is left is the physical
constants, because pgml uses the SI values of ``mu0`` and ``e0`` where OpenDSS truncates
them to ``12.56637e-7`` and ``8.854e-12`` (4.9e-8 and 2.1e-5 relative). The series
impedance is linear in ``mu0`` (and square-root in it through the penetration depth), so
``Z`` agrees to ~4.8e-8 relative, measured; ``C = 2*pi*e0*inv(P)`` is linear in ``e0``
and agrees to 2.1212e-5 relative, which is exactly the constant ratio.

Scope: BELOW 1 kHz. OpenDSS's geometry model changes its conductor spacing term at
exactly 1 kHz (away from the published GMR, toward the physical radius); pgml always uses
the published GMR, so above 1 kHz the two differ by ~1e-2 relative on this geometry.
``FREQS`` therefore stops at 750 Hz, and a separate test pins the step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Optional opendssdirect guard (matches existing reference test conventions)
# ---------------------------------------------------------------------------
try:
    import opendssdirect as dss

    _OPENDSS_AVAILABLE = True
except ImportError:
    _OPENDSS_AVAILABLE = False

if not _OPENDSS_AVAILABLE:
    pytest.skip("opendssdirect not installed", allow_module_level=True)

pytestmark = pytest.mark.opendss

from pgml.geometry.carson import line_constants  # noqa: E402

CDT = torch.float64
FREQS = [50.0, 150.0, 250.0, 350.0, 550.0, 750.0]


def _dss_series_z(cmds, nph, freqs):
    dss.Text.Command("Clear")
    # Base frequency of every element (and the reference for OpenDSS's own frequency
    # scaling); set explicitly so the helper does not inherit another circuit's value.
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


def test_single_conductor_matches_opendss():
    cmds = [
        "New WireData.w1 Rdc=0.12 GMRac=0.0078 radius=0.0102 GMRunits=m radunits=m Runits=km",
        "New LineGeometry.g1 nconds=1 nphases=1 cond=1 wire=w1 x=0 h=10 units=m",
        "New Line.l1 phases=1 bus1=a bus2=b geometry=g1 length=1 units=km",
    ]
    z_dss = _dss_series_z(cmds, 1, FREQS)[:, 0, 0]
    freqs = torch.tensor(FREQS, dtype=CDT)
    z, _ = line_constants(
        torch.tensor([0.0], dtype=CDT),
        torch.tensor([10.0], dtype=CDT),
        torch.tensor([0.0078], dtype=CDT),
        torch.tensor([0.12e-3], dtype=CDT),
        torch.tensor([0.0102], dtype=CDT),
        100.0,
        freqs,
        1,
    )
    z_km = (z * 1000.0).squeeze(-1).squeeze(-1).numpy()
    # Measured 4.66e-8 relative = the mu0 constant difference (see the module docstring).
    np.testing.assert_allclose(z_km, z_dss, rtol=2e-7, atol=1e-8)


def test_three_phase_with_neutral_kron_matches_opendss():
    cmds = [
        "New WireData.ph Rdc=0.1 GMRac=0.0078 radius=0.0102 GMRunits=m radunits=m Runits=km",
        "New WireData.nt Rdc=0.3 GMRac=0.0050 radius=0.0070 GMRunits=m radunits=m Runits=km",
        "New LineGeometry.g3 nconds=4 nphases=3 reduce=y "
        "cond=1 wire=ph x=-1.0 h=10 units=m cond=2 wire=ph x=0 h=10 units=m "
        "cond=3 wire=ph x=1 h=10 units=m cond=4 wire=nt x=0 h=9 units=m",
        "New Line.l1 phases=3 bus1=a.1.2.3 bus2=b.1.2.3 geometry=g3 length=1 units=km",
    ]
    z_dss = _dss_series_z(cmds, 3, FREQS)
    freqs = torch.tensor(FREQS, dtype=CDT)
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=CDT)
    y = torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=CDT)
    gmr = torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=CDT)
    rdc = torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=CDT) * 1e-3
    rad = torch.tensor([0.0102, 0.0102, 0.0102, 0.0070], dtype=CDT)
    z, _ = line_constants(x, y, gmr, rdc, rad, 100.0, freqs, 3)
    # Measured 4.81e-8 relative = the mu0 constant difference.
    np.testing.assert_allclose((z * 1000.0).numpy(), z_dss, rtol=2e-7, atol=1e-8)


def test_batched_equals_per_line():
    """Stacking lines (leading batch) equals solving each individually."""
    freqs = torch.tensor(FREQS, dtype=CDT)
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=CDT)
    y = torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=CDT)
    gmr = torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=CDT)
    rdc = torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=CDT) * 1e-3
    rad = torch.tensor([0.0102, 0.0102, 0.0102, 0.0070], dtype=CDT)
    z1, _ = line_constants(x, y, gmr, rdc, rad, 100.0, freqs, 3)
    zb, _ = line_constants(
        torch.stack([x, x]),
        torch.stack([y, y]),
        torch.stack([gmr, gmr]),
        torch.stack([rdc, rdc]),
        torch.stack([rad, rad]),
        100.0,
        freqs,
        3,
    )
    assert torch.allclose(zb[0], z1) and torch.allclose(zb[1], z1)


def test_reactance_is_not_naive_h_scaling():
    """Regression guard: Carson X(h) differs materially from h * X(f0) (earth return)."""
    freqs = torch.tensor([50.0, 250.0], dtype=CDT)
    z, _ = line_constants(
        torch.tensor([0.0], dtype=CDT),
        torch.tensor([10.0], dtype=CDT),
        torch.tensor([0.0078], dtype=CDT),
        torch.tensor([0.12e-3], dtype=CDT),
        torch.tensor([0.0102], dtype=CDT),
        100.0,
        freqs,
        1,
    )
    r50, x50 = float(z[0, 0, 0].real), float(z[0, 0, 0].imag)
    r250, x250 = float(z[1, 0, 0].real), float(z[1, 0, 0].imag)
    # X is materially below naive h*X50, and R rises with frequency (earth return + skin).
    assert abs(x250 - 5.0 * x50) / (5.0 * x50) > 0.03
    assert r250 > r50 * 1.05


def test_opendss_changes_its_spacing_term_at_1_khz():
    """The agreement is scoped BELOW 1 kHz: OpenDSS steps at exactly 1 kHz.

    OpenDSS's geometry line model stops using the published GMR for the conductor
    spacing term at 1 kHz (the current has crowded toward the surface, so it moves
    toward the physical radius); pgml always uses the published GMR. Measured on the
    single-conductor geometry above: 4.7e-8 relative at 950 Hz (the ``mu0`` constant
    difference alone) and 1.2e-2 at 1050 Hz. The step is about 45 % of a full
    GMR-to-radius substitution, so OpenDSS's high-frequency form is not a plain swap —
    matching it would mean reading its source, not guessing.

    This test exists to FIX that boundary in place: if a future OpenDSS release moves or
    removes the switch, it fails and the scope statement in the docstrings has to change.
    """
    cmds = [
        "New WireData.w1 Rdc=0.12 GMRac=0.0078 radius=0.0102 GMRunits=m radunits=m Runits=km",
        "New LineGeometry.g1 nconds=1 nphases=1 cond=1 wire=w1 x=0 h=10 units=m",
        "New Line.l1 phases=1 bus1=a bus2=b geometry=g1 length=1 units=km",
    ]
    freqs_hz = [950.0, 1050.0]
    z_dss = _dss_series_z(cmds, 1, freqs_hz)[:, 0, 0]
    freqs = torch.tensor(freqs_hz, dtype=CDT)
    z, _ = line_constants(
        torch.tensor([0.0], dtype=CDT),
        torch.tensor([10.0], dtype=CDT),
        torch.tensor([0.0078], dtype=CDT),
        torch.tensor([0.12e-3], dtype=CDT),
        torch.tensor([0.0102], dtype=CDT),
        100.0,
        freqs,
        1,
    )
    z_km = (z * 1000.0).squeeze(-1).squeeze(-1).numpy()
    rel = np.abs(z_km - z_dss) / np.abs(z_dss)
    assert rel[0] < 2e-7, f"below 1 kHz the models must agree, got {rel[0]:.2e}"
    assert 5e-3 < rel[1] < 5e-2, f"expected the 1 kHz step, got {rel[1]:.2e}"
    # The resistance is unaffected: only the spacing (reactance) term steps.
    assert abs(z_km[1].real - z_dss[1].real) / z_dss[1].real < 2e-7


def test_capacitance_matches_opendss_to_the_epsilon0_constant():
    """``C = 2*pi*e0*inv(P)`` matches OpenDSS to the ``e0`` constant ratio exactly.

    The potential-coefficient matrix is geometry only, so the whole deviation is the
    constant: OpenDSS truncates ``e0`` to ``8.854e-12`` where pgml uses
    ``8.8541878128e-12`` (2.1212e-5 relative). Measured 2.1212e-5 on both geometries,
    i.e. the ratio and nothing else.
    """
    cmds = [
        "New WireData.w1 Rdc=0.12 GMRac=0.0078 radius=0.0102 GMRunits=m radunits=m Runits=km",
        "New LineGeometry.g1 nconds=1 nphases=1 cond=1 wire=w1 x=0 h=10 units=m",
        "New Line.l1 phases=1 bus1=a bus2=b geometry=g1 length=1 units=km",
    ]
    _dss_series_z(cmds, 1, [50.0])  # builds and solves the circuit
    c_dss_nf_per_km = np.array(dss.Lines.CMatrix())[0]  # nF/km
    freqs = torch.tensor([50.0], dtype=CDT)
    _z, c = line_constants(
        torch.tensor([0.0], dtype=CDT),
        torch.tensor([10.0], dtype=CDT),
        torch.tensor([0.0078], dtype=CDT),
        torch.tensor([0.12e-3], dtype=CDT),
        torch.tensor([0.0102], dtype=CDT),
        100.0,
        freqs,
        1,
    )
    c_pgml_nf_per_km = float(c.squeeze()) * 1000.0 * 1e9
    rel = abs(c_pgml_nf_per_km - c_dss_nf_per_km) / c_dss_nf_per_km
    expected = (8.8541878128e-12 - 8.854e-12) / 8.854e-12
    assert abs(rel - expected) < 1e-9, f"C rel {rel:.6e} != e0 ratio {expected:.6e}"
