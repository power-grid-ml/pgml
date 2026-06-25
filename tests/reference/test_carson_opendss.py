"""Oracle test: pgml Carson/Deri line constants vs OpenDSS geometry lines.

OpenDSS (default DERI earth model) is the ground truth for frequency-dependent line
impedance. We build geometry lines in OpenDSS, extract the series Z from the element
Yprim at several frequencies, and compare to ``pgml.geometry.carson`` — single
conductor and 3-phase+neutral (Kron-reduced). The match is to floating point
(``docs/pgml/modeling/references/opendss/carson.md``), the whole point being that this closes the
harmonic line-impedance gap.
"""

from __future__ import annotations

import numpy as np
import torch

import opendssdirect as dss

from pgml.geometry.carson import line_constants

CDT = torch.float64
FREQS = [50.0, 150.0, 250.0, 350.0, 550.0, 750.0]


def _dss_series_z(cmds, nph, freqs):
    dss.Text.Command("Clear")
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
    np.testing.assert_allclose(z_km, z_dss, rtol=1e-9, atol=1e-9)


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
    np.testing.assert_allclose((z * 1000.0).numpy(), z_dss, rtol=1e-9, atol=1e-9)


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
