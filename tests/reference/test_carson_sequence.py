"""Positive-sequence-aware harmonic line model (TODO #1, option B).

The physics being asserted: a balanced positive-sequence current produces no net
ground current, so the Carson/Deri earth-return term CANCELS in ``Z1`` and survives
only in ``Z0``. Therefore the positive-sequence harmonic reactance scales ``X1·h``
with NO earth floor, while the zero sequence carries the earth floor. The corrected
model (:mod:`pgml.geometry.sequence`) implements ``Z1`` directly and via a physical
two-conductor go/return loop; both stay physical where the single-conductor
earth-return synthesis blows the GMR past the conductor radius.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from pgml.geometry.carson import kron_reduce, series_impedance
from pgml.geometry.sequence import (
    positive_sequence_z,
    sequence_impedances,
    two_conductor_geometry,
    two_conductor_loop_z,
)
from pgml.geometry.synthesis import synthesize_line_geometry

RDT = torch.float64
F0 = 50.0
ORDERS = [1, 5, 7, 11, 13, 25]


def _freqs(orders=ORDERS):
    return torch.tensor([h * F0 for h in orders], dtype=RDT)


# --- 3-phase geometry: earth return only in Z0 -----------------------------
def _three_phase_geometry_z(freqs):
    """Full Carson Z(h) for a 3-phase overhead line (3 phase + neutral, Kron-reduced)."""
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=RDT)
    y = torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=RDT)
    gmr = torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=RDT)
    rdc = torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=RDT) * 1e-3
    z = series_impedance(x, y, gmr, rdc, 100.0, freqs)
    return kron_reduce(z, 3)  # [H, 3, 3]


def test_positive_sequence_has_no_earth_floor_zero_does():
    """Z1 reactance scales ~∝h (earth cancels); Z0 carries the earth floor."""
    freqs = _freqs()
    zf = _three_phase_geometry_z(freqs)
    z0, z1, _z2 = sequence_impedances(zf)
    h = freqs / F0

    # positive sequence: X1(h) tracks h*X1(f0) to within ~1% (residual mutual earth).
    x1_lin = z1.imag / (z1.imag[0] * h)
    assert torch.allclose(x1_lin, torch.ones_like(x1_lin), atol=2e-3), x1_lin

    # zero sequence: X0(h) departs strongly from h*X0(f0) — the earth floor.
    x0_lin = z0.imag / (z0.imag[0] * h)
    assert (x0_lin[-1] < 0.92) and (x0_lin[-1] > 0.5), x0_lin

    # zero-sequence resistance is dominated by the earth return (R0 >> R1).
    assert (z0.real / z1.real)[-1] > 3.0


def test_zero_sequence_reactance_exceeds_positive_floor():
    """The X0 self-reactance floor is what makes single-conductor synthesis fail."""
    freqs = _freqs([1])
    z0, z1, _ = sequence_impedances(_three_phase_geometry_z(freqs))
    # zero-sequence reactance is several times the positive-sequence value (earth term).
    assert float(z0.imag) > 2.0 * float(z1.imag)


# --- direct positive-sequence model ----------------------------------------
def test_positive_sequence_x_scales_exactly_with_h():
    """X1(h) = X1*(f/f0) to floating point (geometric reactance ∝ frequency)."""
    freqs = _freqs()
    r1, x1 = 3.6e-4, 3.0e-4
    z = positive_sequence_z(r1, x1, F0, freqs)
    expected = x1 * (freqs / F0)
    assert torch.allclose(z.imag, expected, rtol=0, atol=1e-18)


def test_positive_sequence_resistance_grows_with_skin():
    """R1(h) grows monotonically (skin effect) and equals R1 exactly at f0."""
    freqs = _freqs()
    r1, x1 = 3.6e-4, 3.0e-4
    z = positive_sequence_z(r1, x1, F0, freqs)
    assert abs(float(z.real[0]) - r1) < 1e-12  # m(f0) = 1
    assert torch.all(z.real[1:] > z.real[:-1])  # strictly increasing with h
    assert float(z.real[-1]) > r1  # net skin growth

    # skin=False keeps R constant (the naive model).
    z_naive = positive_sequence_z(r1, x1, F0, freqs, skin=False)
    assert torch.allclose(z_naive.real, torch.full_like(z_naive.real, r1))


def test_direct_model_matches_two_conductor_loop():
    """Direct Z1 agrees with the physical go/return Carson loop (earth cancels)."""
    freqs = _freqs([1, 5, 7, 11])
    r1, x1 = 3.6e-4, 3.0e-4
    geom = two_conductor_geometry(r1, x1, F0, radius_m=0.0102)
    z_loop = two_conductor_loop_z(geom, freqs)
    z_dir = positive_sequence_z(r1, x1, F0, freqs)
    # exact at f0 (construction), residual earth coupling stays small at harmonics.
    assert abs(complex(z_loop[0]) - complex(z_dir[0])) / abs(complex(z_dir[0])) < 1e-7
    rel = (z_loop - z_dir).abs() / z_dir.abs()
    assert float(rel.max()) < 0.03


def test_two_conductor_geometry_stays_physical_where_single_blows_up():
    """Guard: low-X cable -> single-conductor GMR is non-physical, two-conductor is not."""
    r1, x1 = 3.6e-4, 3.0e-4  # low-X line (X1 below the single-conductor earth floor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        single = synthesize_line_geometry(r1, x1, f0=F0, line_type="cs")
    c = single.conductors[0]
    # single-conductor synthesis is flagged non-physical (GMR >= radius).
    assert single.provenance.extra["synth_unphysical"] == "True"
    assert float(c.gmr_m) >= float(c.radius_m)

    # two-conductor go/return synthesis is physical for the SAME line.
    geom = two_conductor_geometry(r1, x1, F0, radius_m=0.0102)
    assert geom["gmr_m"] < geom["radius_m"]  # GMR below radius (physical)
    assert geom["spacing_m"] > 2.0 * geom["radius_m"]  # conductors do not overlap
    assert np.isfinite(geom["spacing_m"])

    # and it reproduces R1/X1 at the fundamental.
    z0 = two_conductor_loop_z(geom, _freqs([1]))
    assert abs(float(z0.real) - r1) / r1 < 1e-6
    assert abs(float(z0.imag) - x1) / x1 < 1e-6


# --- batching ---------------------------------------------------------------
def test_positive_sequence_batched_equals_per_line():
    """Stacking lines (leading batch) equals solving each individually."""
    freqs = _freqs()
    r1 = torch.tensor([3.6e-4, 5.0e-4], dtype=RDT)
    x1 = torch.tensor([3.0e-4, 8.0e-4], dtype=RDT)
    zb = positive_sequence_z(r1, x1, F0, freqs)  # [2, H]
    z0 = positive_sequence_z(r1[0], x1[0], F0, freqs)
    z1 = positive_sequence_z(r1[1], x1[1], F0, freqs)
    assert torch.allclose(zb[0], z0) and torch.allclose(zb[1], z1)
