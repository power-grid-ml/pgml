"""Positive-sequence-aware harmonic line model.

The physics being asserted: a balanced positive-sequence current produces no net
ground current, so the Carson/Deri earth-return term CANCELS in ``Z1`` and survives
only in ``Z0``. Therefore the positive-sequence harmonic reactance scales ``X1·h``
with NO earth floor, while the zero sequence carries the earth floor. The corrected
model (:mod:`pgml.geometry.sequence`) implements ``Z1`` directly and via a physical
two-conductor go/return loop; both stay physical where the single-conductor
earth-return synthesis blows the GMR past the conductor radius.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

from pgml.geometry.carson import kron_reduce, series_impedance
from pgml.geometry.sequence import (
    positive_sequence_z,
    sequence_aware_phase_z,
    sequence_impedances,
    sequence_to_phase_z,
    skin_resistance_multiplier,
    two_conductor_geometry,
    two_conductor_loop_z,
    zero_sequence_harmonic_z,
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


# --- the reference frequency is the exact constant 1 ------------------------
def test_skin_multiplier_at_the_reference_frequency_is_exactly_one():
    """m(f0) = 1 to the bit, whether f0 is requested alone or inside a harmonic list.

    The fundamental-frequency assembly of a feeder asks for ``f0`` alone, and the
    multiplier there is a known constant: the value and the gradient must be the same
    as the order-1 entry of a multi-order request, which evaluates the full Bessel
    expression.
    """
    r1 = torch.tensor([3.6e-4, 5.0e-4, 1.2e-3], dtype=RDT)
    alone = skin_resistance_multiplier(r1, F0, _freqs([1]))  # [3, 1]
    with_harmonics = skin_resistance_multiplier(r1, F0, _freqs([1, 5, 13]))  # [3, 3]

    assert alone.shape == (3, 1)
    assert torch.equal(alone[:, 0], torch.ones(3, dtype=RDT))
    assert torch.equal(with_harmonics[:, 0], alone[:, 0])
    assert torch.all(with_harmonics[:, 1:] > 1.0)  # skin growth above f0


def test_skin_multiplier_at_the_reference_frequency_carries_no_gradient():
    """``dm(f0)/dR1 = 0`` exactly: the full expression divides one value by itself."""
    r1 = torch.tensor([3.6e-4, 5.0e-4], dtype=RDT, requires_grad=True)
    m = skin_resistance_multiplier(r1, F0, _freqs([1, 5]))
    (g,) = torch.autograd.grad(m[:, 0].sum(), r1)
    assert torch.equal(g, torch.zeros_like(g))


def test_fundamental_ybus_equals_the_order_one_slice_of_a_harmonic_assembly():
    """A positive-sequence feeder's Y(f0) is the same matrix either way, bit for bit."""
    from pgml.assembly import assemble_network_ybus
    from pgml.grids import synthetic_feeder

    grid = synthetic_feeder(12)
    for ln in grid.branches:
        ln.harmonic_line_model = "positive_sequence"
        ln.harmonic_skin_effect = True
    f0 = float(grid.base_frequency_hz)
    y_fund = assemble_network_ybus(grid, [f0], dtype=torch.complex128).Y
    y_harm = assemble_network_ybus(grid, [f0, 5 * f0], dtype=torch.complex128).Y
    assert torch.equal(y_fund[0], y_harm[0])


# --- native OpenDSS R/X lines (how OpenDSS frequency-adjusts an R/X LineCode) ----
_DSS_F0 = 60.0  # OpenDSS default base frequency
_DSS_ORDERS = [1, 5, 7, 11, 13]


def _dss_series_z(line_cmd: str, nph: int, orders):
    """Series Z(h) [len(orders), nph, nph] (Ω/km) of an OpenDSS R/X line at harmonics."""
    dss = pytest.importorskip("opendssdirect", exc_type=ImportError)

    dss.Text.Command("Clear")
    # OpenDSS scales a sequence-defined line's X by f/basefreq, where basefreq comes
    # from this global setting: set it explicitly (the default is 60 Hz, and another
    # circuit may have changed it).
    dss.Text.Command(f"Set DefaultBaseFrequency={_DSS_F0}")
    dss.Text.Command(
        f"New Circuit.t basekv=12.47 phases={nph} bus1=s frequency={_DSS_F0} "
        "r1=1e-6 x1=1e-6"
    )
    dss.Text.Command(line_cmd)
    dss.Text.Command("Set voltagebases=[12.47]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    out = []
    for h in orders:
        dss.Text.Command(f"set frequency={h * _DSS_F0}")
        dss.Solution.BuildYMatrix(2, 1)
        dss.Circuit.SetActiveElement("Line.l1")
        yp = np.array(dss.CktElement.YPrim())
        n = int(round((len(yp) / 2) ** 0.5))
        yy = (yp[0::2] + 1j * yp[1::2]).reshape(n, n)
        half = n // 2
        out.append(np.linalg.inv(-yy[:half, half:]))
    return np.array(out)


@pytest.mark.opendss
def test_opendss_native_3phase_rx_positive_sequence_is_naive():
    """Native OpenDSS 3-phase R/X line: Z1(h)=R1+jX1·(f/f0) (earth only in Z0).

    This is the standard way to enter a balanced feeder in OpenDSS, and it is exactly the
    pgml DEFAULT R/X behaviour (``X∝h``, ``R`` const) — i.e. the simplified R/X model does
    NOT diverge from native 3-phase OpenDSS.
    """
    r1, x1 = 0.36, 0.30
    zabc = _dss_series_z(
        "New Line.l1 phases=3 bus1=a.1.2.3 bus2=b.1.2.3 "
        f"r1={r1} x1={x1} r0=0.6 x0=1.2 length=1 units=km",
        3,
        _DSS_ORDERS,
    )
    a = np.exp(2j * np.pi / 3)
    A = np.array([[1, 1, 1], [1, a * a, a], [1, a, a * a]])
    Ainv = np.linalg.inv(A)
    z1 = np.array([(Ainv @ zabc[i] @ A)[1, 1] for i in range(len(_DSS_ORDERS))])
    z0 = np.array([(Ainv @ zabc[i] @ A)[0, 0] for i in range(len(_DSS_ORDERS))])

    # OpenDSS positive sequence: R1 constant, X1 ∝ h (earth cancels).
    for i, h in enumerate(_DSS_ORDERS):
        assert abs(z1[i].real - r1) < 1e-4, (h, z1[i].real)
        assert abs(z1[i].imag - x1 * h) / (x1 * h) < 1e-4, (h, z1[i].imag)
    # earth return shows up in Z0 (X0 sub-linear, R0 grows).
    assert (z0[-1].imag / (z0[0].imag * _DSS_ORDERS[-1])) < 0.95
    assert z0[-1].real > z0[0].real * 1.2

    # pgml DEFAULT (naive: skin=False) reproduces the OpenDSS Z1 to floating point.
    freqs = torch.tensor([h * _DSS_F0 for h in _DSS_ORDERS], dtype=RDT)
    z_pgml = positive_sequence_z(r1, x1, _DSS_F0, freqs, skin=False).numpy()
    np.testing.assert_allclose(z_pgml, z1, rtol=1e-4, atol=1e-4)


# --- sequence-aware model (unbalanced / 4-wire: earth return in Z0) -------------
def _seq_line():
    """A representative LV line: R0/X0 > R1/X1 (a real zero-sequence/earth loop)."""
    return dict(r1=0.21e-3, x1=0.08e-3, r0=0.82e-3, x0=0.32e-3)


def test_zero_sequence_carries_earth_damping_positive_does_not():
    """Z0 gains a frequency-growing earth-return resistance; Z1 stays earth-free."""
    freqs = _freqs()
    p = _seq_line()
    z1 = positive_sequence_z(p["r1"], p["x1"], F0, freqs)
    z0 = zero_sequence_harmonic_z(p["r0"], p["x0"], F0, freqs)

    # at f0 both reproduce their inputs.
    assert abs(float(z0[0].real) - p["r0"]) < 1e-12
    assert abs(float(z1[0].real) - p["r1"]) < 1e-12
    # Shipped law: X0 scales as a geometric inductance, exactly like X1.
    torch.testing.assert_close(z0.imag, p["x0"] * (freqs / F0), rtol=1e-14, atol=0.0)
    # The Carson option bends it sub-linear and the guard keeps it non-negative.
    z0_sub = zero_sequence_harmonic_z(
        p["r0"], p["x0"], F0, freqs, x0_frequency="carson_sublinear"
    )
    assert torch.all(z0_sub.imag[1:] < z0.imag[1:])
    assert torch.all(z0_sub.imag >= 0)
    torch.testing.assert_close(z0_sub.real, z0.real)
    r0_growth = float(z0.real[-1] - z0.real[0])
    r1_growth = float(z1.real[-1] - z1.real[0])
    assert (
        r0_growth > 5.0 * r1_growth
    )  # earth-return damping lives in the zero sequence


def test_zero_sequence_skin_rise_is_the_phase_conductors():
    """``R0(h) - R0`` equals ``R1*(m(R1) - 1)`` plus the Carson earth increment.

    The phase conductor contributes ``R1`` to ``R0`` and rises with its own skin curve;
    the return-path remainder ``R0 - R1`` stays constant. Fitting the curve to ``R0``
    instead (a conductor with a quarter of the cross-section for ``R0 = 4*R1``) loses
    most of the rise: ``m(R1) = 1.63`` against ``m(R0) = 1.07`` at order 25.
    """
    freqs = torch.tensor([50.0, 1250.0], dtype=RDT)
    r1, r0, x1, x0 = 0.208e-3, 0.832e-3, 0.080e-3, 0.240e-3  # NAYY 4x150, ohm/m
    m1 = skin_resistance_multiplier(r1, 50.0, freqs)
    m0 = skin_resistance_multiplier(r0, 50.0, freqs)
    assert float(m1[-1]) == pytest.approx(1.63, abs=0.01)
    assert float(m0[-1]) == pytest.approx(1.07, abs=0.01)

    z_abc = sequence_aware_phase_z(r1, x1, r0, x0, 50.0, freqs)
    z0 = z_abc.sum(-1).mean(-1)  # [H]
    z1 = z_abc[..., 0, 0] - z_abc[..., 0, 1]
    earth = 3.0 * math.pi**2 * 1e-7 * (freqs - 50.0)
    torch.testing.assert_close(z1.real, r1 * m1, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(
        z0.real, r1 * m1 + (r0 - r1) + earth, rtol=1e-12, atol=0.0
    )
    # Without the phase resistance the whole R0 is one fictitious conductor.
    legacy = zero_sequence_harmonic_z(r0, x0, 50.0, freqs)
    torch.testing.assert_close(legacy.real, r0 * m0 + earth, rtol=1e-12, atol=0.0)


def test_zero_sequence_reduces_to_positive_without_earth():
    """earth_resistance_coeff=0 -> Z0 is a pure conductor sequence (== positive model)."""
    freqs = _freqs()
    p = _seq_line()
    z0_no_earth = zero_sequence_harmonic_z(
        p["r1"],
        p["x1"],
        F0,
        freqs,
        earth_resistance_coeff=0.0,
        earth_reactance_coeff=0.0,
    )
    z1 = positive_sequence_z(p["r1"], p["x1"], F0, freqs)
    assert torch.allclose(z0_no_earth, z1, atol=1e-15)


def test_sequence_aware_phase_matrix_roundtrips():
    """Z_abc(h) decomposes back to exactly (Z0, Z1, Z1) — recombination is consistent."""
    freqs = _freqs()
    p = _seq_line()
    zabc = sequence_aware_phase_z(p["r1"], p["x1"], p["r0"], p["x0"], F0, freqs)
    assert zabc.shape == (len(ORDERS), 3, 3)
    z1 = positive_sequence_z(p["r1"], p["x1"], F0, freqs)
    z0 = zero_sequence_harmonic_z(p["r0"], p["x0"], F0, freqs, phase_resistance=p["r1"])
    zr0, zr1, zr2 = sequence_impedances(zabc)
    assert torch.allclose(zr1, z1, atol=1e-12)
    assert torch.allclose(zr2, z1, atol=1e-12)
    assert torch.allclose(zr0, z0, atol=1e-12)
    # symmetric/transposed structure: equal diagonals, equal off-diagonals.
    diag = zabc.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, diag[..., :1].expand_as(diag), atol=1e-15)


def test_sequence_to_phase_z_matches_textbook():
    """Z_self=(Z0+2Z1)/3, Z_mutual=(Z0-Z1)/3 (inverse Fortescue, balanced)."""
    freqs = torch.tensor([F0], dtype=RDT)
    z1 = torch.tensor([[1.0 + 2.0j]], dtype=torch.complex128)
    z0 = torch.tensor([[3.0 + 9.0j]], dtype=torch.complex128)
    zabc = sequence_to_phase_z(z1, z0)
    zs = (z0 + 2 * z1) / 3
    zm = (z0 - z1) / 3
    assert torch.allclose(zabc[0, 0, 0, 0], zs[0, 0])
    assert torch.allclose(zabc[0, 0, 0, 1], zm[0, 0])
    _ = freqs


def test_sequence_aware_assembly_recovers_damped_zero_sequence():
    """End-to-end: a tagged 3-phase line's assembled Y(h) -> Z1 earth-free, Z0 damped."""
    import math

    import numpy as np

    from pgml.assembly import assemble_network_ybus
    from pgml.geometry.synthesis import apply_sequence_aware_harmonic_model
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    f0 = 50.0
    r1, x1, r0, x0 = 0.30e-3, 0.30e-3, 0.60e-3, 1.20e-3
    z1c, z0c = complex(r1, x1), complex(r0, x0)
    zs, zm = (z0c + 2 * z1c) / 3, (z0c - z1c) / 3
    w = 2 * math.pi * f0
    rm = [[zs.real if i == j else zm.real for j in range(3)] for i in range(3)]
    lm = [[(zs.imag if i == j else zm.imag) / w for j in range(3)] for i in range(3)]
    ph = (Phase.A, Phase.B, Phase.C)
    grid = Grid(
        base_frequency_hz=f0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ph),
            Node(id=2, u_rated_v=400.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=rm,
                series_inductance_h_per_m=lm,
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0, 230.0, 230.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )
    apply_sequence_aware_harmonic_model(grid)
    assert grid.branches[0].harmonic_line_model == "sequence_aware"

    a = np.exp(2j * np.pi / 3)
    amat = np.array([[1, 1, 1], [1, a * a, a], [1, a, a * a]])
    ainv = np.linalg.inv(amat)

    def seq_z(h):
        yb = assemble_network_ybus(grid, [h * f0], dtype=torch.complex128).Y[0].numpy()
        z = np.linalg.inv(-yb[0:3, 3:6]) / 100.0
        zseq = ainv @ z @ amat
        return zseq[1, 1], zseq[0, 0]

    z1_1, z0_1 = seq_z(1)
    z1_13, z0_13 = seq_z(13)
    # h=1 recovers the input sequence impedances.
    assert abs(z1_1 - z1c) < 1e-9 and abs(z0_1 - z0c) < 1e-9
    # positive sequence: R ~ constant, X ∝ h (earth-free).
    assert abs(z1_13.real - r1) / r1 < 0.2
    assert abs(z1_13.imag - x1 * 13) / (x1 * 13) < 1e-6
    # zero sequence: R grows strongly (earth-return damping concentrated here).
    assert z0_13.real > 3.0 * z0_1.real
    assert (z0_13.real / z1_13.real) > 2.0 * (z0_1.real / z1_1.real)


@pytest.mark.opendss
def test_opendss_native_1phase_rx_carries_earth_floor():
    """Native OpenDSS 1-phase R/X line: the earth term enters the single self-Z (floor).

    With no second phase to cancel against, the Carson Rg/Xg of the line code surface
    directly — R rises and X scales sub-linearly. This is the OpenDSS setup our
    single-conductor synthesis matches, and why it differs from the positive sequence.
    """
    r1, x1 = 0.36, 0.30
    z = _dss_series_z(
        f"New Line.l1 phases=1 bus1=a bus2=b r1={r1} x1={x1} length=1 units=km",
        1,
        _DSS_ORDERS,
    )[:, 0, 0]
    # earth floor: reactance below naive h·X1, resistance grows with frequency.
    assert (z[-1].imag / (x1 * _DSS_ORDERS[-1])) < 0.95
    assert z[-1].real > z[0].real * 1.2
