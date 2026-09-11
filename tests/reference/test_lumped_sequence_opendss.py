"""The lumped sequence harmonic line model against a matched OpenDSS R/X line.

OpenDSS frequency-corrects a sequence-defined (``r1/x1/r0/x0``) line with its
earth-return parameters ``Rg``/``Xg``: per matrix entry it adds ``Rg*(h-1)`` to the
resistance and scales the reactance as ``h*(X - 0.5*KXg*ln(h))`` with
``KXg = Xg/ln(658.5*sqrt(rho/f0))``. In sequence terms the correction cancels in ``Z1``
(earth return is excited by residual current only) and appears three times in ``Z0``::

    R0(h) = R0 + 3*Rg*(h-1)
    X0(h) = h*(X0 - 1.5*KXg*ln(h))

That is exactly pgml's lumped ``sequence_aware`` model with
``earth_resistance_coeff = Rg/f0``, ``earth_reactance_coeff = KXg/f0`` and
``x0_frequency='carson_sublinear'`` (``'linear'`` corresponds to ``Xg = 0``). These
tests feed the SAME earth parameters to both engines and compare the series impedance
``Z_abc(h)`` of one line directly, which isolates the line model from any solve:

- with the skin effect off the two models agree to floating point at every order;
- with the skin effect on (pgml's refinement, which OpenDSS does not apply to an R/X
  line) the deviation is the skin rise itself, bounded and growing with order.

The tolerance for the matched comparison (1e-8 relative) is set by the 10 significant
digits the stub writes into the OpenDSS command strings, not by the physics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_network_ybus
from pgml.defaults import get as cfg
from pgml.evaluation.oracles.opendss_oracle import _build_seq_aware_circuit_stub
from pgml.geometry.synthesis import apply_sequence_aware_harmonic_model
from pgml.schemas.grid_schema import EarthReturnModel, Grid, Line, Node, Phase, Source

pytest.importorskip("opendssdirect")
pytestmark = pytest.mark.opendss

F0 = 50.0
W0 = 2.0 * math.pi * F0
ORDERS = (1, 3, 5, 7, 11, 13, 25)
CDT = torch.complex128
RTOL_MATCHED = 1e-8  # command-string precision (measured ~1e-10)

# A representative LV cable: R0/R1 = 4, X0/X1 = 3 (the converter's invented ratios).
LINE = dict(r1=0.162e-3, x1=0.0554e-3, r0=0.648e-3, x0=0.1662e-3, length=100.0)

_A = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, np.exp(2j * np.pi / 3) ** 2, np.exp(2j * np.pi / 3)],
        [1.0, np.exp(2j * np.pi / 3), np.exp(2j * np.pi / 3) ** 2],
    ]
)
_AINV = np.linalg.inv(_A)


def _two_bus_grid(*, r1, x1, r0, x0, length, skin, law):
    """Two-bus 3-phase grid whose single line carries the sequence-aware model."""
    zs, zm = (r0 + 2.0 * r1) / 3.0, (r0 - r1) / 3.0
    xs, xm = (x0 + 2.0 * x1) / 3.0, (x0 - x1) / 3.0
    r_mat = [[zs if i == j else zm for j in range(3)] for i in range(3)]
    l_mat = [[(xs if i == j else xm) / W0 for j in range(3)] for i in range(3)]
    ph = (Phase.A, Phase.B, Phase.C)
    grid = Grid(
        base_frequency_hz=F0,
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
                length_m=length,
                series_resistance_ohm_per_m=r_mat,
                series_inductance_h_per_m=l_mat,
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0,) * 3,
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
    apply_sequence_aware_harmonic_model(grid, skin=skin)
    grid.branches[0].earth_return = EarthReturnModel(x0_frequency=law)
    return grid


def _pgml_z(grid, h):
    """Series impedance matrix ``Z_abc(h)`` (Ohm) from the assembled Y-bus."""
    y = assemble_network_ybus(grid, [h * F0], dtype=CDT).Y[0].numpy()
    return np.linalg.inv(-y[0:3, 3:6])


def _dss_z(grid, h):
    """Series impedance matrix ``Z_abc(h)`` (Ohm) of the same line inside OpenDSS."""
    import opendssdirect as dss

    _build_seq_aware_circuit_stub(grid, {1: "bus1", 2: "bus2"})
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    dss.Text.Command(f"set frequency={h * F0}")
    dss.Solution.BuildYMatrix(2, 1)
    dss.Circuit.SetActiveElement("Line.l10")
    yp = np.array(dss.CktElement.YPrim())
    n = int(round((len(yp) / 2) ** 0.5))
    yy = (yp[0::2] + 1j * yp[1::2]).reshape(n, n)
    return np.linalg.inv(-yy[: n // 2, n // 2 :])


def _seq(z):
    zs = _AINV @ z @ _A
    return zs[1, 1], zs[0, 0]  # Z1, Z0


@pytest.mark.parametrize("law", ["linear", "carson_sublinear"])
def test_matches_opendss_with_matched_earth_parameters(law):
    """Skin off: pgml's lumped Z_abc(h) equals OpenDSS's to floating point.

    Both the ``linear`` and the ``carson_sublinear`` zero-sequence reactance law are
    reproduced: the reference stub emits ``Xg = 0`` for the former (OpenDSS's own way of
    switching the earth-return reactance correction off) and the physical Carson ``Xg``
    for the latter.
    """
    grid = _two_bus_grid(**LINE, skin=False, law=law)
    for h in ORDERS:
        zp, zd = _pgml_z(grid, h), _dss_z(grid, h)
        err = np.abs(zp - zd).max() / np.abs(zd).max()
        assert err < RTOL_MATCHED, f"h={h} ({law}): Z_abc rel error {err:.3e}"


def test_fundamental_reproduces_the_stored_sequence_values():
    """At f0 both engines return the stored R1/X1/R0/X0 exactly."""
    grid = _two_bus_grid(**LINE, skin=True, law="carson_sublinear")
    z1, z0 = _seq(_pgml_z(grid, 1))
    length = LINE["length"]
    assert abs(z1 - complex(LINE["r1"], LINE["x1"]) * length) < 1e-12
    assert abs(z0 - complex(LINE["r0"], LINE["x0"]) * length) < 1e-12
    z1d, z0d = _seq(_dss_z(grid, 1))
    assert abs(z1d - z1) < 1e-9 * abs(z1)
    assert abs(z0d - z0) < 1e-9 * abs(z0)


def test_zero_sequence_resistance_follows_the_carson_law():
    """``R0(h) = R0 + 3*coeff*(f - f0)`` — the earth-return damping, by hand."""
    grid = _two_bus_grid(**LINE, skin=False, law="linear")
    coeff = cfg("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
    length = LINE["length"]
    for h in ORDERS:
        _z1, z0 = _seq(_pgml_z(grid, h))
        expected = (LINE["r0"] + 3.0 * coeff * (h - 1) * F0) * length
        assert abs(z0.real - expected) < 1e-12 * expected, h


def test_zero_sequence_reactance_sublinear_law():
    """``X0(h) = h*(X0 - 1.5*kx*f0*ln h)`` — the Carson/Deri decay, by hand."""
    grid = _two_bus_grid(**LINE, skin=False, law="carson_sublinear")
    kx = cfg("line.earth_return.reactance_coeff_ohm_per_m_per_hz")
    length = LINE["length"]
    for h in ORDERS:
        _z1, z0 = _seq(_pgml_z(grid, h))
        expected = h * (LINE["x0"] - 1.5 * kx * F0 * math.log(h)) * length
        assert abs(z0.imag - expected) < 1e-12 * max(abs(expected), 1e-12), h


def test_positive_sequence_is_earth_free_and_linear_in_h():
    """``Z1(h) = R1 + j*X1*h`` with the earth return cancelled (skin off)."""
    grid = _two_bus_grid(**LINE, skin=False, law="carson_sublinear")
    length = LINE["length"]
    for h in ORDERS:
        z1, _z0 = _seq(_pgml_z(grid, h))
        assert abs(z1.real - LINE["r1"] * length) < 1e-12
        assert abs(z1.imag - LINE["x1"] * h * length) < 1e-12


def test_skin_effect_is_the_only_deviation_from_opendss():
    """Skin on: the deviation is the skin rise alone, bounded and order-growing.

    OpenDSS does not apply a skin correction to a sequence-defined line, so this is a
    deliberate pgml refinement. The bounds pin its size: a few percent on ``Z1`` at the
    low orders, under 15 % at h=25 for this cable.
    """
    grid = _two_bus_grid(**LINE, skin=True, law="carson_sublinear")
    errs = []
    for h in ORDERS:
        z1p, _ = _seq(_pgml_z(grid, h))
        z1d, _ = _seq(_dss_z(grid, h))
        errs.append(abs(z1p - z1d) / abs(z1d))
    assert errs[0] < 1e-12  # exact at f0
    assert all(b >= a for a, b in zip(errs, errs[1:]))  # grows with order
    assert 0.01 < errs[-1] < 0.15, errs
