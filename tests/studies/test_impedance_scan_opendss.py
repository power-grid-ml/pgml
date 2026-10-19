"""Impedance scan vs OpenDSS: ``Z = Y^-1`` from OpenDSS's own ``SystemY`` at each frequency.

The circuit is a three-phase 400 V feeder: a Vsource with distinct positive- and
zero-sequence impedance, two sequence-defined lines with shunt capacitance, and a
grounded-wye capacitor bank at the far end (the bank and the cable capacitance against the
source and line inductance produce parallel and series resonances inside the band). The
pgml grid carries the same data as phase matrices (self ``(Z0 + 2 Z1)/3``, mutual
``(Z0 - Z1)/3``) on the ``naive`` line model, which is what an OpenDSS line defined by
sequence values computes at another frequency when its earth-return terms are off
(``Rg=0 Xg=0``: R constant, X and B proportional to frequency).

OpenDSS rebuilds ``SystemY`` at each frequency (``set frequency=f`` + ``BuildYMatrix``);
the reference is the dense inverse of that matrix, restricted to the scanned rows in the
scan's row order. Measured with opendssdirect.py 0.9.4, complex128, x86-64: worst
element-wise relative difference 1.8e-13 (at 2450 Hz, a small transfer entry), worst
difference over the largest entry 3.4e-15. Tolerance: 1e-10 element-wise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import opendssdirect as dss
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("opendssdirect not installed", allow_module_level=True)

from pgml.schemas.grid_schema import Grid, Line, Node, Phase, ShuntAppliance, Source
from pgml.studies import impedance_scan

pytestmark = pytest.mark.opendss

F0 = 50.0
W0 = 2.0 * math.pi * F0
ABC = (Phase.A, Phase.B, Phase.C)
SOURCE = {"r1": 0.05, "x1": 0.4, "r0": 0.15, "x0": 1.2}
LINES = [
    (
        1,
        2,
        {"r1": 0.12, "x1": 0.30, "r0": 0.40, "x0": 1.10, "c1": 300e-9, "c0": 150e-9},
    ),
    (
        2,
        3,
        {"r1": 0.20, "x1": 0.25, "r0": 0.55, "x0": 0.90, "c1": 200e-9, "c0": 100e-9},
    ),
]
CAP_NODE, CAP_UF = 3, 80.0
FREQUENCIES_HZ = [50.0, 250.0, 650.0, 1150.0, 2450.0]


def _seq_matrix(z1: float, z0: float) -> list[list[float]]:
    self_, mutual = (z0 + 2.0 * z1) / 3.0, (z0 - z1) / 3.0
    return [[self_ if i == j else mutual for j in range(3)] for i in range(3)]


def _build_opendss() -> None:
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={F0:g}")
    s = SOURCE
    dss.Text.Command(
        "New Circuit.impedance_scan phases=3 basekv=0.4 bus1=bus1.1.2.3 pu=1 angle=0 "
        f"frequency={F0:g} r1={s['r1']} x1={s['x1']} r0={s['r0']} x0={s['x0']}"
    )
    for a, b, p in LINES:
        dss.Text.Command(
            f"New Line.l{a}{b} phases=3 bus1=bus{a}.1.2.3 bus2=bus{b}.1.2.3 "
            f"r1={p['r1']} x1={p['x1']} r0={p['r0']} x0={p['x0']} "
            f"c1={p['c1'] * 1e9} c0={p['c0'] * 1e9} length=1 units=none Rg=0 Xg=0"
        )
    dss.Text.Command(
        f"New Capacitor.bank bus1=bus{CAP_NODE}.1.2.3 phases=3 kv=0.4 "
        f"cuf=[{CAP_UF} {CAP_UF} {CAP_UF}]"
    )
    dss.Solution.Solve()


def _build_pgml() -> Grid:
    branches = [
        Line(
            id=10 + k,
            from_node=a,
            to_node=b,
            from_phases=ABC,
            to_phases=ABC,
            length_m=1.0,
            series_resistance_ohm_per_m=_seq_matrix(p["r1"], p["r0"]),
            series_inductance_h_per_m=_seq_matrix(p["x1"] / W0, p["x0"] / W0),
            shunt_capacitance_f_per_m=_seq_matrix(p["c1"], p["c0"]),
            harmonic_line_model="naive",
        )
        for k, (a, b, p) in enumerate(LINES)
    ]
    source = Source(
        id=100,
        node=1,
        phases=ABC,
        u_ref_v=(400.0 / math.sqrt(3.0),) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=_seq_matrix(SOURCE["r1"], SOURCE["r0"]),
        inductance_h=_seq_matrix(SOURCE["x1"] / W0, SOURCE["x0"] / W0),
    )
    bank = ShuntAppliance(
        id=200,
        node=CAP_NODE,
        phases=ABC,
        conductance_s=(0.0,) * 3,
        capacitance_f=(CAP_UF * 1e-6,) * 3,
    )
    return Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=i, u_rated_v=400.0, phases=ABC) for i in (1, 2, 3)],
        branches=branches,
        appliances=[source, bank],
    )


def _opendss_z(rows, frequency_hz: float) -> np.ndarray:
    """Dense inverse of OpenDSS's SystemY at one frequency, in the scan's row order."""
    dss.Text.Command(f"set frequency={frequency_hz:g}")
    dss.Solution.BuildYMatrix(2, 1)
    order = [e.upper() for e in dss.Circuit.YNodeOrder()]
    flat = np.asarray(dss.Circuit.SystemY(), dtype=np.float64)
    y = (flat[0::2] + 1j * flat[1::2]).reshape(len(order), len(order))
    pos = {name: k for k, name in enumerate(order)}
    perm = [pos[f"BUS{nid}.{'abc'.index(ph.value) + 1}"] for nid, ph in rows]
    return np.linalg.inv(y)[np.ix_(perm, perm)]


@pytest.fixture(autouse=True)
def _restore_opendss_base_frequency():
    """OpenDSS is one process-global engine and ``Clear`` keeps DefaultBaseFrequency:
    hand the next module the engine's own 60 Hz default back."""
    yield
    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=60")


def test_scan_matches_inverse_of_opendss_systemy():
    _build_opendss()
    scan = impedance_scan(_build_pgml(), frequencies_hz=FREQUENCIES_HZ, transfer=True)
    for k, f in enumerate(FREQUENCIES_HZ):
        z_ref = _opendss_z(scan.transfer_rows, f)
        np.testing.assert_allclose(
            scan.z_transfer_ohm[k].numpy(), z_ref, rtol=1e-10, atol=0, err_msg=f"{f} Hz"
        )
