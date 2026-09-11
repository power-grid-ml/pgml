"""Inductive shunt (OpenDSS ``Reactor``) against OpenDSS at h = 1, 5, 13.

An inductive shunt's susceptance magnitude FALLS as ``1/h``; represented as a fixed
capacitance it would RISE as ``h``, i.e. with the wrong sign of slope above the
fundamental. ``ShuntAppliance.inductance_h`` / ``ShuntReactor.inductance_h`` carry the
reactance so the stamp is ``Y(h) = G + 1/(j 2π h f0 L) + j 2π h f0 C``.

Two cases are compared against a live OpenDSS circuit, both as the admittance the
element adds to its bus row (the difference of the system matrices with and without the
element, which isolates the stamp from the rest of the network):

1. a three-phase shunt reactor on the IEEE-33 feeder (the standard compensation idiom);
2. a single-phase grounding reactor tying a four-wire line's neutral to ground (the
   zero-sequence grounding path itself).

OpenDSS's ``Reactor`` is a series ``R + jX`` branch. With ``R = 0`` (its default, and
both cases here) the series and the parallel form are identical at every frequency, so
the comparison is exact; the ``R > 0`` deviation is a separate test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_network_ybus, node_phase_index
from pgml.convert._common import PhaseMode
from pgml.schemas.grid_schema import Phase, ShuntAppliance

pytest.importorskip("opendssdirect")
pytestmark = pytest.mark.opendss

F0 = 50.0
ORDERS = (1, 5, 13)
CDT = torch.complex128
RTOL = 1e-9  # command-string precision; the physics is identical


def _dss_systemy():
    from pgml.evaluation.oracles.opendss_oracle import dss_systemy

    return dss_systemy()


def _dss_row(order_list, bus: str, conductor: int) -> int:
    want = f"{bus}.{conductor}".upper()
    return [e.upper() for e in order_list].index(want)


def _ieee33_circuit(*, with_reactor: bool, kvar: float = 1200.0, r_ohm: float = 0.0):
    """IEEE-33 as a three-phase OpenDSS circuit, optionally with a shunt reactor."""
    import opendssdirect as dss
    import pandapower.networks as pn

    net = pn.case33bw()
    vn = float(net.bus.at[0, "vn_kv"])
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={F0}")
    dss.Text.Command(
        f"New Circuit.ieee33_reactor basekv={vn} pu=1.0 phases=3 bus1=bus0.1.2.3 "
        f"frequency={F0} r1=1e-3 x1=1e-3 r0=1e-3 x0=1e-3"
    )
    for idx, row in net.line[net.line["in_service"]].iterrows():
        dss.Text.Command(
            f"New Line.line{idx} phases=3 "
            f"bus1=bus{int(row['from_bus'])}.1.2.3 bus2=bus{int(row['to_bus'])}.1.2.3 "
            f"r1={float(row['r_ohm_per_km'])} x1={float(row['x_ohm_per_km'])} "
            f"r0={4.0 * float(row['r_ohm_per_km'])} "
            f"x0={3.0 * float(row['x_ohm_per_km'])} c1=0 c0=0 "
            f"length={float(row['length_km'])} units=km"
        )
    if with_reactor:
        dss.Text.Command(
            f"New Reactor.rx1 bus1=bus17.1.2.3 phases=3 kv={vn} kvar={kvar} "
            f"conn=wye R={r_ohm}"
        )
    dss.Text.Command(f"Set voltagebases=[{vn}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    return dss


def _four_wire_circuit(*, with_reactor: bool, x_ohm: float = 5.0):
    """A two-bus four-conductor line with an optional neutral-to-ground reactor."""
    import opendssdirect as dss

    # 4x4 phase matrices (Ohm/km): identical conductors, one shared mutual.
    rm = "|".join(
        " ".join(f"{0.3 if i == j else 0.05:g}" for j in range(i + 1)) for i in range(4)
    )
    xm = "|".join(
        " ".join(f"{0.35 if i == j else 0.12:g}" for j in range(i + 1))
        for i in range(4)
    )
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={F0}")
    dss.Text.Command(
        f"New Circuit.fourwire basekv=0.4 pu=1.0 phases=3 bus1=a.1.2.3 "
        f"frequency={F0} r1=1e-3 x1=1e-3 r0=1e-3 x0=1e-3"
    )
    dss.Text.Command(
        f"New Line.l1 phases=4 bus1=a.1.2.3.4 bus2=b.1.2.3.4 "
        f"rmatrix=[{rm}] xmatrix=[{xm}] cmatrix=[0 | 0 0 | 0 0 0 | 0 0 0 0] "
        "length=0.1 units=km"
    )
    if with_reactor:
        # `bus1=b.4.0`: conductor 4 (the neutral) to the grounded reference.
        dss.Text.Command(f"New Reactor.gr bus1=b.4.0 phases=1 R=0 X={x_ohm}")
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    return dss


def _dss_element_admittance(builder, bus: str, conductor: int, orders, **kw):
    """``{h: Y_added}``: the admittance the element adds to one bus row in OpenDSS."""
    out = {}
    for h in orders:
        vals = []
        for flag in (True, False):
            dss = builder(with_reactor=flag, **kw)
            dss.Text.Command(f"set frequency={h * F0}")
            dss.Solution.BuildYMatrix(2, 1)
            y, order = _dss_systemy()
            vals.append(
                y[_dss_row(order, bus, conductor)][_dss_row(order, bus, conductor)]
            )
        out[h] = vals[0] - vals[1]
    return out


def _pgml_element_admittance(builder, bus: str, conductor: int, orders, **kw):
    """``{h: Y_added}``: the same quantity from the converted pgml grid's Y-bus."""
    from pgml.convert.opendss import to_grid

    phase = {1: Phase.A, 2: Phase.B, 3: Phase.C, 4: Phase.N}[conductor]
    out = {}
    grids = []
    for flag in (True, False):
        dss = builder(with_reactor=flag, **kw)
        grids.append(to_grid(dss, phase_mode=PhaseMode.THREE_PHASE))
    for h in orders:
        vals = []
        for grid, id_map in grids:
            index = node_phase_index(grid)
            row = index.row(id_map["bus"][bus], phase)
            y = assemble_network_ybus(grid, [h * F0], dtype=CDT).Y[0].numpy()
            vals.append(y[row, row])
        out[h] = vals[0] - vals[1]
    return out


def test_three_phase_reactor_on_ieee33_matches_opendss():
    """A 1.2 MVAr wye reactor on IEEE-33 bus 17: same Y at h = 1, 5, 13."""
    y_dss = _dss_element_admittance(_ieee33_circuit, "bus17", 1, ORDERS)
    y_pgml = _pgml_element_admittance(_ieee33_circuit, "bus17", 1, ORDERS)
    for h in ORDERS:
        rel = abs(y_pgml[h] - y_dss[h]) / abs(y_dss[h])
        assert rel < RTOL, f"h={h}: reactor Y rel error {rel:.3e}"
        # Inductive: susceptance is negative and its magnitude falls as 1/h.
        assert y_dss[h].imag < 0.0
        assert abs(y_pgml[h].imag * h - y_pgml[1].imag) < 1e-9 * abs(y_pgml[1].imag)


def test_grounding_reactor_matches_opendss():
    """A neutral-to-ground reactor (the zero-sequence grounding path): same Y."""
    y_dss = _dss_element_admittance(_four_wire_circuit, "b", 4, ORDERS)
    y_pgml = _pgml_element_admittance(_four_wire_circuit, "b", 4, ORDERS)
    for h in ORDERS:
        rel = abs(y_pgml[h] - y_dss[h]) / abs(y_dss[h])
        assert rel < RTOL, f"h={h}: grounding reactor Y rel error {rel:.3e}"
    # X = 5 Ohm at f0 -> Y(h) = 1/(j*h*5).
    for h in ORDERS:
        assert abs(y_pgml[h] - 1.0 / (1j * h * 5.0)) < 1e-9


def test_capacitance_model_would_have_the_wrong_slope():
    """The pre-inductance representation (a negative C) is exact only at h = 1.

    Quantifies what the inductance field buys: an inductive shunt entered as the
    equivalent capacitance ``C = B(f0)/(2*pi*f0) < 0`` matches at the fundamental and
    then moves the wrong way, by ``h**2`` in the susceptance.
    """
    y_dss = _dss_element_admittance(_four_wire_circuit, "b", 4, ORDERS)
    b0 = y_dss[1].imag  # < 0
    for h in ORDERS:
        b_fixed_c = b0 * h  # the old model
        b_true = y_dss[h].imag
        assert abs(b_true - b0 / h) < 1e-9 * abs(b0)
        if h > 1:
            assert abs(b_fixed_c / b_true - h * h) < 1e-6


def test_reactor_with_series_resistance_is_exact_at_f0_and_warns(caplog):
    """R > 0: the parallel conversion is exact at f0, and the deviation is logged."""
    import logging

    with caplog.at_level(logging.WARNING, logger="pgml"):
        y_pgml = _pgml_element_admittance(_ieee33_circuit, "bus17", 1, (1,), r_ohm=13.0)
    y_dss = _dss_element_admittance(_ieee33_circuit, "bus17", 1, (1,), r_ohm=13.0)
    rel = abs(y_pgml[1] - y_dss[1]) / abs(y_dss[1])
    assert rel < RTOL, f"f0 reactor Y rel error {rel:.3e}"
    assert [r for r in caplog.records if "in SERIES" in r.getMessage()], (
        "a series-R reactor must warn about the parallel-form conversion"
    )


def test_shunt_reactor_branch_stamp_matches_the_closed_form():
    """``ShuntReactor`` (the branch form) stamps the same G + 1/(jwL) + jwC."""
    from pgml.schemas.grid_schema import Grid, Node, ShuntReactor, Source

    ph = (Phase.A, Phase.B, Phase.C)
    ell, cap, cond = 0.4, 1e-6, 1e-4
    grid = Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=1, u_rated_v=400.0, phases=ph)],
        branches=[
            ShuntReactor(
                id=5,
                from_node=1,
                to_node=1,
                from_phases=ph,
                to_phases=ph,
                conductance_s=[
                    [cond if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                capacitance_f=[
                    [cap if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [ell if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
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
                    [1.0 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )
    y_net = assemble_network_ybus(grid, [h * F0 for h in ORDERS], dtype=CDT).Y.numpy()
    grid.branches = []
    y_src = assemble_network_ybus(grid, [h * F0 for h in ORDERS], dtype=CDT).Y.numpy()
    for k, h in enumerate(ORDERS):
        w = 2.0 * math.pi * h * F0
        expected = cond + 1.0 / (1j * w * ell) + 1j * w * cap
        got = y_net[k, 0, 0] - y_src[k, 0, 0]
        assert abs(got - expected) < 1e-12 * abs(expected), (h, got, expected)
    _ = np
    _ = ShuntAppliance
