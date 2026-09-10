"""Forward-correctness of the harmonic power flow (`solve_harmonic_flow`).

The harmonic INJECTION convention is OpenDSS-exact (verified in
``docs/pgml/modeling/references/opendss/harmonics.md``). The network harmonic IMPEDANCE uses the
standard ``R const, X∝h`` model (OpenDSS adds a Carson earth-return correction we
postpone), so the rigorous correctness check is an INDEPENDENT numpy reimplementation
of that same model; the OpenDSS bus-voltage match is asserted only to ballpark.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import torch

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
# Line / source ohms at fundamental, and the load.
R_LINE, X_LINE = 0.5, 0.5
R_SRC, X_SRC = 0.1, 0.1
P_LOAD, Q_LOAD = 2000.0, 500.0
SPEC = [(1, 1.0, 0.0), (5, 0.2, 0.0), (7, 0.14, 0.0)]  # (order, mag_pu, phase_deg)


def _grid(spec=SPEC) -> Grid:
    comps = [
        HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a) for o, m, a in spec
    ]
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1.0,
                series_resistance_ohm_per_m=[[R_LINE]],
                series_inductance_h_per_m=[[X_LINE / W0]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[R_SRC]],
                inductance_h=[[X_SRC / W0]],
            ),
            Load(
                id=2,
                node=2,
                phases=(Phase.A,),
                p_nom_w=P_LOAD,
                q_nom_var=Q_LOAD,
                load_model=LoadModel.CONST_POWER,
                spectrum=StaticSpectrum(spectrum=SpectrumPoint(components=comps)),
            ),
        ],
    )


def _numpy_harmonic_v_ld(v_ld_fundamental: complex, orders) -> dict[int, complex]:
    """Independent numpy harmonic solve (simple R-const, X∝h model) at the load bus.

    Uses the fundamental load-bus voltage to form ``I1 = conj(S0)/conj(V1)`` then,
    per harmonic, builds the 2x2 Y (line series + source Norton shunt), injects the
    nodal harmonic current ``-I_h`` at the load bus, and solves.
    """
    s0 = complex(P_LOAD, Q_LOAD)
    i1 = np.conj(s0) / np.conj(v_ld_fundamental)
    spec = {o: (m, a) for o, m, a in SPEC}
    mag1, ang1 = spec[1]
    out = {}
    for h in orders:
        if h == 1:
            out[1] = v_ld_fundamental
            continue
        z_line = R_LINE + 1j * h * X_LINE
        z_src = R_SRC + 1j * h * X_SRC
        y_line = 1.0 / z_line
        y_src = 1.0 / z_src
        Y = np.array([[y_src + y_line, -y_line], [-y_line, y_line]], dtype=complex)
        mag_h, ang_h = spec.get(h, (0.0, 0.0))
        i_drawn = (
            (mag_h / mag1)
            * abs(i1)
            * cmath.exp(
                1j * (math.radians(ang_h) + h * (cmath.phase(i1) - math.radians(ang1)))
            )
        )
        rhs = np.array([0.0, -i_drawn], dtype=complex)  # nodal injection = -I_drawn
        v = np.linalg.solve(Y, rhs)
        out[h] = complex(v[1])
    return out


def test_matches_numpy_oracle():
    grid = _grid()
    orders = [1, 5, 7]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    assert res.pf.converged
    ld = res.index.row(2, Phase.A)
    v_fund = complex(res.v[orders.index(1), ld])
    ref = _numpy_harmonic_v_ld(v_fund, orders)
    for k, h in enumerate(orders):
        got = complex(res.v[k, ld])
        np.testing.assert_allclose(
            [got.real, got.imag],
            [ref[h].real, ref[h].imag],
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"order {h} mismatch vs numpy oracle",
        )


def test_fundamental_matches_power_flow():
    grid = _grid()
    res = solve_harmonic_flow(grid, [1, 5, 7], slack="norton", dtype=CDT)
    # Order 1 is exactly the fundamental power-flow solution carried in res.pf.
    torch.testing.assert_close(res.v[0], res.pf.v, rtol=0, atol=0)
    # And it agrees with an independent solve to fixed-point tolerance.
    pf = solve_power_flow(grid, slack="norton", dtype=CDT)
    torch.testing.assert_close(res.v[0], pf.v, rtol=0, atol=1e-8)


def test_opendss_ballpark():
    """Regression guard against OpenDSS values recorded once (harmonics.md), not a live
    oracle call. Fundamental exact; harmonics within ~4% (the residual is OpenDSS Carson
    earth-return + load-Y, both deferred). The live OpenDSS harmonic comparison is
    `test_cigre_lv_live_opendss.py` / `test_carson_harmonics_feeders.py`."""
    grid = _grid()
    orders = [1, 5, 7]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    ld = res.index.row(2, Phase.A)
    # OpenDSS AllBusVolts (NeglectLoadY=yes), |V_ld| per order, recorded once:
    recorded_mag = {1: 223.24690, 5: 5.51315, 7: 5.30916}
    for k, h in enumerate(orders):
        mag = abs(complex(res.v[k, ld]))
        rel = abs(mag - recorded_mag[h]) / recorded_mag[h]
        tol = 1e-4 if h == 1 else 0.04
        assert rel < tol, (
            f"order {h}: |V|={mag:.5f} vs recorded OpenDSS {recorded_mag[h]} (rel {rel:.4f})"
        )


def test_orders_shapes_and_frequencies():
    grid = _grid()
    orders = [1, 5, 7, 11]
    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    assert res.v.shape == (len(orders), res.index.size)
    np.testing.assert_allclose(
        res.frequencies_hz.numpy(), [o * F0 for o in orders], rtol=0, atol=0
    )


def test_harmonic_only_orders_without_fundamental():
    """Requesting only harmonics (no order 1) still works (PF runs internally)."""
    grid = _grid()
    res = solve_harmonic_flow(grid, [5, 7], slack="norton", dtype=CDT)
    assert res.v.shape == (2, res.index.size)
    # Compare to the full-orders run's 5th/7th.
    full = solve_harmonic_flow(grid, [1, 5, 7], slack="norton", dtype=CDT)
    torch.testing.assert_close(res.v[0], full.v[1], rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(res.v[1], full.v[2], rtol=1e-7, atol=1e-9)
