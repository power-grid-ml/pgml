"""Frequency dependence of the const-Z device fold in ``assemble_ybus``.

``assemble_ybus`` is the LINEAR-model assembler: it folds each load/generator into a
constant-impedance shunt taken at the operating point. That shunt's conductance is a
resistance (frequency-flat), but its susceptance is the equivalent reactive element and
must scale like one::

    y(h) = P/|V0|^2 + j * B(f0) * h     (B(f0) > 0: capacitive, Q < 0)
    y(h) = P/|V0|^2 + j * B(f0) / h     (B(f0) < 0: inductive,  Q > 0)

with ``B(f0) = -Q/|V0|^2``. This is the parallel R-L / R-C branch of the classical
harmonic load model (OpenDSS's ``Load`` with ``%SeriesRL=0``), and it is exact at the
fundamental, so no load-flow result moves.

The harmonic power flow does not fold loads at all (it uses
``assemble_network_ybus`` plus current injections), so these tests are about
``assemble_ybus`` as public API: the values a user gets when building a harmonic Y-bus
by hand. The reference values here are hand-derived closed forms; the h=1 checks pin
that the fundamental is bit-identical to the operating-point admittance.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.assembly import assemble_network_ybus, assemble_ybus
from pgml.schemas.grid_schema import Generator, Grid, Load, Node, Phase, Source

CDT = torch.complex128
F0 = 50.0
ORDERS = (1, 2, 5, 13, 25)
U_LN = 230.0


def _one_bus_grid(p_w: float, q_var: float, *, generator: bool = False) -> Grid:
    """One node, one stiff source, one const-Z device (the Y-bus is 1x1)."""
    ph = (Phase.A,)
    dev = (
        Generator(id=2, node=1, phases=ph, p_nom_w=p_w, q_nom_var=q_var)
        if generator
        else Load(id=2, node=1, phases=ph, p_nom_w=p_w, q_nom_var=q_var)
    )
    return Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=1, u_rated_v=U_LN, phases=ph)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(U_LN,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[1.0]],
                inductance_h=[[1e-6]],
            ),
            dev,
        ],
    )


def _device_admittance(grid: Grid, orders) -> dict:
    """``{h: y_device}`` from the difference of the full and the passive Y-bus."""
    freqs = [h * F0 for h in orders]
    y_full = assemble_ybus(grid, freqs, dtype=CDT).Y.numpy()
    net = Grid(
        base_frequency_hz=grid.base_frequency_hz,
        nodes=grid.nodes,
        branches=grid.branches,
        appliances=[a for a in grid.appliances if isinstance(a, Source)],
    )
    # The source Norton is in both, so the difference is the device alone.
    y_src = assemble_ybus(net, freqs, dtype=CDT).Y.numpy()
    return {h: y_full[k, 0, 0] - y_src[k, 0, 0] for k, h in enumerate(orders)}


@pytest.mark.parametrize("generator", [False, True])
def test_inductive_load_susceptance_falls_with_order(generator):
    """Q > 0 (lagging) folds to a fixed inductance: ``B(h) = B(f0)/h``."""
    p_w, q_var = 2300.0, 1000.0
    grid = _one_bus_grid(p_w, q_var, generator=generator)
    sign = -1.0 if generator else 1.0
    g_expected = sign * p_w / U_LN**2
    b0 = -sign * q_var / U_LN**2
    y = _device_admittance(grid, ORDERS)
    for h in ORDERS:
        assert y[h].real == pytest.approx(g_expected, rel=1e-12)
        assert y[h].imag == pytest.approx(b0 / h if b0 < 0 else b0 * h, rel=1e-12)


def test_capacitive_load_susceptance_rises_with_order():
    """Q < 0 (leading) folds to a fixed capacitance: ``B(h) = B(f0)*h``."""
    p_w, q_var = 2300.0, -1000.0
    y = _device_admittance(_one_bus_grid(p_w, q_var), ORDERS)
    b0 = q_var / U_LN**2 * -1.0
    for h in ORDERS:
        assert y[h].imag == pytest.approx(b0 * h, rel=1e-12)


def test_fundamental_is_the_operating_point():
    """At f0 the fold is ``conj(P + jQ)/|V0|^2``, and asking for more orders changes
    nothing at f0.

    The susceptance is bit-identical (``clamp`` splits ``B`` into one zero and one exact
    term, and adding 0.0 is exact); the conductance is compared at ``1e-16`` absolute
    because isolating the device here subtracts the source Norton, which cancels the
    leading digits.
    """
    for q in (1000.0, -1000.0, 0.0):
        grid = _one_bus_grid(2300.0, q)
        y = _device_admittance(grid, (1,))
        expected = complex(2300.0, -q) / U_LN**2
        assert y[1].real == pytest.approx(expected.real, abs=1e-16)
        assert y[1].imag == expected.imag
        # The f0 slice does not depend on which other orders were requested.
        y_alone = assemble_ybus(grid, [F0], dtype=CDT).Y
        y_many = assemble_ybus(grid, [h * F0 for h in ORDERS], dtype=CDT).Y
        assert complex(y_many[0, 0, 0]) == complex(y_alone[0, 0, 0])


def test_zero_reactive_power_stays_real_at_every_order():
    """Q = 0 has no reactive element, so the fold is a pure conductance."""
    y = _device_admittance(_one_bus_grid(2300.0, 0.0), ORDERS)
    for h in ORDERS:
        assert y[h].imag == 0.0, h


def test_three_phase_delta_load_scales_the_same_way():
    """A DELTA-connected const-Z load keeps the ``1/h`` inductive scaling per leg."""
    from pgml.schemas.grid_schema import WindingConnection

    ph = (Phase.A, Phase.B, Phase.C)
    grid = Grid(
        base_frequency_hz=F0,
        nodes=[Node(id=1, u_rated_v=400.0, phases=ph)],
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
            ),
            Load(
                id=2,
                node=1,
                phases=ph,
                p_nom_w=9000.0,
                q_nom_var=3000.0,
                connection=WindingConnection.DELTA,
            ),
        ],
    )
    freqs = [h * F0 for h in (1, 5)]
    y = assemble_ybus(grid, freqs, dtype=CDT).Y
    y_net = assemble_network_ybus(grid, freqs, dtype=CDT).Y
    # Source Norton is frequency dependent; isolate the device by its off-diagonal
    # (the delta incidence puts -y_leg there and the network has nothing off-diagonal).
    off_1 = complex(y[0, 0, 1] - y_net[0, 0, 1])
    off_5 = complex(y[1, 0, 1] - y_net[1, 0, 1])
    assert off_1.imag != 0.0
    assert off_5.imag == pytest.approx(off_1.imag / 5.0, rel=1e-12)
    assert off_5.real == pytest.approx(off_1.real, rel=1e-12)


def test_harmonic_flow_is_unaffected():
    """The harmonic path never folds loads, so its matrix must not change.

    ``assemble_network_ybus`` (what ``solve_harmonic_flow`` uses) contains no device
    shunt at all, which is what makes the change above safe: it can only be seen by a
    caller of ``assemble_ybus`` at a frequency other than ``f0``.
    """
    grid = _one_bus_grid(2300.0, 1000.0)
    freqs = [h * F0 for h in ORDERS]
    y_net = assemble_network_ybus(grid, freqs, dtype=CDT).Y.numpy()
    # No branches, no device shunt and no source Norton: the passive matrix is zero at
    # every order, with or without the load.
    assert (y_net == 0).all()
    _ = math
