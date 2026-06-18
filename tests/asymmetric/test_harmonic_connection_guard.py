"""Deferred-boundary guard for connection-aware harmonic injection.

The harmonic power flow forms each device's fundamental current ``i1`` from
PHASE-ROW voltages, which only equals the device TERMINAL voltage for a
WYE-to-ground load. Full connection-aware (DELTA / 4-wire) harmonic injection is
deferred to the per-phase-harmonics increment, so a device that actually injects a
harmonic on an unsupported topology must raise ``NotImplementedError`` rather than
silently produce wrong numbers. WYE-to-ground harmonic flow is unaffected.
"""

from __future__ import annotations

import math

import pytest
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
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)


def _spectrum():
    comps = [
        HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
        HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
    ]
    return StaticSpectrum(spectrum=SpectrumPoint(components=comps))


def _source(node_phases):
    n = len(node_phases)
    angles = [0.0, -120.0, 120.0, 0.0][:n]
    mags = [231.0, 231.0, 231.0, 0.0][:n]
    r = [[0.1 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[0.1 / W0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Source(
        id=10,
        node=1,
        phases=node_phases,
        u_ref_v=tuple(mags),
        u_angle_deg=tuple(angles),
        resistance_ohm=r,
        inductance_h=ll,
    )


def _line(node_phases):
    n = len(node_phases)
    r = [[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[0.5 / W0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    c = [[0.0] * n for _ in range(n)]
    return Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=node_phases,
        to_phases=node_phases,
        length_m=1.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=ll,
        shunt_capacitance_f_per_m=c,
    )


def _grid(load, node_phases):
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=node_phases),
            Node(id=2, u_rated_v=400.0, phases=node_phases),
        ],
        branches=[_line(node_phases)],
        appliances=[_source(node_phases), load],
    )


def test_delta_load_with_spectrum_raises():
    """A DELTA load that injects a harmonic raises NotImplementedError."""
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        connection=WindingConnection.DELTA,
        load_model=LoadModel.CONST_POWER,
        spectrum=_spectrum(),
    )
    grid = _grid(load, ABC)
    with pytest.raises(NotImplementedError, match="connection-aware"):
        solve_harmonic_flow(grid, [1, 5], slack="norton", dtype=CDT)


def test_wye_on_abcn_node_with_spectrum_raises():
    """A WYE load on a node carrying Phase.N that injects a harmonic raises."""
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        load_model=LoadModel.CONST_POWER,
        spectrum=_spectrum(),
    )
    grid = _grid(load, ABCN)
    with pytest.raises(NotImplementedError, match="connection-aware"):
        solve_harmonic_flow(grid, [1, 5], slack="norton", dtype=CDT)


def test_wye_to_ground_harmonic_flow_unaffected():
    """A WYE-to-ground load (no Phase.N) with a spectrum solves without raising."""
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        load_model=LoadModel.CONST_POWER,
        spectrum=_spectrum(),
    )
    grid = _grid(load, ABC)
    res = solve_harmonic_flow(grid, [1, 5], slack="norton", dtype=CDT)
    assert res.v.shape[-2] == 2  # two requested orders
    assert torch.isfinite(res.v.real).all() and torch.isfinite(res.v.imag).all()


def test_delta_load_without_spectrum_is_allowed():
    """A DELTA load with NO spectrum does not inject a harmonic, so it is allowed."""
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        connection=WindingConnection.DELTA,
        load_model=LoadModel.CONST_POWER,
    )
    grid = _grid(load, ABC)
    # No spectrum -> _resolve_spectrum returns None -> the guard is never reached.
    res = solve_harmonic_flow(grid, [1, 5], slack="norton", dtype=CDT)
    assert res.v.shape[-2] == 2
