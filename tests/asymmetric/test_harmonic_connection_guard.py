"""The connection-aware harmonic injection accepts every load connection.

DELTA and WYE-on-a-Phase.N-node harmonic injection used to raise
``NotImplementedError`` (the deferred-boundary guard) because the fundamental
current was formed from phase-row voltages, not the device terminal voltage. The
per-phase / connection-aware increment now models the terminal voltage via the
same incidence ``M`` the load flow uses, so these topologies solve instead of
raising. This file pins that the guard is GONE and the solve is finite. The
numerical correctness vs an independent oracle lives in
``test_harmonic_per_phase.py``.
"""

from __future__ import annotations

import math

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


def test_delta_load_without_spectrum_is_allowed():
    """A DELTA load with NO spectrum injects no harmonic, so it stays linear."""
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
    res = solve_harmonic_flow(grid, [1, 5], slack="norton", dtype=CDT)
    assert res.v.shape[-2] == 2
