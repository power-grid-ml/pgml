"""Differentiability gate for the public per-harmonic system assembly.

:func:`pgml.solver.assemble_harmonic_system` exposes the LINEAR harmonic system
``Y(h) V(h) = I(h)`` (orders ``h > 1``) that :func:`solve_harmonic_flow` solves, so
a downstream package can form the physics-consistency residual
``r(V) = Y(h)·V − I(h)`` (``≈ 0`` at the true ``V``).

These checks lock that:

- ``r`` is differentiable w.r.t. the candidate voltage ``V`` and w.r.t. a grid
  parameter (a line ``R``, which enters both ``Y(h)`` and — via the fundamental
  ``v1`` — ``I(h)``);
- ``r ≈ 0`` (within solve tolerance) when ``V`` is the true
  ``solve_harmonic(Y, I)``.
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
)
from pgml.solver import (
    assemble_harmonic_system,
    solve_harmonic,
    solve_power_flow,
)

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
HARM = [5, 7]
torch.manual_seed(0)


def _grid(r) -> Grid:
    comps = [
        HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
        HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
        HarmonicComponent(order=7, magnitude_pu=0.14, phase_deg=0.0),
    ]
    spec = StaticSpectrum(spectrum=SpectrumPoint(components=comps))
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
                series_resistance_ohm_per_m=r,
                series_inductance_h_per_m=[[0.5 / W0]],
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
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W0]],
            ),
            Load(
                id=2,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=500.0,
                load_model=LoadModel.CONST_POWER,
                spectrum=spec,
            ),
        ],
    )


def _residual(Y, Iinj, V):
    """Physics-consistency residual ``r = Y(h)·V − I(h)`` (real + imag stacked)."""
    r = torch.einsum("...hij,...hj->...hi", Y, V) - Iinj
    return torch.stack([r.real, r.imag], dim=-1)


def test_residual_zero_at_true_voltage():
    """``r = Y·V − I`` vanishes (solve tolerance) at the true ``V = solve(Y, I)``."""
    g = _grid([[0.5]])
    pf = solve_power_flow(g, slack="norton", dtype=CDT, symmetry="symmetric")
    Y, Iinj, index = assemble_harmonic_system(
        g, HARM, pf.v, symmetry="symmetric", dtype=CDT
    )
    assert tuple(Y.shape) == (len(HARM), index.size, index.size)
    assert tuple(Iinj.shape) == (len(HARM), index.size)

    V = solve_harmonic(Y, Iinj)
    r = _residual(Y, Iinj, V)
    assert r.abs().max() < 1e-9


def test_residual_gradcheck_wrt_voltage():
    """``r(V)`` is differentiable w.r.t. an arbitrary candidate voltage ``V``."""
    g = _grid([[0.5]])
    pf = solve_power_flow(g, slack="norton", dtype=CDT, symmetry="symmetric")
    Y, Iinj, _ = assemble_harmonic_system(
        g, HARM, pf.v, symmetry="symmetric", dtype=CDT
    )
    Y = Y.detach()
    Iinj = Iinj.detach()

    V0 = solve_harmonic(Y, Iinj).detach()
    V = (V0 + 0.01 * torch.randn_like(V0)).clone().requires_grad_(True)

    def fn(V):
        return _residual(Y, Iinj, V)

    assert torch.autograd.gradcheck(fn, (V,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_residual_gradcheck_wrt_line_r():
    """``r`` at the true ``V`` is differentiable w.r.t. a grid param (line ``R``).

    ``R`` enters both ``Y(h)`` and — through the fundamental ``v1`` it shifts — the
    injection ``I(h)``, exercising the full grid -> (Y, I) -> r path.
    """
    r_leaf = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)

    def fn(r):
        g = _grid(r)
        pf = solve_power_flow(g, slack="norton", dtype=CDT, symmetry="symmetric")
        Y, Iinj, _ = assemble_harmonic_system(
            g, HARM, pf.v, symmetry="symmetric", dtype=CDT
        )
        V = solve_harmonic(Y, Iinj)
        return _residual(Y, Iinj, V)

    assert torch.autograd.gradcheck(fn, (r_leaf,), eps=1e-6, atol=1e-5, rtol=1e-3)
