"""Differentiability gate for the harmonic power flow.

Gradients flow through the fundamental (IFT in `solve_power_flow`) and the linear
per-harmonic solves to: line R/L, load P/Q (tensor duality), and the harmonic
INJECTION magnitudes (`harmonic_injection` override). Batched scenarios too.
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
from pgml.solver import solve_harmonic_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
torch.manual_seed(0)


def _grid(r, ind, p, q, *, spectrum=True) -> Grid:
    comps = [
        HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
        HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
        HarmonicComponent(order=7, magnitude_pu=0.14, phase_deg=0.0),
    ]
    spec = (
        StaticSpectrum(spectrum=SpectrumPoint(components=comps)) if spectrum else None
    )
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
                series_inductance_h_per_m=ind,
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
                p_nom_w=p,
                q_nom_var=q,
                load_model=LoadModel.CONST_POWER,
                spectrum=spec,
            ),
        ],
    )


def test_gradcheck_line_rl():
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[0.5 / W0]], dtype=torch.float64, requires_grad=True)

    def fn(r, ind):
        return solve_harmonic_flow(
            _grid(r, ind, 2000.0, 500.0), [1, 5, 7], slack="norton", dtype=CDT
        ).v

    assert torch.autograd.gradcheck(fn, (r, ind), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_load_pq():
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)

    def fn(p, q):
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], p, q), [1, 5, 7], slack="norton", dtype=CDT
        ).v

    assert torch.autograd.gradcheck(fn, (p, q), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_gradcheck_harmonic_injection_override():
    """Gradient w.r.t. a per-device harmonic injection magnitude (scenario override)."""
    m5 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    m7 = torch.tensor(0.14, dtype=torch.float64, requires_grad=True)

    def fn(m5, m7):
        inj = {2: {1: (1.0, 0.0), 5: (m5, 0.0), 7: (m7, 0.0)}}
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0, spectrum=False),
            [1, 5, 7],
            slack="norton",
            harmonic_injection=inj,
            dtype=CDT,
        ).v

    assert torch.autograd.gradcheck(fn, (m5, m7), eps=1e-4, atol=1e-5, rtol=1e-3)


def test_gradcheck_batched_injection_scenarios():
    """A scenario batch of injection magnitudes -> v [S, H, N], gradchecked."""
    m5 = torch.tensor([0.15, 0.20, 0.25], dtype=torch.float64, requires_grad=True)

    def fn(m5):
        inj = {2: {1: (1.0, 0.0), 5: (m5, 0.0), 7: (0.14, 0.0)}}
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0, spectrum=False),
            [1, 5, 7],
            slack="norton",
            harmonic_injection=inj,
            dtype=CDT,
        ).v

    out = fn(m5)
    assert out.shape == (3, 3, 2)  # [S, H, N]
    assert torch.autograd.gradcheck(fn, (m5,), eps=1e-4, atol=1e-5, rtol=1e-3)


def test_backward_reaches_all_leaves():
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    res = solve_harmonic_flow(
        _grid(r, [[0.5 / W0]], p, 500.0), [1, 5, 7], slack="norton", dtype=CDT
    )
    res.v.abs().sum().backward()
    for leaf in (r, p):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert r.grad.abs().sum() > 0 and p.grad.abs().sum() > 0
