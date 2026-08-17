"""Differentiability gate for per-phase / connection-aware harmonic injection.

float64 gradcheck of the solved harmonic ``V`` w.r.t. a PER-ELEMENT harmonic
injection magnitude (a python LIST of per-entry leaf tensors ``[m_a, m_b, m_c]``
passed via the ``harmonic_injection`` override — the unambiguous per-element
convention) on a WYE 3-phase load AND through a DELTA load. Also covers the
``mag1 == 0`` dead-element guard (finite, zero gradient on the dead leaf). This
locks the ``M^T @ i_elem`` scatter and the per-element coefficient broadcast on
the differentiable path.
"""

from __future__ import annotations

import math

import torch

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
ABC = (Phase.A, Phase.B, Phase.C)
torch.manual_seed(0)


def _source():
    r = [[0.1 if i == j else 0.0 for j in range(3)] for i in range(3)]
    ll = [[0.1 / W0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    return Source(
        id=10,
        node=1,
        phases=ABC,
        u_ref_v=(231.0, 231.0, 231.0),
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=r,
        inductance_h=ll,
    )


def _line():
    r = [[0.5 if i == j else 0.0 for j in range(3)] for i in range(3)]
    ll = [[0.5 / W0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    c = [[0.0] * 3 for _ in range(3)]
    return Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=ABC,
        to_phases=ABC,
        length_m=1.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=ll,
        shunt_capacitance_f_per_m=c,
    )


def _grid(connection=None):
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        connection=connection,
        load_model=LoadModel.CONST_POWER,
    )
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABC),
        ],
        branches=[_line()],
        appliances=[_source(), load],
    )


def test_gradcheck_per_element_injection_wye():
    """V(h) gradchecked w.r.t. a per-element 5th-harmonic magnitude on a WYE load.

    The per-element override is a python LIST of per-entry leaf tensors
    ``[m_a, m_b, m_c]`` (the unambiguous per-element convention), so the gradient must
    flow back into each element's leaf independently.
    """
    m_a = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
    m_b = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    m_c = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)

    def fn(m_a, m_b, m_c):
        inj = {30: {1: (1.0, 0.0), 5: ([m_a, m_b, m_c], 0.0)}}
        return solve_harmonic_flow(
            _grid(),
            [1, 5],
            slack="norton",
            dtype=CDT,
            harmonic_injection=inj,
            symmetry="asymmetric",
        ).v

    assert torch.autograd.gradcheck(fn, (m_a, m_b, m_c), eps=1e-4, atol=1e-5, rtol=1e-3)


def test_gradcheck_dead_element_mag1_zero_branch():
    """Exercise the ``mag1 == 0`` guard with leaves: element 0 has a zero order-1 coeff.

    Element 0 carries ``mag_1 = 0`` (so the guarded 0/0 ratio kicks in and it injects
    NOTHING at any order), elements 1/2 are live. Gradients must be FINITE everywhere
    and exactly ZERO on the dead element's leaf (it never reaches the solved V).
    """
    t_b = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
    t_c = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    m_a = torch.tensor(0.30, dtype=torch.float64, requires_grad=True)
    m_b = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    m_c = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)

    def fn(t_b, t_c, m_a, m_b, m_c):
        # order 1: element 0 magnitude is 0.0 (dead); orders 5: per-element leaves.
        inj = {
            30: {
                1: ([0.0, t_b, t_c], 0.0),
                5: ([m_a, m_b, m_c], 0.0),
            }
        }
        return solve_harmonic_flow(
            _grid(),
            [1, 5],
            slack="norton",
            dtype=CDT,
            harmonic_injection=inj,
            symmetry="asymmetric",
        ).v

    assert torch.autograd.gradcheck(
        fn, (t_b, t_c, m_a, m_b, m_c), eps=1e-4, atol=1e-5, rtol=1e-3
    )

    # The dead element's 5th-harmonic leaf m_a must receive a finite ZERO gradient.
    v = fn(t_b, t_c, m_a, m_b, m_c)
    loss = (v.real**2 + v.imag**2).sum()
    grads = torch.autograd.grad(loss, (m_a, m_b, m_c))
    assert torch.isfinite(grads[0]) and grads[0] == 0.0  # dead element 0
    assert torch.isfinite(grads[1]) and grads[1] != 0.0  # live element 1
    assert torch.isfinite(grads[2]) and grads[2] != 0.0  # live element 2


def test_gradcheck_per_element_injection_delta():
    """V(h) gradchecked w.r.t. a per-element 5th-harmonic magnitude on a DELTA load."""
    m_a = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
    m_b = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    m_c = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)

    def fn(m_a, m_b, m_c):
        inj = {30: {1: (1.0, 0.0), 5: ([m_a, m_b, m_c], 0.0)}}
        return solve_harmonic_flow(
            _grid(connection=WindingConnection.DELTA),
            [1, 5],
            slack="norton",
            dtype=CDT,
            harmonic_injection=inj,
            symmetry="asymmetric",
        ).v

    assert torch.autograd.gradcheck(fn, (m_a, m_b, m_c), eps=1e-4, atol=1e-5, rtol=1e-3)
