"""Differentiability of the Carson/Deri geometry path (float64 gradcheck).

Gradients must flow from conductor geometry (Rdc, GMR, height, x-position, earth
resistivity) through the line constants AND through a full assemble+solve, so the
geometry->impedance path supports gradient-based parameter recovery / geometry tuning.
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import assemble_network_ybus
from pgml.geometry.carson import (
    INTERNAL_INDUCTANCE_MODELS,
    internal_reactance_ratio,
    kron_reduce,
    line_constants,
    series_impedance,
)
from pgml.schemas.grid_schema import (
    ConductorPlacement,
    Grid,
    Line,
    LineGeometry,
    Load,
    Node,
    Phase,
    Source,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128
RDT = torch.float64
PH = (Phase.A,)


def test_gradcheck_line_constants_single():
    freqs = torch.tensor([50.0, 250.0], dtype=RDT)

    def fn(rdc, gmr, y, rho):
        z, _ = line_constants(
            torch.tensor([0.0], dtype=RDT),
            y,
            gmr,
            rdc,
            torch.tensor([0.0102], dtype=RDT),
            rho,
            freqs,
            1,
        )
        return z.abs().sum()

    args = (
        torch.tensor([1.2e-4], dtype=RDT, requires_grad=True),
        torch.tensor([0.0078], dtype=RDT, requires_grad=True),
        torch.tensor([10.0], dtype=RDT, requires_grad=True),
        torch.tensor(100.0, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-5)


def test_gradcheck_three_phase_kron():
    freqs = torch.tensor([250.0], dtype=RDT)
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=RDT)

    def fn(rdc, gmr, y):
        z = series_impedance(x, y, gmr, rdc, 100.0, freqs)
        return kron_reduce(z, 3).abs().sum()

    args = (
        (torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=RDT) * 1e-3).requires_grad_(),
        torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=RDT, requires_grad=True),
        torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("model", INTERNAL_INDUCTANCE_MODELS)
def test_gradcheck_every_internal_inductance_model(model):
    """Each internal-inductance model is differentiable in Rdc, GMR, radius, y and rho.

    The three non-default models make the self-term spacing radius depend on frequency
    (``"gmr_power_frequency"`` through a mask over the frequency axis, ``"gmr_skin"``
    through the Bessel decay ``g(f)``), so the radius and the DC resistance enter the
    series reactance as well as the capacitance. The frequency axis is an INPUT, never a
    differentiated quantity, so the mask carries no gradient.
    """
    # Straddles the power-frequency band so the masked model sees both branches.
    freqs = torch.tensor([50.0, 250.0, 1250.0], dtype=RDT)

    def fn(rdc, gmr, radius, y, rho):
        z, c = line_constants(
            torch.tensor([0.0], dtype=RDT),
            y,
            gmr,
            rdc,
            radius,
            rho,
            freqs,
            1,
            internal_inductance=model,
        )
        return z.abs().sum() + 1e9 * c.abs().sum()

    args = (
        torch.tensor([1.2e-4], dtype=RDT, requires_grad=True),
        torch.tensor([0.0078], dtype=RDT, requires_grad=True),
        torch.tensor([0.0102], dtype=RDT, requires_grad=True),
        torch.tensor([10.0], dtype=RDT, requires_grad=True),
        torch.tensor(100.0, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-5)


def test_gradcheck_internal_reactance_ratio():
    """The internal-inductance decay ``g(f)`` is differentiable in Rdc."""
    freqs = torch.tensor([50.0, 1250.0, 2500.0], dtype=RDT)

    def fn(rdc):
        return internal_reactance_ratio(rdc, freqs).sum()

    rdc = torch.tensor([1.2e-4, 3.0e-4], dtype=RDT, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (rdc,), eps=1e-9, atol=1e-5)


def _geom_grid(rdc_leaf):
    geom = LineGeometry(
        conductors=[
            ConductorPlacement(
                phase=Phase.A,
                x_m=0.0,
                y_m=10.0,
                gmr_m=0.0078,
                radius_m=0.0102,
                r_dc_ohm_per_m=rdc_leaf,
            )
        ]
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=12660.0, phases=PH),
            Node(id=2, u_rated_v=12660.0, phases=PH),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=PH,
                to_phases=PH,
                length_m=1000.0,
                conductor_geometry=geom,
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=PH,
                u_ref_v=(12660.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1e-3]],
            ),
            Load(id=30, node=2, phases=PH, p_nom_w=2e5, q_nom_var=5e4),
        ],
    )


def test_gradient_through_assemble_and_solve():
    rdc = torch.tensor(1.2e-4, dtype=RDT, requires_grad=True)
    # through assembly
    yb = assemble_network_ybus(
        _geom_grid(rdc), torch.tensor([250.0], dtype=RDT), dtype=CDT
    ).Y
    yb.abs().sum().backward()
    assert (
        rdc.grad is not None and torch.isfinite(rdc.grad).all() and rdc.grad.abs() > 0
    )

    # through a nonlinear power-flow solve
    rdc2 = torch.tensor(1.2e-4, dtype=RDT, requires_grad=True)
    res = solve_power_flow(_geom_grid(rdc2), slack="ideal", dtype=CDT)
    res.v.abs().sum().backward()
    assert (
        rdc2.grad is not None
        and torch.isfinite(rdc2.grad).all()
        and rdc2.grad.abs() > 0
    )
