"""Gradients through ``solve_harmonic_flow(param_overrides=...)``.

The harmonic path takes the same parameter-substitution hook as the power flow and the
assemblers, so a gradient w.r.t. a substituted tensor must flow into every order: through
``Y(h)`` (a line impedance), through the source stamp, and through the device power that
sets each injection's fundamental current. Without the hook those gradients reached the
engine only through the float/tensor duality (a tensor stored on the Grid), i.e. through a
different mechanism than the one the fundamental solve is gradchecked on.
"""

from __future__ import annotations

import torch

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.solver import assemble_harmonic_system, solve_harmonic_flow

CDT = torch.complex128
A = (Phase.A,)
torch.manual_seed(0)

#: A small converter-like spectrum on the load, so the harmonic orders carry current.
_SPECTRUM = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
            HarmonicComponent(order=7, magnitude_pu=0.1, phase_deg=0.0),
        ]
    )
)


def _two_bus() -> Grid:
    """Fixed single-phase two-bus grid: every varied parameter comes from an override."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=A),
            Node(id=2, u_rated_v=230.0, phases=A),
        ],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=A,
                to_phases=A,
                length_m=100.0,
                series_resistance_ohm_per_m=[[1.0e-3]],
                series_inductance_h_per_m=[[1.0e-6]],
                shunt_capacitance_f_per_m=[[1.0e-9]],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=A,
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1.0e-3]],
            ),
            Load(
                id=30,
                node=2,
                phases=A,
                p_nom_w=2000.0,
                q_nom_var=500.0,
                spectrum=_SPECTRUM,
            ),
        ],
    )


def test_gradcheck_line_r_override_all_orders():
    r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)

    def fn(r):
        return solve_harmonic_flow(
            _two_bus(),
            [1, 5, 7],
            dtype=CDT,
            param_overrides={("line", 20, "series_resistance_ohm_per_m"): r},
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_source_impedance_override():
    """The source stamp is the harmonic boundary; its impedance must be reachable."""
    rs = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)

    def fn(rs):
        return solve_harmonic_flow(
            _two_bus(),
            [1, 5],
            dtype=CDT,
            param_overrides={("source", 10, "resistance_ohm"): rs},
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (rs,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_load_power_override_drives_the_injection():
    """The device power sets the fundamental current every harmonic scales from."""
    p = torch.tensor([2000.0], dtype=torch.float64, requires_grad=True)

    def fn(p):
        return solve_harmonic_flow(
            _two_bus(),
            [1, 5, 7],
            dtype=CDT,
            param_overrides={("load", 30, "p_nom_per_phase_w"): p},
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p,), eps=1e-2, atol=1e-4, rtol=1e-3)


def test_override_equals_the_value_on_the_grid():
    """The hook is a substitution, not a second model: same numbers as a stored value."""
    grid = _two_bus()
    r_new = 2.0e-3
    overridden = solve_harmonic_flow(
        grid,
        [1, 5, 7],
        dtype=CDT,
        param_overrides={
            ("line", 20, "series_resistance_ohm_per_m"): torch.tensor(
                [[r_new]], dtype=torch.float64
            )
        },
    ).v
    on_grid = grid.model_copy(deep=True)
    next(b for b in on_grid.branches if b.id == 20).series_resistance_ohm_per_m = [
        [r_new]
    ]
    assert torch.allclose(overridden, solve_harmonic_flow(on_grid, [1, 5, 7]).v)
    # and it really moved the network
    assert not torch.allclose(overridden, solve_harmonic_flow(grid, [1, 5, 7]).v)


def test_assemble_harmonic_system_takes_the_same_override():
    """The assembled system a downstream residual uses sees the substitution too."""
    grid = _two_bus()
    v1 = solve_harmonic_flow(grid, [1], dtype=CDT).v[..., 0, :]
    r = torch.tensor([[2.0e-3]], dtype=torch.float64, requires_grad=True)
    y, i, _ = assemble_harmonic_system(
        grid,
        [5],
        v1,
        dtype=CDT,
        param_overrides={("line", 20, "series_resistance_ohm_per_m"): r},
    )
    y.abs().sum().backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()
    assert float(r.grad.abs().sum()) > 0.0
