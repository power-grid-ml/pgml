"""Differentiability of the positive-sequence harmonic line model (float64 gradcheck).

Gradients must flow from the line's R1/X1 through the positive-sequence impedance
``Z1(h)`` (skin-effect resistance + geometric reactance, no earth floor), and through a
full harmonic Y-bus assembly using the ``carson_skin_multiplier`` resistance law — so
the corrected model supports gradient-based parameter recovery just like the Carson
geometry path.
"""

from __future__ import annotations

import torch

from pgml.assembly import assemble_network_ybus
from pgml.geometry.carson import series_impedance
from pgml.geometry.sequence import (
    positive_sequence_z,
    sequence_aware_phase_z,
    skin_resistance_multiplier,
    two_conductor_geometry,
    zero_sequence_harmonic_z,
)
from pgml.geometry.synthesis import apply_positive_sequence_harmonic_model

from tests.fixtures.tiny_grids import single_phase_chain

RDT = torch.float64
CDT = torch.complex128


def test_gradcheck_positive_sequence_z():
    """Z1(h) is differentiable w.r.t. R1 and X1."""
    freqs = torch.tensor([50.0, 250.0, 650.0], dtype=RDT)

    def fn(r1, x1):
        return positive_sequence_z(r1, x1, 50.0, freqs).abs().sum()

    args = (
        torch.tensor(3.6e-4, dtype=RDT, requires_grad=True),
        torch.tensor(3.0e-4, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_skin_multiplier():
    """The skin-effect multiplier m(h) is differentiable w.r.t. R1 (the fit Rdc)."""
    freqs = torch.tensor([50.0, 350.0, 750.0], dtype=RDT)

    def fn(r1):
        return skin_resistance_multiplier(r1, 50.0, freqs).sum()

    args = (torch.tensor(4.0e-4, dtype=RDT, requires_grad=True),)
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_two_conductor_loop_batched():
    """The physical go/return Carson loop is differentiable w.r.t. geometry."""
    geom = two_conductor_geometry(3.6e-4, 3.0e-4, 50.0, radius_m=0.0102)
    freqs = torch.tensor([50.0, 350.0], dtype=RDT)

    def fn(gmr, rdc, spacing):
        x = torch.stack([torch.zeros((), dtype=RDT), spacing])
        y = torch.full((2,), geom["height_m"], dtype=RDT)
        gmrv = torch.stack([gmr, gmr])
        rdcv = torch.stack([rdc, rdc])
        z = series_impedance(x, y, gmrv, rdcv, 100.0, freqs)
        zl = z[..., 0, 0] - z[..., 0, 1] - z[..., 1, 0] + z[..., 1, 1]
        return zl.abs().sum()

    args = (
        torch.tensor(geom["gmr_m"], dtype=RDT, requires_grad=True),
        torch.tensor(geom["rdc_ohm_per_m"], dtype=RDT, requires_grad=True),
        torch.tensor(geom["spacing_m"], dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_sequence_aware_phase_z():
    """The sequence-aware phase matrix Z_abc(h) is differentiable in R1/X1/R0/X0."""
    freqs = torch.tensor([50.0, 250.0, 650.0], dtype=RDT)

    def fn(r1, x1, r0, x0):
        return sequence_aware_phase_z(r1, x1, r0, x0, 50.0, freqs).abs().sum()

    args = (
        torch.tensor(0.21e-3, dtype=RDT, requires_grad=True),
        torch.tensor(0.08e-3, dtype=RDT, requires_grad=True),
        torch.tensor(0.82e-3, dtype=RDT, requires_grad=True),
        torch.tensor(0.32e-3, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_zero_sequence_harmonic_z():
    """The earth-damped zero-sequence impedance is differentiable in R0/X0."""
    freqs = torch.tensor([50.0, 350.0], dtype=RDT)

    def fn(r0, x0):
        return zero_sequence_harmonic_z(r0, x0, 50.0, freqs).abs().sum()

    args = (
        torch.tensor(0.82e-3, dtype=RDT, requires_grad=True),
        torch.tensor(0.32e-3, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradient_through_sequence_aware_assembly():
    """R/L gradients flow through the sequence-aware assembly path at a harmonic."""
    import math

    from pgml.assembly import assemble_network_ybus
    from pgml.geometry.synthesis import apply_sequence_aware_harmonic_model
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    f0 = 50.0
    ph = (Phase.A, Phase.B, Phase.C)
    r_leaf = torch.tensor(
        [[0.5e-3 if i == j else 0.1e-3 for j in range(3)] for i in range(3)],
        dtype=RDT,
        requires_grad=True,
    )
    lm = [[2e-6 if i == j else 0.5e-6 for j in range(3)] for i in range(3)]
    grid = Grid(
        base_frequency_hz=f0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ph),
            Node(id=2, u_rated_v=400.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=[[0.0] * 3 for _ in range(3)],
                series_inductance_h_per_m=lm,
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0, 230.0, 230.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )
    apply_sequence_aware_harmonic_model(grid)
    overrides = {("line", 10, "series_resistance_ohm_per_m"): r_leaf}
    yb = assemble_network_ybus(
        grid, torch.tensor([650.0], dtype=RDT), dtype=CDT, param_overrides=overrides
    ).Y
    yb.abs().sum().backward()
    assert (
        r_leaf.grad is not None
        and torch.isfinite(r_leaf.grad).all()
        and r_leaf.grad.abs().sum() > 0
    )
    _ = math


def test_gradient_through_assemble_with_skin_law():
    """R-gradient flows through assemble at a harmonic with the skin multiplier active."""
    grid = single_phase_chain()
    apply_positive_sequence_harmonic_model(grid)
    line_id = next(
        b.id for b in grid.branches if hasattr(b, "series_resistance_ohm_per_m")
    )

    r_leaf = torch.tensor([[1.0e-3]], dtype=RDT, requires_grad=True)
    overrides = {("line", line_id, "series_resistance_ohm_per_m"): r_leaf}

    yb = assemble_network_ybus(
        grid, torch.tensor([550.0], dtype=RDT), dtype=CDT, param_overrides=overrides
    ).Y  # 11th harmonic -> skin multiplier > 1
    yb.abs().sum().backward()
    assert (
        r_leaf.grad is not None
        and torch.isfinite(r_leaf.grad).all()
        and r_leaf.grad.abs().sum() > 0
    )


def test_gradcheck_zero_sequence_earth_coefficients():
    """The earth-return coefficients are gradient leaves, not frozen constants.

    ``Line.earth_return`` may hold tensors, so the earth-return resistance and reactance
    coefficients and the ``X0`` exponent are part of the differentiable path (a harmonic
    state estimator can fit the earth path of a feeder whose soil is unknown).
    """
    freqs = torch.tensor([50.0, 350.0, 750.0], dtype=RDT)

    def fn(r0, x0, coeff, coeff_x, exponent):
        return (
            zero_sequence_harmonic_z(
                r0,
                x0,
                50.0,
                freqs,
                earth_resistance_coeff=coeff,
                earth_reactance_coeff=coeff_x,
                x0_frequency="carson_sublinear",
                x0_exponent=exponent,
            )
            .abs()
            .sum()
        )

    args = (
        torch.tensor(0.82e-3, dtype=RDT, requires_grad=True),
        torch.tensor(0.32e-3, dtype=RDT, requires_grad=True),
        torch.tensor(9.8696e-7, dtype=RDT, requires_grad=True),
        torch.tensor(1.2566e-6, dtype=RDT, requires_grad=True),
        torch.tensor(0.95, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradcheck_zero_sequence_r0_includes_earth_return():
    """The ``r0_includes_earth_return`` split keeps R0 differentiable."""
    freqs = torch.tensor([50.0, 350.0], dtype=RDT)

    def fn(r0, x0):
        return (
            zero_sequence_harmonic_z(r0, x0, 50.0, freqs, r0_includes_earth_return=True)
            .abs()
            .sum()
        )

    args = (
        torch.tensor(0.82e-3, dtype=RDT, requires_grad=True),
        torch.tensor(0.32e-3, dtype=RDT, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-9, atol=1e-6)


def test_gradient_through_per_line_earth_return_tensor():
    """A tensor earth coefficient on ``Line.earth_return`` receives a gradient."""
    from pgml.schemas.grid_schema import (
        EarthReturnModel,
        Grid,
        Line,
        Node,
        Phase,
        Source,
    )

    f0 = 50.0
    ph = (Phase.A, Phase.B, Phase.C)
    coeff = torch.tensor(9.8696e-7, dtype=RDT, requires_grad=True)
    rm = [[0.5e-3 if i == j else 0.1e-3 for j in range(3)] for i in range(3)]
    lm = [[2e-6 if i == j else 0.5e-6 for j in range(3)] for i in range(3)]
    grid = Grid(
        base_frequency_hz=f0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ph),
            Node(id=2, u_rated_v=400.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=rm,
                series_inductance_h_per_m=lm,
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
                harmonic_line_model="sequence_aware",
                earth_return=EarthReturnModel(
                    resistance_coeff_ohm_per_m_per_hz=coeff,
                    x0_frequency="carson_sublinear",
                ),
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0, 230.0, 230.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )
    yb = assemble_network_ybus(grid, torch.tensor([650.0], dtype=RDT), dtype=CDT).Y
    yb.abs().sum().backward()
    assert coeff.grad is not None and torch.isfinite(coeff.grad).all()
    assert float(coeff.grad.abs()) > 0.0


def test_gradcheck_rx_line_conductor_earth_split():
    """The skin multiplier scales the conductor part only, and stays differentiable.

    ``positive_sequence`` on a 3-phase line derives its skin curve from the line's own
    positive-sequence resistance (mean diagonal minus mean mutual) inside assembly, so
    the whole resistance matrix is one gradient leaf.
    """
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    ph = (Phase.A, Phase.B, Phase.C)
    lm = [[2e-6 if i == j else 0.5e-6 for j in range(3)] for i in range(3)]
    grid = Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ph),
            Node(id=2, u_rated_v=400.0, phases=ph),
        ],
        branches=[
            Line(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=[
                    [0.5e-3 if i == j else 0.1e-3 for j in range(3)] for i in range(3)
                ],
                series_inductance_h_per_m=lm,
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
                harmonic_line_model="positive_sequence",
                harmonic_skin_effect=True,
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ph,
                u_ref_v=(230.0, 230.0, 230.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )

    def fn(r):
        yb = assemble_network_ybus(
            grid,
            torch.tensor([650.0], dtype=RDT),
            dtype=CDT,
            param_overrides={("line", 10, "series_resistance_ohm_per_m"): r},
        ).Y
        return yb.abs().sum()

    r0 = torch.tensor(
        [[0.5e-3 if i == j else 0.1e-3 for j in range(3)] for i in range(3)],
        dtype=RDT,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(fn, (r0,), eps=1e-10, atol=1e-5)
