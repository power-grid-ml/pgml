"""Differentiability gate for the transformer star-point terminal model.

Gradients of solved node voltages must pass ``gradcheck`` (float64) w.r.t. every
parameter of the terminal model: the leakage R / L and tap, the earthing R / X of
both star points (internal point eliminated by Kron reduction, and neutral row), the
centre-tap half-winding R / L, and the zero-sequence leakage of a three-phase unit
on the sequence-aware path. Two frequencies (fundamental and order 3) cover the
frequency-dependent terms. A leaf placed in the schema itself (float / tensor
duality) is checked through the nonlinear power flow (implicit-function adjoint).
"""

from __future__ import annotations

import math

import torch

from pgml.assembly import (
    assemble_ybus,
    base_voltage_per_row,
    build_injections,
    node_phase_index,
)
from pgml.schemas.grid_schema import (
    CentreTap,
    ComplexTap,
    Grid,
    GroundingImpedance,
    Load,
    LoadModel,
    Node,
    Phase,
    ShuntReactor,
    Source,
    Transformer,
    TransformerZeroSeq,
)
from pgml.solver import solve_harmonic, solve_power_flow

P = Phase
F0 = 50.0
FREQS = [F0, 3.0 * F0]


def _source(phases) -> Source:
    n = len(phases)
    return Source(
        id=1,
        node=1,
        phases=phases,
        u_ref_v=(20_000.0 / math.sqrt(3),) * n,
        u_angle_deg=(0.0, -120.0, 120.0)[:n],
        resistance_ohm=[[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)],
        inductance_h=[[5.0e-3 if i == j else 0.0 for j in range(n)] for i in range(n)],
    )


def _centre_tap_grid(schema_leaf=None) -> Grid:
    """Phase-to-earth primary behind an electrode; centre tap on the neutral row."""
    lv = (P.A, P.B, P.N)
    xfmr = Transformer(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(P.A,),
        to_phases=lv,
        s_rated_va=50e3,
        u_rated_from_v=20_000.0 / math.sqrt(3),
        u_rated_to_v=440.0,
        from_connection="wye_grounded",
        to_connection="wye_grounded",
        series_resistance_ohm=0.0465,
        series_inductance_h=1.8e-4,
        from_grounding=GroundingImpedance(r_ohm=10.0, x_ohm=2.0),
        to_grounding=GroundingImpedance(
            r_ohm=0.3 if schema_leaf is None else schema_leaf
        ),
        centre_tap=CentreTap(),
    )
    loads = [
        Load(id=30, node=2, phases=(P.A,), p_nom_w=8e3, q_nom_var=2e3),
        Load(id=31, node=2, phases=(P.B,), p_nom_w=3e3, q_nom_var=1e3),
    ]
    if schema_leaf is None:
        for ld in loads:
            ld.load_model = LoadModel.CONST_IMPEDANCE
    nodes = [
        Node(id=1, u_rated_v=20_000.0, phases=(P.A, P.B, P.C)),
        Node(id=2, u_rated_v=440.0, phases=lv),
    ]
    return Grid(
        base_frequency_hz=F0,
        nodes=nodes,
        branches=[xfmr],
        appliances=[_source((P.A, P.B, P.C)), *loads],
    )


def _ynyn_grid() -> Grid:
    """YNyn: HV star earthed through an impedance (internal), LV star on the N row."""
    abcn = (P.A, P.B, P.C, P.N)
    xfmr = Transformer(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(P.A, P.B, P.C),
        to_phases=abcn,
        s_rated_va=0.4e6,
        u_rated_from_v=20_000.0,
        u_rated_to_v=400.0,
        from_connection="wye_grounded",
        to_connection="wye_grounded",
        series_resistance_ohm=0.01,
        series_inductance_h=1.0e-4,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=0.0),
        from_grounding=GroundingImpedance(r_ohm=20.0, x_ohm=5.0),
        to_grounding=GroundingImpedance(r_ohm=0.3),
        zero_sequence=TransformerZeroSeq(r0_ohm=0.004, x0_ohm=0.0126),
    )
    electrode = ShuntReactor(
        id=21,
        from_node=2,
        to_node=2,
        from_phases=(P.N,),
        to_phases=(P.N,),
        conductance_s=[[1.0 / 25.0]],
        capacitance_f=[[0.0]],
    )
    loads = [
        Load(
            id=30 + k,
            node=2,
            phases=(p,),
            p_nom_w=w,
            q_nom_var=0.2 * w,
            load_model=LoadModel.CONST_IMPEDANCE,
        )
        for k, (p, w) in enumerate(((P.A, 20e3), (P.B, 5e3), (P.C, 10e3)))
    ]
    nodes = [
        Node(id=1, u_rated_v=20_000.0, phases=(P.A, P.B, P.C)),
        Node(id=2, u_rated_v=400.0, phases=abcn),
    ]
    return Grid(
        base_frequency_hz=F0,
        nodes=nodes,
        branches=[xfmr, electrode],
        appliances=[_source((P.A, P.B, P.C)), *loads],
    )


def _voltages(grid, overrides):
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, FREQS, dtype=torch.complex128, param_overrides=overrides)
    i = build_injections(
        grid, FREQS, idx, dtype=torch.complex128, param_overrides=overrides
    )
    # The order-3 RHS is the source Norton current only; perturb it so the harmonic
    # system carries a non-trivial solution whose gradient is checked too.
    i = i + torch.ones_like(i) * 1e-3
    v = solve_harmonic(yb.Y, i) / base_voltage_per_row(grid)  # per unit
    return torch.view_as_real(v).reshape(-1)


def _leaf(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float64, requires_grad=True)


def _check_scaled(grid, nominal: dict) -> None:
    """Gradcheck w.r.t. unit scale factors on each nominal value (well-scaled steps)."""
    names = tuple(nominal)
    scales = tuple(_leaf(1.0) for _ in names)

    def fn(*args):
        overrides = {
            ("transformer", 20, n): a * nominal[n] for n, a in zip(names, args)
        }
        return _voltages(grid, overrides)

    assert torch.autograd.gradcheck(fn, scales, eps=1e-6, atol=1e-7, rtol=1e-4)


def test_gradcheck_centre_tap_terminal_model():
    """Centre tap: leakage, tap, half winding and both earthing impedances."""
    _check_scaled(
        _centre_tap_grid(),
        {
            "series_resistance_ohm": 0.0465,
            "series_inductance_h": 1.8e-4,
            "tap_magnitude": 1.0,
            "centre_tap_half_resistance_ohm": 0.0116,
            "centre_tap_half_inductance_h": 1.8e-5,
            "from_grounding_resistance_ohm": 10.0,
            "from_grounding_reactance_ohm": 2.0,
            "to_grounding_resistance_ohm": 0.3,
            "to_grounding_reactance_ohm": 0.05,
        },
    )


def test_gradcheck_earthed_star_points_sequence_aware():
    """YNyn: Kron-eliminated HV star, earthed LV neutral row, Z0 leakage matrix."""
    _check_scaled(
        _ynyn_grid(),
        {
            "series_resistance_ohm": 0.01,
            "zero_sequence_resistance_ohm": 0.004,
            "zero_sequence_inductance_h": 4.0e-5,
            "from_grounding_resistance_ohm": 20.0,
            "from_grounding_reactance_ohm": 5.0,
            "to_grounding_resistance_ohm": 0.3,
        },
    )


def test_gradcheck_schema_earthing_leaf_through_power_flow():
    """A tensor in ``GroundingImpedance.r_ohm`` reaches the nonlinear solve's adjoint."""

    def fn(r_g):
        grid = _centre_tap_grid(schema_leaf=r_g)
        res = solve_power_flow(grid, slack="norton", dtype=torch.complex128, tol=1e-12)
        return torch.view_as_real(res.v).reshape(-1)

    assert torch.autograd.gradcheck(fn, (_leaf(0.3),), eps=1e-6, atol=1e-6, rtol=1e-3)
