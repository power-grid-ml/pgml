"""Differentiability gate for the per-node harmonic "error" source.

Gradients must flow to ``source_power_va`` (and to the spectrum / grid params)
through the per-node harmonic disturbance source, for BOTH a Thevenin voltage
source (which also stamps ``Y_s`` on the diagonal of ``Y(h)``) and a Norton current
source. Physics: ``references/error_injection.md``.
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
)
from pgml.solver import NodeHarmonicSource, solve_harmonic_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0
torch.manual_seed(0)


def _grid(r, ind, p, q) -> Grid:
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
            ),
        ],
    )


def test_gradcheck_voltage_source_power():
    """Gradient of node voltages w.r.t. ``source_power_va`` of a VOLTAGE source.

    A voltage source stamps ``Y_s`` onto the diagonal of ``Y(h)`` AND ``I_N`` into
    the current, so this exercises both the (batched) ``Y`` add and the current add.
    """
    s_sc = torch.tensor(5.0e5, dtype=torch.float64, requires_grad=True)

    def fn(s_sc):
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 10.0), 7: (0.1, -20.0)},
            source_power_va=s_sc,
            kind="voltage",
        )
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0),
            [1, 5, 7],
            slack="norton",
            node_sources=[src],
            dtype=CDT,
        ).v

    assert torch.autograd.gradcheck(fn, (s_sc,), eps=1e-1, atol=1e-5, rtol=1e-3)


def test_gradcheck_current_source_power():
    """Gradient of node voltages w.r.t. ``source_power_va`` of a CURRENT source."""
    s_sc = torch.tensor(5.0e5, dtype=torch.float64, requires_grad=True)

    def fn(s_sc):
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 10.0)},
            source_power_va=s_sc,
            kind="current",
        )
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0),
            [1, 5],
            slack="norton",
            node_sources=[src],
            dtype=CDT,
        ).v

    assert torch.autograd.gradcheck(fn, (s_sc,), eps=1e-1, atol=1e-5, rtol=1e-3)


def test_gradcheck_source_spectrum_magnitude():
    """Gradient of node voltages w.r.t. the source SPECTRUM magnitude (voltage src)."""
    m5 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def fn(m5):
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (m5, 0.0)},
            source_power_va=1.0e6,
            kind="voltage",
        )
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0),
            [1, 5],
            slack="norton",
            node_sources=[src],
            dtype=CDT,
        ).v

    assert torch.autograd.gradcheck(fn, (m5,), eps=1e-4, atol=1e-5, rtol=1e-3)


def test_gradcheck_grid_params_with_source():
    """Gradient still flows to line R/L through the harmonics WITH a node source."""
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[0.5 / W0]], dtype=torch.float64, requires_grad=True)

    def fn(r, ind):
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
            source_power_va=1.0e6,
            kind="voltage",
        )
        return solve_harmonic_flow(
            _grid(r, ind, 2000.0, 500.0),
            [1, 5],
            slack="norton",
            node_sources=[src],
            dtype=CDT,
        ).v

    assert torch.autograd.gradcheck(fn, (r, ind), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_batched_source_power():
    """A SCENARIO batch of ``source_power_va`` -> batched Y(h) [S,H,N,N], gradchecked."""
    s_sc = torch.tensor([1.0e5, 5.0e5, 1.0e6], dtype=torch.float64, requires_grad=True)

    def fn(s_sc):
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
            source_power_va=s_sc,
            kind="voltage",
        )
        return solve_harmonic_flow(
            _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0),
            [1, 5],
            slack="norton",
            node_sources=[src],
            dtype=CDT,
        ).v

    out = fn(s_sc)
    assert out.shape == (3, 2, 2)  # [S, H, N]
    assert torch.autograd.gradcheck(fn, (s_sc,), eps=1e-1, atol=1e-5, rtol=1e-3)


def test_backward_reaches_source_power():
    s_sc = torch.tensor(5.0e5, dtype=torch.float64, requires_grad=True)
    src = NodeHarmonicSource(
        node_id=2,
        spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
        source_power_va=s_sc,
        kind="voltage",
    )
    res = solve_harmonic_flow(
        _grid([[0.5]], [[0.5 / W0]], 2000.0, 500.0),
        [1, 5],
        slack="norton",
        node_sources=[src],
        dtype=CDT,
    )
    res.v.abs().sum().backward()
    assert s_sc.grad is not None and torch.isfinite(s_sc.grad).all()
    assert s_sc.grad.abs().sum() > 0
