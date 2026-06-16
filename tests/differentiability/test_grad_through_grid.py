"""Schema tensor-duality: gradients flow through a tensor-bearing Grid directly.

This validates the duck-typed float/tensor duality of the schema: physical fields
may hold torch tensors (leaves), which pass through assembly + solve untouched, so
``loss.backward()`` reaches the grid parameters WITHOUT the ``param_overrides`` hook.
"""

from __future__ import annotations

import torch

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_harmonic

F0 = 50.0


def _build(r, ind, cap, *, r_src=None, l_src=None) -> Grid:
    """Two-bus single-phase grid; line R/L/C and source R/L may be tensors."""
    r_src = [[0.1]] if r_src is None else r_src
    l_src = [[1e-4]] if l_src is None else l_src
    return Grid(
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
                length_m=1000.0,
                series_resistance_ohm_per_m=r,
                series_inductance_h_per_m=ind,
                shunt_capacitance_f_per_m=cap,
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=r_src,
                inductance_h=l_src,
            ),
            Load(id=2, node=2, phases=(Phase.A,), p_nom_w=1.0e4, q_nom_var=1.0e3),
        ],
    )


def _solve(grid) -> torch.Tensor:
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, [F0], dtype=torch.complex128)
    i = build_injections(grid, [F0], idx, dtype=torch.complex128)
    return solve_harmonic(yb.Y, i)


def test_backward_reaches_line_and_source_tensors():
    r = torch.tensor([[1e-4]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-6]], dtype=torch.float64, requires_grad=True)
    cap = torch.tensor([[1e-12]], dtype=torch.float64, requires_grad=True)
    r_src = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    grid = _build(r, ind, cap, r_src=r_src)
    _solve(grid).abs().sum().backward()
    for leaf in (r, ind, cap, r_src):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert r.grad.abs().sum() > 0


def test_gradcheck_through_grid_line_rl():
    cap = torch.tensor([[1e-12]], dtype=torch.float64)

    def f(r, ind):
        return _solve(_build(r, ind, cap))

    r = torch.tensor([[1e-3]], dtype=torch.float64, requires_grad=True)
    ind = torch.tensor([[1e-5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (r, ind), eps=1e-9, atol=1e-6)


def test_float_grid_still_works_and_serializes():
    """A plain-float grid is unchanged: builds, solves, and round-trips to JSON."""
    grid = _build([[1e-3]], [[1e-5]], [[1e-12]])
    v = _solve(grid)
    assert v.shape == (1, 2) and torch.isfinite(v.abs()).all()
    # JSON round-trip (the serializable default path).
    dumped = grid.model_dump_json()
    assert '"series_resistance_ohm_per_m"' in dumped
    assert isinstance(Grid.model_validate_json(dumped), Grid)
