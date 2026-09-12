"""Differentiability gate: gradients through an exactly fused (zero-impedance) branch.

Collapsing a branch's terminal rows changes the row layout of the solve, so every
gradient path crosses two new operations: the reduced assembly (a scatter into shared
rows) and the prolongation back to the grid's own rows (a gather, whose adjoint is a
scatter-add). Both are linear and differentiable, which these float64 gradchecks pin —
for the node voltages, for the Kirchhoff-recovered current through the fused branch, and
at a harmonic order.

A fused branch's own impedance is NOT a parameter of the solved system (it has no stamp),
so its gradient is structurally zero. That is checked too: it must come out as a finite
zero rather than as a missing-gradient error.
"""

from __future__ import annotations

import torch

from pgml.assembly import branch_currents, device_current_injections
from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
    Switch,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow

A = (Phase.A,)
CDT = torch.complex128
torch.manual_seed(0)


def _grid(*, switch_r: float = 0.0, load_node: int = 3) -> Grid:
    """Source -- ideal switch -- load bus -- feeder -- load bus (20 kV, 1-phase)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=20_000.0, phases=A) for i in (1, 2, 3)],
        branches=[
            Switch(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=A,
                to_phases=A,
                closed=True,
                resistance_ohm=switch_r,
            ),
            Line(
                id=11,
                from_node=2,
                to_node=3,
                from_phases=A,
                to_phases=A,
                length_m=1_000.0,
                series_resistance_ohm_per_m=[[2.0e-4]],
                series_inductance_h_per_m=[[8.0e-7]],
                shunt_capacitance_f_per_m=[[1.0e-11]],
                harmonic_line_model="naive",
            ),
        ],
        appliances=[
            Source(
                id=20,
                node=1,
                phases=A,
                u_ref_v=(20_000.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[1.0e-6]],
                inductance_h=[[1.0e-12]],
            ),
            Load(id=21, node=2, phases=A, p_nom_w=3.0e5, q_nom_var=1.0e5),
            Load(id=22, node=load_node, phases=A, p_nom_w=9.0e5, q_nom_var=3.0e5),
        ],
    )


def test_gradcheck_voltages_w_r_t_a_line_next_to_a_fused_switch():
    """The reduced solve and the prolongation keep the implicit-function gradients."""
    grid = _grid()
    r = torch.tensor([[2.0e-4]], dtype=torch.float64, requires_grad=True)
    ell = torch.tensor([[8.0e-7]], dtype=torch.float64, requires_grad=True)

    def fn(r, ell):
        res = solve_power_flow(
            grid,
            dtype=CDT,
            tol=1e-12,
            tol_update_pu=1e-12,
            param_overrides={
                ("line", 11, "series_resistance_ohm_per_m"): r,
                ("line", 11, "series_inductance_h_per_m"): ell,
            },
        )
        assert res.fusion is not None
        return res.v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r, ell), eps=1e-9, atol=1e-6, rtol=1e-3)


def test_gradcheck_voltages_w_r_t_a_load_on_a_fused_node():
    """A device on a fused row injects into the shared row; its power stays a leaf."""
    grid = _grid()
    p = torch.tensor([3.0e5], dtype=torch.float64, requires_grad=True)

    def fn(p):
        res = solve_power_flow(
            grid,
            dtype=CDT,
            tol=1e-12,
            tol_update_pu=1e-12,
            param_overrides={("load", 21, "p_nom_per_phase_w"): p},
        )
        return res.v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (p,), eps=1e-2, atol=1e-6, rtol=1e-3)


def test_gradcheck_the_recovered_current_through_the_fused_branch():
    """The Kirchhoff recovery is a linear map on the solved state: gradients flow."""
    grid = _grid()
    r = torch.tensor([[2.0e-4]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([3.0e5], dtype=torch.float64, requires_grad=True)
    f0 = 50.0

    def fn(r, p):
        overrides = {
            ("line", 11, "series_resistance_ohm_per_m"): r,
            ("load", 21, "p_nom_per_phase_w"): p,
        }
        res = solve_power_flow(
            grid, dtype=CDT, tol=1e-12, tol_update_pu=1e-12, param_overrides=overrides
        )
        i_inj = -device_current_injections(
            grid, res.v, res.index, [f0], dtype=CDT, param_overrides=overrides
        )
        currents = branch_currents(
            grid,
            res.v.unsqueeze(-2),
            [f0],
            res.index,
            dtype=CDT,
            param_overrides=overrides,
            fusion=res.fusion,
            i_inj=i_inj,
        )
        fused = next(bc for bc in currents if bc.branch_id == 10)
        return fused.i_from.reshape(-1)

    # eps is kept above the solver's own termination noise: the recovered current is a
    # DERIVED quantity of a converged state, so a perturbation smaller than the
    # convergence tolerance's footprint in V* measures that noise instead of the
    # derivative (a central difference at 1e-6 reproduces the analytical value to 1e-8).
    assert torch.autograd.gradcheck(fn, (r, p), eps=1e-6, atol=1e-6, rtol=1e-3)


def test_gradcheck_harmonic_voltages_with_a_fused_branch():
    """One map serves every order: an ideal conductor is ideal at every frequency."""
    grid = _grid()
    grid = grid.model_copy(
        update={
            "appliances": [
                a.model_copy(
                    update={"spectrum": {"magnitudes": {1: 1.0, 5: 0.1}, "angles": {}}}
                )
                if getattr(a, "id", None) == 22
                else a
                for a in grid.appliances
            ]
        }
    )
    r = torch.tensor([[2.0e-4]], dtype=torch.float64, requires_grad=True)

    def fn(r):
        res = solve_harmonic_flow(
            grid,
            [1, 5],
            dtype=CDT,
            tol=1e-12,
            tol_update_pu=1e-12,
            param_overrides={("line", 11, "series_resistance_ohm_per_m"): r},
        )
        assert res.fusion is not None
        return res.v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r,), eps=1e-9, atol=1e-6, rtol=1e-3)


def test_a_fused_branch_parameter_has_a_structurally_zero_gradient():
    """Its impedance is not a parameter of the solved system — a finite zero, not an error."""
    grid = _grid()
    r_sw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    r_line = torch.tensor([[2.0e-4]], dtype=torch.float64, requires_grad=True)
    res = solve_power_flow(
        grid,
        dtype=CDT,
        tol=1e-12,
        tol_update_pu=1e-12,
        param_overrides={
            ("switch", 10, "resistance_ohm"): r_sw,
            ("line", 11, "series_resistance_ohm_per_m"): r_line,
        },
    )
    assert res.fusion is not None and res.fusion.fused_branch_ids == (10,)
    res.v.abs().sum().backward()
    assert r_sw.grad is not None
    assert float(r_sw.grad) == 0.0
    assert r_line.grad is not None and float(r_line.grad.abs().sum()) > 0.0


def test_a_finite_override_on_the_same_switch_restores_a_non_zero_gradient():
    """The same switch, substituted with a finite resistance, is stamped and differentiable."""
    grid = _grid()
    r_sw = torch.tensor(1.0e-4, dtype=torch.float64, requires_grad=True)
    res = solve_power_flow(
        grid,
        dtype=CDT,
        tol=1e-12,
        tol_update_pu=1e-12,
        param_overrides={("switch", 10, "resistance_ohm"): r_sw},
    )
    assert res.fusion is None  # the effective impedance is finite, so nothing fuses
    res.v.abs().sum().backward()
    assert float(r_sw.grad.abs()) > 0.0


def test_finite_difference_spot_check_of_the_fused_switch_current():
    """One entry of the recovered current against a central difference in the load power."""
    grid = _grid()
    f0 = 50.0

    def current(p_value: float) -> float:
        overrides = {
            ("load", 21, "p_nom_per_phase_w"): torch.tensor(
                [p_value], dtype=torch.float64
            )
        }
        res = solve_power_flow(
            grid, dtype=CDT, tol=1e-13, tol_update_pu=1e-13, param_overrides=overrides
        )
        i_inj = -device_current_injections(
            grid, res.v, res.index, [f0], dtype=CDT, param_overrides=overrides
        )
        bc = next(
            b
            for b in branch_currents(
                grid,
                res.v.unsqueeze(-2),
                [f0],
                res.index,
                dtype=CDT,
                param_overrides=overrides,
                fusion=res.fusion,
                i_inj=i_inj,
            )
            if b.branch_id == 10
        )
        return float(bc.i_from.abs().reshape(-1)[0])

    p0, h = 3.0e5, 1.0e2
    fd = (current(p0 + h) - current(p0 - h)) / (2 * h)

    p = torch.tensor([p0], dtype=torch.float64, requires_grad=True)
    overrides = {("load", 21, "p_nom_per_phase_w"): p}
    res = solve_power_flow(
        grid, dtype=CDT, tol=1e-13, tol_update_pu=1e-13, param_overrides=overrides
    )
    i_inj = -device_current_injections(
        grid, res.v, res.index, [f0], dtype=CDT, param_overrides=overrides
    )
    bc = next(
        b
        for b in branch_currents(
            grid,
            res.v.unsqueeze(-2),
            [f0],
            res.index,
            dtype=CDT,
            param_overrides=overrides,
            fusion=res.fusion,
            i_inj=i_inj,
        )
        if b.branch_id == 10
    )
    bc.i_from.abs().reshape(-1)[0].backward()
    assert abs(float(p.grad) - fd) < 1e-6 * abs(fd)
