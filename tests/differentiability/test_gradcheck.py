"""Differentiability gate: gradcheck of solved voltages w.r.t. grid params.

Gradients are checked through ``assemble_ybus`` + ``build_injections`` +
``solve_harmonic`` in complex128, for BOTH slack modes, w.r.t. line R/L/C and
source Z. Leaf tensors are injected via the ``param_overrides`` hook (additive,
optional keyword) so gradcheck can perturb the physical parameters without editing
the frozen schema.
"""

from __future__ import annotations

import torch

from pgml.assembly import assemble_ybus, build_injections, node_phase_index
from pgml.solver import solve_harmonic
from pgml.solver.power_flow import solve_power_flow

from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus

torch.manual_seed(0)


def _voltages_from_params(grid, f, overrides, *, fixed_rows=None, v_fixed=None):
    """assemble + solve, returning the complex node voltages [N]."""
    idx = node_phase_index(grid)
    yb = assemble_ybus(grid, [f], dtype=torch.complex128, param_overrides=overrides)
    i = build_injections(
        grid, [f], idx, dtype=torch.complex128, param_overrides=overrides
    )
    v = solve_harmonic(yb.Y, i, fixed_rows=fixed_rows, v_fixed=v_fixed)
    return v.reshape(-1)


def test_gradcheck_norton_line_rlc_and_source_z():
    grid = single_phase_chain()
    f = 50.0

    # Leaf tensors for line1 R/L/C and the source R/L (1x1 matrices).
    r1 = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    l1 = torch.tensor([[1.0e-6]], dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor([[1.0e-9]], dtype=torch.float64, requires_grad=True)
    rs = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    ls = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)

    def fn(r1, l1, c1, rs, ls):
        overrides = {
            ("line", 20, "series_resistance_ohm_per_m"): r1,
            ("line", 20, "series_inductance_h_per_m"): l1,
            ("line", 20, "shunt_capacitance_f_per_m"): c1,
            ("source", 10, "resistance_ohm"): rs,
            ("source", 10, "inductance_h"): ls,
        }
        return _voltages_from_params(grid, f, overrides)

    assert torch.autograd.gradcheck(
        fn, (r1, l1, c1, rs, ls), eps=1e-6, atol=1e-5, rtol=1e-3
    )


def test_gradcheck_ideal_slack_mode():
    grid = single_phase_chain()
    f = 50.0
    idx = node_phase_index(grid)
    slack_row = idx.row(1, grid.nodes[0].phases[0])
    fixed_rows = torch.tensor([slack_row], dtype=torch.int64)

    r1 = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    l1 = torch.tensor([[1.0e-6]], dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor([[1.0e-9]], dtype=torch.float64, requires_grad=True)
    v_fixed = torch.tensor([230.0 + 0.0j], dtype=torch.complex128, requires_grad=True)

    def fn(r1, l1, c1, v_fixed):
        overrides = {
            ("line", 20, "series_resistance_ohm_per_m"): r1,
            ("line", 20, "series_inductance_h_per_m"): l1,
            ("line", 20, "shunt_capacitance_f_per_m"): c1,
        }
        return _voltages_from_params(
            grid, f, overrides, fixed_rows=fixed_rows, v_fixed=v_fixed
        )

    assert torch.autograd.gradcheck(
        fn, (r1, l1, c1, v_fixed), eps=1e-6, atol=1e-5, rtol=1e-3
    )


def test_gradcheck_three_phase_line_matrices():
    grid = three_phase_two_bus()
    f = 50.0

    def sym(diag, off):
        return torch.tensor(
            [[diag if i == j else off for j in range(3)] for i in range(3)],
            dtype=torch.float64,
        )

    r = sym(1.0e-3, 1.0e-4).clone().requires_grad_(True)
    ind = sym(1.0e-6, 1.0e-7).clone().requires_grad_(True)

    def fn(r, ind):
        overrides = {
            ("line", 20, "series_resistance_ohm_per_m"): r,
            ("line", 20, "series_inductance_h_per_m"): ind,
        }
        return _voltages_from_params(grid, f, overrides)

    assert torch.autograd.gradcheck(fn, (r, ind), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_newton_power_flow():
    """The NONLINEAR const-P solve is differentiable via the IFT for method='newton'.

    Newton finds V* by a different forward iteration than the current-injection fixed
    point, but the gradient is attached by the SAME implicit-function-theorem backward
    (the converged V* is differentiable regardless of how it was found).
    """
    grid = single_phase_chain()

    r1 = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    l1 = torch.tensor([[1.0e-6]], dtype=torch.float64, requires_grad=True)
    rs = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)

    def fn(r1, l1, rs):
        overrides = {
            ("line", 20, "series_resistance_ohm_per_m"): r1,
            ("line", 20, "series_inductance_h_per_m"): l1,
            ("source", 10, "resistance_ohm"): rs,
        }
        return solve_power_flow(
            grid, slack="ideal", method="newton", tol=1e-12, max_iter=100,
            dtype=torch.complex128, param_overrides=overrides,
        ).v.reshape(-1)

    assert torch.autograd.gradcheck(fn, (r1, l1, rs), eps=1e-6, atol=1e-5, rtol=1e-3)
