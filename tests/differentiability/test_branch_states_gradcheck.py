"""float64 gradcheck of continuous branch states (differentiable topology).

A branch state in ``[0, 1]`` scales the branch's primitive admittance stamp, so
gradients must flow ``state -> Y -> solve -> V`` through both the assembly tape
(linear path) and the IFT backward (nonlinear power flow).
"""

from __future__ import annotations

import torch

from pgml.assembly import assemble_network_ybus
from pgml.grids import synthetic_feeder
from pgml.solver import solve_power_flow

TIE = 30000


def test_gradcheck_assembly_masked_stamp():
    grid = synthetic_feeder(8, n_feeders=2, tie_switches=1)

    def f(state):
        return assemble_network_ybus(grid, [50.0], branch_states={TIE: state}).Y

    s = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (s,), eps=1e-6, atol=1e-8)


def test_gradcheck_power_flow_through_ift():
    grid = synthetic_feeder(8, n_feeders=2, tie_switches=1)

    def f(state):
        # tight tol + a large FD step: the fixed point converges to ~1e-11 of the
        # 1e4-V scale, so eps must sit well above that convergence noise (the
        # convention of the other IFT gradchecks in this suite).
        return solve_power_flow(grid, branch_states={TIE: state}, tol=1e-11).v

    s = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (s,), eps=1e-3, atol=1e-4, rtol=1e-3)


def test_batched_state_gradients_are_per_scenario():
    grid = synthetic_feeder(8, n_feeders=2, tie_switches=1)
    s = torch.tensor([0.3, 0.9], dtype=torch.float64, requires_grad=True)
    res = solve_power_flow(grid, branch_states={TIE: s})
    res.v.abs().sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    # each scenario's gradient must match its own single-state solve
    for i in range(2):
        si = s.detach()[i].clone().requires_grad_(True)
        ri = solve_power_flow(grid, branch_states={TIE: si})
        ri.v.abs().sum().backward()
        assert torch.allclose(si.grad, s.grad[i], rtol=1e-8)
