"""float64 gradcheck of the Woodbury update-solve (linear primitive + power flow).

The low-rank update-solve is pure torch, so gradients must flow to the BASE matrix
that was factored, to the update terms ``U`` / ``C`` / ``V``, to the right-hand side
and to the ideal-slack reference. Wired into a switch-state sweep, the forward is
only an accelerator — the implicit-function-theorem backward rebuilds the per-state
admittance — so ``branch_states_method="woodbury"`` must reproduce the assemble
path's gradients w.r.t. both a continuous switch state and a network parameter.
"""

from __future__ import annotations

import torch

from pgml.assembly import node_phase_index
from pgml.grids import synthetic_feeder
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored
from pgml.solver.lowrank import branch_state_terms, solve_factored_updated

TIE = 30000


def _system(n: int = 8, k: int = 3, seed: int = 0):
    """A small, well-conditioned complex system plus a rank-``k`` update."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.complex128, generator=g) + 4.0 * torch.eye(
        n, dtype=torch.complex128
    )
    u = torch.zeros(n, k, dtype=torch.complex128)
    u[torch.tensor([1, 3, 6]), torch.arange(k)] = 1.0
    c = 0.5 * torch.randn(k, k, dtype=torch.complex128, generator=g)
    b = torch.randn(2, n, dtype=torch.complex128, generator=g)
    return a, u, c, b


def test_gradcheck_updated_solve_norton():
    a, u, c, b = _system()

    def f(a_, c_, u_, b_):
        return solve_factored_updated(lu_factor_system(a_), b_, u=u_, c=c_)

    args = (
        a.clone().requires_grad_(True),
        c.clone().requires_grad_(True),
        u.clone().requires_grad_(True),
        b.clone().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7)


def test_gradcheck_updated_solve_ideal_slack():
    """Gradients also reach ``v_fixed`` — the update modifies the slack coupling."""
    a, u, c, b = _system(seed=1)
    fixed = torch.tensor([0, 6], dtype=torch.int64)  # row 6 carries an update column
    vf = torch.randn(2, dtype=torch.complex128)

    def f(a_, c_, vf_):
        return solve_factored_updated(
            lu_factor_system(a_, fixed_rows=fixed), b, u=u, c=c_, v_fixed=vf_
        )

    args = (
        a.clone().requires_grad_(True),
        c.clone().requires_grad_(True),
        vf.clone().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7)


def test_gradcheck_state_through_branch_state_terms():
    """``state -> C -> updated solve`` on a real grid's switch stamp."""
    grid = synthetic_feeder(8, n_feeders=2, tie_switches=1)
    index = node_phase_index(grid)
    from pgml.assembly import assemble_ybus

    y_base = assemble_ybus(grid, [50.0], branch_states={TIE: 0.0}).Y
    fac = lu_factor_system(y_base)
    b = torch.randn(y_base.shape[-1], dtype=y_base.dtype)

    def f(state):
        u, c = branch_state_terms(grid, index, {TIE: state}, 50.0, base_states=0.0)
        return solve_factored_updated(fac, b, u=u, c=c)

    s = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (s,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_updated_solve_is_the_base_solve_when_the_core_vanishes():
    """A zero core leaves the base solution and its gradient untouched."""
    a, u, c, b = _system(seed=2)
    a = a.clone().requires_grad_(True)
    zero = torch.zeros_like(c)
    fac = lu_factor_system(a)
    (solve_factored_updated(fac, b, u=u, c=zero).abs().sum()).backward()
    g_upd = a.grad.clone()
    a.grad = None
    (solve_factored(lu_factor_system(a), b).abs().sum()).backward()
    assert torch.allclose(g_upd, a.grad, rtol=1e-12, atol=1e-12)


def test_gradcheck_power_flow_sweep_through_the_ift():
    """A continuous switch state stays differentiable through the fast path."""
    grid = synthetic_feeder(8, n_feeders=2, tie_switches=1)

    def f(state):
        # tight tol + a large FD step: the fixed point converges to ~1e-11 of the
        # 1e4-V scale, so eps must sit well above that convergence noise.
        return solve_power_flow(
            grid,
            branch_states={TIE: state},
            branch_states_method="woodbury",
            tol=1e-11,
        ).v

    s = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (s,), eps=1e-3, atol=1e-4, rtol=1e-3)


def test_sweep_gradients_match_the_assemble_path():
    """Batched states + a line parameter: same gradients as assembling per state."""
    grid = synthetic_feeder(8, n_feeders=2, tie_switches=1)
    line = next(b for b in grid.branches if b.id == 10001)
    r0 = torch.as_tensor(line.series_resistance_ohm_per_m, dtype=torch.float64)

    def run(method, s, r):
        res = solve_power_flow(
            grid,
            branch_states={TIE: s},
            branch_states_method=method,
            param_overrides={("line", 10001, "series_resistance_ohm_per_m"): r},
            tol=1e-11,
        )
        res.v.abs().sum().backward()

    grads = {}
    for method in ("assemble", "woodbury"):
        s = torch.tensor([0.3, 0.9], dtype=torch.float64, requires_grad=True)
        r = r0.clone().requires_grad_(True)
        run(method, s, r)
        grads[method] = (s.grad.clone(), r.grad.clone())
    # Tolerances are set by the FORWARD paths, not by the backward: both gradients are
    # evaluated at their own converged V*, and the Woodbury path reaches it through a
    # downdate of the closed-tie base whose measured rounding amplification
    # (LowRankUpdate.amplification) costs digits in the STATE derivative of the switched
    # branch. Measured deviation: ~1e-8 relative on the state gradient (0.22 / 0.025 in
    # magnitude) and ~3e-12 on the line-parameter gradient (~1e5 in magnitude).
    for (a, w), rtol in zip(zip(grads["assemble"], grads["woodbury"]), (1e-6, 1e-9)):
        assert torch.isfinite(w).all()
        assert torch.allclose(a, w, rtol=rtol, atol=1e-9 * float(a.abs().max()))
