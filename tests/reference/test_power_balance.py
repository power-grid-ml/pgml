"""Complex power balance of the nonlinear power flow.

At convergence three independently derived quantities must reconcile:

- each const-P load's complex power, RECOVERED from the solved voltages and the
  KCL-exact per-branch terminal currents (``Yprim @ V_term`` — a different code
  path than the load model), equals its specification;
- the slack injection (again recovered from the branch currents at the source
  node) equals the total load power plus the total branch absorption (losses).

This ties the load stamps, the source model, the solver, and the per-branch
primitive currents together: a sign, scaling, or per-unit error in any one of
them breaks the balance.
"""

from __future__ import annotations

import torch

import pgml
from pgml.schemas.grid_schema import Load
from tests.fixtures.tiny_grids import single_phase_chain

CDT = torch.complex128


def _solved_state():
    grid = single_phase_chain()
    config = pgml.SimulationConfig(calculation="power_flow")
    return grid, pgml.simulate(grid, config, dtype=CDT)


def _node_injection_power(state) -> dict[int, complex]:
    """``{node_id: S_injected}`` recovered from the branch terminal currents.

    KCL at a node: the net device injection current equals the sum of the
    currents flowing INTO the attached branches, so ``S_inj = Σ_p V_p ·
    conj(Σ_branches I_into_branch)`` — positive = delivered into the network
    (a source), negative = consumed (a load).
    """
    index = state.index
    inj: dict[int, complex] = {}
    for bc in state.branch_currents():
        terminals = [(bc.from_node, bc.from_phases, bc.i_from)]
        if bc.to_node is not None:
            terminals.append((bc.to_node, bc.to_phases, bc.i_to))
        for node, phases, i_term in terminals:
            rows = [index.row(int(node), p) for p in phases]
            v = state.v[..., :, rows]  # [H, P]
            s = (v * torch.conj(i_term)).sum()
            inj[int(node)] = inj.get(int(node), 0j) + complex(s)
    return inj


def test_load_power_recovered_from_branch_currents():
    """The const-P spec is realized EXACTLY in the solved state (via KCL)."""
    grid, state = _solved_state()
    inj = _node_injection_power(state)
    loads = [a for a in grid.appliances if isinstance(a, Load)]
    assert loads, "fixture must carry a load"
    for load in loads:
        s_spec = complex(float(load.p_nom_w), float(load.q_nom_var))
        s_recovered = -inj[int(load.node)]  # consumed = minus the injection
        assert abs(s_recovered - s_spec) < 1e-6 * abs(s_spec), (
            f"load {load.id}: recovered {s_recovered:.6f} VA vs spec {s_spec:.6f} VA"
        )


def test_slack_covers_loads_plus_losses():
    """P_slack == Σ P_loads + Σ branch absorption, each from its own path."""
    grid, state = _solved_state()
    inj = _node_injection_power(state)
    loads = [a for a in grid.appliances if isinstance(a, Load)]
    src_node = int(next(a.node for a in grid.appliances if a.component == "source"))

    p_slack = inj[src_node].real
    p_loads = sum(float(ld.p_nom_w) for ld in loads)
    p_loss = 0.0
    for _bid, s_from, s_to in state.branch_flows():
        p_loss += float((s_from.sum() + s_to.sum()).real)

    assert p_slack > p_loads > 0.0  # losses are strictly positive
    assert abs(p_slack - (p_loads + p_loss)) < 1e-6 * p_slack, (
        f"P_slack={p_slack:.6f} W vs loads {p_loads:.6f} W + losses {p_loss:.6f} W"
    )
