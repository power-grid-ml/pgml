"""Differentiability gate of the enhancement sensitivities (float64 gradcheck).

The study's total stress is checked as a function of every leaf family at once: the
log impedance upgrade of each line and transformer, the transformer tap, the log
rating upgrade of each rated line and transformer, and new active and reactive power
at every bus. The base point keeps every stress term away from its onset and the two
transformer terminals apart (off-nominal tap, core branch), where the stress is smooth;
the solve runs at 1e-13 pu so the finite differences of gradcheck resolve the gradient.
"""

from __future__ import annotations

import torch

from pgml import SimulationConfig
from pgml.schemas.grid_schema import Load, Phase, WindingConnection
from pgml.studies import HostingLimits
from pgml.studies._common import base_operating_values
from pgml.studies.enhancement import _Problem
from tests.studies.grids import four_wire_two_bus, stressed_feeder

TIGHT = SimulationConfig(calculation="power_flow", tol=1e-13, tol_update_pu=1e-13)


def _problem(grid, limits: HostingLimits, config: SimulationConfig) -> _Problem:
    label, values = base_operating_values(
        grid, "generation", "nameplate", torch.float64, None
    )
    return _Problem(
        grid,
        limits=limits,
        base_label=label,
        base_values=values,
        config=config,
        onset=0.5,
        dtype=torch.complex128,
        device=None,
    )


def _check(problem: _Problem) -> None:
    leaves = problem.initial_leaves()
    assert float(problem.forward(*leaves).total.detach()) > 0.0

    def total(*x):
        return problem.forward(*x).total

    assert torch.autograd.gradcheck(total, leaves, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_stress_wrt_every_enhancement():
    grid = stressed_feeder(tap=1.02, magnetizing=True)
    _check(_problem(grid, HostingLimits(), TIGHT))


def test_gradcheck_stress_harmonic_calculation():
    grid = stressed_feeder(tap=1.02, magnetizing=True)
    cfg = TIGHT.model_copy(
        update={"calculation": "harmonic", "harmonic_orders": [1, 5]}
    )
    _check(_problem(grid, HostingLimits(), cfg))


def test_gradcheck_stress_neutral_and_unbalance_terms():
    grid = four_wire_two_bus(i_max_a=60.0)
    load = Load(
        id=7,
        node=2,
        phases=(Phase.A,),
        connection=WindingConnection.WYE,
        p_nom_w=9.0e3,
        q_nom_var=1.0e3,
    )
    grid = grid.model_copy(update={"appliances": [*grid.appliances, load]})
    limits = HostingLimits(neutral_loading=1.0, vuf_max=0.004)
    _check(_problem(grid, limits, TIGHT))
