"""High-level public API: simulate, SimulationConfig, SolvedState, errors, facade."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

import pgml
from pgml.errors import (
    ComputationError,
    ConvergenceError,
    InputError,
    ModelingError,
    PgmError,
)
from pgml.schemas.grid_schema import Phase
from pgml.simulation import (
    ResultBundle,
    SimulationConfig,
    SolvedState,
    simulate,
    simulate_serializable,
)
from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus


# --------------------------------------------------------------------------- #
# facade
# --------------------------------------------------------------------------- #
def test_facade_reexports_public_surface():
    assert pgml.simulate is simulate
    assert pgml.SimulationConfig is SimulationConfig
    assert pgml.Grid.__name__ == "Grid"
    for name in (
        "PgmlError",
        "PgmError",  # deprecated alias, kept importable
        "InputError",
        "ComputationError",
        "ConvergenceError",
        "ConfigurationError",
        "ConversionError",
        "ModelingError",
    ):
        assert hasattr(pgml, name)
    assert isinstance(pgml.__version__, str)


# --------------------------------------------------------------------------- #
# exception hierarchy + REST status hints
# --------------------------------------------------------------------------- #
def test_exception_hierarchy_and_http_status():
    assert pgml.PgmError is pgml.PgmlError  # deprecated alias stays identical
    assert issubclass(InputError, PgmError) and InputError.http_status == 422
    assert (
        issubclass(ComputationError, PgmError) and ComputationError.http_status == 500
    )
    assert issubclass(ConvergenceError, ComputationError)
    # ModelingError is an InputError (422) AND backward-compatible NotImplementedError.
    assert issubclass(ModelingError, InputError)
    assert issubclass(ModelingError, NotImplementedError)
    assert ModelingError.http_status == 422
    err = ConvergenceError("nope", iterations=3, residual=1e-2)
    assert err.iterations == 3 and err.residual == 1e-2 and err.http_status == 500


# --------------------------------------------------------------------------- #
# SimulationConfig validation (pydantic ValidationError, not wrapped)
# --------------------------------------------------------------------------- #
def test_config_defaults_and_validation():
    assert SimulationConfig().calculation == "harmonic"
    with pytest.raises(ValidationError):
        SimulationConfig(calculation="harmonic", harmonic_orders=[])
    with pytest.raises(ValidationError):
        SimulationConfig(harmonic_orders=[1, -5])
    with pytest.raises(ValidationError):
        SimulationConfig(tol=0.0)
    with pytest.raises(ValidationError):
        SimulationConfig(unknown_field=1)  # extra="forbid"


# --------------------------------------------------------------------------- #
# simulate dispatch + SolvedState
# --------------------------------------------------------------------------- #
def test_simulate_power_flow():
    st = simulate(single_phase_chain(), SimulationConfig(calculation="power_flow"))
    assert isinstance(st, SolvedState)
    assert st.converged
    assert st.v.shape == (1, 3)  # [H=1, N=3]
    assert st.frequencies_hz.tolist() == [50.0]


def test_simulate_harmonic_orders_and_spectrum():
    st = simulate(
        three_phase_two_bus(),
        SimulationConfig(calculation="harmonic", harmonic_orders=[1, 3, 5]),
    )
    assert st.v.shape[0] == 3  # H
    assert st.frequencies_hz.tolist() == [50.0, 150.0, 250.0]
    spec = st.spectrum_at(2, Phase.A)
    assert spec.shape == (3,)
    assert float(st.thd(2, Phase.A)) >= 0.0


def test_default_config_is_harmonic():
    st = simulate(single_phase_chain())  # no config -> default harmonic
    assert st.config.calculation == "harmonic"
    assert st.v.dim() == 2


# --------------------------------------------------------------------------- #
# convergence: strict raises, non-strict returns the flag
# --------------------------------------------------------------------------- #
def test_convergence_error_strict_vs_nonstrict():
    cfg = SimulationConfig(calculation="power_flow", max_iter=1)
    with pytest.raises(ConvergenceError) as exc:
        simulate(single_phase_chain(), cfg)
    assert exc.value.iterations is not None
    # non-strict returns the (non-converged) state instead of raising
    st = simulate(single_phase_chain(), cfg, strict=False)
    assert st.converged is False


# --------------------------------------------------------------------------- #
# differentiability through the high-level entry point
# --------------------------------------------------------------------------- #
def test_simulate_is_differentiable():
    grid = single_phase_chain()
    r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)

    def loss(r):
        st = simulate(
            grid,
            SimulationConfig(calculation="power_flow"),
            param_overrides={("line", 20, "series_resistance_ohm_per_m"): r},
        )
        return st.node_voltages().abs().sum()

    out = loss(r)
    out.backward()
    assert r.grad is not None and float(r.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# serialization (node records; JSON round-trip)
# --------------------------------------------------------------------------- #
def test_to_result_set_nodes_serialize():
    st = simulate(single_phase_chain(), SimulationConfig(calculation="power_flow"))
    bundle = st.to_result_set(include_branches=False)
    assert isinstance(bundle, ResultBundle)
    assert len(bundle.nodes) == 3  # 3 nodes x 1 order
    assert bundle.diagnostics.converged is True
    assert bundle.result_set.base_frequency_hz == 50.0
    # round-trips through JSON
    restored = ResultBundle.model_validate_json(bundle.model_dump_json())
    assert restored.nodes[0].v_re == bundle.nodes[0].v_re


def test_to_result_set_rejects_batched():
    st = simulate(single_phase_chain(), SimulationConfig(calculation="power_flow"))
    st.v = st.v.unsqueeze(0)  # fake a batch dim
    with pytest.raises(ValueError):
        st.to_result_set()


def test_simulate_serializable_wrapper():
    bundle = simulate_serializable(
        single_phase_chain(),
        SimulationConfig(calculation="power_flow"),
    )
    assert isinstance(bundle, ResultBundle)
    assert bundle.nodes  # node voltages present


# --------------------------------------------------------------------------- #
# branch quantities (currents / flows) on the solved state
# --------------------------------------------------------------------------- #
def test_branch_currents_and_flows():
    st = simulate(
        three_phase_two_bus(),
        SimulationConfig(calculation="harmonic", harmonic_orders=[1, 5]),
    )
    currents = st.branch_currents()
    assert len(currents) == 1  # one line
    bc = currents[0]
    assert bc.i_from.shape == (2, 3)  # [H, P]
    flows = st.branch_flows()
    assert len(flows) == 1
    _bid, s_from, _s_to = flows[0]
    assert s_from.shape == (2, 3) and s_from.is_complex()


def test_solved_state_branch_currents_satisfy_kcl():
    """Sum of branch terminal currents into each node == the network Y @ V (KCL)."""
    from pgml.assembly import assemble_network_ybus

    grid = three_phase_two_bus()
    st = simulate(grid, SimulationConfig(calculation="power_flow"))
    v = st.v[0]  # [N] at f0
    yb = assemble_network_ybus(grid, [grid.base_frequency_hz], dtype=st.dtype)
    i_net = (yb.Y[0] @ v).detach()
    i_scattered = torch.zeros_like(i_net)
    for bc in st.branch_currents():
        for k, ph in enumerate(bc.from_phases):
            i_scattered[st.index.row(bc.from_node, ph)] += bc.i_from[0, k]
        if bc.to_node is not None:
            for k, ph in enumerate(bc.to_phases):
                i_scattered[st.index.row(bc.to_node, ph)] += bc.i_to[0, k]
    # KCL holds at non-source/non-load nodes; compare the branch-only contribution to
    # the network Y@V (both exclude device shunts since we used assemble_network_ybus).
    assert torch.allclose(i_scattered.detach(), i_net, atol=1e-6)


def test_to_result_set_with_branches_round_trips():
    st = simulate(
        three_phase_two_bus(),
        SimulationConfig(calculation="harmonic", harmonic_orders=[1, 5]),
    )
    bundle = st.to_result_set()
    assert bundle.branches  # branch records present
    assert bundle.branches[0].branch_kind == "line"
    restored = ResultBundle.model_validate_json(bundle.model_dump_json())
    assert restored.branches[0].i_from_re == bundle.branches[0].i_from_re


def test_branch_current_is_differentiable():
    grid = three_phase_two_bus()
    r = torch.tensor(
        [[1.0e-3 if i == j else 1.0e-4 for j in range(3)] for i in range(3)],
        dtype=torch.float64,
        requires_grad=True,
    )

    def loss(r):
        st = simulate(
            grid,
            SimulationConfig(calculation="power_flow"),
            param_overrides={("line", 20, "series_resistance_ohm_per_m"): r},
        )
        return st.branch_currents()[0].i_from.abs().sum()

    out = loss(r)
    out.backward()
    assert r.grad is not None and float(r.grad.abs().sum()) > 0.0
