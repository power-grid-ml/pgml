"""Harmonics must initialize from the PV solution, including its phase allocation."""

import pytest
import torch

from pgml import SimulationConfig, simulate
from pgml.schemas import (
    HarmonicComponent,
    HarmonicImpedance,
    Phase,
    RegulatedQuantity,
    SpectrumPoint,
    StaticSpectrum,
    VoltageRegulation,
)
from pgml.solver import assemble_harmonic_system, solve_harmonic_flow
from tests.reference.test_voltage_regulating_generator import _grid


def harmonic_grid(reference="current", **kwargs):
    grid = _grid(**kwargs)
    gen = grid.appliances[2]
    gen.spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=0.1, phase_deg=17.0),
                HarmonicComponent(order=7, magnitude_pu=0.05, phase_deg=-9.0),
            ]
        )
    )
    gen.harmonic_impedance = HarmonicImpedance(
        resistance_ohm=2.0,
        inductance_h=0.01,
        spectrum_reference=reference,
        frequency_model="opendss_admittance"
        if reference == "opendss_voltage"
        else "series_rl",
    )
    return grid


def solve(grid, **kwargs):
    result = solve_harmonic_flow(
        grid, [1, 5, 7], tol=1e-11, max_iter=100, method="newton", **kwargs
    )
    assert result.converged
    return result


@pytest.mark.parametrize(
    "reference", ["current", "internal_voltage", "opendss_voltage"]
)
@pytest.mark.parametrize("setpoint,limit", [(1.0, None), (1.02, 5e5), (0.98, 5e5)])
def test_pv_and_equivalent_pq_initialize_identical_harmonics(
    reference, setpoint, limit
):
    grid = harmonic_grid(
        reference,
        regulation=VoltageRegulation(
            v_set_pu=setpoint, q_min_var=-limit if limit else None, q_max_var=limit
        ),
    )
    op = {22: {"q_var": 123.0}}  # ignored PV input must not leak into harmonics
    pv = solve(grid, operating_point=op)
    resolved = pv.pf.resolved_operating_point(op)
    assert op == {22: {"q_var": 123.0}}
    pq_grid = grid.model_copy(deep=True)
    pq_grid.appliances[2].voltage_regulation = None
    pq = solve(pq_grid, operating_point=resolved)
    torch.testing.assert_close(pv.v, pq.v, atol=2e-8, rtol=2e-10)
    y, i, _ = assemble_harmonic_system(grid, [5, 7], pv.pf.v, operating_point=resolved)
    torch.testing.assert_close(
        y @ pv.v[1:].unsqueeze(-1), i.unsqueeze(-1), atol=1e-8, rtol=1e-10
    )


@pytest.mark.parametrize("regulated", list(RegulatedQuantity))
@pytest.mark.parametrize("symmetry", ["asymmetric", "symmetric"])
def test_batched_unbalanced_phase_allocation_and_simulation_readout(
    regulated, symmetry, caplog
):
    abc = (Phase.A, Phase.B, Phase.C)
    grid = harmonic_grid(
        phases=abc,
        load_per_phase=[2e5, 3e5, 5e5],
        regulation=VoltageRegulation(v_set_pu=1.0, regulated=regulated),
    )
    if symmetry == "symmetric":
        # Symmetric input powers do not make an unbalanced physical network balanced.
        grid.appliances[1].p_nom_per_phase_w = None
        grid.branches[0].series_resistance_ohm_per_m[0][0] *= 0.5
        grid.branches[0].series_resistance_ohm_per_m[2][2] *= 1.5
    op = {22: {"p_w": torch.tensor([2e5, 3e5], dtype=torch.float64)}}
    pv = solve(grid, operating_point=op, symmetry=symmetry)
    resolved = pv.pf.resolved_operating_point(op)
    q_phase = pv.pf.regulation.q_per_phase_var[22]
    assert q_phase.shape == (2, 3)
    torch.testing.assert_close(q_phase.sum(-1), pv.pf.regulation.q_var[22])
    if regulated == RegulatedQuantity.PER_PHASE:
        assert (q_phase.max(-1).values - q_phase.min(-1).values > 1e4).all()
        if symmetry == "symmetric":
            assert (
                "symmetric calculation balances prescribed input powers, not solved outputs"
                in caplog.text
            )
    pq_grid = grid.model_copy(deep=True)
    pq_grid.appliances[2].voltage_regulation = None
    pq = solve(pq_grid, operating_point=resolved, symmetry="asymmetric")
    torch.testing.assert_close(pv.v, pq.v, atol=2e-8, rtol=2e-10)
    state = simulate(
        grid,
        SimulationConfig(
            harmonic_orders=[1, 5, 7], operating_point=op, symmetry=symmetry, tol=1e-11
        ),
    )
    expected = simulate(
        pq_grid,
        SimulationConfig(
            harmonic_orders=[1, 5, 7],
            operating_point=resolved,
            symmetry="asymmetric",
            tol=1e-11,
        ),
    )
    torch.testing.assert_close(
        state._nodal_injection(), expected._nodal_injection(), atol=2e-8, rtol=2e-10
    )


def test_symmetric_user_q_split_is_ignored_in_both_resolvers(caplog):
    from pgml.assembly._params import resolve_operating_power
    from pgml.assembly.ybus import build_injection_plan, injections_from_plan

    grid = harmonic_grid(
        phases=(Phase.A, Phase.B, Phase.C),
        regulation=VoltageRegulation(v_set_pu=1.0),
    )
    generator = grid.appliances[2]
    raw = {22: {"q_per_phase_var": (10.0, 20.0, 30.0)}}
    _, q = resolve_operating_power(generator, raw, asymmetric=False)
    assert q == [20.0, 20.0, 20.0]
    assert "phase allocation is ignored for a symmetric calculation" in caplog.text
    solved = solve(grid, operating_point=raw, symmetry="symmetric")
    assert "VOLTAGE-REGULATING generator; it is ignored" in caplog.text
    caplog.clear()
    plan = build_injection_plan(
        grid, solved.index, [50.0], operating_point=raw, symmetry="symmetric"
    )
    equal = build_injection_plan(
        grid,
        solved.index,
        [50.0],
        operating_point={22: {"q_var": 60.0}},
        symmetry="symmetric",
    )
    torch.testing.assert_close(
        injections_from_plan(plan, solved.pf.v),
        injections_from_plan(equal, solved.pf.v),
    )
    assert "phase allocation is ignored for a symmetric calculation" in caplog.text


def test_only_solved_readout_preserves_q_and_survives_harmonic_chunking(monkeypatch):
    from pgml.assembly._params import resolve_operating_power
    import pgml.solver.harmonic_flow as harmonic

    grid = harmonic_grid(
        phases=(Phase.A, Phase.B, Phase.C),
        regulation=VoltageRegulation(
            v_set_pu=1.0, regulated=RegulatedQuantity.PER_PHASE
        ),
    )
    grid.branches[0].series_resistance_ohm_per_m[0][0] *= 0.5
    grid.branches[0].series_resistance_ohm_per_m[2][2] *= 1.5
    op = {22: {"p_w": torch.tensor([2e5, 3e5], dtype=torch.float64)}}
    expected = solve(grid, operating_point=op, symmetry="symmetric")
    resolved = expected.pf.resolved_operating_point(op)
    _, q = resolve_operating_power(grid.appliances[2], resolved, asymmetric=False)
    torch.testing.assert_close(
        torch.stack(q, -1), expected.pf.regulation.q_per_phase_var[22]
    )
    # Copying just the values into a normal input dict must not grant an exemption.
    plain = {22: dict(resolved[22])}
    _, averaged = resolve_operating_power(grid.appliances[2], plain, asymmetric=False)
    torch.testing.assert_close(averaged[0], averaged[1])
    assert not torch.allclose(q[0], averaged[0])
    monkeypatch.setattr(harmonic, "_harmonic_system_budget_bytes", lambda: 1)
    chunked = solve(grid, operating_point=op, symmetry="symmetric")
    torch.testing.assert_close(chunked.v, expected.v, atol=2e-8, rtol=2e-10)


def test_solved_q_drives_optional_derived_generation_shunt(monkeypatch):
    monkeypatch.setattr(
        "pgml.assembly._load_shunt._generation_model", lambda: "load_style"
    )
    grid = harmonic_grid(regulation=VoltageRegulation(v_set_pu=1.0))
    grid.appliances[2].harmonic_impedance = None
    pv = solve(grid)
    resolved = pv.pf.resolved_operating_point()
    pq_grid = grid.model_copy(deep=True)
    pq_grid.appliances[2].voltage_regulation = None
    pq = solve(pq_grid, operating_point=resolved)
    torch.testing.assert_close(pv.v, pq.v, atol=2e-8, rtol=2e-10)


def test_nominal_reactive_parameter_override_cannot_replace_pv_output():
    grid = harmonic_grid(regulation=VoltageRegulation(v_set_pu=1.02, q_max_var=5e5))
    reference = solve(grid)
    result = solve(
        grid,
        param_overrides={
            ("generator", 22, "q_nom_per_phase_var"): torch.tensor(
                [9e5], dtype=torch.float64
            )
        },
    )
    torch.testing.assert_close(
        result.pf.regulation.q_var[22], reference.pf.regulation.q_var[22]
    )
    torch.testing.assert_close(result.v, reference.v)


def test_zeroed_island_retains_regulation_for_current_readout():
    from pgml.schemas import Node

    grid = harmonic_grid(regulation=VoltageRegulation(v_set_pu=1.0))
    expected = solve(grid)
    grid.nodes.append(Node(id=99, u_rated_v=20000.0, phases=(Phase.A,)))
    result = solve(grid, on_disconnected="zero")
    assert result.pf.regulation is not None
    torch.testing.assert_close(result.v[..., :2], expected.v)
    torch.testing.assert_close(
        result.pf.regulation.q_var[22], expected.pf.regulation.q_var[22]
    )


@pytest.mark.parametrize("limited", [False, True])
def test_harmonic_gradient_through_solved_reactive_power(limited):
    from tests.differentiability.test_pv_bus_gradcheck import _grid as small_grid

    def voltage(parameter):
        grid = small_grid(
            v_set=1.02 if limited else parameter, q_max=parameter if limited else None
        )
        grid.appliances[2].spectrum = harmonic_grid().appliances[2].spectrum
        return solve(grid).v[..., 1:, :]

    value = torch.tensor(
        100.0 if limited else 1.001, dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(voltage, (value,), eps=1e-6, atol=2e-5, rtol=2e-4)
