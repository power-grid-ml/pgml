"""The emission-spread study: consistency with the harmonic flow, closed forms on a
two-bus feeder, random-phase aggregation, primary emission, batching and contracts."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from pgml.errors import InputError
from pgml.inverter import InverterState
from pgml.inverter._common import laplace
from pgml.schemas.grid_schema import Generator, HarmonicImpedance, Phase
from pgml.solver import NodeHarmonicSource, solve_harmonic_flow
from pgml.studies import EmissionFlow, emission_flow, emission_frequencies

from tests.inverter.grids import (
    default_models,
    default_states,
    lv_feeder,
    single_phase_inverter,
)
from tests.studies.grids import two_bus

CDT = torch.complex128
F1 = 50.0


def _pv_two_bus(p_w: float = 3000.0, **kw):
    grid = two_bus(r_ohm=0.25, x_ohm=0.12, **kw)
    gen = Generator(id=7, node=2, phases=(Phase.A,), p_nom_w=p_w, q_nom_var=0.0)
    return grid.model_copy(update={"appliances": [*grid.appliances, gen]})


def test_without_inverters_the_study_is_the_harmonic_flow():
    grid = lv_feeder()
    orders = list(range(2, 26))
    background = [
        NodeHarmonicSource(
            node_id=1, spectrum={5: (0.02, 10.0), 7: (0.01, -30.0)}, source_power_va=5e6
        )
    ]
    for sources in (None, background):
        res = emission_flow(grid, {}, harmonic_orders=orders, node_sources=sources)
        ref = solve_harmonic_flow(grid, [1, *orders], node_sources=sources)
        torch.testing.assert_close(res.v, ref.v[1:], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(res.v_rms, ref.v[1:].abs(), rtol=1e-12, atol=1e-12)


def test_two_bus_closed_form():
    """V2 = I_N Z_th / (1 + Z_th Y) with Z_th the source plus line impedance."""
    grid = _pv_two_bus()
    model = single_phase_inverter()
    res = emission_flow(
        grid, {7: model}, states={7: InverterState(u_dc_v=400.0)}, f_max_hz=40000.0
    )
    nor = res.norton[7]
    f = res.frequencies_hz
    s = laplace(f, CDT)
    z_th = (0.25 + 1.0e-3) + s * (0.12 / (2 * math.pi * F1) + 1.0e-6)
    y = nor.admittance[:, 0, 0]
    v2 = nor.source[:, 0] * z_th / (1.0 + z_th * y)
    row = res.rows(2, "a")[0]
    torch.testing.assert_close(res.v[:, row], v2, rtol=1e-10, atol=1e-12)
    # A single device: its current is all primary emission, i = I_N - Y V2.
    torch.testing.assert_close(res.device_current[7][:, 0], nor.source[:, 0] - y * v2)
    torch.testing.assert_close(res.device_primary[7], res.device_current[7])
    # One unit: random-phase and coherent magnitudes coincide.
    torch.testing.assert_close(res.v_rms, res.v.abs(), rtol=1e-12, atol=1e-14)
    # Stability margin: |1 + Z_th Y| for a single-phase unit on a Thevenin source.
    torch.testing.assert_close(
        res.stability_margin[7], (1.0 + z_th * y).abs(), rtol=1e-9, atol=1e-12
    )


def test_random_phase_aggregation_is_the_power_sum():
    """v_rms^2 = sum over units of |V| from each unit's unlocked sources alone."""
    grid = lv_feeder()
    states = default_states()
    pwm_only = {
        k: dataclasses.replace(m, mechanisms=("pwm",))
        for k, m in default_models().items()
    }
    silent = {k: dataclasses.replace(m, mechanisms=()) for k, m in pwm_only.items()}
    kw = dict(states=states, f_max_hz=40000.0, harmonic_orders=range(2, 41))
    both = emission_flow(grid, pwm_only, **kw)
    kw["frequencies_hz"] = both.frequencies_hz.tolist()
    power = torch.zeros_like(both.v_rms)
    for dev in pwm_only:
        alone = {k: (pwm_only[k] if k == dev else silent[k]) for k in pwm_only}
        part = emission_flow(grid, alone, **kw)
        power = power + part.v.abs() ** 2
        # The loads' spectra are locked and common to every run.
        locked = part.v_locked
    torch.testing.assert_close(both.v_locked, locked, rtol=1e-12, atol=1e-12)
    mask = both.frequencies_hz > 2000.0  # PWM band only: no load spectra there
    torch.testing.assert_close(
        both.v_rms[mask] ** 2, power[mask], rtol=1e-9, atol=1e-14
    )


def test_primary_and_secondary_split_on_the_feeder():
    grid = lv_feeder()
    res = emission_flow(
        grid, default_models(), states=default_states(), harmonic_orders=range(2, 41)
    )
    for dev, i_tot in res.device_current.items():
        secondary = i_tot - res.device_primary[dev]
        # Secondary current exists: other units and the loads' harmonics drive it.
        assert float(secondary.abs().max()) > 0.0
        assert torch.isfinite(i_tot).all()
    assert all(float(m.min()) > 0.0 for m in res.stability_margin.values())


def test_batched_states_equal_serial_runs():
    grid = _pv_two_bus()
    model = single_phase_inverter()
    kw = dict(f_max_hz=40000.0)
    batched = emission_flow(
        grid,
        {7: model},
        states={
            7: InverterState(u_dc_v=torch.tensor([380.0, 450.0], dtype=torch.float64))
        },
        operating_point={
            7: {"p_w": torch.tensor([3000.0, 800.0], dtype=torch.float64)}
        },
        **kw,
    )
    for k, (u, p) in enumerate([(380.0, 3000.0), (450.0, 800.0)]):
        one = emission_flow(
            grid,
            {7: model},
            states={7: InverterState(u_dc_v=u)},
            operating_point={7: {"p_w": p}},
            **kw,
        )
        torch.testing.assert_close(batched.v[k], one.v, rtol=1e-10, atol=1e-13)
        torch.testing.assert_close(batched.v_rms[k], one.v_rms, rtol=1e-10, atol=1e-13)


def test_night_state_draws_only_secondary_current():
    grid = lv_feeder()
    states = default_states()
    states[102] = InverterState(u_dc_v=0.0, bridge_on=False, connected=False)
    res = emission_flow(
        grid,
        default_models(),
        states=states,
        operating_point={102: {"p_w": 0.0}},
        harmonic_orders=range(2, 41),
    )
    assert float(res.device_primary[102].abs().max()) == 0.0
    assert float(res.device_current[102].abs().max()) > 0.0  # EMC capacitor current


def test_serialization_round_trip():
    res = emission_flow(
        _pv_two_bus(),
        {7: single_phase_inverter()},
        states={7: InverterState(u_dc_v=400.0)},
    )
    back = EmissionFlow.from_dict(res.to_dict())
    torch.testing.assert_close(back.v, res.v)
    torch.testing.assert_close(back.v_rms, res.v_rms)
    torch.testing.assert_close(back.device_current[7], res.device_current[7])
    assert back.node_ids == res.node_ids and back.phases == res.phases


def test_frequency_grid_contains_every_line_and_no_fundamental():
    grid = _pv_two_bus()
    model = single_phase_inverter()
    f = emission_frequencies(grid, {7: model}, harmonic_orders=[3, 5], f_max_hz=20000.0)
    assert F1 not in f and f == sorted(f) and max(f) <= 20000.0
    for line in model.emission_frequencies(F1, 20000.0):
        assert round(line, 6) in {round(x, 6) for x in f}


def test_contract_errors():
    grid = _pv_two_bus()
    model = single_phase_inverter()
    with pytest.raises(InputError, match="no InverterState"):
        emission_flow(grid, {7: model})
    with pytest.raises(InputError, match="not an in-service"):
        emission_flow(grid, {99: model}, states={99: InverterState(u_dc_v=400.0)})
    gen = Generator(
        id=7,
        node=2,
        phases=(Phase.A,),
        p_nom_w=1000.0,
        harmonic_impedance=HarmonicImpedance(resistance_ohm=0.1, inductance_h=1e-3),
    )
    bad = grid.model_copy(update={"appliances": [grid.appliances[0], gen]})
    with pytest.raises(InputError, match="harmonic_impedance"):
        emission_flow(bad, {7: model}, states={7: InverterState(u_dc_v=400.0)})
    three = lv_feeder()
    with pytest.raises(InputError, match="element"):
        emission_flow(three, {101: model}, states={101: InverterState(u_dc_v=400.0)})


def test_admittance_modes_separate_control_and_filter():
    """'none' is the pure current-source model, 'passive' the filter-only admittance."""
    grid = _pv_two_bus()
    model = single_phase_inverter()
    kw = dict(states={7: InverterState(u_dc_v=400.0)}, f_max_hz=40000.0)
    full = emission_flow(grid, {7: model}, **kw)
    none = emission_flow(grid, {7: model}, admittance="none", **kw)
    passive = emission_flow(grid, {7: model}, admittance="passive", **kw)
    f = full.frequencies_hz
    z_th = (0.25 + 1.0e-3) + laplace(f, CDT) * (0.12 / (2 * math.pi * F1) + 1.0e-6)
    row = full.rows(2, "a")[0]
    src = full.norton[7].source[:, 0]
    torch.testing.assert_close(none.v[:, row], src * z_th, rtol=1e-10, atol=1e-12)
    y_p = full.norton[7].admittance_passive[:, 0, 0]
    torch.testing.assert_close(
        passive.v[:, row], src * z_th / (1.0 + z_th * y_p), rtol=1e-10, atol=1e-12
    )
    # Above the controller's Nyquist frequency the control has no authority.
    high = f > 16000.0
    torch.testing.assert_close(full.v[high], passive.v[high], rtol=1e-12, atol=1e-14)
    with pytest.raises(InputError, match="admittance"):
        emission_flow(grid, {7: model}, admittance="nonsense", **kw)
