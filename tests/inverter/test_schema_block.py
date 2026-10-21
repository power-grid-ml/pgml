"""The optional ``inverter_model`` schema block: validation, serialization, conversion
to the model, and the two harmonic modes side by side."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.inverter import InverterState, from_schema
from pgml.schemas import SCHEMA_VERSION
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    HarmonicImpedance,
    InverterCurrentControl,
    InverterDCLink,
    InverterLCLFilter,
    InverterModel,
    InverterMPPT,
    InverterPLL,
    InverterPWM,
    InverterSeriesStage,
    InverterShuntStage,
    Phase,
)
from pgml.solver import solve_harmonic_flow
from pgml.studies import emission_flow

from tests.inverter.grids import single_phase_inverter
from tests.studies.grids import two_bus


def _spec(**kw) -> InverterModel:
    """The schema form of tests.inverter.grids.single_phase_inverter()."""
    base = dict(
        filter=InverterLCLFilter(
            l1_h=1.5e-3, r1_ohm=0.05, c_f=4.7e-6, r_d_ohm=1.0, l2_h=0.5e-3, r2_ohm=0.05
        ),
        control=InverterCurrentControl(
            kind="pr",
            kp=6.0,
            resonant={1: 500.0, 3: 150.0, 5: 100.0, 7: 80.0},
            resonant_bandwidth_rad_s=5.0,
            sampling_hz=32000.0,
            feedforward_gain=1.0,
            feedforward_cutoff_hz=1000.0,
        ),
        pwm=InverterPWM(
            topology="single_phase_unipolar",
            carrier_hz=16000.0,
            carrier_groups=4,
            sidebands=10,
        ),
        pll=InverterPLL(kp=0.5, ki=40.0),
        emc=[
            InverterSeriesStage(l_h=20e-6, r_ohm=0.01),
            InverterShuntStage(c_f=1e-6, c_ground_f=4.7e-9),
        ],
        dead_time_s=1e-6,
        dc_link=InverterDCLink(c_dc_f=1100e-6, kp=0.05, ki=2.0, sampling_hz=32000.0),
        mppt=InverterMPPT(v_step_v=4.0, rate_hz=5.0, max_line_hz=200.0),
        max_harmonic_order=40,
        u_dc_v=400.0,
    )
    base.update(kw)
    return InverterModel(**base)


def _grid(spec=None, **gen_kw) -> Grid:
    grid = two_bus(r_ohm=0.25, x_ohm=0.12)
    gen = Generator(
        id=7,
        node=2,
        phases=(Phase.A,),
        p_nom_w=3000.0,
        q_nom_var=0.0,
        inverter_model=spec,
        **gen_kw,
    )
    return grid.model_copy(update={"appliances": [*grid.appliances, gen]})


def test_schema_version_bumped_for_the_block():
    assert SCHEMA_VERSION == "0.4.0"


def test_json_round_trip():
    grid = _grid(_spec())
    back = Grid.model_validate_json(grid.model_dump_json())
    assert back.appliances[-1].inverter_model == grid.appliances[-1].inverter_model


def test_validation_rules():
    with pytest.raises(ValidationError, match="alternative harmonic models"):
        _grid(
            _spec(),
            harmonic_impedance=HarmonicImpedance(resistance_ohm=0.1, inductance_h=1e-3),
        )
    with pytest.raises(ValidationError, match="three-phase PWM"):
        _spec(control=InverterCurrentControl(kind="pi_dq", kp=10.0, ki=100.0))
    with pytest.raises(ValidationError):
        InverterLCLFilter(l1_h=0.0)  # a bridge-side inductance is required
    with pytest.raises(ValidationError):
        _spec(mechanisms=["pwm", "flicker"])


def test_schema_block_is_the_same_model_as_the_explicit_one():
    f = torch.as_tensor(
        single_phase_inverter().emission_frequencies(50.0, 40000.0), dtype=torch.float64
    )
    kw = dict(
        f1_hz=50.0,
        v1=torch.tensor([230.0 + 0j], dtype=torch.complex128),
        i1=torch.tensor([13.0 + 0j], dtype=torch.complex128),
        state=InverterState(u_dc_v=400.0),
    )
    a = from_schema(_spec()).norton(f, **kw)
    b = single_phase_inverter(
        mppt=from_schema(_spec()).mppt, harmonic_orders=tuple(range(2, 41))
    ).norton(f, **kw)
    torch.testing.assert_close(a.admittance, b.admittance)
    torch.testing.assert_close(a.source, b.source)


def test_emission_flow_reads_the_grid_block_and_its_default_state():
    grid = _grid(_spec())
    from_grid = emission_flow(grid, f_max_hz=40000.0)
    plain = _grid(None)
    explicit = emission_flow(
        plain,
        {7: from_schema(_spec())},
        states={7: InverterState(u_dc_v=400.0)},
        f_max_hz=40000.0,
    )
    torch.testing.assert_close(from_grid.v, explicit.v)
    # An explicit model overrides the grid block.
    other = from_schema(_spec(dead_time_s=3e-6))
    over = emission_flow(grid, {7: other}, f_max_hz=40000.0)
    assert not torch.allclose(over.v, from_grid.v)


def test_simple_mode_ignores_the_block():
    """solve_harmonic_flow keeps the injection mode: the block changes nothing there."""
    with_block = solve_harmonic_flow(_grid(_spec()), [1, 3, 5, 7])
    without = solve_harmonic_flow(_grid(None), [1, 3, 5, 7])
    torch.testing.assert_close(with_block.v, without.v)
