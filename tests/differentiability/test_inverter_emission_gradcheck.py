"""Differentiability of the inverter models and the emission study (float64 gradcheck).

Checked: the closed-loop admittance and disturbance transfer w.r.t. filter inductance,
capacitance and controller gain; the Norton source w.r.t. the DC-link voltage and the dead
time path; the PWM residues w.r.t. the modulation index; and the study's node voltages
w.r.t. a line resistance, a filter inductance and the DC-link voltage, plus a
finite-difference spot check.
"""

from __future__ import annotations

import dataclasses

import torch
from torch.autograd import gradcheck

from pgml.inverter import (
    CurrentController,
    InverterState,
    LCLFilter,
    PhaseLockedLoop,
    PWMModulator,
    closed_loop,
)
from pgml.schemas.grid_schema import Generator, Phase
from pgml.studies import emission_flow

from tests.inverter.grids import single_phase_inverter
from tests.studies.grids import two_bus

RDT = torch.float64


def test_closed_loop_gradcheck():
    f = torch.tensor([70.0, 230.0, 1150.0, 4300.0], dtype=RDT)

    def fn(l1, c_f, kp):
        filt = LCLFilter(l1_h=l1, c_f=c_f, l2_h=0.5e-3, r1_ohm=0.05, r_d_ohm=1.0)
        ctrl = CurrentController(
            "pr", kp=kp, resonant={1: 300.0, 5: 80.0}, sampling_hz=20000.0
        )
        cl = closed_loop(
            filt,
            ctrl,
            f,
            f1_hz=50.0,
            pll=PhaseLockedLoop(kp=0.3, ki=20.0),
            v1d=torch.tensor(325.0, dtype=RDT),
            i1dq=torch.tensor(20.0 + 1.0j, dtype=torch.complex128),
        )
        return torch.view_as_real(torch.stack([cl.y, cl.g_dist, cl.y_mirror]))

    args = (
        torch.tensor(1.5e-3, dtype=RDT, requires_grad=True),
        torch.tensor(4.7e-6, dtype=RDT, requires_grad=True),
        torch.tensor(6.0, dtype=RDT, requires_grad=True),
    )
    assert gradcheck(fn, args, eps=1e-9, atol=1e-7, rtol=1e-5)


def test_pwm_residues_gradcheck():
    pwm = PWMModulator("three_phase", carrier_hz=10000.0, carrier_groups=2, sidebands=4)

    def fn(m, u):
        _, lines = pwm.bridge_spectrum(m, u, 50.0, torch.tensor(0.2, dtype=RDT), 0.4)
        return torch.view_as_real(lines)

    args = (
        torch.tensor(0.81, dtype=RDT, requires_grad=True),
        torch.tensor(700.0, dtype=RDT, requires_grad=True),
    )
    assert gradcheck(fn, args, eps=1e-7, atol=1e-6, rtol=1e-5)


def _study(r_line, l2, u_dc):
    grid = two_bus(r_ohm=0.25, x_ohm=0.12)
    line = grid.branches[0].model_copy(
        update={"series_resistance_ohm_per_m": r_line.reshape(1, 1)}
    )
    gen = Generator(id=7, node=2, phases=(Phase.A,), p_nom_w=3000.0, q_nom_var=0.0)
    grid = grid.model_copy(
        update={"branches": [line], "appliances": [*grid.appliances, gen]}
    )
    base = single_phase_inverter()
    model = dataclasses.replace(
        base,
        filter=dataclasses.replace(base.filter, l2_h=l2),
        mechanisms=("dead_time", "dc_ripple"),
        harmonic_orders=(3, 5, 7),
    )
    res = emission_flow(
        grid,
        {7: model},
        states={7: InverterState(u_dc_v=u_dc)},
        f_max_hz=400.0,
        stability=True,
    )
    return torch.cat(
        [torch.view_as_real(res.v).reshape(-1), res.stability_margin[7].reshape(-1)]
    )


def test_emission_flow_gradcheck():
    args = (
        torch.tensor(0.25, dtype=RDT, requires_grad=True),
        torch.tensor(0.5e-3, dtype=RDT, requires_grad=True),
        torch.tensor(400.0, dtype=RDT, requires_grad=True),
    )
    assert gradcheck(_study, args, eps=1e-7, atol=1e-7, rtol=1e-4)


def test_emission_flow_finite_difference_spot_check():
    u = torch.tensor(400.0, dtype=RDT, requires_grad=True)
    r = torch.tensor(0.25, dtype=RDT)
    l2 = torch.tensor(0.5e-3, dtype=RDT)
    out = _study(r, l2, u)
    (grad,) = torch.autograd.grad(out.sum(), u)
    h = 1e-3
    with torch.no_grad():
        fd = (_study(r, l2, u + h).sum() - _study(r, l2, u - h).sum()) / (2 * h)
    torch.testing.assert_close(grad, fd, rtol=1e-6, atol=1e-12)


def test_dead_zone_gradcheck_at_low_power():
    """The ripple dead zone is active when the current is comparable to the ripple."""
    grid = two_bus(r_ohm=0.25, x_ohm=0.12)
    gen = Generator(id=7, node=2, phases=(Phase.A,), p_nom_w=150.0, q_nom_var=0.0)
    grid = grid.model_copy(update={"appliances": [*grid.appliances, gen]})
    base = single_phase_inverter()

    def fn(l1, u_dc):
        model = dataclasses.replace(
            base,
            filter=dataclasses.replace(base.filter, l1_h=l1),
            mechanisms=("dead_time",),
            harmonic_orders=(3, 5),
        )
        res = emission_flow(
            grid,
            {7: model},
            states={7: InverterState(u_dc_v=u_dc)},
            f_max_hz=300.0,
            stability=False,
        )
        return torch.view_as_real(res.v).reshape(-1)

    args = (
        torch.tensor(1.5e-3, dtype=RDT, requires_grad=True),
        torch.tensor(400.0, dtype=RDT, requires_grad=True),
    )
    assert gradcheck(fn, args, eps=1e-9, atol=1e-6, rtol=1e-4)
