"""Batched (tensor) parameters: one call over a fleet equals one call per parameter set."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pgml.inverter import (
    CurrentController,
    DCLinkControl,
    InverterState,
    LCLFilter,
    PerturbObserveMPPT,
    PhaseLockedLoop,
    SeriesStage,
    ShuntStage,
)
from pgml.studies import emission_flow

from tests.inverter.grids import single_phase_inverter, three_phase_inverter
from tests.inverter.test_emission_flow import _pv_two_bus

RDT = torch.float64
CDT = torch.complex128
F1 = 50.0


def _fleet(values: dict, k: int):
    """The k-th scalar of each batched value."""
    return {
        name: (v[k] if isinstance(v, torch.Tensor) else v) for name, v in values.items()
    }


def _single_phase(p):
    base = single_phase_inverter()
    return dataclasses.replace(
        base,
        filter=LCLFilter(
            l1_h=p["l1"],
            c_f=p["cf"],
            l2_h=p["l2"],
            r1_ohm=0.05,
            r_d_ohm=1.0,
            r2_ohm=0.05,
        ),
        control=CurrentController(
            "pr",
            kp=p["kp"],
            resonant={1: 500.0, 3: 150.0, 5: 100.0},
            resonant_bandwidth_rad_s=5.0,
            sampling_hz=32000.0,
            feedforward_gain=p["kff"],
            feedforward_cutoff_hz=1000.0,
        ),
        pll=PhaseLockedLoop(kp=p["pll_kp"], ki=40.0),
        emc=(
            SeriesStage(l_h=20e-6, r_ohm=0.01),
            ShuntStage(c_f=p["cx"], c_ground_f=4.7e-9),
        ),
        dead_time_s=p["td"],
        dc_link=DCLinkControl(c_dc_f=p["cdc"], kp=0.05, ki=2.0, sampling_hz=32000.0),
        mppt=PerturbObserveMPPT(v_step_v=p["vstep"], rate_hz=5.0, max_line_hz=200.0),
    )


VALUES = {
    "l1": torch.tensor([1.2e-3, 1.5e-3, 2.5e-3], dtype=RDT),
    "cf": torch.tensor([3.3e-6, 4.7e-6, 8.0e-6], dtype=RDT),
    "l2": torch.tensor([0.3e-3, 0.5e-3, 0.8e-3], dtype=RDT),
    "kp": torch.tensor([4.0, 6.0, 9.0], dtype=RDT),
    "kff": torch.tensor([0.0, 1.0, 0.8], dtype=RDT),
    "pll_kp": torch.tensor([0.2, 0.5, 1.0], dtype=RDT),
    "cx": torch.tensor([0.5e-6, 1.0e-6, 4.7e-6], dtype=RDT),
    "td": torch.tensor([0.5e-6, 1.0e-6, 2.5e-6], dtype=RDT),
    "cdc": torch.tensor([800e-6, 1100e-6, 2000e-6], dtype=RDT),
    "vstep": torch.tensor([2.0, 4.0, 8.0], dtype=RDT),
}


@pytest.mark.parametrize("phase_model", ["single", "three"])
def test_batched_norton_equals_serial(phase_model):
    if phase_model == "single":
        build = _single_phase
        v1 = torch.tensor([[230.0 + 0j]], dtype=CDT).expand(3, 1)
        i1 = torch.tensor([[15.0 - 2.0j]], dtype=CDT).expand(3, 1)
        u_dc = 400.0
    else:

        def build(p):
            base = three_phase_inverter()
            return dataclasses.replace(
                base,
                filter=dataclasses.replace(base.filter, l1_h=p["l1"], c_f=p["cf"]),
                control=dataclasses.replace(
                    base.control, kp=p["kp"], active_damping_kc=p["kff"] * 4
                ),
                pll=PhaseLockedLoop(kp=p["pll_kp"], ki=30.0),
                dead_time_s=p["td"],
            )

        a = complex(-0.5, -(3**0.5) / 2)
        v = torch.tensor([230.0, 230.0 * a, 230.0 * a.conjugate()], dtype=CDT)
        v1 = v.expand(3, 3)
        i1 = (v / 230.0 * 14.0).expand(3, 3)
        u_dc = 700.0
    batched = build(VALUES)
    f = torch.as_tensor(batched.emission_frequencies(F1, 40000.0), dtype=RDT)
    nb = batched.norton(f, f1_hz=F1, v1=v1, i1=i1, state=InverterState(u_dc_v=u_dc))
    for k in range(3):
        one = build(_fleet(VALUES, k))
        n1 = one.norton(
            f, f1_hz=F1, v1=v1[k], i1=i1[k], state=InverterState(u_dc_v=u_dc)
        )
        torch.testing.assert_close(
            nb.admittance[k], n1.admittance, rtol=1e-10, atol=1e-14
        )
        torch.testing.assert_close(nb.source[k], n1.source, rtol=1e-10, atol=1e-14)
        torch.testing.assert_close(
            nb.admittance_passive[k], n1.admittance_passive, rtol=1e-10, atol=1e-14
        )


def test_batched_fleet_in_emission_flow():
    grid = _pv_two_bus()
    batched = _single_phase(VALUES)
    state = InverterState(u_dc_v=torch.full((3,), 400.0, dtype=RDT))
    res = emission_flow(grid, {7: batched}, states={7: state}, f_max_hz=40000.0)
    assert res.v.shape[0] == 3
    for k in range(3):
        one = emission_flow(
            grid,
            {7: _single_phase(_fleet(VALUES, k))},
            states={7: InverterState(u_dc_v=400.0)},
            f_max_hz=40000.0,
            frequencies_hz=res.frequencies_hz.tolist(),
        )
        torch.testing.assert_close(res.v[k], one.v, rtol=1e-9, atol=1e-13)
        torch.testing.assert_close(
            res.stability_margin[7][k], one.stability_margin[7], rtol=1e-9, atol=1e-12
        )
