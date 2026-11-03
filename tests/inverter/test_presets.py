"""Generic inverter designs: design rules, the stability check, batching and ranges."""

from __future__ import annotations

import math

import pytest
import torch

from pgml.inverter import (
    GENERIC_PARAMETER_RANGES,
    CurrentController,
    GenericInverterDesign,
    InverterState,
    LCLFilter,
    generic_single_phase_inverter,
    generic_three_phase_inverter,
)
from pgml.inverter.presets import current_loop_stability
from pgml.studies import emission_flow

from tests.inverter.test_emission_flow import _pv_two_bus

RDT = torch.float64


@pytest.mark.parametrize("factor, stable", [(0.9, True), (1.1, False)])
def test_stability_count_matches_the_l_filter_closed_form(factor, stable):
    """P control of an L filter with delay T_d: stable iff k_p T_d / L < pi / 2."""
    l_h, fs = 2e-3, 10000.0
    td = 1.5 / fs
    kp = factor * (math.pi / 2) * l_h / td
    ctrl = CurrentController("pr", kp=kp, resonant={}, sampling_hz=fs)
    n_bad, _ = current_loop_stability(LCLFilter(l1_h=l_h), ctrl, n_points=40001)
    assert (int(n_bad) == 0) is stable


def test_design_rules():
    d = GenericInverterDesign(
        4600.0,
        phases=1,
        carrier_hz=16000.0,
        ripple_fraction=0.2,
        cf_reactive_fraction=0.04,
        l2_ratio=0.5,
    )
    fv = d.filter_values()
    w1 = 2 * math.pi * 50.0
    # Filter capacitor reactive power = 4 % of the rated active power (Liserre et al.).
    assert float(w1 * fv["c_f"] * 230.0**2) == pytest.approx(0.04 * 4600.0)
    # Unipolar ripple: U_dc / (8 L1 f_c) = 20 % of the rated peak current.
    i_peak = math.sqrt(2) * 4600.0 / 230.0
    assert float(400.0 / (8 * fv["l1_h"] * 16000.0)) == pytest.approx(0.2 * i_peak)
    assert float(fv["l2_h"]) == pytest.approx(0.5 * float(fv["l1_h"]))
    # Damping resistor one third of the capacitor impedance at the resonance.
    w_res = 2 * math.pi * float(fv["f_res_hz"])
    assert float(fv["r_d_ohm"]) == pytest.approx(1.0 / (3 * w_res * float(fv["c_f"])))
    # PLL tuned for a damping ratio 0.707: s^2 + k_p V s + k_i V.
    pll = d.build().pll
    v = math.sqrt(2) * 230.0
    zeta = float(pll.kp) * v / (2 * math.sqrt(float(pll.ki) * v))
    assert zeta == pytest.approx(0.707)


@pytest.mark.parametrize(
    "build", [generic_single_phase_inverter, generic_three_phase_inverter]
)
def test_default_designs_are_stable(build):
    model = build()
    n_bad, margin = current_loop_stability(model.filter, model.control)
    assert int(n_bad) == 0 and float(margin) > 0.2


def test_batched_knobs_equal_serial_designs():
    knobs = {
        "ripple_fraction": torch.tensor([0.1, 0.25, 0.4], dtype=RDT),
        "cf_reactive_fraction": torch.tensor([0.01, 0.03, 0.05], dtype=RDT),
        "dead_time_s": torch.tensor([0.5e-6, 1e-6, 3e-6], dtype=RDT),
        "pll_bandwidth_hz": torch.tensor([5.0, 20.0, 50.0], dtype=RDT),
    }
    batched = GenericInverterDesign(4600.0, phases=1, **knobs)
    rep = batched.report()
    assert bool(rep["stable"].all())
    f = torch.as_tensor(batched.build().emission_frequencies(50.0, 40000.0), dtype=RDT)
    kw = dict(
        f1_hz=50.0,
        v1=torch.tensor([[230.0 + 0j]], dtype=torch.complex128).expand(3, 1),
        i1=torch.tensor([[18.0 + 0j]], dtype=torch.complex128).expand(3, 1),
        state=InverterState(u_dc_v=400.0),
    )
    nb = batched.build().norton(f, **kw)
    for k in range(3):
        one = GenericInverterDesign(
            4600.0, phases=1, **{n: float(v[k]) for n, v in knobs.items()}
        )
        n1 = one.build().norton(
            f, f1_hz=50.0, v1=kw["v1"][k], i1=kw["i1"][k], state=kw["state"]
        )
        torch.testing.assert_close(
            nb.admittance[k], n1.admittance, rtol=1e-10, atol=1e-14
        )
        torch.testing.assert_close(nb.source[k], n1.source, rtol=1e-10, atol=1e-14)


def test_ranges_are_documented():
    for name, r in GENERIC_PARAMETER_RANGES.items():
        assert (
            r["low"] < r["high"] and r["scale"] in ("linear", "log") and r["basis"]
        ), name
        if not r.get("structural"):
            GenericInverterDesign(4600.0, **{name: r["low"]}).filter_values()


def test_generic_unit_in_the_emission_study():
    res = emission_flow(
        _pv_two_bus(),
        {7: generic_single_phase_inverter(3000.0)},
        states={7: InverterState(u_dc_v=400.0)},
        f_max_hz=40000.0,
    )
    assert torch.isfinite(res.v).all() and float(res.stability_margin[7].min()) > 0
