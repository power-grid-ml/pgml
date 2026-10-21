"""Closed-form checks of the inverter building blocks (filters, control, PWM, dead time,
DC link) against the equations they implement and published values."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import jv

from pgml.errors import InputError, ModelingError
from pgml.inverter import (
    CurrentController,
    DCLinkControl,
    InverterState,
    LCLFilter,
    PerturbObserveMPPT,
    PhaseLockedLoop,
    PWMModulator,
    SeriesStage,
    ShuntStage,
    bessel_j,
    closed_loop,
    dead_time_voltage,
    mppt_current_lines,
    propagate_norton,
    ripple_third_harmonic,
)
from pgml.inverter._common import laplace, sequence_to_element

from tests.inverter.grids import single_phase_inverter, three_phase_inverter

CDT = torch.complex128
RDT = torch.float64
F1 = 50.0


def _f(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=RDT)


# ---- filters ------------------------------------------------------------------------
def test_lcl_plant_matches_nodal_solution():
    filt = LCLFilter(
        l1_h=2e-3, c_f=4.7e-6, l2_h=1e-3, r1_ohm=0.1, r_d_ohm=1.5, r2_ohm=0.05
    )
    s = laplace(_f([150.0, 1234.0, 9000.0]), CDT)
    yp, yo = filt.plant(s)
    z1, yc, z2 = filt.impedances(s)
    for k in range(s.shape[0]):
        # Unknowns [v_c, i2] for (v_b, v) = (1, 0) and (0, 1).
        a = torch.tensor(
            [[1.0 / z1[k] + yc[k], 1.0], [1.0, -z2[k]]], dtype=CDT
        )  # KCL at C (i1 = (v_b - v_c)/Z1 = Yc v_c + i2), KVL grid side v_c - Z2 i2 = v
        x_b = torch.linalg.solve(a, torch.tensor([1.0 / z1[k], 0.0], dtype=CDT))
        x_v = torch.linalg.solve(a, torch.tensor([0.0, 1.0], dtype=CDT))
        torch.testing.assert_close(x_b[1], yp[k], rtol=1e-12, atol=0)
        torch.testing.assert_close(-x_v[1], yo[k], rtol=1e-12, atol=0)


def test_norton_through_series_then_shunt():
    s = laplace(_f([400.0, 20000.0]), CDT)
    i_n = torch.tensor([2.0 - 1.0j, 0.5 + 0.2j], dtype=CDT)
    y = torch.tensor([0.01 - 0.3j, 0.002 - 0.01j], dtype=CDT)
    series, shunt = SeriesStage(l_h=50e-6, r_ohm=0.02), ShuntStage(c_f=1e-6, r_ohm=0.1)
    i_out, y_out = propagate_norton(i_n, y, [series, shunt], s)
    z = series.impedance(s)
    y_sh = shunt.admittance(s)
    # Short-circuit current at the terminal: the inner source drives 1/Y || z shorted.
    torch.testing.assert_close(i_out, i_n / (1.0 + z * y), rtol=1e-13, atol=0)
    torch.testing.assert_close(y_out, 1.0 / (1.0 / y + z) + y_sh, rtol=1e-13, atol=0)


# ---- control ------------------------------------------------------------------------
def _l_filter_case():
    filt = LCLFilter(l1_h=3e-3, r1_ohm=0.05)
    ctrl = CurrentController(
        "pr", kp=12.0, resonant={1: 600.0, 5: 200.0}, sampling_hz=10000.0
    )
    return filt, ctrl


def test_closed_loop_l_filter_matches_wang_eq7():
    filt, ctrl = _l_filter_case()
    f = _f([75.0, 250.0, 1000.0, 3000.0, 4900.0])
    cl = closed_loop(filt, ctrl, f, f1_hz=F1)
    s = laplace(f, CDT)
    yp, yo = filt.plant(s)
    t = ctrl.delay(s) * ctrl.gain(s, F1) * yp
    torch.testing.assert_close(cl.y, yo / (1.0 + t), rtol=1e-12, atol=0)
    torch.testing.assert_close(cl.g_ref, t / (1.0 + t), rtol=1e-12, atol=0)
    torch.testing.assert_close(cl.g_dist, yp / (1.0 + t), rtol=1e-12, atol=0)
    torch.testing.assert_close(cl.loop_gain, t, rtol=1e-12, atol=0)


def test_closed_loop_has_no_authority_above_nyquist():
    filt = LCLFilter(l1_h=2e-3, c_f=4.7e-6, l2_h=1e-3, r_d_ohm=1.0)
    ctrl = CurrentController("pr", kp=8.0, sampling_hz=20000.0, feedforward_gain=1.0)
    f = _f([10500.0, 32000.0])
    cl = closed_loop(filt, ctrl, f, f1_hz=F1)
    yp, yo = filt.plant(laplace(f, CDT))
    torch.testing.assert_close(cl.y, yo, rtol=1e-13, atol=0)
    torch.testing.assert_close(cl.y_passive, yo, rtol=1e-13, atol=0)
    torch.testing.assert_close(cl.g_dist, yp, rtol=1e-13, atol=0)
    assert bool((cl.g_ref == 0).all())


def test_closed_loop_pll_alpha_beta_closed_form():
    """Stationary-frame control with a PLL on the reference (Wang et al. 2018, Fig. 7)."""
    filt, ctrl = _l_filter_case()
    pll = PhaseLockedLoop(kp=0.2, ki=20.0)
    v1d = torch.tensor(325.0, dtype=RDT)
    i1dq = torch.tensor(15.0 - 2.0j, dtype=CDT)
    f = _f([20.0, 90.0, 140.0, 400.0, -60.0, -230.0])
    cl = closed_loop(filt, ctrl, f, f1_hz=F1, pll=pll, v1d=v1d, i1dq=i1dq)
    s = laplace(f, CDT)
    yp, yo = filt.plant(s)
    t = ctrl.delay(s) * ctrl.gain(s, F1) * yp
    h = pll.transfer(s - 1j * 2 * math.pi * F1, v1d)
    y_pll = h * i1dq / 2.0
    torch.testing.assert_close(
        cl.y, yo / (1 + t) - t / (1 + t) * y_pll, rtol=1e-11, atol=0
    )
    torch.testing.assert_close(cl.y_mirror, t / (1 + t) * y_pll, rtol=1e-11, atol=0)


def test_closed_loop_dq_pi_with_pll_closed_form():
    """dq-frame PI: the PLL also rotates the modulating voltage (Wang et al., eq. 38)."""
    filt = LCLFilter(l1_h=3e-3, r1_ohm=0.05)
    ctrl = CurrentController("pi_dq", kp=16.0, ki=600.0, sampling_hz=10000.0)
    pll = PhaseLockedLoop(kp=0.1, ki=10.0)
    v1d = torch.tensor(326.6, dtype=RDT)
    i1dq = torch.tensor(15.0 + 0.0j, dtype=CDT)
    vb1dq = torch.tensor(327.0 + 14.0j, dtype=CDT)
    f = _f([30.0, 110.0, 196.0, 600.0, -96.0])
    cl = closed_loop(filt, ctrl, f, f1_hz=F1, pll=pll, v1d=v1d, i1dq=i1dq, vb1dq=vb1dq)
    s = laplace(f, CDT)
    yp, yo = filt.plant(s)
    gdel = ctrl.delay(s)
    t = gdel * ctrl.gain(s, F1) * yp
    h = pll.transfer(s - 1j * 2 * math.pi * F1, v1d)
    y_expect = (yo - yp * gdel * h * vb1dq / 2.0) / (1 + t) - t / (
        1 + t
    ) * h * i1dq / 2.0
    torch.testing.assert_close(cl.y, y_expect, rtol=1e-11, atol=0)


def test_ideal_resonator_is_refused():
    with pytest.raises(InputError, match="bandwidth"):
        CurrentController("pr", resonant={1: 100.0}, resonant_bandwidth_rad_s=0.0)


def test_symmetric_three_phase_model_has_equal_sequence_admittances():
    """Without a PLL a stationary-frame PR loop is real-coefficient: Y- = Y+."""
    model = three_phase_inverter(
        control=CurrentController(
            "pr", kp=10.0, resonant={1: 400.0}, sampling_hz=20000.0
        ),
        pll=None,
    )
    f = _f([150.0, 250.0, 1250.0, 10000.0])
    u = 400.0 / math.sqrt(3.0)
    a = complex(math.cos(-2 * math.pi / 3), math.sin(-2 * math.pi / 3))
    v1 = torch.tensor([u, u * a, u * a.conjugate()], dtype=CDT)
    i1 = v1 / u * 14.0
    nor = model.norton(
        torch.cat([f, _f(model.emission_frequencies(F1, 40000.0))]).unique(),
        f1_hz=F1,
        v1=v1,
        i1=i1,
        state=InverterState(u_dc_v=700.0),
    )
    y = nor.admittance
    # Element admittance = A diag(Y0, Y+, Y-) A^-1 with Y+ = Y-: diagonal entries equal,
    # off-diagonal entries equal (a circulant-symmetric matrix).
    torch.testing.assert_close(y[:, 0, 1], y[:, 1, 0], rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(y[:, 0, 0], y[:, 2, 2], rtol=1e-12, atol=1e-15)


def test_sequence_to_element_round_trip():
    d = torch.tensor([[0.1 + 0.2j, 2.0 - 1.0j, 0.5 + 0.0j]], dtype=CDT)
    m = sequence_to_element(d)[0]
    a = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    pos = torch.tensor([1.0, a * a, a], dtype=CDT)  # positive-sequence vector (b lags)
    torch.testing.assert_close(m @ pos, d[0, 1] * pos, rtol=1e-13, atol=1e-15)


# ---- PWM ----------------------------------------------------------------------------
def test_bessel_matches_scipy():
    n = torch.arange(-15, 16, dtype=RDT)[:, None]
    x = torch.linspace(0.0, 25.0, 11, dtype=RDT)[None, :]
    expect = jv(n.numpy(), x.numpy())
    np.testing.assert_allclose(bessel_j(n, x, 160).numpy(), expect, atol=1e-14)


def test_unipolar_first_band_matches_klatt_fig3():
    """Klatt et al. 2019 (IET GTD 13(24), eq. 7, fig. 3): U_CO = 230 V, U_DCL = 400 V,
    f_c = 1 kHz gives an RMS of about 136 V in the first emission band."""
    pwm = PWMModulator(
        "single_phase_unipolar", carrier_hz=1000.0, carrier_groups=2, sidebands=15
    )
    m = torch.tensor(230.0 * math.sqrt(2.0) / 400.0, dtype=RDT)
    f, lines = pwm.bridge_spectrum(
        m, torch.tensor(400.0, dtype=RDT), F1, torch.tensor(0.0, dtype=RDT)
    )
    band = (f > 1000.0) & (f < 3000.0)
    rms = lines[band, 0].abs().pow(2).sum().sqrt()
    assert abs(float(rms) - 136.0) < 0.5
    # Klatt's eq. 7 in closed form: (2 U / pi) sqrt(sum_k J_{2k-1}(pi M)^2).
    k = np.arange(1, 30)
    expect = (
        2 * 400.0 / math.pi * math.sqrt(np.sum(jv(2 * k - 1, math.pi * float(m)) ** 2))
    )
    assert abs(float(rms) - expect) < 1e-9 * expect


def test_unipolar_cancels_odd_carrier_groups_and_three_phase_cancels_carrier():
    m = torch.tensor(0.8, dtype=RDT)
    u = torch.tensor(400.0, dtype=RDT)
    t0 = torch.tensor(0.3, dtype=RDT)
    uni = PWMModulator(
        "single_phase_unipolar", carrier_hz=16000.0, carrier_groups=3, sidebands=8
    )
    f, lines = uni.bridge_spectrum(m, u, F1, t0)
    odd_group = (f > 15000) & (f < 17000)
    assert float(lines[odd_group].abs().max()) < 1e-9
    three = PWMModulator(
        "three_phase", carrier_hz=10000.0, carrier_groups=1, sidebands=6
    )
    f3, l3 = three.bridge_spectrum(m, u, F1, t0)
    ms, ns = three.line_orders()
    for k, n in enumerate(ns):
        if n % 3 == 0:  # carrier line and triplen sidebands are common mode
            assert float(l3[k].abs().max()) < 1e-9


def test_bipolar_bridge_is_twice_the_leg():
    m = torch.tensor(0.7, dtype=RDT)
    u = torch.tensor(400.0, dtype=RDT)
    bi = PWMModulator(
        "single_phase_bipolar", carrier_hz=8000.0, carrier_groups=1, sidebands=4
    )
    f, lines = bi.bridge_spectrum(m, u, F1, torch.tensor(0.0, dtype=RDT))
    # Leg line (m=1, n=0): (2 U / pi) J0(pi M / 2) sin(pi/2), bridge doubles it.
    k = bi.line_orders()[1].index(0)
    expect = 2.0 * (2.0 * 400.0 / math.pi) * jv(0, math.pi * 0.7 / 2.0) / math.sqrt(2.0)
    assert abs(float(lines[k, 0].abs()) - expect) < 1e-10 * expect


# ---- dead time ----------------------------------------------------------------------
def test_dead_time_square_wave_amplitude_and_locking():
    """Mohan et al. eq. 8-77: full-bridge error 2 t_d f_s U_dc sgn(i)."""
    u = torch.tensor(400.0, dtype=RDT)
    phi = 0.4
    i1 = torch.polar(torch.tensor([10.0], dtype=RDT), torch.tensor([phi], dtype=RDT))
    d = dead_time_voltage(
        i1, u, 2e-6, 16000.0, [2, 3, 5, 7], full_bridge=True, three_phase=False
    )
    dv = 2 * 2e-6 * 16000.0 * 400.0
    for k, h in enumerate([2, 3, 5, 7]):
        expect = 0.0 if h % 2 == 0 else dv * 4 / (math.pi * h) / math.sqrt(2.0)
        assert abs(float(d[k, 0].abs()) - expect) < 1e-12 * max(expect, 1.0)
    # -dV (4/pi) (-1)^((h-1)/2)/h cos(h(wt + phi)): phase h phi + pi (h=1,5) or h phi (h=3,7).
    ang3 = float(torch.angle(d[1, 0]))
    assert abs(math.remainder(ang3 - 3 * phi, 2 * math.pi)) < 1e-12


def test_dead_time_three_phase_removes_triplens():
    u = torch.tensor(700.0, dtype=RDT)
    ang = torch.tensor([0.1, 0.1 - 2 * math.pi / 3, 0.1 + 2 * math.pi / 3], dtype=RDT)
    i1 = torch.polar(torch.full((3,), 15.0, dtype=RDT), ang)
    d = dead_time_voltage(
        i1, u, 1.5e-6, 10000.0, [3, 5, 9], full_bridge=False, three_phase=True
    )
    assert float(d[0].abs().max()) < 1e-12 and float(d[2].abs().max()) < 1e-12
    leg = 1.5e-6 * 10000.0 * 700.0
    assert abs(float(d[1, 0].abs()) - leg * 4 / (5 * math.pi) / math.sqrt(2)) < 1e-12


def test_dead_time_ripple_dead_zone():
    """Large current: the dead zone is negligible; current inside the ripple: no error."""
    u = torch.tensor(400.0, dtype=RDT)
    kw = dict(full_bridge=True, three_phase=False)
    loc = dict(
        ripple="local",
        l1_h=1.5e-3,
        modulation_index=torch.tensor(0.8, dtype=RDT),
        bridge_angle=torch.tensor([0.25], dtype=RDT),
        topology="single_phase_bipolar",
        n_samples=1 << 14,
    )
    big = torch.polar(torch.tensor([200.0], dtype=RDT), torch.tensor([0.2], dtype=RDT))
    hard = dead_time_voltage(big, u, 2e-6, 16000.0, [3, 5], **kw)
    soft = dead_time_voltage(big, u, 2e-6, 16000.0, [3, 5], **kw, **loc)
    torch.testing.assert_close(soft, hard, rtol=2e-2, atol=1e-3)
    # Half ripple at least 400 (1 - 0.64) / (4 * 1.5e-3 * 16e3) = 1.5 A everywhere.
    tiny = torch.polar(torch.tensor([0.5], dtype=RDT), torch.tensor([0.2], dtype=RDT))
    none = dead_time_voltage(tiny, u, 2e-6, 16000.0, [3, 5], **kw, **loc)
    assert float(none.abs().max()) == 0.0


def test_overmodulation_is_refused():
    pwm = PWMModulator("single_phase_unipolar", carrier_hz=16000.0)
    with pytest.raises(ModelingError, match="modulation index"):
        pwm.bridge_spectrum(
            torch.tensor(1.15, dtype=RDT),
            torch.tensor(400.0, dtype=RDT),
            F1,
            torch.tensor(0.0, dtype=RDT),
        )


# ---- DC link ------------------------------------------------------------------------
def test_mppt_reference_lines_match_sangwongwanich():
    """1.25 Hz line of a 12 V step at 5 Hz: 2 sqrt2 12 / pi = 10.80 V (TPEL 2018, fig. 17a)."""
    mppt = PerturbObserveMPPT(v_step_v=12.0, rate_hz=5.0, n_terms=4)
    fn, amp, _ = mppt.reference_lines(RDT, "cpu")
    torch.testing.assert_close(fn, _f([1.25, 3.75, 6.25, 8.75]))
    torch.testing.assert_close(
        amp, 2 * math.sqrt(2) * 12.0 / (math.pi * _f([1, 3, 5, 7])), rtol=1e-12, atol=0
    )
    # The waveform {0, +v, 0, -v} reconstructs from the (amplitude, phase) series.
    t = torch.linspace(0.0, 0.8, 801, dtype=RDT)[:-1]
    mppt8 = PerturbObserveMPPT(v_step_v=12.0, rate_hz=5.0, n_terms=400)
    fn8, a8, p8 = mppt8.reference_lines(RDT, "cpu")
    wave = (a8[:, None] * torch.sin(2 * math.pi * fn8[:, None] * t + p8[:, None])).sum(
        0
    )
    level = torch.tensor([0.0, 12.0, 0.0, -12.0], dtype=RDT)
    expect = level[(t / 0.2).floor().long() % 4]
    inner = ((t % 0.2) > 0.02) & ((t % 0.2) < 0.18)  # away from the steps (Gibbs)
    assert float((wave - expect)[inner].abs().max()) < 0.2


def test_ripple_third_harmonic_blocked_by_notch_and_scales_with_power():
    dc = DCLinkControl(c_dc_f=1100e-6, kp=0.05, ki=2.0, sampling_hz=20000.0)
    v = torch.tensor(230.0 + 0j, dtype=CDT)
    u = torch.tensor(400.0, dtype=RDT)
    i_lo, i_hi = torch.tensor(5.0 + 0j, dtype=CDT), torch.tensor(10.0 + 0j, dtype=CDT)
    assert (
        float(ripple_third_harmonic(dc, f1_hz=F1, v_t=v, i_t=i_hi, u_dc=u).abs())
        < 1e-12
    )
    raw = DCLinkControl(
        c_dc_f=1100e-6, kp=0.05, ki=2.0, sampling_hz=20000.0, notch_quality_rad_s=None
    )
    lo = ripple_third_harmonic(raw, f1_hz=F1, v_t=v, i_t=i_lo, u_dc=u)
    hi = ripple_third_harmonic(raw, f1_hz=F1, v_t=v, i_t=i_hi, u_dc=u)
    assert float(hi.abs()) > 0
    torch.testing.assert_close(hi, 2.0 * lo, rtol=1e-12, atol=0)


# ---- the assembled model ------------------------------------------------------------
def _single_phase_norton(state: InverterState, model=None):
    model = model or single_phase_inverter()
    f = _f(model.emission_frequencies(F1, 70000.0))
    v1 = torch.tensor([230.0 + 0j], dtype=CDT)
    i1 = torch.tensor([20.0 + 0j], dtype=CDT)
    return model, f, model.norton(f, f1_hz=F1, v1=v1, i1=i1, state=state)


def test_night_state_leaves_only_the_emc_shunt():
    model, f, nor = _single_phase_norton(
        InverterState(u_dc_v=0.0, bridge_on=False, connected=False)
    )
    shunt = model.emc[1]
    expect = shunt.admittance(laplace(f, CDT))
    torch.testing.assert_close(nor.admittance[:, 0, 0], expect, rtol=1e-12, atol=0)
    assert float(nor.source.abs().max()) == 0.0


def test_idle_bridge_is_the_passive_filter():
    _, _, idle = _single_phase_norton(InverterState(u_dc_v=400.0, bridge_on=False))
    _, _, on = _single_phase_norton(InverterState(u_dc_v=400.0))
    torch.testing.assert_close(
        idle.admittance, on.admittance_passive, rtol=1e-12, atol=0
    )
    assert float(idle.source.abs().max()) == 0.0


def test_source_split_and_line_placement():
    model, f, nor = _single_phase_norton(InverterState(u_dc_v=400.0))
    assert set(nor.components) == {"pwm", "dead_time", "mppt", "dc_ripple"}
    torch.testing.assert_close(
        nor.source_locked, nor.components["dead_time"] + nor.components["dc_ripple"]
    )
    torch.testing.assert_close(
        nor.source_unlocked, nor.components["pwm"] + nor.components["mppt"]
    )
    # Dead-time harmonics are odd only for a single-phase bridge.
    f2 = (f - 100.0).abs().argmin()
    f3 = (f - 150.0).abs().argmin()
    assert float(nor.components["dead_time"][f2].abs()) == 0.0
    assert float(nor.components["dead_time"][f3].abs()) > 0.0
    with pytest.raises(InputError, match="not in the frequency grid"):
        model.norton(
            _f([150.0, 151.0, 70000.0]),
            f1_hz=F1,
            v1=torch.tensor([230.0 + 0j], dtype=CDT),
            i1=torch.tensor([20.0 + 0j], dtype=CDT),
            state=InverterState(u_dc_v=400.0),
        )


def test_mppt_lower_sidebands_below_zero_fold():
    """f_1 - n f_MPPT / 4 < 0 is the real line at n f_MPPT / 4 - f_1, conjugated."""
    mppt = PerturbObserveMPPT(v_step_v=12.0, rate_hz=20.0, n_terms=8)
    dc = DCLinkControl(c_dc_f=1100e-6, kp=0.185, ki=12.2, sampling_hz=20000.0)
    kw = dict(
        f1_hz=F1,
        v_g=torch.tensor(230.0, dtype=RDT),
        u_dc=torch.tensor(450.0, dtype=RDT),
        current_angle=torch.tensor(0.3, dtype=RDT),
    )
    freqs, lines = mppt_current_lines(mppt, dc, **kw)
    assert min(freqs) > 0 and 5.0 in freqs  # n = 11: |50 - 55| = 5 Hz
    assert freqs == mppt.line_frequencies(F1)
    # Folding conjugates: the lower line of n = 11 rebuilt from the unfolded formula.
    fn, a_n, phi_n = mppt.reference_lines(RDT, "cpu")
    g = -dc.reference_to_amplitude(laplace(fn, CDT), F1, kw["v_g"], kw["u_dc"])
    k = mppt.orders().index(11)
    unfolded = torch.polar(
        a_n[k] * g[k].abs() / (2 * math.sqrt(2)),
        0.3 + math.pi / 2 - (phi_n[k] + torch.angle(g[k])),
    )
    # n = 9 (50 - 45) and the folded n = 11 (55 - 50) share 5 Hz; the folded one is last.
    at_5 = [j for j, f in enumerate(freqs) if f == 5.0]
    assert len(at_5) == 2
    torch.testing.assert_close(lines[at_5[-1], 0], torch.conj(unfolded))


@pytest.mark.parametrize("topology", ["single_phase_unipolar", "three_phase"])
def test_dead_zone_tends_to_the_hard_sign_at_large_current(topology):
    three = topology == "three_phase"
    u = torch.tensor(700.0 if three else 400.0, dtype=RDT)
    shift = torch.tensor(
        [0.0, -2 * math.pi / 3, 2 * math.pi / 3] if three else [0.0], dtype=RDT
    )
    i1 = torch.polar(torch.full_like(shift, 300.0), 0.2 + shift)
    kw = dict(full_bridge=not three, three_phase=three)
    hard = dead_time_voltage(i1, u, 2e-6, 16000.0, [5, 7], **kw)
    soft = dead_time_voltage(
        i1,
        u,
        2e-6,
        16000.0,
        [5, 7],
        **kw,
        ripple="local",
        l1_h=1.5e-3,
        modulation_index=torch.tensor(0.8, dtype=RDT),
        bridge_angle=0.25 + shift,
        topology=topology,
        n_samples=1 << 14,
    )
    torch.testing.assert_close(soft, hard, rtol=3e-2, atol=2e-3)
