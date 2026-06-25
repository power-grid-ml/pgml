"""Inverter / DER control laws: unit + integration behaviour.

The unit tests pin the control math (``resolve_injection_power`` /
``evaluate_characteristic`` / ``smooth_clamp``) against hand-computed values; the
integration tests confirm a controlled :class:`Generator` solves and moves the voltage
in the physically correct direction. See ``docs/pgml/modeling/der-pv-storage.md`` §4.2-4.3.
"""

from __future__ import annotations

import math

import torch

from pgml.assembly._control import (
    evaluate_characteristic,
    resolve_injection_power,
    smooth_clamp,
)
from pgml.schemas.grid_schema import (
    Characteristic,
    ConstantPowerFactorControl,
    ConstantReactivePowerControl,
    Generator,
    Grid,
    Line,
    Node,
    Phase,
    PowerFactorWattControl,
    QReference,
    Source,
    VoltVarControl,
    VoltVarVoltWattControl,
    VoltWattControl,
)
from pgml.solver import solve_power_flow

RDT = torch.float64
ABC = (Phase.A, Phase.B, Phase.C)


def _p_v(p, v):
    """Helper: p_avail [n_elem], v_pu [H=1, n_elem] from python scalars."""
    return (
        torch.tensor([float(p)], dtype=RDT),
        torch.tensor([[float(v)]], dtype=RDT),
    )


# --------------------------------------------------------------------------- #
# Characteristic evaluation
# --------------------------------------------------------------------------- #
def test_characteristic_linear_interp_and_extrap():
    c = Characteristic(x_values=[0.95, 1.0, 1.05], y_values=[1.0, 0.0, -1.0])
    x = torch.tensor([0.95, 0.975, 1.0, 1.025, 1.05], dtype=RDT)
    y = evaluate_characteristic(x, c, RDT, None)
    assert torch.allclose(y, torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0], dtype=RDT))
    # constant extrapolation holds the endpoints
    out = evaluate_characteristic(torch.tensor([0.8, 1.2], dtype=RDT), c, RDT, None)
    assert torch.allclose(out, torch.tensor([1.0, -1.0], dtype=RDT))


def test_characteristic_cubic_matches_at_knots():
    c = Characteristic(
        x_values=[0.95, 1.0, 1.05], y_values=[1.0, 0.0, -1.0], interpolation="cubic"
    )
    x = torch.tensor([0.95, 1.0, 1.05], dtype=RDT)
    y = evaluate_characteristic(x, c, RDT, None)
    assert torch.allclose(y, torch.tensor([1.0, 0.0, -1.0], dtype=RDT), atol=1e-9)


# --------------------------------------------------------------------------- #
# smooth_clamp
# --------------------------------------------------------------------------- #
def test_smooth_clamp_hard_when_beta_inf():
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=RDT)
    lo = torch.tensor(-1.0, dtype=RDT)
    hi = torch.tensor(1.0, dtype=RDT)
    hard = smooth_clamp(x, lo, hi, math.inf)
    assert torch.allclose(hard, torch.tensor([-1.0, 0.0, 1.0], dtype=RDT))
    # A finite beta stays within (slightly inside) the bounds and is smooth.
    soft = smooth_clamp(x, lo, hi, 50.0)
    assert (soft <= hi + 1e-9).all() and (soft >= lo - 1e-9).all()
    assert abs(soft[1].item()) < 1e-3  # midpoint roughly unchanged


# --------------------------------------------------------------------------- #
# resolve_injection_power — reactive modes
# --------------------------------------------------------------------------- #
def test_constant_power_factor_overexcited_and_under():
    p, v = _p_v(5000.0, 1.0)
    tanphi = math.sqrt(1 - 0.95**2) / 0.95
    over = ConstantPowerFactorControl(power_factor=0.95, overexcited=True)
    _, q = resolve_injection_power(over, p, v, rdt=RDT, device=None)
    assert math.isclose(q.item(), 5000.0 * tanphi, rel_tol=1e-9)
    under = ConstantPowerFactorControl(power_factor=0.95, overexcited=False)
    _, q2 = resolve_injection_power(under, p, v, rdt=RDT, device=None)
    assert math.isclose(q2.item(), -5000.0 * tanphi, rel_tol=1e-9)


def test_constant_reactive_power():
    p, v = _p_v(5000.0, 1.02)
    c = ConstantReactivePowerControl(q_var=2000.0)
    pe, q = resolve_injection_power(c, p, v, rdt=RDT, device=None)
    assert math.isclose(q.item(), 2000.0, rel_tol=1e-12)
    assert math.isclose(pe.item(), 5000.0, rel_tol=1e-12)  # P unchanged


def test_cosphi_p_characteristic():
    # cosphi(P): unity below half rating, 0.95 absorb at full output.
    c = PowerFactorWattControl(
        characteristic=Characteristic(
            x_values=[0.0, 0.5, 1.0], y_values=[1.0, 1.0, -0.95]
        ),
        p_ref_w=10000.0,
    )
    p, v = _p_v(10000.0, 1.0)  # x = P/p_ref = 1.0 -> pf = -0.95 (absorb)
    _, q = resolve_injection_power(c, p, v, rdt=RDT, device=None)
    tanphi = math.sqrt(1 - 0.95**2) / 0.95
    assert math.isclose(q.item(), -10000.0 * tanphi, rel_tol=1e-6)


def test_volt_var_deadband_and_slope_rated():
    c = VoltVarControl(
        s_rated_va=10000.0,
        q_reference=QReference.RATED,
        characteristic=Characteristic(
            x_values=[0.95, 1.0, 1.05], y_values=[1.0, 0.0, -1.0]
        ),
    )
    p, _ = _p_v(0.0, 0.0)  # P=0 so the capability clamp does not bind
    # deadband centre -> no reactive power
    _, q0 = resolve_injection_power(
        c, p, torch.tensor([[1.0]], dtype=RDT), rdt=RDT, device=None
    )
    assert abs(q0.item()) < 1e-9
    # half-slope point -> 0.5 * rating injected
    _, qhi = resolve_injection_power(
        c, p, torch.tensor([[0.975]], dtype=RDT), rdt=RDT, device=None
    )
    assert math.isclose(qhi.item(), 5000.0, rel_tol=1e-9)


def test_volt_var_capability_clamp_binds():
    # At full P, |Q| is bounded by sqrt(S^2 - P^2); the curve would ask for the rating.
    c = VoltVarControl(
        s_rated_va=10000.0,
        q_reference=QReference.RATED,
        characteristic=Characteristic(x_values=[1.0, 1.05], y_values=[0.0, -1.0]),
    )
    p, _ = _p_v(8000.0, 0.0)
    _, q = resolve_injection_power(
        c, p, torch.tensor([[1.05]], dtype=RDT), rdt=RDT, device=None
    )
    qmax = math.sqrt(10000.0**2 - 8000.0**2)  # 6000
    assert math.isclose(q.item(), -qmax, rel_tol=1e-9)


def test_volt_var_available_reference():
    # q_reference="available": base = sqrt(S^2 - P^2), so curve y=1 injects exactly that.
    c = VoltVarControl(
        s_rated_va=10000.0,
        q_reference=QReference.AVAILABLE,
        characteristic=Characteristic(x_values=[0.95, 1.0], y_values=[1.0, 1.0]),
    )
    p, _ = _p_v(6000.0, 0.0)
    _, q = resolve_injection_power(
        c, p, torch.tensor([[0.95]], dtype=RDT), rdt=RDT, device=None
    )
    assert math.isclose(q.item(), 8000.0, rel_tol=1e-9)  # sqrt(1e8 - 3.6e7)


def test_volt_watt_curtailment():
    c = VoltWattControl(
        characteristic=Characteristic(x_values=[1.0, 1.1], y_values=[1.0, 0.0])
    )
    p, _ = _p_v(10000.0, 1.05)  # halfway up the ramp -> 50 % curtailment
    pe, q = resolve_injection_power(
        c, p, torch.tensor([[1.05]], dtype=RDT), rdt=RDT, device=None
    )
    assert math.isclose(pe.item(), 5000.0, rel_tol=1e-9)
    assert abs(q.item()) < 1e-12  # pure active control, no reactive


def test_combined_vv_vw():
    c = VoltVarVoltWattControl(
        s_rated_va=12000.0,
        volt_var=Characteristic(x_values=[1.0, 1.05], y_values=[0.0, -1.0]),
        volt_watt=Characteristic(x_values=[1.0, 1.1], y_values=[1.0, 0.0]),
    )
    p, _ = _p_v(10000.0, 1.05)
    pe, q = resolve_injection_power(
        c, p, torch.tensor([[1.05]], dtype=RDT), rdt=RDT, device=None
    )
    assert math.isclose(pe.item(), 5000.0, rel_tol=1e-9)  # VW curtails to 50 %
    # VV asks for -rating; clamp to sqrt(S^2 - P_eff^2) with P_eff = 5000
    qmax = math.sqrt(12000.0**2 - 5000.0**2)
    assert math.isclose(q.item(), -qmax, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# Integration: a controlled generator solves and regulates voltage
# --------------------------------------------------------------------------- #
def _feeder(control, p_gen=30000.0):
    nodes = [
        Node(id=0, u_rated_v=400.0, phases=ABC),
        Node(id=1, u_rated_v=400.0, phases=ABC),
    ]
    src = Source(
        id=0,
        node=0,
        phases=ABC,
        u_ref_v=[230.94] * 3,
        u_angle_deg=[0.0, -120.0, 120.0],
        resistance_ohm=[[0.05 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[2e-4 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    line = Line(
        id=0,
        from_node=0,
        to_node=1,
        from_phases=ABC,
        to_phases=ABC,
        length_m=300.0,
        series_resistance_ohm_per_m=[
            [3e-4 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        series_inductance_h_per_m=[
            [1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
        ],
        shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
    )
    gen = Generator(id=2, node=1, phases=ABC, p_nom_w=p_gen, control=control)
    return Grid(nodes=nodes, branches=[line], appliances=[src, gen])


def _mean_vpu(res):
    return (res.v.abs()[3:] / (400.0 / math.sqrt(3.0))).mean().item()


def test_volt_var_lowers_overvoltage():
    """A PV inverter at high voltage absorbs reactive power, lowering the voltage."""
    no_ctrl = _mean_vpu(solve_power_flow(_feeder(None)))
    vv = VoltVarControl(
        s_rated_va=35000.0,
        characteristic=Characteristic(
            x_values=[0.95, 1.0, 1.05], y_values=[1.0, 0.0, -1.0]
        ),
    )
    res = solve_power_flow(_feeder(vv), method="newton")
    assert res.converged
    assert _mean_vpu(res) < no_ctrl  # reactive absorption pulls V down


def test_volt_watt_curtails_overvoltage():
    no_ctrl = _mean_vpu(solve_power_flow(_feeder(None)))
    vw = VoltWattControl(
        s_rated_va=35000.0,
        characteristic=Characteristic(x_values=[1.0, 1.05], y_values=[1.0, 0.0]),
    )
    res = solve_power_flow(_feeder(vw), method="newton")
    assert res.converged
    assert _mean_vpu(res) < no_ctrl  # active curtailment pulls V down


def test_constant_pf_absorbs_reactive():
    """Underexcited constant PF absorbs Q -> lower voltage than unity PF."""
    cpf = ConstantPowerFactorControl(
        power_factor=0.95, overexcited=False, s_rated_va=35000.0
    )
    res = solve_power_flow(_feeder(cpf), method="newton")
    assert res.converged
    no_ctrl = _mean_vpu(solve_power_flow(_feeder(None)))
    assert _mean_vpu(res) < no_ctrl
