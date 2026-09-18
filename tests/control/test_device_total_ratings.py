"""Inverter control ratings are device totals, split equally over the elements.

``s_rated_va``, ``q_var`` and ``p_ref_w`` describe the whole device. A device with
``n_elem`` connection elements (phases for WYE, phase pairs for DELTA) gives each
element ``1 / n_elem`` of them, so a balanced three-phase device and its
positive-sequence single-phase equivalent inject the same totals.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.assembly import device_current_injections
from pgml.assembly._control import resolve_injection_power
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
    Source,
    VoltVarControl,
    VoltVarVoltWattControl,
    VoltWattControl,
    WindingConnection,
)
from pgml.solver import solve_power_flow

RDT = torch.float64
CDT = torch.complex128
A = (Phase.A,)
ABC = (Phase.A, Phase.B, Phase.C)
U_LL = 400.0
WYE = WindingConnection.WYE
DELTA = WindingConnection.DELTA

Q_OF_V = Characteristic(x_values=[0.95, 1.0, 1.05], y_values=[1.0, 0.0, -1.0])
P_OF_V = Characteristic(x_values=[1.0, 1.05], y_values=[1.0, 0.0])
COSPHI_OF_P = Characteristic(x_values=[0.0, 0.5, 1.0], y_values=[1.0, 1.0, -0.9])

#: Every control law, with the ratings of one 30 kW device.
CONTROLS = {
    "constant_q": ConstantReactivePowerControl(q_var=-9_000.0),
    "constant_q_clamped": ConstantReactivePowerControl(
        q_var=-30_000.0, s_rated_va=33_000.0
    ),
    "constant_pf": ConstantPowerFactorControl(
        power_factor=0.9, overexcited=False, s_rated_va=31_000.0
    ),
    "cosphi_p": PowerFactorWattControl(characteristic=COSPHI_OF_P, p_ref_w=36_000.0),
    "volt_var_rated": VoltVarControl(characteristic=Q_OF_V, s_rated_va=35_000.0),
    "volt_var_available": VoltVarControl(
        characteristic=Q_OF_V, s_rated_va=35_000.0, q_reference="available"
    ),
    "volt_watt": VoltWattControl(characteristic=P_OF_V, s_rated_va=35_000.0),
    "volt_var_volt_watt": VoltVarVoltWattControl(
        volt_var=Q_OF_V, volt_watt=P_OF_V, s_rated_va=35_000.0
    ),
    "active_power_clamp": ConstantPowerFactorControl(
        power_factor=1.0, s_rated_va=24_000.0
    ),
    "soft_clamp": ConstantReactivePowerControl(
        q_var=-14_000.0, s_rated_va=33_000.0, smoothing=0.05
    ),
}


def _diag(value, n):
    return [[value if i == j else 0.0 for j in range(n)] for i in range(n)]


def _grid(control, phases, connection=None, p_gen=30_000.0):
    """Source, one line and one generator; ``phases=A`` is the positive-sequence
    equivalent (line-to-line voltage, total power on one conductor)."""
    n = len(phases)
    u_src = U_LL if n == 1 else U_LL / math.sqrt(3.0)
    source = Source(
        id=0,
        node=0,
        phases=phases,
        u_ref_v=[u_src] * n,
        u_angle_deg=[0.0, -120.0, 120.0][:n],
        resistance_ohm=_diag(0.05, n),
        inductance_h=_diag(2e-4, n),
    )
    line = Line(
        id=0,
        from_node=0,
        to_node=1,
        from_phases=phases,
        to_phases=phases,
        length_m=300.0,
        series_resistance_ohm_per_m=_diag(3e-4, n),
        series_inductance_h_per_m=_diag(1e-6, n),
        shunt_capacitance_f_per_m=_diag(0.0, n),
    )
    gen = Generator(
        id=2,
        node=1,
        phases=phases,
        connection=connection,
        p_nom_w=p_gen,
        control=control,
    )
    return Grid(
        nodes=[Node(id=i, u_rated_v=U_LL, phases=phases) for i in range(2)],
        branches=[line],
        appliances=[source, gen],
    )


def _solve(grid):
    result = solve_power_flow(
        grid, method="newton", dtype=CDT, tol=1e-12, tol_update_pu=1e-12
    )
    assert result.converged
    return result


def _device_injection(grid):
    """Total complex power the generator injects, and its terminal voltage in pu."""
    result = _solve(grid)
    n = len(grid.nodes[1].phases)
    v = result.v[-n:]
    drawn = device_current_injections(grid, result.v, result.index, [50.0], dtype=CDT)
    s = -(v * torch.conj(drawn.reshape(-1)[-n:])).sum()
    v_nom = U_LL if n == 1 else U_LL / math.sqrt(3.0)
    return s, float(v.abs().mean() / v_nom)


@pytest.mark.parametrize("connection", [WYE, DELTA], ids=["wye", "delta"])
def test_constant_q_is_the_device_total(connection):
    control = ConstantReactivePowerControl(q_var=3_000.0)
    s, _ = _device_injection(_grid(control, ABC, connection, p_gen=6_000.0))
    assert float(s.imag) == pytest.approx(3_000.0, rel=1e-9)
    assert float(s.real) == pytest.approx(6_000.0, rel=1e-9)


@pytest.mark.parametrize("connection", [WYE, DELTA], ids=["wye", "delta"])
def test_rating_limits_the_device_total(connection):
    """A 6 kW three-phase unit behind a 5 kVA inverter delivers 5 kW."""
    unity = ConstantPowerFactorControl(power_factor=1.0, s_rated_va=5_000.0)
    s, _ = _device_injection(_grid(unity, ABC, connection, p_gen=6_000.0))
    assert float(s.real) == pytest.approx(5_000.0, rel=1e-9)

    # 4 kW leaves 3 kvar under the 5 kVA circle, whatever the request.
    q_request = ConstantReactivePowerControl(q_var=9_000.0, s_rated_va=5_000.0)
    s, _ = _device_injection(_grid(q_request, ABC, connection, p_gen=4_000.0))
    assert float(s.real) == pytest.approx(4_000.0, rel=1e-9)
    assert float(s.imag) == pytest.approx(3_000.0, rel=1e-9)


@pytest.mark.parametrize("connection", [WYE, DELTA], ids=["wye", "delta"])
@pytest.mark.parametrize("law", list(CONTROLS))
def test_three_phase_solve_matches_the_positive_sequence_equivalent(law, connection):
    control = CONTROLS[law]
    s_eq, v_eq = _device_injection(_grid(control, A))
    s_3p, v_3p = _device_injection(_grid(control, ABC, connection))
    assert v_3p == pytest.approx(v_eq, rel=1e-9)
    assert float(s_3p.real) == pytest.approx(float(s_eq.real), rel=1e-8)
    assert float(s_3p.imag) == pytest.approx(float(s_eq.imag), rel=1e-8, abs=1e-6)


def test_the_laws_are_active_in_the_comparison():
    """The equivalence above is only meaningful where the laws do something."""
    _, v = _device_injection(_grid(None, A))
    assert 1.01 < v < 1.04  # inside the Q(V) slope and the P(V) curtailment band
    s, _ = _device_injection(_grid(CONTROLS["volt_watt"], A))
    assert 3_000.0 < float(s.real) < 27_000.0
    s, _ = _device_injection(_grid(CONTROLS["volt_var_rated"], A))
    assert -30_000.0 < float(s.imag) < -3_000.0
    s, _ = _device_injection(_grid(CONTROLS["cosphi_p"], A))
    assert float(s.imag) < -1_000.0
    s, _ = _device_injection(_grid(CONTROLS["active_power_clamp"], A))
    assert float(s.real) == pytest.approx(24_000.0, rel=1e-9)


def _resolve(control, p_total, v_pu, n_elem, batch=()):
    p = torch.full((*batch, n_elem), p_total / n_elem, dtype=RDT)
    v = torch.full((*batch, 1, n_elem), v_pu, dtype=RDT)
    return resolve_injection_power(control, p, v, rdt=RDT, device=None)


@pytest.mark.parametrize("n_elem", [1, 2, 3])
@pytest.mark.parametrize("law", list(CONTROLS))
def test_element_powers_sum_to_the_single_element_device(law, n_elem):
    control = CONTROLS[law]
    p_one, q_one = _resolve(control, 30_000.0, 1.03, 1)
    p_n, q_n = _resolve(control, 30_000.0, 1.03, n_elem)
    assert p_n.shape == (1, n_elem)
    # Equal elements, and their sum is the device the single element describes.
    assert torch.allclose(p_n, p_one / n_elem, rtol=1e-12, atol=0.0)
    assert torch.allclose(q_n, q_one / n_elem, rtol=1e-12, atol=1e-9)


def test_single_phase_device_keeps_the_full_rating():
    control = ConstantReactivePowerControl(q_var=9_000.0, s_rated_va=5_000.0)
    p, q = _resolve(control, 4_000.0, 1.0, 1)
    assert float(p) == pytest.approx(4_000.0)
    assert float(q) == pytest.approx(3_000.0)


def test_soft_clamp_width_is_a_fraction_of_the_device_rating():
    """At the corner the DEVICE falls short of its limit by ``smoothing*S*ln 2``."""
    s_rated, p_total, smoothing = 20_000.0, 12_000.0, 0.05
    q_max = math.sqrt(s_rated**2 - p_total**2)
    control = ConstantReactivePowerControl(
        q_var=q_max, s_rated_va=s_rated, smoothing=smoothing
    )
    for n_elem in (1, 3):
        _, q = _resolve(control, p_total, 1.0, n_elem)
        shortfall = q_max - float(q.sum())
        assert shortfall == pytest.approx(smoothing * s_rated * math.log(2.0), rel=1e-3)


def test_split_is_vectorised_over_a_scenario_batch():
    control = CONTROLS["volt_var_volt_watt"]
    p, q = _resolve(control, 30_000.0, 1.03, 3, batch=(4, 2))
    p_one, q_one = _resolve(control, 30_000.0, 1.03, 1)
    assert p.shape == q.shape == (4, 2, 1, 3)
    assert torch.allclose(p.sum(-1), p_one.sum(-1).expand(4, 2, 1))
    assert torch.allclose(q.sum(-1), q_one.sum(-1).expand(4, 2, 1))


def test_split_honours_dtype_and_tensor_ratings():
    s_rated = torch.tensor(5_000.0, dtype=torch.float32, requires_grad=True)
    control = ConstantReactivePowerControl(q_var=9_000.0, s_rated_va=s_rated)
    p = torch.full((3,), 4_000.0 / 3, dtype=torch.float32)
    v = torch.ones((1, 3), dtype=torch.float32)
    _, q = resolve_injection_power(control, p, v, rdt=torch.float32, device=None)
    assert q.dtype == torch.float32
    assert float(q.sum().detach()) == pytest.approx(3_000.0, rel=1e-5)
    (grad,) = torch.autograd.grad(q.sum(), s_rated)
    # d sqrt(S² - P²) / dS = S / Q_max for the device as a whole.
    assert float(grad) == pytest.approx(5_000.0 / 3_000.0, rel=1e-5)
