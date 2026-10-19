"""Continuation power flow (``pv_curve``) and Q-V curve (``qv_curve``).

Validation against closed forms on a two-bus feeder (stiff source, line ``R + jX``,
constant-power load ``P + jP·tanφ`` per phase):

- nose ``P_max = E² / (2 (R + X tanφ + |Z| secφ))`` (``E²/(2X)`` lossless at unity pf),
  and every curve point on both branches against
  ``V² = [E² − 2(RP + XQ) ± √((E² − 2(RP + XQ))² − 4|Z|²(P² + Q²))] / 2``;
- reverse power flow (``direction="generation"``): ``P_max = E² / (2(|Z| − R))``;
- a voltage-regulating generator with a reactive limit: the PV-to-PQ switch at
  ``cos δ = 1 − q_max X / E²`` and the nose of the limited curve, or a limit-induced nose
  when the switch lands on the lower branch;
- the Q-V curve ``Q(V) = Q_L + V²/X − √((EV/X)² − P²)`` of the lossless feeder and its
  minimum at ``V² = X²P²/E² + E²/4``;

and against plain Newton power-flow solves (``solve_power_flow``) along the upper branch of
an unbalanced, mutually coupled three-phase feeder with single-phase, ZIP and generating
appliances and a fused zero-impedance switch.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from pgml.errors import ConvergenceError, InputError
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    Switch,
    VoltageRegulation,
    ZipCoefficients,
)
from pgml.simulation import SimulationConfig
from pgml.solver import solve_power_flow
from pgml.studies import PVCurve, QVCurve, pv_curve, qv_curve

E = 230.0  # source EMF, line-to-neutral [V]
F0 = 50.0
ABC = (Phase.A, Phase.B, Phase.C)


def _line(lid, a, b, r_ohm, x_ohm, phases=(Phase.A,)):
    k = len(phases)
    l_h = x_ohm / (2.0 * math.pi * F0)
    return Line(
        id=lid,
        from_node=a,
        to_node=b,
        from_phases=phases,
        to_phases=phases,
        length_m=1.0,
        series_resistance_ohm_per_m=[
            [r_ohm if i == j else 0.0 for j in range(k)] for i in range(k)
        ],
        series_inductance_h_per_m=[
            [l_h if i == j else 0.0 for j in range(k)] for i in range(k)
        ],
        shunt_capacitance_f_per_m=[[0.0] * k for _ in range(k)],
    )


def _source(phases=(Phase.A,)):
    k = len(phases)
    return Source(
        id=10,
        node=1,
        phases=phases,
        u_ref_v=(E,) * k,
        u_angle_deg=(0.0, -120.0, 120.0)[:k],
        resistance_ohm=[
            [1.0e-9 if i == j else 0.0 for j in range(k)] for i in range(k)
        ],
        inductance_h=[[1.0e-12 if i == j else 0.0 for j in range(k)] for i in range(k)],
    )


def _two_bus(p_w, q_var=0.0, *, r=0.0, x=0.5, extra=()) -> Grid:
    """Single-phase source --(r + jx)-- bus 2 with a constant-power load."""
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=E, phases=(Phase.A,)),
            Node(id=2, u_rated_v=E, phases=(Phase.A,)),
        ],
        branches=[_line(20, 1, 2, r, x)],
        appliances=[
            _source(),
            Load(id=30, node=2, phases=(Phase.A,), p_nom_w=p_w, q_nom_var=q_var),
            *extra,
        ],
    )


def _nose_power(tan_phi, r, x):
    z = math.hypot(r, x)
    sec = math.sqrt(1.0 + tan_phi**2)
    return E**2 / (2.0 * (r + x * tan_phi + z * sec))


def _v_closed_form(p, q, r, x):
    """Upper / lower branch receiving-end voltages [V] of the two-bus feeder."""
    b = E**2 - 2.0 * (r * p + x * q)
    disc = b * b - 4.0 * (r * r + x * x) * (p * p + q * q)
    root = math.sqrt(max(disc, 0.0))
    return math.sqrt((b + root) / 2.0), math.sqrt((b - root) / 2.0)


def _p_closed_form(v, tan_phi, r, x):
    """Load power [W] at receiving-end voltage ``v`` [V] (the root with ``P >= 0``)."""
    a = (r * r + x * x) * (1.0 + tan_phi**2)
    b = 2.0 * (r + x * tan_phi) * v * v
    c = v**4 - E**2 * v * v
    return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


# ---------------------------------------------------------------------------
# P-V: closed forms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tan_phi", "r", "x"),
    [(0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (-0.3, 0.0, 0.5), (0.2, 0.3, 0.5)],
)
def test_nose_matches_closed_form(tan_phi, r, x):
    p_max = _nose_power(tan_phi, r, x)
    p0 = 0.5 * p_max
    curve = pv_curve(_two_bus(p0, p0 * tan_phi, r=r, x=x))
    assert curve.converged and curve.stop_reason in ("v_stop", "lambda_start")
    assert curve.nose_kind == "saddle_node"
    assert curve.lambda_max == pytest.approx(2.0, rel=1e-6)
    assert curve.lambdas[curve.nose_index] == curve.lambda_max
    assert curve.lambdas[0] == 1.0  # starts at the base operating point
    assert curve.critical_node == 2 and curve.critical_phase == "a"
    # λ rises to the nose and falls after it
    lam = curve.lambdas
    assert np.all(np.diff(lam[: curve.nose_index + 1]) > 0)
    assert np.all(np.diff(lam[curve.nose_index :]) < 0)


@pytest.mark.parametrize(("tan_phi", "r"), [(0.0, 0.0), (0.2, 0.3)])
def test_every_point_lies_on_the_closed_form_curve(tan_phi, r):
    x = 0.5
    p0 = 0.5 * _nose_power(tan_phi, r, x)
    curve = pv_curve(_two_bus(p0, p0 * tan_phi, r=r, x=x), lambda_start=0.0)
    v2 = curve.voltage_pu(2, "a")
    for k, lam in enumerate(curve.lambdas):
        # the load the solved voltage carries (well conditioned at the nose too)
        assert _p_closed_form(v2[k] * E, tan_phi, r, x) == pytest.approx(
            lam * p0, rel=1e-9, abs=1e-6
        )
        if lam < 0.999 * curve.lambda_max:  # V(λ) is ill-conditioned at the fold
            hi, lo = _v_closed_form(lam * p0, lam * p0 * tan_phi, r, x)
            branch = hi if k <= curve.nose_index else lo
            assert v2[k] == pytest.approx(branch / E, abs=1e-9)
    # p_total_w is the scaled constant power
    np.testing.assert_allclose(curve.p_total_w, curve.lambdas * p0, rtol=1e-12)


def test_generation_direction_reverse_flow_nose():
    r, x = 0.5, 0.5
    gen = Generator(id=40, node=2, phases=(Phase.A,), p_nom_w=10_000.0, q_nom_var=0.0)
    grid = _two_bus(0.0, r=r, x=x, extra=(gen,))
    curve = pv_curve(grid, direction="generation", stop_at_nose=True)
    p_max = E**2 / (2.0 * (math.hypot(r, x) - r))
    assert curve.appliance_ids == (40,)
    assert curve.stop_reason == "nose"
    assert curve.nose_index == len(curve.lambdas) - 1
    assert curve.lambda_max * 10_000.0 == pytest.approx(p_max, rel=1e-6)
    assert curve.p_total_w[-1] == pytest.approx(p_max, rel=1e-6)  # generation positive
    assert curve.voltage_pu(2, "a")[-1] > 1.0  # reverse flow raises the voltage


def test_balanced_three_phase_nose_is_three_times_the_phase_nose():
    x = 0.5
    grid = Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=E * math.sqrt(3.0), phases=ABC),
            Node(id=2, u_rated_v=E * math.sqrt(3.0), phases=ABC),
        ],
        branches=[_line(20, 1, 2, 0.0, x, ABC)],
        appliances=[
            _source(ABC),
            Load(id=30, node=2, phases=ABC, p_nom_w=30_000.0, q_nom_var=0.0),
        ],
    )
    curve = pv_curve(grid, stop_at_nose=True)
    assert curve.lambda_max * 30_000.0 == pytest.approx(3 * E**2 / (2 * x), rel=1e-6)
    assert curve.v_pu.shape == (len(curve.lambdas), 6)
    nose = curve.voltage_pu(2)[curve.nose_index]
    np.testing.assert_allclose(nose, 1.0 / math.sqrt(2.0), atol=1e-6)


# ---------------------------------------------------------------------------
# P-V: voltage-regulating generator with a reactive limit
# ---------------------------------------------------------------------------
def _regulated(q_max, p0):
    reg = VoltageRegulation(v_set_pu=1.0, q_max_var=q_max)
    gen = Generator(
        id=40, node=2, phases=(Phase.A,), p_nom_w=0.0, voltage_regulation=reg
    )
    return _two_bus(p0, x=0.5, extra=(gen,))


def test_reactive_limit_switch_then_saddle_node():
    x, q, p0 = 0.5, 20_000.0, 30_000.0
    curve = pv_curve(_regulated(q, p0), stop_at_nose=True)
    cos_d = 1.0 - q * x / E**2
    p_switch = E**2 * math.sqrt(1.0 - cos_d**2) / x
    p_nose = math.sqrt(((E**2 + 2 * x * q) / (2 * x)) ** 2 - q**2)
    (event,) = curve.events
    assert event["generator_ids"] == [40]
    assert event["lambda"] * p0 == pytest.approx(p_switch, rel=1e-5)
    # held at the setpoint until the switch
    v2 = curve.voltage_pu(2, "a")
    np.testing.assert_allclose(v2[: event["index"]], 1.0, atol=1e-9)
    assert curve.nose_kind == "saddle_node"
    assert curve.lambda_max * p0 == pytest.approx(p_nose, rel=1e-6)


def test_limit_induced_nose_when_the_switch_lands_on_the_lower_branch():
    x, q, p0 = 0.5, 60_000.0, 30_000.0  # q > E²/(2X): the PQ curve is past its nose
    curve = pv_curve(_regulated(q, p0))
    cos_d = 1.0 - q * x / E**2
    p_switch = E**2 * math.sqrt(1.0 - cos_d**2) / x
    assert curve.nose_kind == "limit_induced"
    assert curve.nose_index == curve.events[0]["index"]
    assert curve.lambda_max * p0 == pytest.approx(p_switch, rel=1e-5)
    assert curve.lambdas.max() == pytest.approx(curve.lambda_max, rel=1e-6)


def test_without_limit_enforcement_the_terminal_keeps_regulating():
    curve = pv_curve(
        _regulated(20_000.0, 30_000.0),
        config=SimulationConfig(calculation="power_flow", enforce_q_limits=False),
        lambda_max=2.5,
    )
    assert curve.events == () and curve.stop_reason == "lambda_max"
    np.testing.assert_allclose(curve.voltage_pu(2, "a"), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# P-V: agreement with plain power-flow solves on an unbalanced feeder
# ---------------------------------------------------------------------------
def _coupled(diag, off):
    return [[diag if i == j else off for j in range(3)] for i in range(3)]


def _unbalanced_feeder() -> Grid:
    u = E * math.sqrt(3.0)
    nodes = [Node(id=i, u_rated_v=u, phases=ABC) for i in range(1, 6)]

    def line(lid, a, b, length):
        return Line(
            id=lid,
            from_node=a,
            to_node=b,
            from_phases=ABC,
            to_phases=ABC,
            length_m=length,
            series_resistance_ohm_per_m=_coupled(0.4e-3, 0.05e-3),
            series_inductance_h_per_m=_coupled(0.9e-6, 0.3e-6),
            shunt_capacitance_f_per_m=_coupled(2.0e-10, -4.0e-11),
        )

    zip_q = ZipCoefficients(z_p=0.3, i_p=0.3, p_p=0.4, z_q=0.2, i_q=0.3, p_q=0.5)
    return Grid(
        base_frequency_hz=F0,
        nodes=nodes,
        branches=[
            line(21, 1, 2, 150.0),
            line(22, 2, 3, 200.0),
            line(23, 3, 4, 150.0),
            Switch(
                id=24,
                from_node=4,
                to_node=5,
                from_phases=ABC,
                to_phases=ABC,
                default_closed=True,
            ),
        ],
        appliances=[
            _source(ABC),
            Load(id=31, node=3, phases=ABC, p_nom_w=30_000.0, q_nom_var=9_000.0),
            Load(id=32, node=4, phases=(Phase.A,), p_nom_w=12_000.0, q_nom_var=3_000.0),
            Load(
                id=33,
                node=5,
                phases=(Phase.B,),
                p_nom_w=8_000.0,
                q_nom_var=2_000.0,
                load_model=LoadModel.ZIP,
                zip_coefficients=zip_q,
            ),
            Generator(id=41, node=2, phases=ABC, p_nom_w=6_000.0, q_nom_var=0.0),
        ],
    )


def test_upper_branch_matches_plain_newton_solves():
    grid = _unbalanced_feeder()
    curve = pv_curve(grid, lambda_start=0.0, stop_at_nose=True)
    assert curve.appliance_ids == (31, 32, 33)
    assert curve.stop_reason == "nose" and curve.lambda_max > 1.0
    loads = {a.id: a for a in grid.appliances if isinstance(a, Load)}
    worst = 0.0
    picks = [k for k, lam in enumerate(curve.lambdas) if lam < 0.97 * curve.lambda_max]
    for k in picks:
        lam = float(curve.lambdas[k])
        op = {
            aid: {"p_w": lam * a.p_nom_w, "q_var": lam * a.q_nom_var}
            for aid, a in loads.items()
        }
        ref = solve_power_flow(grid, method="newton", operating_point=op)
        assert bool(ref.converged)
        v_pu_ref = ref.v.reshape(-1).abs().numpy() / E  # every row's L-N base is E
        worst = max(worst, float(np.max(np.abs(v_pu_ref - curve.v_pu[k]))))
    assert len(picks) >= 5
    assert worst < 1e-8
    # the switch fuses nodes 4 and 5: their rows carry one voltage
    np.testing.assert_array_equal(curve.voltage_pu(4), curve.voltage_pu(5))
    assert curve.critical_node in (4, 5)


def test_lambda_cap_lands_exactly_on_the_cap():
    curve = pv_curve(_two_bus(10_000.0), lambda_max=2.5)
    assert curve.stop_reason == "lambda_max" and curve.converged
    assert curve.nose_index is None and curve.lambda_max is None
    assert curve.lambdas[-1] == pytest.approx(2.5, abs=1e-12)
    hi, _ = _v_closed_form(25_000.0, 0.0, 0.0, 0.5)
    assert curve.voltage_pu(2, "a")[-1] == pytest.approx(hi / E, abs=1e-9)


def test_selectors_choose_the_scaled_appliances():
    extra = Load(id=31, node=2, phases=(Phase.A,), p_nom_w=10_000.0, q_nom_var=0.0)
    grid = _two_bus(10_000.0, extra=(extra,))
    p_max = E**2 / (2 * 0.5)
    by_id = pv_curve(grid, selector=[30], stop_at_nose=True)
    by_pred = pv_curve(grid, selector=lambda a: a.id == 30, stop_at_nose=True)
    assert by_id.appliance_ids == by_pred.appliance_ids == (30,)
    # only load 30 scales; load 31 stays at 10 kW
    assert by_id.lambda_max * 10_000.0 + 10_000.0 == pytest.approx(p_max, rel=1e-6)
    assert by_pred.lambda_max == pytest.approx(by_id.lambda_max, rel=1e-12)


# ---------------------------------------------------------------------------
# Q-V
# ---------------------------------------------------------------------------
def test_qv_curve_matches_closed_form():
    x = 0.5
    p, q_l = 0.8 * E**2 / (2 * x), 3_000.0
    curve = qv_curve(_two_bus(p, q_l, x=x), 2, v_range_pu=(0.5, 1.1), steps=61)
    assert curve.converged.all() and curve.phases == ("a",)
    v = curve.v_pu * E
    q_exact = q_l + v**2 / x - np.sqrt((E * v / x) ** 2 - p**2)
    np.testing.assert_allclose(curve.q_var, q_exact, rtol=1e-8, atol=1e-6)
    v_min = math.sqrt(x**2 * p**2 / E**2 + E**2 / 4.0)
    q_min = q_l + v_min**2 / x - E**2 / (2 * x)
    assert curve.minimum_in_range
    assert curve.v_at_min_pu == pytest.approx(v_min / E, abs=2e-3)
    assert curve.q_margin_var == pytest.approx(-q_min, rel=1e-4)


def test_qv_curve_reports_non_convergence_below_the_feasible_voltage():
    x = 0.5
    p = 0.8 * E**2 / (2 * x)  # no solution for V < P X / E = 0.4 E
    curve = qv_curve(_two_bus(p, x=x), 2, v_range_pu=(0.3, 1.0), steps=15)
    assert not curve.converged[0] and curve.converged[-1]
    assert np.isnan(curve.q_var[0])
    assert np.all(curve.v_pu[~curve.converged] < 0.4)


def test_qv_on_a_three_phase_bus_holds_every_phase():
    grid = _unbalanced_feeder()
    curve = qv_curve(grid, 3, v_range_pu=(0.9, 1.05), steps=7)
    assert curve.phases == ("a", "b", "c") and curve.converged.all()
    np.testing.assert_allclose(curve.q_var, curve.q_phase_var.sum(axis=1))
    # more support is needed to hold a higher voltage
    assert np.all(np.diff(curve.q_var) > 0)


# ---------------------------------------------------------------------------
# contracts: serialization, inputs, options, dtype
# ---------------------------------------------------------------------------
def test_pv_round_trip_through_json():
    curve = pv_curve(_regulated(20_000.0, 30_000.0))
    data = json.loads(json.dumps(curve.to_dict(), allow_nan=False))
    assert data["study"] == "pv_curve"
    back = PVCurve.from_dict(data)
    np.testing.assert_array_equal(back.v_pu, curve.v_pu)
    np.testing.assert_array_equal(back.lambdas, curve.lambdas)
    assert back.to_dict() == curve.to_dict()


def test_qv_round_trip_through_json_keeps_missing_points():
    curve = qv_curve(_two_bus(0.8 * E**2, x=0.5), 2, v_range_pu=(0.3, 1.0), steps=8)
    data = json.loads(json.dumps(curve.to_dict(), allow_nan=False))
    assert data["study"] == "qv_curve"
    assert data["q_var"][0] is None
    assert data["q_phase_var"][0] == [None] and "q_var_phase" not in data
    back = QVCurve.from_dict(data)
    np.testing.assert_array_equal(back.converged, curve.converged)
    assert back.to_dict() == curve.to_dict()


def test_from_dict_refuses_another_study():
    pv = pv_curve(_two_bus(10_000.0), stop_at_nose=True).to_dict()
    qv = qv_curve(_two_bus(10_000.0), 2, steps=5).to_dict()
    with pytest.raises(InputError, match="expected a 'pv_curve'.*found.*'qv_curve'"):
        PVCurve.from_dict(qv)
    with pytest.raises(InputError, match="expected a 'qv_curve'.*found.*'pv_curve'"):
        QVCurve.from_dict(pv)
    with pytest.raises(InputError, match='found "study": None'):
        QVCurve.from_dict({k: v for k, v in qv.items() if k != "study"})


def test_phases_are_schema_values():
    grid = _unbalanced_feeder()
    curve = pv_curve(grid, stop_at_nose=True)
    assert set(curve.phases) == {"a", "b", "c"}
    assert curve.critical_phase in ("a", "b", "c")
    assert curve.to_dict()["phases"] == list(curve.phases)
    # a Phase, its letter and the upper-case letter select the same row
    v_a = curve.voltage_pu(3, Phase.A)
    np.testing.assert_array_equal(curve.voltage_pu(3, "a"), v_a)
    np.testing.assert_array_equal(curve.voltage_pu(3, "A"), v_a)
    with pytest.raises(InputError, match="phase must be"):
        curve.voltage_pu(3, "x")


def test_inputs_are_not_mutated():
    grid = _regulated(20_000.0, 30_000.0)
    cfg = SimulationConfig(calculation="power_flow", operating_point={30: {"p_w": 2e4}})
    before = (grid.model_dump_json(), cfg.model_dump_json())
    pv_curve(grid, config=cfg, stop_at_nose=True)
    qv_curve(_two_bus(10_000.0), 2, config=cfg, steps=5)
    assert (grid.model_dump_json(), cfg.model_dump_json()) == before


def test_operating_point_is_the_base():
    cfg = SimulationConfig(calculation="power_flow", operating_point={30: {"p_w": 2e4}})
    curve = pv_curve(_two_bus(50_000.0), config=cfg, stop_at_nose=True)
    assert curve.p_total_w[0] == pytest.approx(2e4, rel=1e-12)
    assert curve.lambda_max * 2e4 == pytest.approx(E**2, rel=1e-6)


@pytest.mark.parametrize("dtype", ["complex64", torch.complex64])
def test_complex64(dtype):
    curve = pv_curve(_two_bus(26_450.0), dtype=dtype, stop_at_nose=True)
    assert curve.lambda_max == pytest.approx(2.0, rel=1e-3)


def test_qv_accepts_a_dtype_name():
    grid = _two_bus(10_000.0)
    by_name = qv_curve(grid, 2, steps=5, dtype="complex128")
    by_dtype = qv_curve(grid, 2, steps=5, dtype=torch.complex128)
    np.testing.assert_array_equal(by_name.q_var, by_dtype.q_var)


@pytest.mark.parametrize("dtype", [torch.float64, "float32", "complex256"])
def test_unknown_dtype_raises(dtype):
    grid = _two_bus(10_000.0)
    with pytest.raises(InputError, match="dtype"):
        pv_curve(grid, dtype=dtype)
    with pytest.raises(InputError, match="dtype"):
        qv_curve(grid, 2, dtype=dtype)


def test_norton_slack():
    cfg = SimulationConfig(calculation="power_flow", slack="norton")
    curve = pv_curve(_two_bus(26_450.0), config=cfg, stop_at_nose=True)
    assert curve.lambda_max == pytest.approx(2.0, rel=1e-6)


def test_bad_inputs_raise():
    grid = _two_bus(10_000.0)
    with pytest.raises(InputError):
        pv_curve(grid, direction="sideways")
    with pytest.raises(InputError):
        pv_curve(grid, direction="generation")  # no generator to scale
    with pytest.raises(InputError):
        pv_curve(grid, selector=[999])
    with pytest.raises(InputError):
        pv_curve(grid, selector=lambda a: False)
    with pytest.raises(InputError):
        pv_curve(grid, lambda_max=0.5)
    with pytest.raises(InputError):
        qv_curve(grid, 1)  # the slack bus
    with pytest.raises(InputError):
        qv_curve(grid, 7)
    with pytest.raises(InputError):
        qv_curve(_regulated(1e4, 1e4), 2)  # a regulated bus
    with pytest.raises(InputError):
        qv_curve(grid, 2, v_range_pu=(1.1, 0.9))


def test_infeasible_start_raises():
    with pytest.raises(ConvergenceError):
        pv_curve(_two_bus(3.0 * E**2))
    # the same grid traces from no load
    curve = pv_curve(_two_bus(3.0 * E**2), lambda_start=0.0, stop_at_nose=True)
    assert curve.lambda_max == pytest.approx(1.0 / 3.0, rel=1e-6)
