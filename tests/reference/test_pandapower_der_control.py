"""Oracle test: DER control modes and Storage vs pandapower.

Three small 2-bus feeders (single-phase positive-sequence, 50 Hz, 0.4 kV) validate
that pgml's inverter control laws and the Storage element agree with pandapower.

Feeder topology (Tests 1 and 3)
--------------------------------
- Bus 0 (slack): pandapower ``ext_grid`` / pgml ``Source`` with ideal slack
  (``slack="ideal"``).  The tiny Thevenin impedance on the ``Source`` (1e-6 Ω,
  ~2e-15 H) is irrelevant when ``slack="ideal"`` is used.
- Bus 1 (load + DER): 3 kW / 1 kVAr load plus one DER element per test.
- Line 0-1: R=0.3 Ω/km, X=0.15 Ω/km, C=0, length=0.1 km (100 m).

Feeder topology (Test 2 — Volt-VAr, longer line)
-------------------------------------------------
Same components, but length=1.0 km (1 km) and generation P=10 kW so the
uncurtailed bus-1 voltage rises to ~1.013 pu.  With the Q(V) curve active the DER
absorbs vars and the two tools' converged voltages agree to ~1e-9 pu.

pandapower reference setup
--------------------------
- Test 1 (Constant PF): ``net.sgen`` with ``q_mvar`` set to
  ``P * tan(acos(pf))``; positive ``q_mvar`` = overexcited (inject Q);
  negative = underexcited (absorb Q).  A plain ``pp.runpp`` is sufficient.
- Test 2 (Volt-VAr): ``pp.control.Characteristic`` (y-axis in MVAR) +
  ``pp.control.CharacteristicControl`` mapping ``res_bus.vm_pu → sgen.q_mvar``,
  converged via ``pp.control.run_control(tol=1e-9)``.  Tight controller
  tolerance is needed to match pgml's continuous Newton evaluation of Q(V).
- Test 3 (Storage snapshot): ``pp.create_storage`` with ``p_mw < 0``
  (negative = discharging = inject, the reverse of pgml convention); Q=0.
  Plain ``pp.runpp`` is sufficient.

pgml models
-----------
- Test 1: ``Generator(control=ConstantPowerFactorControl(power_factor=0.95,
  overexcited=True|False, s_rated_va=5000))``.
- Test 2: ``Generator(control=VoltVarControl(s_rated_va=12000,
  q_reference=QReference.RATED, characteristic=Characteristic(...)))``.
  ``method="newton"`` is used because the current-injection fixed point
  oscillates on stiff Volt-VAr curves.
- Test 3: ``Storage(p_nom_w=+2000)``.  Positive = discharging (generator
  convention), which is the OPPOSITE of pandapower ``storage.p_mw`` (load
  convention): ``pgml.p_nom_w = −pandapower.p_mw``.

Slack convention
----------------
Both tools use an IDEAL slack at bus 0 fixed to 400 V∠0°.  pandapower
``ext_grid`` is always ideal; pgml uses ``slack="ideal"``.

Sign conventions
----------------
- ``sgen.q_mvar > 0`` (pandapower) = inject Q = overexcited (same as pgml
  ``overexcited=True``).
- ``storage.p_mw > 0`` (pandapower) = CHARGING (draw from grid) — the load
  convention.  pgml ``Storage.p_nom_w > 0`` = DISCHARGING (inject) — the
  generator convention.  To compare: ``pgml.p_nom_w = −pandapower.storage.p_mw``.

Tolerance targets
-----------------
- Test 1 (Constant PF): ``atol = 1e-6 pu`` on voltage magnitude.  The PF
  control is voltage-independent (Q is fixed), so both tools solve the same
  linear constant-power system; agreement is near machine precision.
- Test 2 (Volt-VAr): ``atol = 1e-6 pu`` on voltage magnitude and
  ``atol = 1.0 var`` on reactive power.  With ``tol=1e-9`` on the pandapower
  controller the outer-loop residual is ~1e-9 pu; the residual in Q is < 1 var.
- Test 3 (Storage): ``atol = 1e-6 pu`` on voltage magnitude.  The storage is
  a fixed (P, Q) injection; agreement is near machine precision.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Optional pandapower guard (matches existing reference test conventions)
# ---------------------------------------------------------------------------
try:
    import numpy as _np

    _np.Inf = _np.inf
    _np.in1d = _np.isin
    import pandapower as pp
    import pandapower.control as ppctrl

    _PP_AVAILABLE = True
except ImportError:
    _PP_AVAILABLE = False

if not _PP_AVAILABLE:
    pytest.skip("pandapower not installed", allow_module_level=True)

import torch

from pgml.schemas.grid_schema import (
    Characteristic,
    ConstantPowerFactorControl,
    Generator,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    QReference,
    Source,
    Storage,
    VoltVarControl,
)
from pgml.solver import solve_power_flow

# ---------------------------------------------------------------------------
# Shared feeder parameters
# ---------------------------------------------------------------------------

_F0: float = 50.0  # Hz
_TWO_PI_F0: float = 2.0 * math.pi * _F0

# Voltage base: line-to-line for the 0.4 kV LV network
_U_LL_V: float = 400.0
_U_LL_KV: float = _U_LL_V / 1_000.0

# Source: tiny Thevenin impedance (irrelevant under ideal slack)
_R_S_OHM: float = 1e-6
_L_S_H: float = 1e-12 / _TWO_PI_F0

# Short feeder (Tests 1 and 3)
_LINE_KM_SHORT: float = 0.1
_R_OHM_PER_KM: float = 0.3
_X_OHM_PER_KM: float = 0.15

# Load at bus 1
_P_LOAD_W: float = 3_000.0
_Q_LOAD_VAR: float = 1_000.0

# DER: PV generator (Tests 1 and 2)
_P_GEN_W_SHORT: float = 4_000.0  # 4 kW on the short feeder (Test 1)
_S_RATED_VA_SHORT: float = 5_000.0  # 5 kVA inverter (Test 1)

# Volt-VAr feeder (Test 2): longer line + larger generator so Q(V) activates
_LINE_KM_LONG: float = 1.0
_P_GEN_W_LONG: float = 10_000.0  # 10 kW PV
_S_RATED_VA_LONG: float = 12_000.0  # 12 kVA inverter
_P_LOAD_W_VVC: float = 2_000.0  # 2 kW load on the Volt-VAr feeder
_Q_LOAD_VAR_VVC: float = 1_000.0

# Storage (Test 3)
_P_STORAGE_W: float = 2_000.0  # 2 kW discharge (inject)


# ---------------------------------------------------------------------------
# pgml grid builders
# ---------------------------------------------------------------------------


def _pgml_line(length_km: float) -> Line:
    """Return a single-phase line with the shared R/X parameters."""
    return Line(
        id=10,
        from_node=0,
        to_node=1,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=length_km * 1_000.0,
        series_resistance_ohm_per_m=[[_R_OHM_PER_KM * 1e-3]],
        series_inductance_h_per_m=[[_X_OHM_PER_KM * 1e-3 / _TWO_PI_F0]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


def _pgml_slack() -> Source:
    """Ideal-slack Thevenin source (negligible impedance, used with ``slack='ideal'``)."""
    return Source(
        id=100,
        node=0,
        phases=(Phase.A,),
        u_ref_v=[_U_LL_V],
        u_angle_deg=[0.0],
        resistance_ohm=[[_R_S_OHM]],
        inductance_h=[[_L_S_H]],
    )


def _pgml_feeder(
    der_appliance,
    *,
    line_km: float = _LINE_KM_SHORT,
    p_load_w: float = _P_LOAD_W,
    q_load_var: float = _Q_LOAD_VAR,
) -> Grid:
    """2-bus single-phase feeder with a DER at bus 1."""
    return Grid(
        base_frequency_hz=_F0,
        nodes=[
            Node(id=0, u_rated_v=_U_LL_V, phases=(Phase.A,)),
            Node(id=1, u_rated_v=_U_LL_V, phases=(Phase.A,)),
        ],
        branches=[_pgml_line(line_km)],
        appliances=[
            _pgml_slack(),
            Load(
                id=101,
                node=1,
                phases=(Phase.A,),
                p_nom_w=p_load_w,
                q_nom_var=q_load_var,
            ),
            der_appliance,
        ],
    )


def _pgml_solve(grid: Grid) -> dict[int, complex]:
    """Return ``{node_id: complex_voltage_V}`` from ``solve_power_flow``."""
    res = solve_power_flow(grid, slack="ideal", method="newton", dtype=torch.complex128)
    assert res.converged, (
        f"pgml solve_power_flow did not converge (residual={float(res.residual):.3e})"
    )
    return {
        node.id: res.v[res.index.row(node.id, Phase.A)].item() for node in grid.nodes
    }


def _vm_pu(v: complex, u_rated_v: float = _U_LL_V) -> float:
    """Voltage magnitude in per unit."""
    return abs(v) / u_rated_v


# ---------------------------------------------------------------------------
# pandapower net builders
# ---------------------------------------------------------------------------


def _pp_base_net(
    line_km: float = _LINE_KM_SHORT,
    p_load_mw: float = _P_LOAD_W / 1e6,
    q_load_mvar: float = _Q_LOAD_VAR / 1e6,
) -> tuple[pp.pandapowerNet, int, int]:
    """Return ``(net, slack_bus_idx, load_bus_idx)`` for the shared 2-bus feeder."""
    net = pp.create_empty_network(f_hz=_F0)
    b0 = pp.create_bus(net, vn_kv=_U_LL_KV)
    b1 = pp.create_bus(net, vn_kv=_U_LL_KV)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=line_km,
        r_ohm_per_km=_R_OHM_PER_KM,
        x_ohm_per_km=_X_OHM_PER_KM,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, bus=b1, p_mw=p_load_mw, q_mvar=q_load_mvar)
    return net, b0, b1


# ---------------------------------------------------------------------------
# Test 1: Constant power factor
# ---------------------------------------------------------------------------


class TestConstantPowerFactor:
    """Constant-PF DER: fixed Q = ±|P| tan(acos(pf)) vs pandapower sgen.

    pandapower: ``sgen.q_mvar`` set explicitly.  Positive q_mvar = inject Q
    (overexcited); negative = absorb (underexcited).
    pgml: ``Generator(control=ConstantPowerFactorControl(...))``.
    Both tools solve a constant-power (P,Q) injection — the result is exact.
    """

    ATOL_VM_PU: float = 1e-6  # V/V_rated; tight because the system is linear in Q
    PF: float = 0.95

    def _pp_run(self, overexcited: bool) -> dict[int, float]:
        """pandapower constant-PF: set q_mvar explicitly and runpp."""
        net, b0, b1 = _pp_base_net()
        q_sign = +1.0 if overexcited else -1.0
        q_mvar = q_sign * _P_GEN_W_SHORT * math.tan(math.acos(self.PF)) / 1e6
        pp.create_sgen(net, bus=b1, p_mw=_P_GEN_W_SHORT / 1e6, q_mvar=q_mvar)
        pp.runpp(net, numba=False)
        assert net.converged, "pandapower runpp did not converge"
        return {b0: net.res_bus.vm_pu[b0], b1: net.res_bus.vm_pu[b1]}

    def _pgml_run(self, overexcited: bool) -> dict[int, complex]:
        gen = Generator(
            id=102,
            node=1,
            phases=(Phase.A,),
            p_nom_w=_P_GEN_W_SHORT,
            q_nom_var=0.0,
            control=ConstantPowerFactorControl(
                power_factor=self.PF,
                overexcited=overexcited,
                s_rated_va=_S_RATED_VA_SHORT,
            ),
        )
        return _pgml_solve(_pgml_feeder(gen))

    def test_overexcited_voltage_matches(self) -> None:
        """Overexcited PF=0.95: pgml and pandapower voltage magnitudes agree."""
        pp_v = self._pp_run(overexcited=True)
        pgml_v = self._pgml_run(overexcited=True)

        for bus_idx, node_id in [(0, 0), (1, 1)]:
            vm_pgml = _vm_pu(pgml_v[node_id])
            vm_pp = pp_v[bus_idx]
            err = abs(vm_pgml - vm_pp)
            assert err < self.ATOL_VM_PU, (
                f"ConstantPF overexcited bus {bus_idx}: |V| mismatch "
                f"pgml={vm_pgml:.8f} pp={vm_pp:.8f} err={err:.2e} pu"
            )

    def test_underexcited_voltage_matches(self) -> None:
        """Underexcited PF=0.95: pgml and pandapower voltage magnitudes agree."""
        pp_v = self._pp_run(overexcited=False)
        pgml_v = self._pgml_run(overexcited=False)

        for bus_idx, node_id in [(0, 0), (1, 1)]:
            vm_pgml = _vm_pu(pgml_v[node_id])
            vm_pp = pp_v[bus_idx]
            err = abs(vm_pgml - vm_pp)
            assert err < self.ATOL_VM_PU, (
                f"ConstantPF underexcited bus {bus_idx}: |V| mismatch "
                f"pgml={vm_pgml:.8f} pp={vm_pp:.8f} err={err:.2e} pu"
            )

    def test_overexcited_raises_voltage(self) -> None:
        """Overexcited DER raises bus-1 voltage above unity (Q injected)."""
        pgml_v = self._pgml_run(overexcited=True)
        # With net P injection + Q injection the voltage at bus 1 should exceed
        # the slack bus voltage.
        assert _vm_pu(pgml_v[1]) > _vm_pu(pgml_v[0]), (
            f"ConstantPF overexcited: expected V_bus1 > V_bus0, "
            f"got {_vm_pu(pgml_v[1]):.6f} pu vs {_vm_pu(pgml_v[0]):.6f} pu"
        )

    def test_underexcited_lowers_voltage_vs_overexcited(self) -> None:
        """Underexcited Q absorption lowers bus-1 voltage compared to overexcited."""
        v_over = self._pgml_run(overexcited=True)
        v_under = self._pgml_run(overexcited=False)
        assert _vm_pu(v_under[1]) < _vm_pu(v_over[1]), (
            f"ConstantPF: expected V_under < V_over at bus 1, "
            f"got V_under={_vm_pu(v_under[1]):.6f} V_over={_vm_pu(v_over[1]):.6f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Volt-VAr Q(V) control
# ---------------------------------------------------------------------------


class TestVoltVar:
    """Volt-VAr Q(V) DER: reactive power tracks bus voltage via a Q(V) curve.

    pandapower: ``pp.control.Characteristic`` (y-axis in MVAR) +
    ``pp.control.CharacteristicControl`` mapping ``res_bus.vm_pu`` to
    ``sgen.q_mvar``.  The outer control loop uses ``tol=1e-9`` to drive the
    controller residual below the Newton residual so that both tools converge
    to the same self-consistent fixed point.

    pgml: ``VoltVarControl`` with ``q_reference=QReference.RATED``; the
    ``y_values`` are fractions of ``s_rated_va`` (not MVAR directly).

    Curve design
    ------------
    x = V_pu (relative to 400 V L-L rated); y = Q/Qref fraction::

        V_pu = 0.90 -> y = +0.30 (inject 30% of rated → capacitive)
        V_pu = 1.00 -> y =  0.00 (no reactive power)
        V_pu = 1.04 -> y = -0.20 (absorb 20% of rated → inductive)
        V_pu = 1.10 -> y = -0.30 (absorb 30% of rated)

    With 10 kW generation and a 1 km line the uncurtailed V rises to ~1.013 pu,
    which places the operating point in the absorbing region of the curve.  The
    self-consistent Q is ~−786 var and the converged V is ~1.013 pu.
    """

    ATOL_VM_PU: float = (
        1e-6  # pu; tight because both tools converge to the same fixed point
    )
    ATOL_Q_VAR: float = 1.0  # var; margin for the piecewise-linear curve residual

    # Q(V) curve: x = V_pu, y = Q/S_rated fraction
    CURVE_X: list[float] = [0.90, 1.00, 1.04, 1.10]
    CURVE_Y_FRAC: list[float] = [0.30, 0.00, -0.20, -0.30]

    def _pp_run(self) -> tuple[dict[int, float], float]:
        """pandapower Volt-VAr via CharacteristicControl; returns (vm_pu map, q_sgen var)."""
        net, b0, b1 = _pp_base_net(
            line_km=_LINE_KM_LONG,
            p_load_mw=_P_LOAD_W_VVC / 1e6,
            q_load_mvar=_Q_LOAD_VAR_VVC / 1e6,
        )
        sgen_idx = pp.create_sgen(net, bus=b1, p_mw=_P_GEN_W_LONG / 1e6, q_mvar=0.0)

        # Characteristic: y-values in MVAR (pandapower unit)
        y_q_mvar = [y * _S_RATED_VA_LONG / 1e6 for y in self.CURVE_Y_FRAC]
        char = ppctrl.Characteristic(net, x_values=self.CURVE_X, y_values=y_q_mvar)
        ppctrl.CharacteristicControl(
            net,
            output_element="sgen",
            output_variable="q_mvar",
            output_element_index=sgen_idx,
            input_element="res_bus",
            input_variable="vm_pu",
            input_element_index=b1,
            characteristic_index=char.index,
            tol=1e-9,  # tight outer-loop tolerance to match Newton residual
        )
        ppctrl.run_control(net, numba=False, max_iter=1000)
        assert net.converged, "pandapower run_control did not converge"

        q_var = float(net.sgen.q_mvar[sgen_idx]) * 1e6
        return {b0: net.res_bus.vm_pu[b0], b1: net.res_bus.vm_pu[b1]}, q_var

    def _pgml_run(self) -> tuple[dict[int, complex], float]:
        """pgml Volt-VAr; returns (voltage map, Q_gen_var at converged V)."""
        gen = Generator(
            id=102,
            node=1,
            phases=(Phase.A,),
            p_nom_w=_P_GEN_W_LONG,
            q_nom_var=0.0,
            control=VoltVarControl(
                s_rated_va=_S_RATED_VA_LONG,
                q_reference=QReference.RATED,
                characteristic=Characteristic(
                    x_values=self.CURVE_X,
                    y_values=self.CURVE_Y_FRAC,
                ),
            ),
        )
        grid = _pgml_feeder(
            gen,
            line_km=_LINE_KM_LONG,
            p_load_w=_P_LOAD_W_VVC,
            q_load_var=_Q_LOAD_VAR_VVC,
        )
        v_map = _pgml_solve(grid)
        # Derive Q from the curve at the converged voltage
        import numpy as np

        v_pu = _vm_pu(v_map[1])
        q_frac = float(np.interp(v_pu, self.CURVE_X, self.CURVE_Y_FRAC))
        q_var = q_frac * _S_RATED_VA_LONG
        return v_map, q_var

    def test_voltage_matches(self) -> None:
        """VoltVar: converged bus voltages agree within 1e-6 pu."""
        pp_v, _ = self._pp_run()
        pgml_v, _ = self._pgml_run()

        for bus_idx, node_id in [(0, 0), (1, 1)]:
            vm_pgml = _vm_pu(pgml_v[node_id])
            vm_pp = pp_v[bus_idx]
            err = abs(vm_pgml - vm_pp)
            assert err < self.ATOL_VM_PU, (
                f"VoltVar bus {bus_idx}: |V| mismatch "
                f"pgml={vm_pgml:.8f} pp={vm_pp:.8f} err={err:.2e} pu"
            )

    def test_reactive_power_matches(self) -> None:
        """VoltVar: converged reactive injection (curve output) agrees within 1 var."""
        _, q_pp_var = self._pp_run()
        _, q_pgml_var = self._pgml_run()
        err = abs(q_pgml_var - q_pp_var)
        assert err < self.ATOL_Q_VAR, (
            f"VoltVar: Q mismatch pgml={q_pgml_var:.4f} var "
            f"pp={q_pp_var:.4f} var err={err:.2f} var"
        )

    def test_q_absorbs_when_high_voltage(self) -> None:
        """VoltVar: Q is negative (absorbing) when V > 1.0 pu (curve absorbing region)."""
        _, q_pgml_var = self._pgml_run()
        assert q_pgml_var < 0.0, (
            f"VoltVar: expected Q absorption (Q < 0) at high V, got Q={q_pgml_var:.2f} var"
        )


# ---------------------------------------------------------------------------
# Test 3: Storage snapshot (discharge vs charge)
# ---------------------------------------------------------------------------


class TestStorageSnapshot:
    """Storage (P,Q) injection vs pandapower storage element.

    Sign convention reminder
    ------------------------
    - pandapower ``storage.p_mw > 0`` = CHARGING (draw from grid, load convention).
    - pgml ``Storage.p_nom_w > 0`` = DISCHARGING (inject into grid, generator convention).
    - Mapping: ``pgml.p_nom_w = −pandapower.storage.p_mw``.

    pandapower ``storage`` is modelled as a signed PQ element in ``pp.runpp``.
    At ``p_mw < 0`` (discharging) pandapower injects power at the bus, exactly
    like pgml ``Storage(p_nom_w > 0)``.
    """

    ATOL_VM_PU: float = 1e-6  # pu; tight because both are fixed (P,Q) injections

    def _pp_run(self, *, pgml_p_nom_w: float) -> dict[int, float]:
        """pandapower storage: ``p_mw = −pgml_p_nom_w`` (sign flip)."""
        net, b0, b1 = _pp_base_net()
        # pandapower p_mw is OPPOSITE sign of pgml p_nom_w
        pp.create_storage(
            net, bus=b1, p_mw=-pgml_p_nom_w / 1e6, max_e_mwh=1.0, q_mvar=0.0
        )
        pp.runpp(net, numba=False)
        assert net.converged, "pandapower runpp did not converge"
        return {b0: net.res_bus.vm_pu[b0], b1: net.res_bus.vm_pu[b1]}

    def _pgml_run(self, *, p_nom_w: float) -> dict[int, complex]:
        storage = Storage(
            id=102, node=1, phases=(Phase.A,), p_nom_w=p_nom_w, q_nom_var=0.0
        )
        return _pgml_solve(_pgml_feeder(storage))

    def test_discharge_voltage_matches(self) -> None:
        """Discharging storage: pgml p_nom_w>0 matches pandapower p_mw<0 voltages."""
        pp_v = self._pp_run(pgml_p_nom_w=_P_STORAGE_W)
        pgml_v = self._pgml_run(p_nom_w=_P_STORAGE_W)

        for bus_idx, node_id in [(0, 0), (1, 1)]:
            vm_pgml = _vm_pu(pgml_v[node_id])
            vm_pp = pp_v[bus_idx]
            err = abs(vm_pgml - vm_pp)
            assert err < self.ATOL_VM_PU, (
                f"Storage discharge bus {bus_idx}: |V| mismatch "
                f"pgml={vm_pgml:.8f} pp={vm_pp:.8f} err={err:.2e} pu"
            )

    def test_charge_voltage_matches(self) -> None:
        """Charging storage: pgml p_nom_w<0 matches pandapower p_mw>0 voltages."""
        pp_v = self._pp_run(pgml_p_nom_w=-_P_STORAGE_W)
        pgml_v = self._pgml_run(p_nom_w=-_P_STORAGE_W)

        for bus_idx, node_id in [(0, 0), (1, 1)]:
            vm_pgml = _vm_pu(pgml_v[node_id])
            vm_pp = pp_v[bus_idx]
            err = abs(vm_pgml - vm_pp)
            assert err < self.ATOL_VM_PU, (
                f"Storage charge bus {bus_idx}: |V| mismatch "
                f"pgml={vm_pgml:.8f} pp={vm_pp:.8f} err={err:.2e} pu"
            )

    def test_discharge_raises_voltage_vs_load_only(self) -> None:
        """Discharging storage raises bus-1 voltage vs load-only baseline."""
        # Load-only reference (no storage)
        grid_ref = Grid(
            base_frequency_hz=_F0,
            nodes=[
                Node(id=0, u_rated_v=_U_LL_V, phases=(Phase.A,)),
                Node(id=1, u_rated_v=_U_LL_V, phases=(Phase.A,)),
            ],
            branches=[_pgml_line(_LINE_KM_SHORT)],
            appliances=[
                _pgml_slack(),
                Load(
                    id=101,
                    node=1,
                    phases=(Phase.A,),
                    p_nom_w=_P_LOAD_W,
                    q_nom_var=_Q_LOAD_VAR,
                ),
            ],
        )
        v_ref = _pgml_solve(grid_ref)
        v_discharge = self._pgml_run(p_nom_w=_P_STORAGE_W)

        assert _vm_pu(v_discharge[1]) > _vm_pu(v_ref[1]), (
            f"Storage discharge: expected V_bus1 to rise above load-only baseline, "
            f"got V_discharge={_vm_pu(v_discharge[1]):.6f} pu "
            f"<= V_ref={_vm_pu(v_ref[1]):.6f} pu"
        )

    def test_charge_lowers_voltage_vs_load_only(self) -> None:
        """Charging storage (drawing extra power) lowers bus-1 voltage vs load-only."""
        grid_ref = Grid(
            base_frequency_hz=_F0,
            nodes=[
                Node(id=0, u_rated_v=_U_LL_V, phases=(Phase.A,)),
                Node(id=1, u_rated_v=_U_LL_V, phases=(Phase.A,)),
            ],
            branches=[_pgml_line(_LINE_KM_SHORT)],
            appliances=[
                _pgml_slack(),
                Load(
                    id=101,
                    node=1,
                    phases=(Phase.A,),
                    p_nom_w=_P_LOAD_W,
                    q_nom_var=_Q_LOAD_VAR,
                ),
            ],
        )
        v_ref = _pgml_solve(grid_ref)
        v_charge = self._pgml_run(p_nom_w=-_P_STORAGE_W)

        assert _vm_pu(v_charge[1]) < _vm_pu(v_ref[1]), (
            f"Storage charge: expected V_bus1 to fall below load-only baseline, "
            f"got V_charge={_vm_pu(v_charge[1]):.6f} pu "
            f">= V_ref={_vm_pu(v_ref[1]):.6f} pu"
        )
