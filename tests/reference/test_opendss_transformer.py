"""Oracle test: OpenDSS ``Transformer`` -> pgml ``Transformer`` conversion.

Validates the DSS -> pgml two-winding transformer conversion
(``pgml.convert.opendss.to_grid``) against a LIVE OpenDSS ``Solve`` on the same
circuit: MV source -> transformer -> LV load, for both a phase-shifting
(Dyn11) and a non-shifting (Yy0) vector group.

Test strategy
-------------
1. Build a small 2-bus DSS circuit programmatically (the same pattern as
   ``tests/convert/test_opendss_phase_mode.py``): a near-ideal 3-phase Vsource
   at ``src``, a two-winding ``Transformer`` to ``lv``, and a balanced
   constant-power (``model=1``) load at ``lv``.
2. ``Solve`` the DSS circuit (its own nonlinear AC power flow).
3. Convert with ``to_grid(dss, phase_mode=THREE_PHASE)`` and solve with
   ``solve_power_flow(grid, slack="ideal")`` (pgml's nonlinear const-power
   solve — the DSS Vsource carries a near-zero Thevenin impedance so
   ``slack="ideal"`` and DSS's own Vsource treatment coincide).
4. Compare every bus's per-phase voltage magnitude (pu, on OpenDSS's own
   ``Bus.kVBase()``) and angle against DSS's ``Bus.puVmagAngle()``.

Both transformers here carry NO magnetizing branch (``%noloadloss=%imag=0``)
so the comparison isolates the leakage-referral (``%R``/``XHL`` -> LV-referred
R/L) and vector-group (connections + ``LeadLag`` -> clock/shift) conversion.
A separate class (``TestMagnetizingBranchConversion``) validates the
``%noloadloss``/``%imag`` -> ``magnetizing_conductance_s``/
``magnetizing_inductance_h`` field conversion directly against the closed-form
formula; live-voltage parity is not asserted for it because OpenDSS's own
internal transformer model places the magnetizing branch inside its leakage
"T" (splitting current between the two windings' half-impedances) rather than
as a pure shunt at the external HV terminal — pgml's ``assembly._transformer``
stamps it as a simple HV-terminal shunt (a documented simplification, see
``docs/pgml/modeling/transformer.md``), which reproduces the same ORDER of
voltage change but not an exact match. That residual is on the order of
1e-3 pu for a typical (~0.5 %) magnetizing current — three orders of
magnitude looser than the leakage-only tolerance below, and unrelated to the
leakage/vector-group formulas this file's tight tolerances are meant to catch.

Tolerance targets (leakage + vector-group path, no magnetizing branch)
------------------------------------------------------------------------
- Voltage magnitude: atol = 1e-6 pu (empirically ~1e-7 pu).
- Voltage angle:      atol = 1e-4 deg (empirically ~1e-5 deg).
"""

from __future__ import annotations

import math

import pytest
import torch

import opendssdirect as dss  # noqa: E402

from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import Phase, Transformer, WindingConnection  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

_F0 = 60.0
_KV_HV = 20.0  # kV, line-to-line
_KV_LV = 0.4  # kV, line-to-line
_KVA = 1000.0
_XHL_PCT = 6.0
_LOADLOSS_PCT = 1.5  # %R per winding = 0.75 each
_LOAD_KW = 300.0
_LOAD_KVAR = 100.0


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


def _dss_clear() -> None:
    """Reset the (process-global) OpenDSS engine for a fresh circuit.

    ``Clear`` does NOT reset every engine-wide setting -- notably
    ``DefaultBaseFrequency`` persists across ``Clear`` (opendssdirect wraps a
    single global engine instance shared by every test module in the
    process). A prior test module that changes it (e.g. to 50 Hz) would
    otherwise silently leak into this module's ``frequency=60`` circuits, since
    ``New Circuit ... frequency=60`` does not by itself override a
    already-changed default. Reasserting it here makes this module's tests
    independent of pytest collection order.
    """
    dss.Text.Command("Clear")
    dss.Text.Command(f"set DefaultBaseFrequency={int(_F0)}")


def _build_transformer_circuit(
    *,
    conn_hv: str,
    conn_lv: str,
    leadlag: str = "Lag",
    noloadloss_pct: float = 0.0,
    imag_pct: float = 0.0,
) -> None:
    """Build a 2-bus DSS circuit: Vsource(src) -> Transformer(t1) -> Load(lv)."""
    _dss_clear()
    dss.Text.Command(
        f"New Circuit.trafo_test phases=3 basekv={_KV_HV} bus1=src.1.2.3 pu=1.0 "
        f"angle=0.0 frequency={_F0} r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6"
    )
    dss.Text.Command(
        f"New Transformer.t1 windings=2 phases=3 xhl={_XHL_PCT} "
        f"%loadloss={_LOADLOSS_PCT} %noloadloss={noloadloss_pct} %imag={imag_pct} "
        f"leadlag={leadlag}"
    )
    dss.Text.Command(f"~ wdg=1 bus=src.1.2.3 conn={conn_hv} kV={_KV_HV} kVA={_KVA}")
    dss.Text.Command(f"~ wdg=2 bus=lv.1.2.3.0 conn={conn_lv} kV={_KV_LV} kVA={_KVA}")
    dss.Text.Command(
        f"New Load.load1 bus1=lv.1.2.3 kV={_KV_LV} kW={_LOAD_KW} kvar={_LOAD_KVAR} "
        "conn=wye phases=3 model=1"
    )
    dss.Text.Command(f"Set voltagebases=[{_KV_HV}, {_KV_LV}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "DSS transformer test circuit did not converge"


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b in degrees, wrapped to (-180, 180]."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _solve_pgml() -> tuple:
    """Convert the currently-loaded DSS circuit and solve with pgml. Returns (grid, id_map, result)."""
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
    )
    assert result.converged, (
        f"solve_power_flow did not converge (residual={float(result.residual):.3e})"
    )
    return grid, id_map, result


def _compare_all_buses(
    id_map: dict, result, *, atol_vm_pu: float, atol_va_deg: float
) -> None:
    """Assert every converted bus's per-phase |V|/angle matches DSS's own solve."""
    for bus_name, node_id in id_map["bus"].items():
        dss.Circuit.SetActiveBus(bus_name)
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        for k, phase in enumerate((Phase.A, Phase.B, Phase.C)):
            row = result.index.row(node_id, phase)
            v = result.v.reshape(-1)[row].item()
            vm_pu_ours = abs(v) / (kvbase_ln_kv * 1_000.0)
            va_deg_ours = math.degrees(math.atan2(v.imag, v.real))

            vm_pu_ref = dss_va[2 * k]
            va_deg_ref = dss_va[2 * k + 1]

            vm_err = abs(vm_pu_ours - vm_pu_ref)
            va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_ref))

            assert vm_err < atol_vm_pu, (
                f"bus {bus_name} phase {phase}: |V| mismatch "
                f"ours={vm_pu_ours:.8f} dss={vm_pu_ref:.8f} err={vm_err:.2e} pu"
            )
            assert va_err < atol_va_deg, (
                f"bus {bus_name} phase {phase}: angle mismatch "
                f"ours={va_deg_ours:.6f} dss={va_deg_ref:.6f} err={va_err:.2e} deg"
            )


# ---------------------------------------------------------------------------
# Dyn11 (phase-shifting) oracle
# ---------------------------------------------------------------------------


class TestDyn11Oracle:
    """MV source -> Dyn11 20/0.4 kV transformer -> LV load, vs live OpenDSS."""

    ATOL_VM_PU = 1e-6  # achieved ~1e-7 pu
    ATOL_VA_DEG = 1e-4  # achieved ~1e-5 deg

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        # leadlag=Lead -> Dyn11 (verified empirically: LV leads HV by ~30 deg).
        _build_transformer_circuit(conn_hv="delta", conn_lv="wye", leadlag="Lead")
        request.cls._grid, request.cls._id_map, request.cls._result = _solve_pgml()

    def test_node_voltages_match_opendss(self) -> None:
        _compare_all_buses(
            self._id_map,
            self._result,
            atol_vm_pu=self.ATOL_VM_PU,
            atol_va_deg=self.ATOL_VA_DEG,
        )

    def test_transformer_field_conversion(self) -> None:
        """Connections, tap and LV-referred leakage R/L land as expected."""
        trafo = next(b for b in self._grid.branches if isinstance(b, Transformer))
        assert trafo.from_connection == WindingConnection.DELTA
        assert trafo.to_connection == WindingConnection.WYE_GROUNDED
        assert trafo.tap.ratio_magnitude == pytest.approx(1.0)
        # Dyn11 -> clock 11 -> shift_deg = 330 (LV leads HV by 30 deg).
        assert trafo.tap.shift_deg == pytest.approx(330.0)
        assert trafo.u_rated_from_v == pytest.approx(_KV_HV * 1_000.0)
        assert trafo.u_rated_to_v == pytest.approx(_KV_LV * 1_000.0)
        assert trafo.s_rated_va == pytest.approx(_KVA * 1_000.0)

        # LV-referred leakage: Z_base_LV = kV_lv^2*1000/kVA; %R total = loadloss%.
        z_base_lv = (_KV_LV**2 * 1_000.0) / _KVA
        expected_r = _LOADLOSS_PCT / 100.0 * z_base_lv
        expected_x = _XHL_PCT / 100.0 * z_base_lv
        expected_l = expected_x / (2.0 * math.pi * _F0)
        assert trafo.series_resistance_ohm == pytest.approx(expected_r, rel=1e-9)
        assert trafo.series_inductance_h == pytest.approx(expected_l, rel=1e-9)
        # No magnetizing branch requested for this circuit.
        assert trafo.magnetizing_conductance_s == pytest.approx(0.0)
        assert trafo.magnetizing_inductance_h is None

    def test_vector_group_shift_is_30_degrees(self) -> None:
        """The converted grid's LV bus phase-A voltage leads HV by ~30 deg.

        Not exactly 30 deg because of the small angle drop introduced by the
        loaded LV bus (matches DSS's own solve, verified above to ~1e-5 deg).
        """
        node_hv = self._id_map["bus"]["src"]
        node_lv = self._id_map["bus"]["lv"]
        v_hv = self._result.v.reshape(-1)[
            self._result.index.row(node_hv, Phase.A)
        ].item()
        v_lv = self._result.v.reshape(-1)[
            self._result.index.row(node_lv, Phase.A)
        ].item()
        ang_hv = math.degrees(math.atan2(v_hv.imag, v_hv.real))
        ang_lv = math.degrees(math.atan2(v_lv.imag, v_lv.real))
        shift = _angle_diff_deg(ang_lv, ang_hv)
        assert shift == pytest.approx(30.0, abs=1.0), (
            f"Dyn11: expected LV to lead HV by ~30 deg, got {shift:.4f} deg"
        )


# ---------------------------------------------------------------------------
# Yy0 (non-shifting) oracle
# ---------------------------------------------------------------------------


class TestYy0Oracle:
    """MV source -> Yy0 20/0.4 kV transformer -> LV load, vs live OpenDSS."""

    ATOL_VM_PU = 1e-6  # achieved ~1e-7 pu
    ATOL_VA_DEG = 1e-4  # achieved ~1e-5 deg

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        # Both windings wye -> no inherent phase shift regardless of leadlag.
        _build_transformer_circuit(conn_hv="wye", conn_lv="wye")
        request.cls._grid, request.cls._id_map, request.cls._result = _solve_pgml()

    def test_node_voltages_match_opendss(self) -> None:
        _compare_all_buses(
            self._id_map,
            self._result,
            atol_vm_pu=self.ATOL_VM_PU,
            atol_va_deg=self.ATOL_VA_DEG,
        )

    def test_transformer_field_conversion(self) -> None:
        trafo = next(b for b in self._grid.branches if isinstance(b, Transformer))
        assert trafo.from_connection == WindingConnection.WYE_GROUNDED
        assert trafo.to_connection == WindingConnection.WYE_GROUNDED
        assert trafo.tap.shift_deg == pytest.approx(0.0)

    def test_vector_group_shift_is_zero(self) -> None:
        """The converted grid's LV bus phase-A voltage is in phase with HV."""
        node_hv = self._id_map["bus"]["src"]
        node_lv = self._id_map["bus"]["lv"]
        v_hv = self._result.v.reshape(-1)[
            self._result.index.row(node_hv, Phase.A)
        ].item()
        v_lv = self._result.v.reshape(-1)[
            self._result.index.row(node_lv, Phase.A)
        ].item()
        ang_hv = math.degrees(math.atan2(v_hv.imag, v_hv.real))
        ang_lv = math.degrees(math.atan2(v_lv.imag, v_lv.real))
        shift = _angle_diff_deg(ang_lv, ang_hv)
        assert shift == pytest.approx(0.0, abs=1.0), (
            f"Yy0: expected LV in phase with HV, got {shift:.4f} deg"
        )


# ---------------------------------------------------------------------------
# SINGLE_PHASE_EQUIV mode: the positive-sequence scalar-tap path
# ---------------------------------------------------------------------------


class TestSinglePhaseEquivTransformer:
    """The positive-sequence-equivalent (1-phase) transformer path also solves
    and matches DSS's phase-A voltage (the historical/default converter mode)."""

    ATOL_VM_PU = 1e-6
    ATOL_VA_DEG = 1e-4

    def test_matches_opendss_phase_a(self) -> None:
        _build_transformer_circuit(conn_hv="delta", conn_lv="wye", leadlag="Lead")
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        result = solve_power_flow(
            grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
        )
        assert result.converged

        node_lv = id_map["bus"]["lv"]
        row = result.index.row(node_lv, Phase.A)
        v = result.v.reshape(-1)[row].item()

        dss.Circuit.SetActiveBus("lv")
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        # SINGLE_PHASE_EQUIV stores u_rated_v as the recovered L-L nameplate
        # (kVBase*sqrt(3)*1000), matching Node.u_rated_v == _KV_LV*1000.
        vm_pu_ours = abs(v) / (_KV_LV * 1_000.0)
        va_deg_ours = math.degrees(math.atan2(v.imag, v.real))

        assert abs(vm_pu_ours - dss_va[0]) < self.ATOL_VM_PU
        assert abs(_angle_diff_deg(va_deg_ours, dss_va[1])) < self.ATOL_VA_DEG
        assert kvbase_ln_kv == pytest.approx(_KV_LV / math.sqrt(3.0))


# ---------------------------------------------------------------------------
# Magnetizing branch: field-level formula correctness (not a live-voltage
# comparison -- see the module docstring for why).
# ---------------------------------------------------------------------------


class TestMagnetizingBranchConversion:
    NOLOADLOSS_PCT = 0.2
    IMAG_PCT = 0.5

    def test_magnetizing_fields_match_closed_form(self) -> None:
        _build_transformer_circuit(
            conn_hv="delta",
            conn_lv="wye",
            leadlag="Lead",
            noloadloss_pct=self.NOLOADLOSS_PCT,
            imag_pct=self.IMAG_PCT,
        )
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        trafo = next(b for b in grid.branches if isinstance(b, Transformer))

        s_rated_va = _KVA * 1_000.0
        u_hv_v = _KV_HV * 1_000.0
        pfe_w = self.NOLOADLOSS_PCT / 100.0 * s_rated_va
        expected_g_m = pfe_w / (u_hv_v**2)

        i0_amp = self.IMAG_PCT / 100.0 * s_rated_va / u_hv_v
        s_nl = u_hv_v * i0_amp
        q_nl = math.sqrt(s_nl**2 - pfe_w**2)
        b_m = q_nl / (u_hv_v**2)
        two_pi_f0 = 2.0 * math.pi * _F0
        expected_l_m = 1.0 / (two_pi_f0 * b_m)

        assert trafo.magnetizing_conductance_s == pytest.approx(expected_g_m, rel=1e-9)
        assert trafo.magnetizing_inductance_h == pytest.approx(expected_l_m, rel=1e-9)

    def test_zero_noloadloss_and_imag_gives_no_magnetizing_branch(self) -> None:
        _build_transformer_circuit(conn_hv="delta", conn_lv="wye", leadlag="Lead")
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        trafo = next(b for b in grid.branches if isinstance(b, Transformer))
        assert trafo.magnetizing_conductance_s == pytest.approx(0.0)
        assert trafo.magnetizing_inductance_h is None


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------


class TestConversionScopeGuards:
    def test_three_winding_transformer_raises(self) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.trafo3w phases=3 basekv={_KV_HV} bus1=src.1.2.3 pu=1.0 "
            f"angle=0.0 frequency={_F0}"
        )
        dss.Text.Command(
            "New Transformer.t3w windings=3 phases=3 xhl=6 xht=7 xlt=8 %loadloss=1.5"
        )
        dss.Text.Command("~ wdg=1 bus=src.1.2.3 conn=delta kV=20 kVA=1000")
        dss.Text.Command("~ wdg=2 bus=lv1.1.2.3.0 conn=wye kV=0.4 kVA=500")
        dss.Text.Command("~ wdg=3 bus=lv2.1.2.3.0 conn=wye kV=0.4 kVA=500")
        dss.Text.Command("Set voltagebases=[20, 0.4]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        with pytest.raises(ConversionError, match="windings"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)

    def test_ungrounded_wye_neutral_raises(self) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.trafo_ungnd phases=3 basekv={_KV_HV} bus1=src.1.2.3 pu=1.0 "
            f"angle=0.0 frequency={_F0}"
        )
        dss.Text.Command("New Transformer.tu windings=2 phases=3 xhl=6 %loadloss=1.5")
        dss.Text.Command("~ wdg=1 bus=src.1.2.3 conn=wye kV=20 kVA=1000")
        # Explicit non-zero (n_phases+1)-th conductor -> floating/impedance
        # grounding, not solidly grounded -- out of scope.
        dss.Text.Command("~ wdg=2 bus=lv.1.2.3.4 conn=wye kV=0.4 kVA=1000")
        dss.Text.Command("New Load.load1 bus1=lv.1.2.3 kV=0.4 kW=10 kvar=3 model=1")
        dss.Text.Command("Set voltagebases=[20, 0.4]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        with pytest.raises(ConversionError, match="ungrounded"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)

    def test_differing_winding_kva_raises(self) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.trafo_kva phases=3 basekv={_KV_HV} bus1=src.1.2.3 pu=1.0 "
            f"angle=0.0 frequency={_F0}"
        )
        dss.Text.Command(
            "New Transformer.tk windings=2 phases=3 xhl=6 %loadloss=1.5 "
            "buses=[src.1.2.3, lv.1.2.3.0] conns=[delta, wye] "
            "kvs=[20, 0.4] kvas=[1000, 500]"
        )
        dss.Text.Command("New Load.load1 bus1=lv.1.2.3 kV=0.4 kW=10 kvar=3 model=1")
        dss.Text.Command("Set voltagebases=[20, 0.4]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        with pytest.raises(ConversionError, match="kVA"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
