"""Oracle test: OpenDSS ``Transformer`` -> pgml ``Transformer`` conversion.

Validates the DSS -> pgml two-winding transformer conversion
(``pgml.convert.opendss.to_grid``) against a LIVE OpenDSS ``Solve`` on the same
circuit: MV source -> transformer -> LV load, for a phase-shifting (Dyn11), a
non-shifting (Yy0), and two ROTATED-bus-connection vector groups (Dyn5,
YNd5) that OpenDSS can only express by cyclically rotating one winding's bus
conductor order (it has no explicit clock parameter beyond the binary
``LeadLag`` Dy/Yd toggle).

Test strategy
-------------
1. Build a small 2-bus DSS circuit programmatically (the same pattern as
   ``tests/convert/test_opendss_phase_mode.py``): a near-ideal 3-phase Vsource
   at ``src``, a two-winding ``Transformer`` to ``lv``, and a balanced
   load at ``lv``.
2. ``Solve`` the DSS circuit (its own nonlinear AC power flow).
3. Convert with ``to_grid(dss, phase_mode=THREE_PHASE)`` and solve with pgml
   (``solve_power_flow(grid, slack="ideal")`` — the DSS Vsource carries a
   near-zero Thevenin impedance so ``slack="ideal"`` and DSS's own Vsource
   treatment coincide — EXCEPT ``TestYNd5RotatedOracle``, which uses a
   different solve path for a documented reason; see its docstring).
4. Compare every bus's per-phase voltage magnitude (pu, on OpenDSS's own
   ``Bus.kVBase()``) and angle against DSS's ``Bus.puVmagAngle()``.

Both transformers here carry NO magnetizing branch (``%noloadloss=%imag=0``)
so the comparison isolates the leakage-referral (``%R``/``XHL`` -> LV-referred
R/L) and vector-group (connections + ``LeadLag`` + bus rotation -> clock/shift)
conversion. A separate class (``TestMagnetizingBranchConversion``) validates
the ``%noloadloss``/``%imag`` -> ``magnetizing_conductance_s``/
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
- Voltage magnitude: atol = 1e-6 pu (empirically ~1e-7 pu, Dyn5 ~1e-8 pu).
- Voltage angle:      atol = 1e-4 deg (empirically ~1e-5 deg, Dyn5 ~1e-5 deg).
"""

from __future__ import annotations

import math

import pytest
import torch

import opendssdirect as dss  # noqa: E402

from pgml.assembly import assemble_ybus, build_injections, node_phase_index  # noqa: E402
from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.convert.opendss.converter import (  # noqa: E402
    _cyclic_rotation_steps,
    _parse_transformer_winding_bus,
)
from pgml.errors import ConversionError  # noqa: E402
from pgml.evaluation.oracles.opendss_oracle import (  # noqa: E402
    _build_circuit_with_real_transformer,
)
from pgml.schemas.grid_schema import (  # noqa: E402
    ComplexTap,
    Grid,
    Node,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)
from pgml.solver import solve_harmonic, solve_power_flow  # noqa: E402

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
    bus_hv: str = "src.1.2.3",
    bus_lv: str = "lv.1.2.3.0",
    load_conn: str = "wye",
    load_model: int = 1,
) -> None:
    """Build a 2-bus DSS circuit: Vsource(src) -> Transformer(t1) -> Load(lv).

    ``bus_hv``/``bus_lv`` override the TRANSFORMER's winding bus-conductor
    order (the Vsource's own ``bus1=src.1.2.3`` stays canonical -- only the
    transformer's connection to that bus can rotate). A cyclically rotated
    string (e.g. ``lv.3.1.2.0``, rotation ``r=2``) exercises the clock-fold
    path (``_cyclic_rotation_steps`` in ``pgml.convert.opendss.converter``).

    ``load_conn``/``load_model`` select the LV load's connection and DSS load
    model (``1`` constant power, ``2`` constant impedance) -- see
    ``TestYNd5RotatedOracle`` for why the delta-LV vector group needs a WYE
    load and ``model=2``.
    """
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
    dss.Text.Command(f"~ wdg=1 bus={bus_hv} conn={conn_hv} kV={_KV_HV} kVA={_KVA}")
    dss.Text.Command(f"~ wdg=2 bus={bus_lv} conn={conn_lv} kV={_KV_LV} kVA={_KVA}")
    dss.Text.Command(
        f"New Load.load1 bus1=lv.1.2.3 kV={_KV_LV} kW={_LOAD_KW} kvar={_LOAD_KVAR} "
        f"conn={load_conn} phases=3 model={load_model}"
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


def _compare_all_buses_v(
    id_map: dict, index, v, *, atol_vm_pu: float, atol_va_deg: float
) -> None:
    """Assert every converted bus's per-phase |V|/angle matches DSS's own solve.

    ``index``/``v`` are a raw :class:`~pgml.assembly.index.NodePhaseIndex` and
    complex voltage vector (from either the nonlinear ``solve_power_flow``
    result or a direct linear ``solve_harmonic`` solve).
    """
    for bus_name, node_id in id_map["bus"].items():
        dss.Circuit.SetActiveBus(bus_name)
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        for k, phase in enumerate((Phase.A, Phase.B, Phase.C)):
            row = index.row(node_id, phase)
            raw = v[row]
            v_val = raw.item() if hasattr(raw, "item") else complex(raw)
            vm_pu_ours = abs(v_val) / (kvbase_ln_kv * 1_000.0)
            va_deg_ours = math.degrees(math.atan2(v_val.imag, v_val.real))

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


def _compare_all_buses(
    id_map: dict, result, *, atol_vm_pu: float, atol_va_deg: float
) -> None:
    """Assert every converted bus's per-phase |V|/angle matches DSS's own solve."""
    _compare_all_buses_v(
        id_map,
        result.index,
        result.v.reshape(-1),
        atol_vm_pu=atol_vm_pu,
        atol_va_deg=atol_va_deg,
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
# Dyn5 (rotated-bus phase-shifting) oracle
# ---------------------------------------------------------------------------


class TestDyn5RotatedOracle:
    """MV source -> Dyn5 transformer -> LV load, vs live OpenDSS.

    OpenDSS has no explicit clock parameter; Dyn5 is expressed as Dyn1
    (``conn_hv=delta``, ``conn_lv=wye``, ``leadlag=Lag``) with the LV winding's
    bus connection cyclically rotated by ``r=2`` (``bus_lv="lv.3.1.2.0"``):
    ``30 deg (Dyn1 base) + 120*2 deg (mod 360) = 150 deg = clock 5``, pinned
    against a live OpenDSS solve (see ``pgml.convert.opendss.converter``'s
    "Cyclic winding-bus rotation" CONTEXT.md section for the full derivation
    and sign convention). The LV winding stays grounded-wye, so this isolates
    the rotation fold from the delta-LV coil-referral factor (see
    ``TestYNd5RotatedOracle`` for that combination).
    """

    ATOL_VM_PU = 1e-6  # achieved ~1e-7 pu
    ATOL_VA_DEG = 1e-4  # achieved ~1e-5 deg

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_transformer_circuit(
            conn_hv="delta", conn_lv="wye", leadlag="Lag", bus_lv="lv.3.1.2.0"
        )
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
        assert trafo.from_connection == WindingConnection.DELTA
        assert trafo.to_connection == WindingConnection.WYE_GROUNDED
        # Phases are NORMALIZED to canonical (A, B, C) -- the physical rotation
        # is folded into the clock, never left in the row order.
        assert trafo.from_phases == (Phase.A, Phase.B, Phase.C)
        assert trafo.to_phases == (Phase.A, Phase.B, Phase.C)
        assert trafo.tap.shift_deg == pytest.approx(150.0)  # clock 5

    def test_vector_group_shift_is_150_degrees(self) -> None:
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
        lag = (ang_hv - ang_lv) % 360.0
        assert lag == pytest.approx(150.0, abs=1.0), (
            f"Dyn5: expected LV to lag HV by ~150 deg, got {lag:.4f} deg"
        )


# ---------------------------------------------------------------------------
# YNd5 (rotated-bus, delta-LV) oracle -- factor-3 coil-referral fix
# ---------------------------------------------------------------------------


class TestYNd5RotatedOracle:
    """MV source -> YNd5 transformer -> LV load, vs live OpenDSS.

    Same rotated-bus mechanism as Dyn5 (``conn_hv=wye``, ``conn_lv=delta``,
    ``leadlag=Lag``, LV rotated ``r=2`` via ``bus_lv="lv.3.1.2.0"``:
    ``30 + 120*2 (mod 360) = 150 deg = clock 5``), but with the LV winding
    DELTA instead of wye -- this ALSO exercises the delta-LV coil-referral fix
    (``series_resistance_ohm``/``series_inductance_h`` are 3x the standard
    line-to-line value `%R`/`XHL` recover, because pgml stores the leakage
    referred to the ACTUAL delta coil; see
    ``pgml.convert.opendss.converter``'s transformer section and
    ``docs/pgml/modeling/references/opendss/index.md``). Without the fix, the
    LV-side voltage magnitude is off by ~7e-3 pu (vs ~2e-8 pu with it) on this
    exact circuit.

    **Why this test uses a different solve path than the rest of this file.**
    A delta-only LV secondary with no other grounded element is an isolated
    island: OpenDSS's own delta winding blocks zero-sequence TRANSFER from the
    grounded-wye HV side (textbook delta physics -- see
    ``pgml.assembly._transformer``'s module docstring), so the LV bus's
    absolute (line-to-GROUND) voltage is not physically determined by the
    load flow at all -- only its line-to-line quantities are. pgml's nonlinear
    constant-power ``solve_power_flow`` has no mechanism to anchor that
    reference (verified: it silently returns an internally "residual-zero" but
    physically meaningless solution, with a magnitude that diverges instead of
    stabilizing as the load shrinks toward zero -- a genuine, orthogonal
    solver gap, NOT something this converter/oracle change fixes). Two
    substitutions route around it while keeping the comparison meaningful:

    - The LV load is WYE, not delta (``load_conn="wye"``). With no ``Phase.N``
      row on the LV node, pgml's connection-aware load incidence reduces WYE
      to a direct per-phase admittance BACK TO SYSTEM GROUND (see the
      ``pgml.assembly._incidence`` module docstring: "WYE, node has NO
      Phase.N ... M = I_n" -- "the diagonal const-Z/device stamp"), which
      anchors the missing reference. OpenDSS does the identical thing for a
      ``conn=wye`` load with no explicit 4th conductor. (This is the same
      device used to pin the clock sign in
      ``tests/reference/test_transformer_clock_matrix.py::test_solved_lv_angle_matches_clock``.)
    - The load uses DSS ``model=2`` (constant impedance) and pgml is solved
      with the LINEAR path (``assemble_ybus`` + ``solve_harmonic`` at the
      fundamental) instead of the nonlinear iteration -- this matches pgml's
      own linear const-Z load assembler exactly, so both sides solve the
      IDENTICAL linear problem (no nonlinear iteration on either side).
    """

    ATOL_VM_PU = 1e-6  # achieved ~2e-8 pu
    ATOL_VA_DEG = 1e-4  # achieved ~4e-7 deg

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_transformer_circuit(
            conn_hv="wye",
            conn_lv="delta",
            leadlag="Lag",
            bus_lv="lv.3.1.2.0",
            load_conn="wye",
            load_model=2,
        )
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        index = node_phase_index(grid)
        yb = assemble_ybus(grid, [_F0], dtype=torch.complex128)
        inj = build_injections(grid, [_F0], index, dtype=torch.complex128)
        v = solve_harmonic(yb.Y, inj)[0]
        request.cls._grid = grid
        request.cls._id_map = id_map
        request.cls._index = index
        request.cls._v = v

    def test_node_voltages_match_opendss(self) -> None:
        _compare_all_buses_v(
            self._id_map,
            self._index,
            self._v,
            atol_vm_pu=self.ATOL_VM_PU,
            atol_va_deg=self.ATOL_VA_DEG,
        )

    def test_transformer_field_conversion(self) -> None:
        trafo = next(b for b in self._grid.branches if isinstance(b, Transformer))
        assert trafo.from_connection == WindingConnection.WYE_GROUNDED
        assert trafo.to_connection == WindingConnection.DELTA
        assert trafo.from_phases == (Phase.A, Phase.B, Phase.C)
        assert trafo.to_phases == (Phase.A, Phase.B, Phase.C)
        assert trafo.tap.shift_deg == pytest.approx(150.0)  # clock 5

        # Delta-LV coil referral: R/L are 3x the standard line-to-line value
        # OpenDSS's %R/XHL recover on Z_base_LV -- the factor-3 fix.
        z_base_lv = (_KV_LV**2 * 1_000.0) / _KVA
        expected_r_ll = _LOADLOSS_PCT / 100.0 * z_base_lv
        expected_x_ll = _XHL_PCT / 100.0 * z_base_lv
        expected_l_ll = expected_x_ll / (2.0 * math.pi * _F0)
        assert trafo.series_resistance_ohm == pytest.approx(
            3.0 * expected_r_ll, rel=1e-9
        )
        assert trafo.series_inductance_h == pytest.approx(3.0 * expected_l_ll, rel=1e-9)

    def test_delta_lv_referral_bug_would_fail_tolerance(self) -> None:
        """Without the factor-3 fix, the LV voltage magnitude is off by ~7e-3 pu."""
        trafo = next(b for b in self._grid.branches if isinstance(b, Transformer))
        buggy_grid = self._grid.model_copy(deep=True)
        for b in buggy_grid.branches:
            if isinstance(b, Transformer):
                b.series_resistance_ohm = trafo.series_resistance_ohm / 3.0
                b.series_inductance_h = trafo.series_inductance_h / 3.0
        index = node_phase_index(buggy_grid)
        yb = assemble_ybus(buggy_grid, [_F0], dtype=torch.complex128)
        inj = build_injections(buggy_grid, [_F0], index, dtype=torch.complex128)
        v_buggy = solve_harmonic(yb.Y, inj)[0]

        node_lv = self._id_map["bus"]["lv"]
        dss.Circuit.SetActiveBus("lv")
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        max_err = 0.0
        for k, phase in enumerate((Phase.A, Phase.B, Phase.C)):
            row = index.row(node_lv, phase)
            vm_pu = abs(v_buggy[row].item()) / (kvbase_ln_kv * 1_000.0)
            max_err = max(max_err, abs(vm_pu - dss_va[2 * k]))
        assert max_err > 1e-3, (
            f"expected the pre-fix (no factor-3) leakage to miss by > 1e-3 pu, "
            f"got {max_err:.2e} pu"
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

    def test_non_cyclic_winding_permutation_raises(self) -> None:
        """A transposition (not a cyclic rotation) reverses the phase sequence and raises."""
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.trafo_swap phases=3 basekv={_KV_HV} bus1=src.1.2.3 pu=1.0 "
            f"angle=0.0 frequency={_F0} r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6"
        )
        dss.Text.Command(
            f"New Transformer.tswap windings=2 phases=3 xhl={_XHL_PCT} "
            f"%loadloss={_LOADLOSS_PCT} leadlag=Lag"
        )
        dss.Text.Command(f"~ wdg=1 bus=src.1.2.3 conn=wye kV={_KV_HV} kVA={_KVA}")
        # 1.3.2 swaps phases B and C -- a transposition, not a cyclic rotation.
        dss.Text.Command(f"~ wdg=2 bus=lv.1.3.2.0 conn=wye kV={_KV_LV} kVA={_KVA}")
        dss.Text.Command(
            f"New Load.load1 bus1=lv.1.2.3 kV={_KV_LV} kW=10 kvar=3 conn=wye "
            "phases=3 model=1"
        )
        dss.Text.Command(f"Set voltagebases=[{_KV_HV}, {_KV_LV}]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

        with pytest.raises(ConversionError, match="cyclic rotation"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)


# ---------------------------------------------------------------------------
# Forward-direction: cyclic winding-bus rotation parsing
# ---------------------------------------------------------------------------


class TestCyclicRotationParsing:
    """Unit coverage for ``_cyclic_rotation_steps`` / ``_parse_transformer_winding_bus``."""

    @pytest.mark.parametrize(
        "phase_nums,expected_r",
        [([1, 2, 3], 0), ([2, 3, 1], 1), ([3, 1, 2], 2)],
    )
    def test_cyclic_rotation_detected(self, phase_nums, expected_r) -> None:
        assert _cyclic_rotation_steps(phase_nums, 3, "lv.x.x.x") == expected_r

    @pytest.mark.parametrize("phase_nums", [[1, 3, 2], [2, 1, 3], [3, 2, 1]])
    def test_non_cyclic_permutation_raises(self, phase_nums) -> None:
        with pytest.raises(ConversionError, match="cyclic rotation"):
            _cyclic_rotation_steps(phase_nums, 3, "lv.x.x.x")

    def test_parse_transformer_winding_bus_normalizes_phases(self) -> None:
        bus_name, phases, grounded, rotation = _parse_transformer_winding_bus(
            "lv.2.3.1.0", 3
        )
        assert bus_name == "lv"
        # NORMALIZED to canonical (A, B, C) -- never the raw [B, C, A] DSS order.
        assert phases == [Phase.A, Phase.B, Phase.C]
        assert grounded is True
        assert rotation == 1

    def test_parse_transformer_winding_bus_identity(self) -> None:
        bus_name, phases, grounded, rotation = _parse_transformer_winding_bus(
            "hv.1.2.3", 3
        )
        assert phases == [Phase.A, Phase.B, Phase.C]
        assert rotation == 0

    def test_from_side_rotation_r1_live_conversion(self) -> None:
        """HV winding rotated r=1: base(Lag)=clock1, +120*1 deg -> clock 5 (150 deg)."""
        _build_transformer_circuit(
            conn_hv="delta", conn_lv="wye", leadlag="Lag", bus_hv="src.2.3.1"
        )
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        trafo = next(b for b in grid.branches if isinstance(b, Transformer))
        assert trafo.from_phases == (Phase.A, Phase.B, Phase.C)
        assert trafo.to_phases == (Phase.A, Phase.B, Phase.C)
        assert trafo.tap.shift_deg == pytest.approx(150.0)  # clock 5

    def test_to_side_rotation_r1_live_conversion(self) -> None:
        """LV winding rotated r=1: base(Lag)=clock1, -120*1 deg -> clock 9 (270 deg)."""
        _build_transformer_circuit(
            conn_hv="delta", conn_lv="wye", leadlag="Lag", bus_lv="lv.2.3.1.0"
        )
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        trafo = next(b for b in grid.branches if isinstance(b, Transformer))
        assert trafo.tap.shift_deg == pytest.approx(270.0)  # clock 9


# ---------------------------------------------------------------------------
# Oracle direction (pgml -> DSS): what OpenDSS's Transformer element cannot
# express at all, regardless of bus rotation.
# ---------------------------------------------------------------------------

_ABC = (Phase.A, Phase.B, Phase.C)


def _tiny_transformer_grid(
    from_conn: WindingConnection, to_conn: WindingConnection, shift_deg: float
) -> Grid:
    """Minimal 2-node 3-phase grid with one transformer, for oracle unit tests."""
    src = Source(
        id=1,
        node=1,
        phases=_ABC,
        u_ref_v=(_KV_HV * 1_000.0 / math.sqrt(3.0),) * 3,
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=[[1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[1e-8 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    xfmr = Transformer(
        id=2,
        from_node=1,
        to_node=2,
        from_phases=_ABC,
        to_phases=_ABC,
        s_rated_va=_KVA * 1_000.0,
        u_rated_from_v=_KV_HV * 1_000.0,
        u_rated_to_v=_KV_LV * 1_000.0,
        from_connection=from_conn,
        to_connection=to_conn,
        series_resistance_ohm=0.01,
        series_inductance_h=1e-4,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=shift_deg),
    )
    return Grid(
        base_frequency_hz=_F0,
        nodes=[
            Node(id=1, u_rated_v=_KV_HV * 1_000.0, phases=_ABC),
            Node(id=2, u_rated_v=_KV_LV * 1_000.0, phases=_ABC),
        ],
        branches=[xfmr],
        appliances=[src],
    )


class TestOracleUnsupportedClocks:
    """``pgml -> DSS`` direction: what OpenDSS's Transformer element cannot express."""

    def test_zigzag_winding_raises(self) -> None:
        grid = _tiny_transformer_grid(
            WindingConnection.WYE_GROUNDED, WindingConnection.ZIGZAG_GROUNDED, 30.0
        )
        with pytest.raises(NotImplementedError, match="zigzag"):
            _build_circuit_with_real_transformer(grid, {1: "src", 2: "lv"})

    def test_polarity_flip_clock_raises(self) -> None:
        # Yy6: a matching (non-shifting) pairing at clock 6 needs a reversed
        # winding polarity -- no bus-connection rotation can express it.
        grid = _tiny_transformer_grid(
            WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 180.0
        )
        with pytest.raises(NotImplementedError, match="clock 6"):
            _build_circuit_with_real_transformer(grid, {1: "src", 2: "lv"})
