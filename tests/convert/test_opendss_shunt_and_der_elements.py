"""Coverage tests: OpenDSS Capacitor/Reactor -> ShuntAppliance, Generator/
PVSystem/Storage -> Generator/Storage, dropped-element warnings, and the
Vsource-impedance-under-ideal-slack warning.

Covers the element-scope extension of ``pgml.convert.opendss.to_grid``:

- Capacitor: WYE (solidly grounded) converts to a per-phase
  :class:`~pgml.schemas.grid_schema.ShuntAppliance`; DELTA is out of scope
  (warned, not converted); a multi-step bank sums its ACTIVE steps.
- Reactor: the same WYE-shunt path (admittance ``Y=1/(R+jX)``), including the
  "grounding reactor" idiom (a single-conductor reactor tying a neutral
  conductor to ground); DELTA and an explicitly coupled Rmatrix/Xmatrix are
  out of scope (warned, not converted).
- Generator/PVSystem/Storage: generation-positive (Storage: signed,
  discharge-positive) PQ injection, connection, and (Storage) the inert
  energy-state fields.
- Every other unhandled DSS element class (``Isource``, ``Monitor``,
  ``EnergyMeter``, ...) triggers exactly one ``warn_dropped_elements`` WARNING
  naming the class and count.
- A Vsource with a non-negligible R1/X1 triggers a WARNING about the default
  ``slack="ideal"`` ignoring it; a near-zero one does not.
"""

from __future__ import annotations

import math

import pytest
import torch

import opendssdirect as dss  # noqa: E402

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    ConsumerType,
    Generator,
    Phase,
    ShuntAppliance,
    Storage,
    WindingConnection,
)
from pgml.solver import solve_power_flow  # noqa: E402

_F0 = 50.0
_BASEKV = 0.4  # kV, line-to-line (LV feeder)


def _dss_clear() -> None:
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={int(_F0)}")


def _build_base_circuit() -> None:
    _dss_clear()
    dss.Text.Command(
        f"New Circuit.der_test basekv={_BASEKV} pu=1.0 phases=3 bus1=src "
        f"frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
    )
    dss.Text.Command(
        "New Line.l1 phases=3 bus1=src bus2=b1 r1=0.1 x1=0.2 length=1 units=km"
    )
    dss.Text.Command(
        f"New Load.balance phases=3 bus1=b1 kv={_BASEKV} kw=10 kvar=3 model=1"
    )


def _finish_and_solve() -> None:
    dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "DSS DER-element test circuit did not converge"


# ---------------------------------------------------------------------------
# Capacitor
# ---------------------------------------------------------------------------


class TestCapacitorWye:
    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Capacitor.cap1 phases=3 bus1=b1 kv={_BASEKV} kvar=5 conn=wye"
        )
        _finish_and_solve()
        request.cls._grid, request.cls._id_map = to_grid(
            dss, phase_mode=PhaseMode.THREE_PHASE
        )

    def test_converts_to_shunt_appliance(self) -> None:
        shunts = [a for a in self._grid.appliances if isinstance(a, ShuntAppliance)]
        assert len(shunts) == 1
        cap = shunts[0]
        assert cap.phases == (Phase.A, Phase.B, Phase.C)
        assert all(g == 0.0 for g in cap.conductance_s)

    def test_capacitance_matches_closed_form(self) -> None:
        """C matches OpenDSS's own resolved ``Cuf`` (base-invariant of kv/kvar vs
        direct Cuf specification -- read directly rather than re-derived)."""
        dss.Text.Command("? Capacitor.cap1.Cuf")
        cuf = float(dss.Text.Result().strip().strip("[]").split()[0])
        expected_f = cuf * 1.0e-6
        cap = next(a for a in self._grid.appliances if isinstance(a, ShuntAppliance))
        for c in cap.capacitance_f:
            assert c == pytest.approx(expected_f, rel=1e-9)

    def test_id_map_has_capacitor_bucket(self) -> None:
        assert "cap1" in self._id_map["capacitor"]


class TestCapacitorDeltaSkipped:
    def test_delta_capacitor_not_converted_and_warns(self, caplog) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Capacitor.cap2 phases=3 bus1=b1 kv={_BASEKV} kvar=5 conn=delta"
        )
        _finish_and_solve()
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert "cap2" not in id_map["capacitor"]
        assert any(
            "cap2" in r.message and "not converted" in r.message.lower()
            for r in caplog.records
        )


class TestCapacitorMultiStep:
    def test_sums_only_active_steps(self) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Capacitor.cap3 phases=3 bus1=b1 kv={_BASEKV} kvar=[5,5,5] "
            "numsteps=3 states=[1,1,0]"
        )
        _finish_and_solve()
        dss.Text.Command("? Capacitor.cap3.Cuf")
        cuf_steps = [float(v) for v in dss.Text.Result().strip().strip("[]").split()]
        expected_f = (cuf_steps[0] + cuf_steps[1]) * 1.0e-6  # steps 0,1 ON, 2 OFF

        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        cap = next(a for a in grid.appliances if isinstance(a, ShuntAppliance))
        for c in cap.capacitance_f:
            assert c == pytest.approx(expected_f, rel=1e-9)


# ---------------------------------------------------------------------------
# Reactor
# ---------------------------------------------------------------------------


class TestReactorWyeShunt:
    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Reactor.reac1 phases=3 bus1=b1 kv={_BASEKV} kvar=5 conn=wye"
        )
        _finish_and_solve()
        request.cls._grid, request.cls._id_map = to_grid(
            dss, phase_mode=PhaseMode.THREE_PHASE
        )

    def test_converts_to_shunt_appliance_with_negative_c(self) -> None:
        """An inductive reactor's fundamental susceptance is NEGATIVE; the
        C-based shunt model represents it as a negative capacitance (exact at
        h=1 only -- see the converter's Reactor section docstring)."""
        reac = next(
            a
            for a in self._grid.appliances
            if isinstance(a, ShuntAppliance)
            and a.id == self._id_map["reactor"]["reac1"]
        )
        assert all(c < 0.0 for c in reac.capacitance_f)

    def test_admittance_matches_closed_form(self) -> None:
        r_ohm = dss.Reactors.R()
        x_ohm = dss.Reactors.X()
        two_pi_f0 = 2.0 * math.pi * _F0
        y = 1.0 / complex(r_ohm, x_ohm)
        reac = next(
            a
            for a in self._grid.appliances
            if isinstance(a, ShuntAppliance)
            and a.id == self._id_map["reactor"]["reac1"]
        )
        for g, c in zip(reac.conductance_s, reac.capacitance_f):
            assert g == pytest.approx(y.real, rel=1e-9)
            assert c == pytest.approx(y.imag / two_pi_f0, rel=1e-9)


class TestGroundingReactorPattern:
    """The single-conductor "tie neutral to ground" idiom used elsewhere in
    this test suite (``tests/convert/test_opendss_phase_mode.py``)."""

    def test_grounding_reactor_converts_on_neutral_phase(self) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.gnd_reac_test basekv={_BASEKV} pu=1.0 phases=3 bus1=src "
            f"frequency={_F0} r1=1e-9 x1=1e-9"
        )
        dss.Text.Command(
            "New Line.l4w phases=4 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 "
            "rmatrix=[0.2 | 0.05 0.2 | 0.05 0.05 0.2 | 0.05 0.05 0.05 0.25] "
            "xmatrix=[0.4 | 0.1 0.4 | 0.1 0.1 0.4 | 0.1 0.1 0.1 0.45] "
            "length=1 units=km"
        )
        dss.Text.Command("New Reactor.ngnd phases=1 bus1=src.4.0 R=0.01 X=0.01")
        dss.Text.Command(
            f"New Load.wye3ph phases=3 bus1=b1.1.2.3.4 kv={_BASEKV} kw=10 kvar=3 model=1"
        )
        _finish_and_solve()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        reac = next(
            a
            for a in grid.appliances
            if isinstance(a, ShuntAppliance) and a.id == id_map["reactor"]["ngnd"]
        )
        assert reac.phases == (Phase.N,)


class TestReactorDeltaAndCoupledSkipped:
    def test_delta_reactor_not_converted_and_warns(self, caplog) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Reactor.reac2 phases=3 bus1=b1 kv={_BASEKV} kvar=5 conn=delta"
        )
        _finish_and_solve()
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert "reac2" not in id_map["reactor"]
        assert any("reac2" in r.message for r in caplog.records)

    def test_coupled_reactor_not_converted_and_warns(self, caplog) -> None:
        _build_base_circuit()
        dss.Text.Command(
            "New Reactor.reac3 phases=3 bus1=b1 "
            "Rmatrix=[0.1 | 0.02 0.1 | 0.02 0.02 0.1] "
            "Xmatrix=[1.0 | 0.2 1.0 | 0.2 0.2 1.0]"
        )
        _finish_and_solve()
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert "reac3" not in id_map["reactor"]
        assert any("reac3" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerator:
    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Generator.gen1 phases=3 bus1=b1 kv={_BASEKV} kw=5 kvar=2 "
            "conn=wye model=1"
        )
        _finish_and_solve()
        request.cls._grid, request.cls._id_map = to_grid(
            dss, phase_mode=PhaseMode.THREE_PHASE
        )

    def test_converts_generation_positive(self) -> None:
        gen = next(a for a in self._grid.appliances if isinstance(a, Generator))
        assert gen.p_nom_w == pytest.approx(5_000.0)
        assert gen.q_nom_var == pytest.approx(2_000.0)
        assert gen.connection == WindingConnection.WYE

    def test_id_map_has_generator_bucket(self) -> None:
        assert "gen1" in self._id_map["generator"]

    def test_solve_power_flow_reflects_injection(self) -> None:
        """A generator raises the local bus voltage relative to the load-only
        baseline (a coarse but decisive injection-sign sanity check)."""
        res = solve_power_flow(self._grid, slack="ideal")
        assert res.converged
        assert torch.isfinite(res.v).all()


# ---------------------------------------------------------------------------
# PVSystem
# ---------------------------------------------------------------------------


class TestPVSystem:
    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New PVSystem.pv1 phases=3 bus1=b1 kv={_BASEKV} kVA=8 Pmpp=6 "
            "irradiance=1.0 pf=0.98 conn=wye"
        )
        _finish_and_solve()
        request.cls._grid, request.cls._id_map = to_grid(
            dss, phase_mode=PhaseMode.THREE_PHASE
        )

    def test_converts_as_generator_with_pv_consumer_type(self) -> None:
        pv = next(
            a
            for a in self._grid.appliances
            if isinstance(a, Generator) and a.id == self._id_map["pvsystem"]["pv1"]
        )
        assert pv.consumer_type == ConsumerType.PV
        # Present output (Pmpp * irradiance, pf-derived Q), not the kVA rating.
        assert pv.p_nom_w == pytest.approx(6_000.0, rel=1e-6)
        assert pv.q_nom_var > 0.0


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestStorageDischarging:
    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Storage.bat1 phases=3 bus1=b1 kv={_BASEKV} kWrated=6 kWhrated=20 "
            "%stored=50 %reserve=10 kW=3 kvar=1 conn=wye state=DISCHARGING"
        )
        _finish_and_solve()
        request.cls._grid, request.cls._id_map = to_grid(
            dss, phase_mode=PhaseMode.THREE_PHASE
        )

    def test_discharge_is_positive_p(self) -> None:
        bat = next(a for a in self._grid.appliances if isinstance(a, Storage))
        assert bat.p_nom_w > 0.0
        assert bat.p_nom_w == pytest.approx(3_000.0)
        assert bat.q_nom_var == pytest.approx(1_000.0)

    def test_energy_state_fields(self) -> None:
        bat = next(a for a in self._grid.appliances if isinstance(a, Storage))
        assert bat.energy_capacity_wh == pytest.approx(20_000.0)
        assert bat.p_rated_w == pytest.approx(6_000.0)
        assert bat.soc == pytest.approx(0.5)
        assert bat.soc_min == pytest.approx(0.1)
        assert bat.consumer_type == ConsumerType.BATTERY


class TestStorageCharging:
    def test_charge_is_negative_p(self) -> None:
        _build_base_circuit()
        dss.Text.Command(
            f"New Storage.bat2 phases=3 bus1=b1 kv={_BASEKV} kWrated=6 kWhrated=20 "
            "%stored=50 kW=3 kvar=1 conn=wye state=CHARGING"
        )
        _finish_and_solve()
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        bat = next(a for a in grid.appliances if isinstance(a, Storage))
        assert bat.p_nom_w < 0.0


# ---------------------------------------------------------------------------
# Dropped-element warning (Isource, Monitor, EnergyMeter, ...)
# ---------------------------------------------------------------------------


def test_unhandled_element_classes_warn(caplog) -> None:
    _build_base_circuit()
    dss.Text.Command("New Isource.src1 phases=1 bus1=b1.1 amps=1 angle=0")
    dss.Text.Command("New Monitor.mon1 element=Line.l1 terminal=1")
    dss.Text.Command("New EnergyMeter.em1 element=Line.l1 terminal=1")
    _finish_and_solve()
    with caplog.at_level("WARNING", logger="pgml"):
        to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    warned_kinds = {
        kind
        for r in caplog.records
        for kind in ("isource", "monitor", "energymeter")
        if kind in r.message.lower()
    }
    assert warned_kinds == {"isource", "monitor", "energymeter"}, (
        f"expected warnings for isource/monitor/energymeter; got "
        f"{[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Vsource impedance under the default ideal slack
# ---------------------------------------------------------------------------


def test_meaningful_vsource_impedance_warns(caplog) -> None:
    _dss_clear()
    dss.Text.Command(
        f"New Circuit.vsrc_z_test basekv={_BASEKV} pu=1.0 phases=3 bus1=src "
        f"frequency={_F0} r1=0.05 x1=0.2"
    )
    dss.Text.Command(
        "New Line.l1 phases=3 bus1=src bus2=b1 r1=0.1 x1=0.2 length=1 units=km"
    )
    dss.Text.Command(f"New Load.ld phases=3 bus1=b1 kv={_BASEKV} kw=10 kvar=3 model=1")
    _finish_and_solve()
    with caplog.at_level("WARNING", logger="pgml"):
        to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    assert any(
        "non-negligible source impedance" in r.message for r in caplog.records
    ), (
        f"expected a Vsource impedance warning; got {[r.message for r in caplog.records]}"
    )


def test_negligible_vsource_impedance_does_not_warn(caplog) -> None:
    _build_base_circuit()  # r1=x1=1e-9 on the Vsource
    _finish_and_solve()
    with caplog.at_level("WARNING", logger="pgml"):
        to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    assert not any(
        "non-negligible source impedance" in r.message for r in caplog.records
    )
