"""Phase-mode integration tests for the OpenDSS converter.

Covers:
- ``SINGLE_PHASE_EQUIV`` (the default) collapses everything to ``phases=(Phase.A,)``
  and 1×1 line matrices (the ``[0][0]`` entry of the DSS matrix), matching the
  historical converter output.
- ``THREE_PHASE`` emits the real DSS phases (``(A, B, C)`` for a 3-phase bus,
  ``(A,)`` for a single-phase bus), the full n×n line matrices, and preserves load
  ``connection`` from ``IsDelta()``.
- A delta load and a single-phase L-N load are captured correctly in THREE_PHASE.
- The converted grid (THREE_PHASE) assembles and solves without error.
- ``u_rated_v`` equals the line-to-line voltage (``kVBase * sqrt(3) * 1000``) for all
  buses, matching the pandapower/pgm convention.

The DSS circuit is built programmatically inside the test using opendssdirect so the
test carries no external file dependency (the same pattern as
``tests/reference/test_ieee33_opendss.py``).
"""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Optional opendssdirect guard (matches existing reference test conventions)
# ---------------------------------------------------------------------------
try:
    import opendssdirect as dss  # noqa: E402

    _OPENDSS_AVAILABLE = True
except ImportError:
    _OPENDSS_AVAILABLE = False

if not _OPENDSS_AVAILABLE:
    pytest.skip("opendssdirect not installed", allow_module_level=True)

pytestmark = pytest.mark.opendss

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.convert.opendss.converter import _parse_bus_connection  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    Line,
    Load,
    Phase,
    Source,
    WindingConnection,
)
from pgml.solver import solve_power_flow  # noqa: E402

ABC = (Phase.A, Phase.B, Phase.C)

# ---------------------------------------------------------------------------
# Shared circuit builder
# ---------------------------------------------------------------------------

_BASEKV = 12.66  # kV, line-to-line (the OpenDSS ``basekv=`` parameter)
_SQRT3 = math.sqrt(3.0)  # used only to verify diagonal-vs-mutual structure


def _build_dss_circuit() -> None:
    """Build a small 2-bus 3-phase DSS circuit with three load types.

    Elements:
    - 3-phase Vsource at bus ``src`` (balanced, 12.66 kV LL, 60 Hz).
    - 3-phase line ``l1`` connecting ``src`` -> ``b1``.
    - 3-phase WYE load ``wye3ph`` at ``b1`` (300 kW / 100 kVAR).
    - 3-phase DELTA load ``delta3ph`` at ``b1`` (600 kW / 200 kVAR).
    - Single-phase (phase A) load ``ph_a`` at ``b1.1`` (100 kW / 30 kVAR).
    """
    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.phase_mode_test basekv={_BASEKV} pu=1.0 phases=3 "
        "bus1=src frequency=60"
    )
    # 3-phase line: gives a non-trivial 3x3 R matrix via Carson earth-return
    dss.Text.Command(
        "New Line.l1 phases=3 bus1=src bus2=b1 r1=0.1 x1=0.2 length=1 units=km"
    )
    # 3-phase WYE load
    dss.Text.Command(
        f"New Load.wye3ph phases=3 bus1=b1 kv={_BASEKV} "
        "kw=300 kvar=100 conn=wye model=1"
    )
    # 3-phase DELTA load
    dss.Text.Command(
        f"New Load.delta3ph phases=3 bus1=b1 kv={_BASEKV} "
        "kw=600 kvar=200 conn=delta model=1"
    )
    # Single-phase (phase A only) load
    dss.Text.Command(
        f"New Load.ph_a phases=1 bus1=b1.1 kv={_BASEKV} kw=100 kvar=30 model=1"
    )
    dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "DSS test circuit did not converge"


# ---------------------------------------------------------------------------
# Tests — SINGLE_PHASE_EQUIV (default)
# ---------------------------------------------------------------------------


class TestSinglePhaseEquiv:
    """Verify the default SINGLE_PHASE_EQUIV output shape and conventions."""

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_dss_circuit()
        grid, id_map = to_grid(dss)  # default = SINGLE_PHASE_EQUIV
        request.cls._grid = grid
        request.cls._id_map = id_map

    def test_all_nodes_are_phase_a(self) -> None:
        """Every node collapses to ``phases=(Phase.A,)``."""
        for n in self._grid.nodes:
            assert n.phases == (Phase.A,), f"Node {n.name} has phases {n.phases}"

    def test_line_is_1x1(self) -> None:
        """Line carries a 1×1 series-resistance matrix (the diagonal scalar)."""
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        assert line.from_phases == (Phase.A,)
        r = line.series_resistance_ohm_per_m
        assert len(r) == 1 and len(r[0]) == 1

    def test_source_is_single_phase(self) -> None:
        """Source is single-phase ``(Phase.A,)``."""
        src = next(a for a in self._grid.appliances if isinstance(a, Source))
        assert src.phases == (Phase.A,)
        assert len(src.u_ref_v) == 1

    def test_load_phases_all_phase_a(self) -> None:
        """All loads (including delta and single-phase) collapse to ``(Phase.A,)``."""
        loads = [a for a in self._grid.appliances if isinstance(a, Load)]
        for ld in loads:
            assert ld.phases == (Phase.A,)

    def test_load_connection_is_none_under_single_phase_equiv(self) -> None:
        """Under SINGLE_PHASE_EQUIV the load connection is not set."""
        loads = [a for a in self._grid.appliances if isinstance(a, Load)]
        for ld in loads:
            assert ld.connection is None

    def test_u_rated_v_is_line_to_line(self) -> None:
        """u_rated_v equals the L-L voltage (BasekV_LL * 1000).

        ``_BASEKV`` is the line-to-line kilovoltage specified in the ``New Circuit``
        command.  OpenDSS stores ``kVBase = _BASEKV / sqrt(3)`` internally (L-N).
        The converter recovers L-L via ``kVBase * sqrt(3) * 1000 = _BASEKV * 1000``.
        """
        expected_v = _BASEKV * 1_000.0  # L-L in volts
        for n in self._grid.nodes:
            assert abs(n.u_rated_v - expected_v) < 1.0, (
                f"Node {n.name}: u_rated_v={n.u_rated_v:.2f} V, "
                f"expected {expected_v:.2f} V (L-L)"
            )

    def test_slack_v_complex_in_id_map(self) -> None:
        """id_map carries a 'slack_v_complex' entry from the first Vsource."""
        assert "slack_v_complex" in self._id_map
        assert self._id_map["slack_v_complex"] is not None
        assert abs(self._id_map["slack_v_complex"]) == pytest.approx(
            _BASEKV * 1_000.0, rel=1e-6
        )

    def test_id_map_has_all_required_buckets(self) -> None:
        """id_map contains all expected element-type buckets."""
        for key in ("bus", "line", "load", "vsource", "slack_v_complex"):
            assert key in self._id_map


# ---------------------------------------------------------------------------
# Tests — THREE_PHASE
# ---------------------------------------------------------------------------


class TestThreePhase:
    """Verify the THREE_PHASE output captures real phases and connections."""

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_dss_circuit()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        request.cls._grid = grid
        request.cls._id_map = id_map

    def test_three_phase_nodes_have_abc(self) -> None:
        """3-phase DSS buses produce abc nodes."""
        n_src = next(n for n in self._grid.nodes if n.name == "src")
        n_b1 = next(n for n in self._grid.nodes if n.name == "b1")
        assert n_src.phases == ABC
        assert n_b1.phases == ABC

    def test_line_is_3x3(self) -> None:
        """The 3-phase line carries a 3×3 series-resistance matrix."""
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        assert line.from_phases == ABC
        r = line.series_resistance_ohm_per_m
        assert len(r) == 3 and all(len(row) == 3 for row in r)

    def test_line_r_self_and_mutual_match_sequence_expansion(self) -> None:
        """Self/mutual R match the symmetric-component expansion of the DSS line.

        OpenDSS builds the phase ``RMatrix`` from the sequence impedances
        (``R1``/``R0``) via ``R_self = (R0 + 2*R1)/3`` and
        ``R_mut = (R0 - R1)/3``.  The converter divides the per-unit-length DSS
        matrix by ``length_m`` (here 1 km = 1000 m), so the pgml per-metre matrix
        is the same expansion scaled by ``1/length_m``.
        """
        dss.Lines.Name("l1")
        r1 = dss.Lines.R1()
        r0 = dss.Lines.R0()
        length_m = dss.Lines.Length() * 1_000.0  # units=km -> metres
        r_self = (r0 + 2.0 * r1) / 3.0 / length_m
        r_mut = (r0 - r1) / 3.0 / length_m

        line = next(b for b in self._grid.branches if isinstance(b, Line))
        r = line.series_resistance_ohm_per_m
        assert r[0][0] == pytest.approx(r_self, rel=1e-9)
        assert r[0][1] == pytest.approx(r_mut, rel=1e-9)
        assert r[1][0] == pytest.approx(r_mut, rel=1e-9)
        assert r[0][0] > abs(r[0][1])  # self dominates mutual

    def test_source_is_three_phase(self) -> None:
        """Source has 3 phases and balanced 120-degree-apart angles."""
        src = next(a for a in self._grid.appliances if isinstance(a, Source))
        assert src.phases == ABC
        assert len(src.u_ref_v) == 3
        assert src.u_angle_deg[1] == pytest.approx(0.0 - 120.0)
        assert src.u_angle_deg[2] == pytest.approx(0.0 - 240.0)

    def test_wye_load_connection(self) -> None:
        """The 3-phase WYE load has ``connection=WindingConnection.WYE``."""
        ld = next(
            a
            for a in self._grid.appliances
            if isinstance(a, Load) and a.name == "wye3ph"
        )
        assert ld.phases == ABC
        assert ld.connection == WindingConnection.WYE

    def test_delta_load_connection(self) -> None:
        """The 3-phase DELTA load has ``connection=WindingConnection.DELTA``."""
        ld = next(
            a
            for a in self._grid.appliances
            if isinstance(a, Load) and a.name == "delta3ph"
        )
        assert ld.phases == ABC
        assert ld.connection == WindingConnection.DELTA

    def test_single_phase_load_placement(self) -> None:
        """The single-phase L-N load (bus suffix .1) lands on ``phases=(Phase.A,)``."""
        ld = next(
            a for a in self._grid.appliances if isinstance(a, Load) and a.name == "ph_a"
        )
        assert ld.phases == (Phase.A,)
        assert ld.connection == WindingConnection.WYE  # L-N, two-conductor default

    def test_load_p_values(self) -> None:
        """Converted load active powers match the DSS kW values."""
        loads = {a.name: a for a in self._grid.appliances if isinstance(a, Load)}
        assert loads["wye3ph"].p_nom_w == pytest.approx(300_000.0)
        assert loads["delta3ph"].p_nom_w == pytest.approx(600_000.0)
        assert loads["ph_a"].p_nom_w == pytest.approx(100_000.0)

    def test_u_rated_v_is_line_to_line(self) -> None:
        """u_rated_v equals the L-L voltage for all nodes (BasekV_LL * 1000)."""
        expected_v = _BASEKV * 1_000.0  # L-L in volts
        for n in self._grid.nodes:
            assert abs(n.u_rated_v - expected_v) < 1.0, (
                f"Node {n.name}: u_rated_v={n.u_rated_v:.2f} V, "
                f"expected {expected_v:.2f} V"
            )

    def test_id_map_has_all_required_buckets(self) -> None:
        """id_map contains all expected element-type buckets."""
        for key in ("bus", "line", "load", "vsource", "slack_v_complex"):
            assert key in self._id_map


# ---------------------------------------------------------------------------
# Assembly + solve tests
# ---------------------------------------------------------------------------


class TestConverterSolvability:
    """Verify that the converted grid assembles and solves finite voltages."""

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_dss_circuit()

    def test_three_phase_assembles_and_solves(self) -> None:
        """THREE_PHASE converted grid solves with finite voltages."""
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        res = solve_power_flow(grid, slack="ideal")
        assert res.v is not None
        assert torch.isfinite(res.v.real).all()
        assert torch.isfinite(res.v.imag).all()

    def test_single_phase_equiv_assembles_and_solves(self) -> None:
        """SINGLE_PHASE_EQUIV converted grid solves with finite voltages."""
        grid, _ = to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        res = solve_power_flow(grid, slack="ideal")
        assert res.v is not None
        assert torch.isfinite(res.v.real).all()
        assert torch.isfinite(res.v.imag).all()

    def test_three_phase_node_count_is_2(self) -> None:
        """The 2-bus circuit yields 2 nodes in THREE_PHASE mode."""
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert len(grid.nodes) == 2


# ---------------------------------------------------------------------------
# delta-load log test
# ---------------------------------------------------------------------------


def test_delta_load_collapsed_logs_info(caplog) -> None:
    """Under SINGLE_PHASE_EQUIV, a delta load triggers an INFO log."""
    _build_dss_circuit()
    with caplog.at_level("INFO", logger="pgml"):
        to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    assert any("delta" in r.message.lower() for r in caplog.records), (
        "Expected INFO about delta load collapse; got: "
        + str([r.message for r in caplog.records])
    )


# ---------------------------------------------------------------------------
# Bus-connection parsing (DSS node index -> Phase)
# ---------------------------------------------------------------------------


class TestBusConnectionParsing:
    """DSS bus-node indices map 1=A, 2=B, 3=C, 4=N; 0 (ground) is dropped."""

    def test_four_wire_suffix_maps_to_neutral(self) -> None:
        name, phases = _parse_bus_connection("b1.1.2.3.4", 4)
        assert name == "b1"
        assert phases == [Phase.A, Phase.B, Phase.C, Phase.N]

    def test_ground_suffix_zero_is_dropped(self) -> None:
        """A ``.0`` conductor is tied to the grounded reference — no phase row."""
        name, phases = _parse_bus_connection("b1.1.0", 2)
        assert name == "b1"
        assert phases == [Phase.A]

    def test_unknown_suffix_raises(self) -> None:
        with pytest.raises(ConversionError):
            _parse_bus_connection("b1.1.5", 2)


# ---------------------------------------------------------------------------
# Four-wire circuit with an explicit neutral conductor
# ---------------------------------------------------------------------------


def _build_four_wire_circuit() -> None:
    """Build a 2-bus circuit whose line carries an explicit 4th (neutral) wire.

    The neutral is grounded at the source through a small reactor -- since
    ``Reactor`` now converts to a :class:`~pgml.schemas.grid_schema.ShuntAppliance`
    (a series R+X branch to OpenDSS's own universal ground reference is
    electrically identical to a shunt admittance there), this reactor is
    itself part of the converted grid and anchors the neutral rail's absolute
    voltage in pgml exactly as it does in the live DSS solve. The load ties
    its return explicitly to the 4th (neutral) conductor (``bus1=b1.1.2.3.4``)
    -- pgml's WYE incidence always routes a load's return through the node's
    ``Phase.N`` row when the node carries one, which is exactly what this
    explicit tie means; see ``test_grounded_load_on_neutral_carrying_node_warns``
    for the (out-of-scope, warned) case where the load grounds implicitly
    despite the node carrying a neutral.
    """
    dss.Text.Command("Clear")
    # DSS's own factory default base frequency is 60 Hz; without explicitly
    # reasserting it here (a documented engine gotcha -- see
    # `src/pgml/convert/opendss/CONTEXT.md` and `_dss_clear()` in
    # `tests/reference/test_opendss_transformer.py`), a `frequency=50` circuit
    # solves to a degenerate ALL-ZERO voltage state (still reports
    # `Converged() == True`) on this opendssdirect version.
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(
        "New Circuit.four_wire_test basekv=0.4 pu=1.0 phases=3 bus1=src "
        "frequency=50 r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
    )
    dss.Text.Command(
        "New Line.l4w phases=4 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 "
        "rmatrix=[0.2 | 0.05 0.2 | 0.05 0.05 0.2 | 0.05 0.05 0.05 0.25] "
        "xmatrix=[0.4 | 0.1 0.4 | 0.1 0.1 0.4 | 0.1 0.1 0.1 0.45] "
        "length=1 units=km"
    )
    dss.Text.Command("New Reactor.ngnd phases=1 bus1=src.4.0 R=0.01 X=0.01")
    dss.Text.Command(
        "New Load.wye3ph phases=3 bus1=b1.1.2.3.4 kv=0.4 kw=10 kvar=3 model=1"
    )
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    # DSS's own default solve tolerance (1e-4) is far looser than pgml's
    # nonlinear solve below; tighten it so the voltage-parity comparison
    # isolates the CONVERSION, not DSS's own residual.
    dss.Text.Command("Set Tolerance=1e-13")
    dss.Text.Command("Set maxiterations=1000")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "DSS four-wire test circuit did not converge"


class TestFourWire:
    """An explicit 4th conductor converts to ``Phase.N``, not a phase alias."""

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _build_four_wire_circuit()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        request.cls._grid = grid
        request.cls._id_map = id_map

    def test_nodes_carry_neutral_phase(self) -> None:
        """Both buses register the neutral conductor as a ``Phase.N`` row."""
        for name in ("src", "b1"):
            node = next(n for n in self._grid.nodes if n.name == name)
            assert set(node.phases) == {Phase.A, Phase.B, Phase.C, Phase.N}, (
                f"Node {name}: phases {node.phases}"
            )

    def test_line_phases_include_neutral(self) -> None:
        """The 4-wire line keeps all four conductors, the 4th mapped to N."""
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        assert line.from_phases == (Phase.A, Phase.B, Phase.C, Phase.N)
        assert line.to_phases == (Phase.A, Phase.B, Phase.C, Phase.N)

    def test_line_matrices_are_4x4(self) -> None:
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        r = line.series_resistance_ohm_per_m
        assert len(r) == 4 and all(len(row) == 4 for row in r)

    def test_voltage_parity_vs_live_opendss(self) -> None:
        """Every (bus, phase incl. N) voltage matches the live DSS solve.

        The grounding ``Reactor`` (now converted to a ``ShuntAppliance``) and
        the load's explicit neutral tie (``.4``) together anchor the neutral
        rail's absolute voltage the same way OpenDSS itself does; without the
        Reactor conversion the neutral would float relative to true ground
        and this comparison would fail.
        """
        result = solve_power_flow(
            self._grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
        )
        assert result.converged, (
            f"solve_power_flow did not converge (residual={float(result.residual):.3e})"
        )
        for name in ("src", "b1"):
            node = next(n for n in self._grid.nodes if n.name == name)
            dss.Circuit.SetActiveBus(name)
            dss_va = dss.Bus.puVmagAngle()
            kvbase_ln_kv = dss.Bus.kVBase()
            for k, phase in enumerate((Phase.A, Phase.B, Phase.C, Phase.N)):
                row = result.index.row(node.id, phase)
                v_val = complex(result.v.reshape(-1)[row].item())
                vm_pu_ours = abs(v_val) / (kvbase_ln_kv * 1_000.0)
                vm_pu_ref = dss_va[2 * k]
                assert abs(vm_pu_ours - vm_pu_ref) < 1.0e-6, (
                    f"bus {name} phase {phase}: |V| mismatch ours={vm_pu_ours:.8f} "
                    f"dss={vm_pu_ref:.8f}"
                )


def _build_grounded_load_on_neutral_carrying_bus() -> None:
    """Same 4-wire feeder, but the load grounds IMPLICITLY (no ``.4`` tie).

    The load's own bus string (``b1.1.2.3``, exactly ``Phases=3`` suffixes)
    means "return to true ground" in OpenDSS -- even though bus ``b1`` ALSO
    carries a ``Phase.N`` row (from the line's explicit 4th conductor). The
    converter now expresses this per-appliance via ``return_path='ground'`` so
    pgml routes the load's return to true ground rather than the shared
    neutral, reproducing the OpenDSS circuit exactly.
    """
    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(
        "New Circuit.grounded_on_neutral_bus basekv=0.4 pu=1.0 phases=3 bus1=src "
        "frequency=50 r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
    )
    dss.Text.Command(
        "New Line.l4w phases=4 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 "
        "rmatrix=[0.2 | 0.05 0.2 | 0.05 0.05 0.2 | 0.05 0.05 0.05 0.25] "
        "xmatrix=[0.4 | 0.1 0.4 | 0.1 0.1 0.4 | 0.1 0.1 0.1 0.45] "
        "length=1 units=km"
    )
    dss.Text.Command("New Reactor.ngnd phases=1 bus1=src.4.0 R=0.01 X=0.01")
    dss.Text.Command(
        "New Load.wye3ph phases=3 bus1=b1.1.2.3 kv=0.4 kw=10 kvar=3 model=1"
    )
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Set Tolerance=1e-13")
    dss.Text.Command("Set maxiterations=1000")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), (
        "DSS grounded-on-neutral-bus circuit did not converge"
    )


def test_grounded_load_on_neutral_carrying_node_uses_return_path_ground(caplog) -> None:
    """A WYE load grounded despite its node carrying Phase.N converts with
    ``return_path='ground'`` -- no warning, and the return pins to true ground.

    pgml's WYE incidence is a per-NODE property (returns through the node's
    ``Phase.N`` row whenever the node carries one); ``return_path='ground'``
    overrides that for this appliance so the converted grid reproduces the
    OpenDSS circuit (which grounds this load's return) rather than mis-wiring
    it through the shared neutral.
    """
    _build_grounded_load_on_neutral_carrying_bus()
    with caplog.at_level("WARNING", logger="pgml"):
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    load = next(a for a in grid.appliances if isinstance(a, Load))
    assert load.return_path == "ground"
    assert not any(
        "miswired" in r.message.lower() or "grounded despite" in r.message.lower()
        for r in caplog.records
    ), "the grounded-despite-neutral case is now expressible; no warning expected"


def test_grounded_load_on_neutral_carrying_node_voltage_parity() -> None:
    """The grounded (``.1.2.3``) load on a 4-wire bus matches the live DSS solve.

    Previously inexpressible (pgml routed the return through the shared neutral
    and diverged); ``return_path='ground'`` fixes it.
    """
    _build_grounded_load_on_neutral_carrying_bus()
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
    )
    assert result.converged
    for name in ("src", "b1"):
        node = next(n for n in grid.nodes if n.name == name)
        dss.Circuit.SetActiveBus(name)
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        for k, phase in enumerate((Phase.A, Phase.B, Phase.C, Phase.N)):
            row = result.index.row(node.id, phase)
            v_val = complex(result.v.reshape(-1)[row].item())
            vm_pu_ours = abs(v_val) / (kvbase_ln_kv * 1_000.0)
            vm_pu_ref = dss_va[2 * k]
            assert abs(vm_pu_ours - vm_pu_ref) < 1.0e-6, (
                f"bus {name} phase {phase}: |V| mismatch ours={vm_pu_ours:.8f} "
                f"dss={vm_pu_ref:.8f}"
            )


def _build_four_wire_mixed_return_circuit() -> None:
    """A 4-wire bus carrying BOTH a grounded (``.1.2.3``) and a neutral-returning
    (``.1.2.3.4``) WYE load -- the previously-unrepresentable mix.

    OpenDSS resolves each load's return conductor independently; the converter
    reproduces that with per-appliance ``return_path`` ('ground' for the first,
    'neutral' for the second), so both loads coexist on the same 4-wire node with
    different terminal incidences.
    """
    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(
        "New Circuit.mixed_return basekv=0.4 pu=1.0 phases=3 bus1=src "
        "frequency=50 r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
    )
    dss.Text.Command(
        "New Line.l4w phases=4 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 "
        "rmatrix=[0.2 | 0.05 0.2 | 0.05 0.05 0.2 | 0.05 0.05 0.05 0.25] "
        "xmatrix=[0.4 | 0.1 0.4 | 0.1 0.1 0.4 | 0.1 0.1 0.1 0.45] "
        "length=1 units=km"
    )
    dss.Text.Command("New Reactor.ngnd phases=1 bus1=src.4.0 R=0.01 X=0.01")
    # Grounded load (implicit ground return) and a neutral-tied load on the SAME bus.
    dss.Text.Command(
        "New Load.grounded phases=3 bus1=b1.1.2.3 kv=0.4 kw=8 kvar=2 model=1"
    )
    dss.Text.Command(
        "New Load.neutral phases=3 bus1=b1.1.2.3.4 kv=0.4 kw=6 kvar=1.5 model=1"
    )
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Set Tolerance=1e-13")
    dss.Text.Command("Set maxiterations=1000")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged(), "DSS mixed-return 4-wire circuit did not converge"


def test_four_wire_mixed_return_paths_convert_and_match_opendss() -> None:
    """Grounded + neutral-returning loads on one 4-wire bus: return_path is
    assigned per load and every (bus, phase incl. N) voltage matches live DSS."""
    _build_four_wire_mixed_return_circuit()
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)

    by_name = {a.name: a for a in grid.appliances if isinstance(a, Load)}
    assert by_name["grounded"].return_path == "ground"
    assert by_name["neutral"].return_path == "neutral"

    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
    )
    assert result.converged
    for name in ("src", "b1"):
        node = next(n for n in grid.nodes if n.name == name)
        dss.Circuit.SetActiveBus(name)
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        for k, phase in enumerate((Phase.A, Phase.B, Phase.C, Phase.N)):
            row = result.index.row(node.id, phase)
            v_val = complex(result.v.reshape(-1)[row].item())
            vm_pu_ours = abs(v_val) / (kvbase_ln_kv * 1_000.0)
            vm_pu_ref = dss_va[2 * k]
            assert abs(vm_pu_ours - vm_pu_ref) < 1.0e-6, (
                f"bus {name} phase {phase}: |V| mismatch ours={vm_pu_ours:.8f} "
                f"dss={vm_pu_ref:.8f}"
            )
