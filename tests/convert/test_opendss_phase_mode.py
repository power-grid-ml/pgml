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

import opendssdirect as dss  # noqa: E402

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
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

    def test_line_r_diagonal_dominates(self) -> None:
        """Diagonal entries of R are larger than off-diagonal (self > mutual)."""
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        r = line.series_resistance_ohm_per_m
        assert r[0][0] > abs(r[0][1])

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

    def test_single_phase_equiv_shape_matches_three_phase_node_count(self) -> None:
        """SINGLE_PHASE_EQUIV and THREE_PHASE yield the same number of nodes."""
        grid_1ph, _ = to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
        grid_3ph, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert len(grid_1ph.nodes) == len(grid_3ph.nodes)


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
