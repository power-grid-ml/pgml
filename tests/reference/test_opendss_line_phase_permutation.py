"""Oracle test: a line wired with a PERMUTED terminal phase order.

OpenDSS lets a line's two bus-connection strings list their phase conductors
in a DIFFERENT order at each end, e.g. ``bus1=a.1.2.3 bus2=b.3.2.1`` -- the
line's physical conductor ``k`` (row/column ``k`` of the R/X/C matrix) ties
``a``'s phase ``k`` to whichever phase ``b`` lists in position ``k``. This is
a genuine phase-transposing connection (distinct from a transformer's
vector-group clock): ``pgml.convert.opendss.to_grid`` must carry the FROM and
TO terminal phase tuples independently (``Line.from_phases``/``to_phases``)
rather than assuming both terminals share the same phase order.

``pgml.assembly.ybus._series_terminal_indices`` already indexes a series
branch's two terminals independently (``from_rows``/`to_rows`` built from
``b.from_phases``/``b.to_phases`` respectively) -- the SAME mechanism a
Transformer's differing ``from_phases``/``to_phases`` already relies on -- so
no assembly change is needed; this file is the numeric proof.

Test strategy
-------------
1. Build a small DSS circuit: a near-ideal 3-phase Vsource, ONE line wired
   with a permuted terminal-phase order, and deliberately UNBALANCED
   single-phase loads at the far bus (so a mis-wired permutation would show
   up as a systematic per-phase voltage error, not just a benign relabeling
   of an otherwise-symmetric solve).
2. ``Solve`` the DSS circuit (its own nonlinear AC power flow).
3. Convert with ``to_grid(dss, phase_mode=THREE_PHASE)`` and solve with pgml
   (``solve_power_flow(grid, slack="ideal")``; the Vsource carries a
   near-zero Thevenin impedance).
4. Compare every bus's per-phase voltage magnitude (pu) and angle against
   DSS's own ``Bus.puVmagAngle()``.

A second, single-phase circuit (``bus1=a.1 bus2=b.2``) exercises the
degenerate n=1 case: a lateral tapped off phase A but landing on bus b's
phase B row.

Tolerance targets
------------------
- Voltage magnitude: atol = 1e-6 pu.
- Voltage angle:      atol = 1e-4 deg.
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

from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.schemas.grid_schema import Line, Phase  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

_F0 = 60.0
_BASEKV = 4.16  # kV, line-to-line
ABC = (Phase.A, Phase.B, Phase.C)

ATOL_VM_PU = 1.0e-6
ATOL_VA_DEG = 1.0e-4


def _dss_clear() -> None:
    """Reset the process-global OpenDSS engine and reassert the base frequency.

    ``Clear`` does NOT reset ``DefaultBaseFrequency`` (it persists across
    ``Clear`` within the same ``opendssdirect`` process) -- see
    ``src/pgml/convert/opendss/CONTEXT.md``.
    """
    dss.Text.Command("Clear")
    dss.Text.Command(f"set DefaultBaseFrequency={int(_F0)}")


def _angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _compare_all_buses(id_map: dict, result, phases: tuple) -> None:
    for bus_name, node_id in id_map["bus"].items():
        dss.Circuit.SetActiveBus(bus_name)
        dss_va = dss.Bus.puVmagAngle()
        kvbase_ln_kv = dss.Bus.kVBase()
        for k, phase in enumerate(phases):
            row = result.index.row(node_id, phase)
            v_val = complex(result.v.reshape(-1)[row].item())
            vm_pu_ours = abs(v_val) / (kvbase_ln_kv * 1_000.0)
            va_deg_ours = math.degrees(math.atan2(v_val.imag, v_val.real))

            vm_pu_ref = dss_va[2 * k]
            va_deg_ref = dss_va[2 * k + 1]

            vm_err = abs(vm_pu_ours - vm_pu_ref)
            va_err = abs(_angle_diff_deg(va_deg_ours, va_deg_ref))

            assert vm_err < ATOL_VM_PU, (
                f"bus {bus_name} phase {phase}: |V| mismatch "
                f"ours={vm_pu_ours:.8f} dss={vm_pu_ref:.8f} err={vm_err:.2e} pu"
            )
            assert va_err < ATOL_VA_DEG, (
                f"bus {bus_name} phase {phase}: angle mismatch "
                f"ours={va_deg_ours:.6f} dss={va_deg_ref:.6f} err={va_err:.2e} deg"
            )


def _solve_pgml():
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    result = solve_power_flow(
        grid, slack="ideal", tol=1e-12, max_iter=300, dtype=torch.complex128
    )
    assert result.converged, (
        f"solve_power_flow did not converge (residual={float(result.residual):.3e})"
    )
    return grid, id_map, result


# ---------------------------------------------------------------------------
# 3-phase permutation: bus1=a.1.2.3 bus2=b.3.2.1
# ---------------------------------------------------------------------------


class TestThreePhasePermutedLine:
    """A line whose two terminals list a cyclically-reversed phase order."""

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.perm3 basekv={_BASEKV} pu=1.0 phases=3 bus1=a "
            f"angle=0.0 frequency={_F0} r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9"
        )
        # Conductor k of the line ties a's phase (k+1) to b's phase per the
        # REVERSED list below: conductor 0 -> a.1 <-> b.3, conductor 1 ->
        # a.2 <-> b.2, conductor 2 -> a.3 <-> b.1.
        dss.Text.Command(
            "New Line.l1 phases=3 bus1=a.1.2.3 bus2=b.3.2.1 "
            "r1=0.3 x1=0.6 length=1 units=km"
        )
        # Deliberately UNBALANCED single-phase loads at b -- a mis-wired
        # permutation would feed the wrong phase's current back through the
        # source, showing up as a systematic per-phase voltage error.
        dss.Text.Command(
            f"New Load.la phases=1 bus1=b.1 kv={_BASEKV / math.sqrt(3.0)} "
            "kw=300 kvar=100 model=1"
        )
        dss.Text.Command(
            f"New Load.lb phases=1 bus1=b.2 kv={_BASEKV / math.sqrt(3.0)} "
            "kw=50 kvar=20 model=1"
        )
        dss.Text.Command(
            f"New Load.lc phases=1 bus1=b.3 kv={_BASEKV / math.sqrt(3.0)} "
            "kw=150 kvar=50 model=1"
        )
        dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
        dss.Text.Command("Calcvoltagebases")
        # DSS's own default solve tolerance (1e-4 relative power mismatch) is
        # far looser than pgml's nonlinear solve (tol=1e-12 below); tighten it
        # so the comparison isolates the CONVERSION, not DSS's own residual.
        dss.Text.Command("Set Tolerance=1e-13")
        dss.Text.Command("Set maxiterations=1000")
        dss.Text.Command("Solve")
        assert dss.Solution.Converged(), "DSS permuted-line circuit did not converge"

        request.cls._grid, request.cls._id_map, request.cls._result = _solve_pgml()

    def test_line_carries_permuted_to_phases(self) -> None:
        """The converted Line keeps a's canonical order but b's reversed one."""
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        assert line.from_phases == ABC
        assert line.to_phases == (Phase.C, Phase.B, Phase.A)

    def test_node_voltages_match_opendss(self) -> None:
        _compare_all_buses(self._id_map, self._result, ABC)

    def test_unbalanced_solve_is_actually_unbalanced(self) -> None:
        """Sanity check: the loads really do produce distinct per-phase voltages
        (otherwise a mis-wired permutation could pass by coincidence)."""
        dss.Circuit.SetActiveBus("b")
        vm = dss.Bus.puVmagAngle()[0::2]
        assert max(vm) - min(vm) > 1.0e-3, (
            f"expected a meaningfully unbalanced solve, got per-phase |V| pu {vm}"
        )


# ---------------------------------------------------------------------------
# Single-phase permutation: bus1=a.1 bus2=b.2
# ---------------------------------------------------------------------------


class TestSinglePhasePermutedLine:
    """A 1-phase lateral tapped off phase A but landing on bus b's phase B."""

    @pytest.fixture(autouse=True, scope="class")
    def _circuit(self, request) -> None:
        _dss_clear()
        dss.Text.Command(
            f"New Circuit.perm1 basekv={_BASEKV} pu=1.0 phases=3 bus1=a "
            f"angle=0.0 frequency={_F0} r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6"
        )
        dss.Text.Command(
            "New Line.l1 phases=1 bus1=a.1 bus2=b.2 r1=0.3 x1=0.6 length=1 units=km"
        )
        dss.Text.Command(
            f"New Load.lb phases=1 bus1=b.2 kv={_BASEKV / math.sqrt(3.0)} "
            "kw=40 kvar=15 model=1"
        )
        dss.Text.Command(f"Set voltagebases=[{_BASEKV}]")
        dss.Text.Command("Calcvoltagebases")
        # DSS's own default solve tolerance (1e-4 relative power mismatch) is
        # far looser than pgml's nonlinear solve (tol=1e-12 below); tighten it
        # so the comparison isolates the CONVERSION, not DSS's own residual.
        dss.Text.Command("Set Tolerance=1e-13")
        dss.Text.Command("Set maxiterations=1000")
        dss.Text.Command("Solve")
        assert dss.Solution.Converged(), (
            "DSS 1-phase permuted-line circuit did not converge"
        )

        request.cls._grid, request.cls._id_map, request.cls._result = _solve_pgml()

    def test_line_carries_permuted_to_phase(self) -> None:
        line = next(b for b in self._grid.branches if isinstance(b, Line))
        assert line.from_phases == (Phase.A,)
        assert line.to_phases == (Phase.B,)

    def test_bus_b_registers_only_phase_b(self) -> None:
        node_b = next(n for n in self._grid.nodes if n.name == "b")
        assert node_b.phases == (Phase.B,)

    def test_node_voltages_match_opendss(self) -> None:
        _compare_all_buses(
            {
                "bus": {
                    name: nid
                    for name, nid in self._id_map["bus"].items()
                    if name == "a"
                }
            },
            self._result,
            (Phase.A,),
        )
        _compare_all_buses(
            {
                "bus": {
                    name: nid
                    for name, nid in self._id_map["bus"].items()
                    if name == "b"
                }
            },
            self._result,
            (Phase.B,),
        )
