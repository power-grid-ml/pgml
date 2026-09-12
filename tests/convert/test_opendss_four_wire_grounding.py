"""Four-wire neutral grounding on an OpenDSS import: where the ground tie comes from.

pgml's compact node-phase index gives a four-wire bus its own ``Phase.N`` ROW. Nothing in
assembly ties that row to the ground reference implicitly: a WYE appliance's return
current flows INTO it (``return_path``), a four-conductor line carries it as a series
conductor, and the transformer stamp only touches the phases in ``to_phases``. The ground
tie must therefore come from a converted grounding element, which is exactly how the DSS
circuit expresses it (``New Reactor.ng phases=1 bus1=bus.4``, whose implicit second
terminal is node 0 = ground).

This file pins the three cases that matter on an import:

1. A grounding ``Reactor`` on the neutral conductor converts to a WYE
   ``ShuntAppliance`` on ``Phase.N``, and the solved neutral voltages reproduce a live
   OpenDSS solve at every bus of a three-bus feeder whose neutral is grounded at the
   source end only (measured 9e-12 / 2.6e-10 / 5.2e-10 V on a 231 V base).
2. A transformer whose LV winding names an explicit fourth conductor (the usual way a DSS
   model grounds a four-wire feeder INSIDE the transformer) raises ``ConversionError``:
   pgml stamps windings solidly grounded to the reference, so that wiring would silently
   move the tie.
3. A four-wire feeder with NO grounding element anywhere leaves the ``Phase.N`` rows
   without a path to the reference. The const-power solve's effective admittance (the
   passive network plus the source Norton shunt) is then rank-deficient — measured rank 8
   of 12 on the three-bus feeder, against 9 of 12 with the tie — and the solve does not
   return a usable answer: it reports ``converged=False`` with a diverged voltage update
   (measured 4.9e16 V, i.e. 1.2e14 per unit, at a power mismatch of 1.3e-2 pu), or raises
   out of the factorization when the singular structure produces an exact zero pivot. pgml has no anti-float stabiliser (OpenDSS adds a
   ``ppm_antifloat`` shunt to every transformer winding and then answers with every node
   at nominal voltage), so such a grid must carry an explicit ground tie. A pgml-level
   diagnostic naming the unreferenced rows would be friendlier than either failure mode.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.assembly import assemble_network_ybus  # noqa: E402
from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    Phase,
    ShuntAppliance,
    WindingConnection,
)
from pgml.solver import solve_power_flow  # noqa: E402

pytestmark = pytest.mark.opendss

_F0 = 50.0
_ATOL_V = 1.0e-6

_LINECODE = """
New Linecode.lv4 nphases=4 units=km Rg=0 Xg=0
~ rmatrix=[0.32 | 0.04 0.32 | 0.04 0.04 0.32 | 0.04 0.04 0.04 0.32]
~ xmatrix=[0.08 | 0.03 0.08 | 0.03 0.03 0.08 | 0.03 0.03 0.03 0.08]
~ cmatrix=[0 | 0 0 | 0 0 0 | 0 0 0 0]
"""


def _run(script: str) -> None:
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())


def _four_wire_feeder(*, ground_reactor: bool) -> None:
    """Three-bus four-wire LV feeder fed directly by a Vsource with Z0 != Z1."""
    grounding = (
        "New Reactor.ng phases=1 bus1=b1.4 R=0.001 X=0" if ground_reactor else ""
    )
    _run(f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.fw basekv=0.4 pu=1.0 phases=3 bus1=b1 angle=0
    Edit Vsource.source R1=0.01 X1=0.04 R0=0.05 X0=0.3
    {_LINECODE}
    New Line.l1 bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 linecode=lv4 length=0.1 units=km Rg=0 Xg=0
    New Line.l2 bus1=b2.1.2.3.4 bus2=b3.1.2.3.4 linecode=lv4 length=0.1 units=km Rg=0 Xg=0
    {grounding}
    New Load.la bus1=b3.1.4 phases=1 kv=0.231 kw=6 pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000
    Set VoltageBases=[0.4]
    CalcVoltageBases
    Set Tolerance=1e-10
    Set MaxIterations=100
    Solve
    """)


def _dss_neutral_voltages() -> dict[str, complex]:
    out = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v = np.array(dss.Bus.Voltages())
        per_node = dict(zip(list(dss.Bus.Nodes()), v[0::2] + 1j * v[1::2]))
        if 4 in per_node:
            out[bus] = per_node[4]
    return out


class TestGroundingReactorIsTheTie:
    def test_reactor_converts_to_a_neutral_shunt(self) -> None:
        _four_wire_feeder(ground_reactor=True)
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        ties = [
            a
            for a in grid.appliances
            if isinstance(a, ShuntAppliance) and a.phases == (Phase.N,)
        ]
        assert len(ties) == 1
        assert ties[0].connection is WindingConnection.WYE
        # Every four-wire bus carries a Phase.N row; only one of them is tied to ground.
        n_nodes = [nd.id for nd in grid.nodes if Phase.N in nd.phases]
        assert len(n_nodes) == 3
        assert ties[0].node in n_nodes

    def test_neutral_voltages_match_live_opendss(self) -> None:
        _four_wire_feeder(ground_reactor=True)
        v_dss = _dss_neutral_voltages()
        assert len(v_dss) == 3
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        res = solve_power_flow(
            grid, slack="norton", tol=1e-13, max_iter=300, dtype=torch.complex128
        )
        assert res.converged
        for bus, v_ref in v_dss.items():
            row = res.index.row(id_map["bus"][bus], Phase.N)
            assert complex(res.v[row]) == pytest.approx(v_ref, abs=_ATOL_V)
        # The case must actually carry a neutral displacement, else it proves nothing.
        assert max(abs(v) for v in v_dss.values()) > 0.5


class TestTransformerNeutralWiring:
    def test_explicit_fourth_conductor_raises(self) -> None:
        """A four-wire LV winding (neutral on `lv.4`) is refused, not re-grounded."""
        _run(f"""
        Clear
        Set DefaultBaseFrequency={_F0:g}
        New Circuit.fwt basekv=20 pu=1.0 phases=3 bus1=hv angle=0
        Edit Vsource.source R1=1e-6 X1=1e-6 R0=1e-6 X0=1e-6
        New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3.4] conns=[delta wye]
        ~ kvs=[20 0.4] kvas=[400 400] XHL=4 %Rs=[0.5 0.5] LeadLag=Lead
        New Reactor.ng phases=1 bus1=lv.4 R=0.001 X=0
        New Load.la bus1=lv.1.4 phases=1 kv=0.231 kw=6 pf=1 model=1 conn=wye
        Set VoltageBases=[20, 0.4]
        CalcVoltageBases
        Solve
        """)
        with pytest.raises(ConversionError, match="neutral node"):
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)


class TestUngroundedNeutralFailsLoud:
    @staticmethod
    def _network_rank(grid) -> tuple[int, int]:
        yb = assemble_network_ybus(grid, [_F0], dtype=torch.complex128)
        y = yb.Y[0] if yb.Y.dim() == 3 else yb.Y
        return int(torch.linalg.matrix_rank(y, rtol=1e-10)), int(y.shape[-1])

    def test_no_grounding_element_is_more_rank_deficient(self) -> None:
        """The missing ground tie shows up as a lower rank of the passive network."""
        _four_wire_feeder(ground_reactor=False)
        grid_free, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert not [a for a in grid_free.appliances if isinstance(a, ShuntAppliance)]
        rank_free, n = self._network_rank(grid_free)

        _four_wire_feeder(ground_reactor=True)
        grid_tied, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        rank_tied, n_tied = self._network_rank(grid_tied)

        assert (n, n_tied) == (12, 12)
        assert rank_free == 8
        assert rank_tied == rank_free + 1

    def test_solve_does_not_return_a_silent_result(self) -> None:
        """No silent answer: the const-power solve reports failure (or raises).

        Which of the two happens depends on the singular structure: an exact zero pivot
        raises out of the factorization, while a merely unreferenced block (this feeder)
        reports ``converged=False`` with both convergence criteria far above their
        tolerances. Measured on this feeder: a per-unit power mismatch of 1.3e-2 pu
        against the 1e-13 pu requested, and a voltage update of 1.2e14 pu (4.9e16 V).
        OpenDSS answers the same circuit with every node at nominal voltage, because its
        anti-float shunt references the floating neutral and the load then carries no
        current.
        """
        _four_wire_feeder(ground_reactor=False)
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        try:
            res = solve_power_flow(
                grid, slack="norton", tol=1e-13, max_iter=50, dtype=torch.complex128
            )
        except RuntimeError:
            return
        assert not res.converged
        # `residual` is the per-unit power mismatch; 1e-4 pu is nine orders above the
        # requested tolerance and far outside any converged solve on this feeder.
        assert float(res.residual) > 1.0e-4
        assert float(res.diagnostics.update_max_pu) > 1.0


def test_phase_voltages_are_unaffected_by_the_neutral_row_count() -> None:
    """Sanity: the phase rows still match OpenDSS on the four-wire import."""
    _four_wire_feeder(ground_reactor=True)
    v_ref = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v = np.array(dss.Bus.Voltages())
        v_ref[bus] = dict(zip(list(dss.Bus.Nodes()), v[0::2] + 1j * v[1::2]))
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    res = solve_power_flow(
        grid, slack="norton", tol=1e-13, max_iter=300, dtype=torch.complex128
    )
    phase_of = {1: Phase.A, 2: Phase.B, 3: Phase.C}
    worst = 0.0
    for bus, per_node in v_ref.items():
        for dss_node, phase in phase_of.items():
            row = res.index.row(id_map["bus"][bus], phase)
            worst = max(worst, abs(complex(res.v[row]) - per_node[dss_node]))
    assert worst < _ATOL_V, f"max |dV| = {worst:.3e} V"
    assert math.isfinite(worst)
