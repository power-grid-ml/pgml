"""Oracle test: the zero-sequence-aware transformer leakage against live OpenDSS.

OpenDSS's two-winding ``Transformer`` has NO zero-sequence leakage input: it builds one
primitive from the per-winding ``%R`` and ``XHL`` and the winding connections, so its
zero-sequence impedance IS the positive-sequence one (the only zero-sequence levers are
``Rneut``/``Xneut``, a neutral earthing impedance pgml rejects rather than silently
flattening). That makes OpenDSS the reference for two statements:

1. pgml's DEFAULT (``transformer.zero_sequence.*`` = 1.0, i.e. Z0 = Z1) reproduces a
   live OpenDSS solve on a Dyn and a YNyn unit with a single-phase LV load, which is
   exactly the unbalanced case that excites the zero sequence.
2. The new per-phase leakage MATRIX path is a strict generalization: with an explicit
   ``zero_sequence`` equal to the positive-sequence pair it reproduces the scalar stamp
   (and therefore the OpenDSS parity) to 6.4e-14 V on an 11.5 kV / 231 V circuit, while a
   genuinely different Z0 moves the solution by a physically significant amount that
   OpenDSS cannot express at all.

Measured (this environment, float64/complex128, CPU, OpenDSS via opendssdirect 0.9.4),
20/0.4 kV 400 kVA, ``XHL = 4 %``, ``%R = 0.5`` per winding, no magnetizing branch,
40 kW on LV phase a plus 10 kW on phase b:

=====================  =====================  ====================================
vector group           scalar / matrix Z0=Z1  matrix path, Z0 = 0.4*Z1
=====================  =====================  ====================================
Dyn1  (delta / wye)    4.80e-6 V vs OpenDSS   0.521 V away from the Z0 = Z1 solution
YNyn0 (wye / wye)      4.80e-6 V vs OpenDSS   0.521 V away from the Z0 = Z1 solution
=====================  =====================  ====================================

The scalar and matrix paths agree to 6.4e-14 V, i.e. the generalization is exact.

Tolerances: ``atol = 1e-4 V`` against OpenDSS (20x the 4.8e-6 V residual, which is
OpenDSS's own snap-solve tolerance on an 11.5 kV / 231 V circuit), and ``1e-10 V``
between the scalar and matrix paths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    Phase,
    Transformer,
    TransformerZeroSeq,
)
from pgml.solver import solve_power_flow  # noqa: E402

pytestmark = [pytest.mark.opendss, pytest.mark.usefixtures("opendss_model_defaults")]

_F0 = 50.0
_ABC = (Phase.A, Phase.B, Phase.C)
_NODE_PHASE = {1: Phase.A, 2: Phase.B, 3: Phase.C, 4: Phase.N}
_ATOL_V = 1.0e-4
_ATOL_PATH_V = 1.0e-10


def _build_circuit(conn_hv: str, conn_lv: str, *, extra: str = "") -> None:
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.tz basekv=20 pu=1.0 phases=3 bus1=hv angle=0
    Edit Vsource.source R1=1e-6 X1=1e-6 R0=1e-6 X0=1e-6
    New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3] conns=[{conn_hv} {conn_lv}]
    ~ kvs=[20 0.4] kvas=[400 400] XHL=4 %Rs=[0.5 0.5] %noloadloss=0 %imag=0 {extra}
    New Load.l1 bus1=lv.1 phases=1 kv=0.231 kw=40 pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000
    New Load.l2 bus1=lv.2 phases=1 kv=0.231 kw=10 pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000
    Set VoltageBases=[20, 0.4]
    CalcVoltageBases
    Set Tolerance=1e-10
    Set MaxIterations=100
    Solve
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())
    assert dss.Solution.Converged(), "OpenDSS did not converge"


def _bus_voltages() -> dict[str, dict[int, complex]]:
    out: dict[str, dict[int, complex]] = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v = np.array(dss.Bus.Voltages())
        out[bus] = dict(zip(list(dss.Bus.Nodes()), v[0::2] + 1j * v[1::2]))
    return out


def _solve(grid) -> dict:
    res = solve_power_flow(
        grid, slack="ideal", tol=1e-13, max_iter=300, dtype=torch.complex128
    )
    assert res.converged
    return res


def _deviation(res, id_map, v_dss) -> float:
    worst = 0.0
    for bus, per_node in v_dss.items():
        node_id = id_map["bus"][bus]
        for dss_node, v_ref in per_node.items():
            row = res.index.row(node_id, _NODE_PHASE[dss_node])
            worst = max(worst, abs(complex(res.v[row]) - v_ref))
    return worst


def _with_zero_sequence(grid, factor: float):
    """Copy of ``grid`` whose transformer carries ``Z0 = factor * Z1`` explicitly."""
    out = grid.model_copy(deep=True)
    for branch in out.branches:
        if isinstance(branch, Transformer):
            r1 = float(branch.series_resistance_ohm)
            x1 = float(branch.series_inductance_h) * 2.0 * math.pi * _F0
            branch.zero_sequence = TransformerZeroSeq(
                r0_ohm=factor * r1, x0_ohm=factor * x1
            )
    return out


@pytest.mark.parametrize(
    ("conn_hv", "conn_lv", "label"),
    [("delta", "wye", "Dyn"), ("wye", "wye", "YNyn")],
)
class TestLiveOpenDSSParity:
    def test_default_matches_opendss(self, conn_hv, conn_lv, label) -> None:
        """Z0 = Z1 (the shipped default) is OpenDSS's own transformer model."""
        _build_circuit(conn_hv, conn_lv)
        v_dss = _bus_voltages()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        dev = _deviation(_solve(grid), id_map, v_dss)
        assert dev < _ATOL_V, f"{label}: {dev:.3e} V"

    def test_matrix_path_with_z0_equal_z1_is_the_scalar_path(
        self, conn_hv, conn_lv, label
    ) -> None:
        """An explicit Z0 == Z1 keeps both the OpenDSS parity and the scalar solution."""
        _build_circuit(conn_hv, conn_lv)
        v_dss = _bus_voltages()
        grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        res_scalar = _solve(grid)
        res_matrix = _solve(_with_zero_sequence(grid, 1.0))
        assert _deviation(res_matrix, id_map, v_dss) < _ATOL_V
        assert (
            torch.max(torch.abs(res_matrix.v - res_scalar.v)).item() < _ATOL_PATH_V
        ), f"{label}: matrix and scalar leakage paths disagree"

    def test_a_different_z0_moves_the_solution(self, conn_hv, conn_lv, label) -> None:
        """The knob has authority: a three-limb-like Z0 = 0.4*Z1 is far outside atol."""
        _build_circuit(conn_hv, conn_lv)
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        res_ref = _solve(grid)
        res_z0 = _solve(_with_zero_sequence(grid, 0.4))
        shift = torch.max(torch.abs(res_z0.v - res_ref.v)).item()
        assert shift > 1000.0 * _ATOL_V, f"{label}: {shift:.3e} V"


class TestNeutralEarthingImpedance:
    """A winding neutral earthing impedance is refused, not silently flattened."""

    def test_rneut_raises(self) -> None:
        with pytest.raises(ConversionError, match="neutral earthing impedance"):
            _build_circuit("wye", "wye", extra="wdg=2 Rneut=5 Xneut=2")
            to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)

    def test_solidly_grounded_neutral_converts(self) -> None:
        """An explicit Rneut=0/Xneut=0 is solid grounding and must still convert."""
        _build_circuit("wye", "wye", extra="wdg=2 Rneut=0 Xneut=0")
        grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
        assert any(isinstance(b, Transformer) for b in grid.branches)
