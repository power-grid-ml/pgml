"""Oracle test: branch currents against live OpenDSS element currents.

OpenDSS solves the three-phase circuit and reports each element's phase currents
(``CktElement.CurrentsMagAng``, amperes, one value per conductor). The default
``SINGLE_PHASE_EQUIV`` converter output is one line-to-neutral phase of the same system, so
its branch current is that conductor current; ``THREE_PHASE`` carries all three phases.

Circuit: 20 kV Vsource (Thevenin source, solved with ``slack="norton"``) -> Dyn 20/0.4 kV
transformer (no magnetizing branch) -> 0.4 kV line (R, X, C) -> balanced constant-power load.

Alignment: the pgml branch is matched by element name through ``id_map`` and its first-terminal
current is compared with the OpenDSS terminal-1 phase-A magnitude (all three phases are equal
in this balanced circuit, asserted for OpenDSS); the transformer is compared on both windings
(``CurrentsMagAng`` lists terminal 1 then terminal 2). The ``opendss`` modeling preset selects
OpenDSS's magnetizing placement.

Tolerance: relative 2e-6 on every current. Measured (CPU, complex128, OpenDSS via
opendssdirect, both phase modes): transformer 6.2e-7, line 2.8e-8. The residual is OpenDSS's
iterative power-flow tolerance (it solves a constant-power load by fixed-point iteration), not a
scaling effect: a sqrt(3) error would show as 0.73 / 1.73.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml import defaults  # noqa: E402
from pgml.assembly import branch_currents, device_current_injections  # noqa: E402
from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

pytestmark = [pytest.mark.opendss, pytest.mark.usefixtures("opendss_model_defaults")]

_F0 = 50.0
CDT = torch.complex128
RTOL = 2.0e-6


def _build_circuit() -> None:
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.cur basekv=20 pu=1.01 phases=3 bus1=hv angle=0
    Edit Vsource.source R1=0.05 X1=0.4 R0=0.05 X0=0.4
    New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3.0] conns=[delta wye]
    ~ kvs=[20 0.4] kvas=[630 630] XHL=4 %Rs=[0.5 0.5] %noloadloss=0 %imag=0
    New Line.l1 bus1=lv.1.2.3 bus2=end.1.2.3 phases=3 r1=0.2 x1=0.08 c1=250 r0=0.6 x0=0.24 c0=250 length=0.35 units=km
    New Load.ld1 bus1=end.1.2.3 phases=3 kv=0.4 kw=230 kvar=70 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000
    Set VoltageBases=[20, 0.4]
    CalcVoltageBases
    Set Tolerance=1e-10
    Set MaxIterations=100
    Solve
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())
    assert dss.Solution.Converged()


def _dss_terminal_currents(element: str) -> np.ndarray:
    dss.Circuit.SetActiveElement(element)
    mag = np.array(dss.CktElement.CurrentsMagAng())[0::2]
    n = dss.CktElement.NumPhases()
    nterm = dss.CktElement.NumTerminals()
    per_terminal = mag.reshape(nterm, -1)[:, :n]
    np.testing.assert_allclose(
        per_terminal,
        np.broadcast_to(per_terminal[:, :1], per_terminal.shape),
        rtol=1e-7,
    )
    return per_terminal[:, 0]


@pytest.mark.parametrize("mode", list(PhaseMode), ids=lambda m: m.value)
def test_element_currents_match_opendss(mode) -> None:
    _build_circuit()
    ref = {
        "transformer": _dss_terminal_currents("Transformer.t1"),
        "line": _dss_terminal_currents("Line.l1"),
    }
    with defaults.use_preset("opendss"):
        grid, id_map = to_grid(dss, phase_mode=mode)
        res = solve_power_flow(grid, slack="norton", tol=1e-12, max_iter=200, dtype=CDT)
        assert res.converged
        i_inj = -device_current_injections(grid, res.v, res.index, [_F0], dtype=CDT)
        cur = {
            bc.branch_id: bc
            for bc in branch_currents(
                grid, res.v.unsqueeze(-2), [_F0], res.index, dtype=CDT, i_inj=i_inj
            )
        }
    for kind, key in (("transformer", "t1"), ("line", "l1")):
        bc = cur[id_map["trafo" if kind == "transformer" else "line"][key]]
        ours = np.array(
            [bc.i_from.abs().reshape(-1)[0].item(), bc.i_to.abs().reshape(-1)[0].item()]
        )
        assert np.max(np.abs(ours / ref[kind] - 1.0)) < RTOL, (kind, ours / ref[kind])
