"""Oracle test: transformer winding-resistance frequency law vs live OpenDSS.

OpenDSS's ``Transformer.XRConst`` selects how the leakage impedance scales with
frequency. Measured on its own ``Yprim`` (20/0.4 kV, 400 kVA, ``XHL = 4 %``,
``%R = 0.5`` per winding, so ``Z = 500 + j2000`` Ohm referred to the HV side at 50 Hz):

=========  ==========================  ==========================
order      ``XRConst=No`` (default)    ``XRConst=Yes``
=========  ==========================  ==========================
1          ``500 + j2000``, X/R = 4    ``500 + j2000``, X/R = 4
5          ``500 + j10000``, X/R = 20  ``2500 + j10000``, X/R = 4
13         ``500 + j26000``, X/R = 52  ``6500 + j26000``, X/R = 4
=========  ==========================  ==========================

So ``XRConst=Yes`` scales R with the harmonic order (holding X/R constant) and
``XRConst=No`` keeps R at its fundamental value. ``Transformer.harmonic_xr_constant``
carries exactly that flag (the OpenDSS converter already reads it), and the stamp now
applies it, composed with the shared ``ResistanceFrequencyModel`` multiplier
(``R(f) = R · m(f) · (f/f0 if harmonic_xr_constant else 1)``).

This test compares pgml's assembled transformer primitive against OpenDSS's own
``Yprim`` at orders 1, 5 and 13 for both settings. The pgml grid is assembled WITHOUT
its source so the 6x6 block is the transformer primitive alone; OpenDSS's 8x8 ``Yprim``
is reduced to the same six phase conductors by dropping its two grounded-neutral rows.

Measured agreement: 1.25e-6 S absolute at every order and both settings (2e-8 to 2.7e-7
relative). The residual is a constant additive offset, OpenDSS's ``ppm_Antifloat``
anti-float shunt on the winding neutrals, not a frequency-law difference; ignoring
``harmonic_xr_constant`` instead moves the same entries by ~2e-2 S at order 13.

Tolerance: ``atol = 1e-5 S`` (8x the anti-float offset, 1000x below the error of
ignoring the flag).
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
from pgml.evaluation.oracles.opendss_scenario_oracle import (  # noqa: E402
    _scratch_datapath,
)
from pgml.schemas.grid_schema import Grid, Phase, Transformer  # noqa: E402

pytestmark = pytest.mark.opendss

_F0 = 50.0
_ABC = (Phase.A, Phase.B, Phase.C)
_ORDERS = (1, 5, 13)
_ATOL_S = 1.0e-5


def _build_circuit(xrconst: str) -> None:
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.xrc basekv=20 pu=1.0 phases=3 bus1=hv angle=0
    New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3] conns=[wye wye]
    ~ kvs=[20 0.4] kvas=[400 400] XHL=4 %Rs=[0.5 0.5] %noloadloss=0 %imag=0 XRConst={xrconst}
    New Load.l1 bus1=lv.1.2.3 phases=3 kv=0.4 kw=100 pf=1 model=1 conn=wye
    Set VoltageBases=[20, 0.4]
    CalcVoltageBases
    Set Mode=Snap
    Solve
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())
    assert dss.Solution.Converged()


def _dss_yprim_phase_block() -> np.ndarray:
    """The transformer ``Yprim`` reduced to its six PHASE conductors [S]."""
    dss.Circuit.SetActiveElement("Transformer.t1")
    raw = np.array(dss.CktElement.YPrim())
    n = int(round(math.sqrt(len(raw) / 2)))
    y = (raw[0::2] + 1j * raw[1::2]).reshape(n, n)
    order = dss.CktElement.NodeOrder()
    keep = [i for i, node in enumerate(order) if node != 0]
    return y[np.ix_(keep, keep)]


def _pgml_block(grid, id_map, orders) -> np.ndarray:
    """pgml's transformer primitive ``[H, 6, 6]`` (source excluded) [S]."""
    passive = Grid(
        base_frequency_hz=_F0,
        nodes=grid.nodes,
        branches=grid.branches,
        appliances=[],
    )
    f = torch.tensor([h * _F0 for h in orders], dtype=torch.float64)
    yb = assemble_network_ybus(passive, f, dtype=torch.complex128)
    rows = [yb.index.row(id_map["bus"]["hv"], p) for p in _ABC]
    rows += [yb.index.row(id_map["bus"]["lv"], p) for p in _ABC]
    return yb.Y.numpy()[:, np.ix_(rows, rows)[0], np.ix_(rows, rows)[1]]


def _dss_blocks_per_order() -> dict[int, np.ndarray]:
    out = {1: _dss_yprim_phase_block()}
    with _scratch_datapath():
        dss.Text.Command("Set Mode=Harmonics")
        for h in _ORDERS[1:]:
            dss.Text.Command(f"Set Harmonic={h:g}")
            dss.Text.Command("Solve")
            out[h] = _dss_yprim_phase_block()
        dss.Text.Command("Set Mode=Snap")
        dss.Text.Command("Solve")
    return out


@pytest.mark.parametrize(("xrconst", "expected_flag"), [("No", False), ("Yes", True)])
def test_yprim_matches_opendss_at_every_order(xrconst, expected_flag) -> None:
    _build_circuit(xrconst)
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
    assert bool(xfmr.harmonic_xr_constant) is expected_flag
    y_dss = _dss_blocks_per_order()
    y_pgml = _pgml_block(grid, id_map, _ORDERS)
    for k, h in enumerate(_ORDERS):
        dev = np.abs(y_dss[h] - y_pgml[k]).max()
        assert dev < _ATOL_S, f"XRConst={xrconst}, h={h}: {dev:.3e} S"


def test_ignoring_the_flag_is_visible_at_high_order() -> None:
    """Dropping ``harmonic_xr_constant`` misses OpenDSS by 1000x the tolerance."""
    _build_circuit("Yes")
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    y_dss = _dss_blocks_per_order()
    for branch in grid.branches:
        if isinstance(branch, Transformer):
            branch.harmonic_xr_constant = False
    y_pgml = _pgml_block(grid, id_map, _ORDERS)
    dev_h13 = np.abs(y_dss[13] - y_pgml[_ORDERS.index(13)]).max()
    assert dev_h13 > 1000.0 * _ATOL_S, f"{dev_h13:.3e} S"


def test_xr_ratio_is_held_constant() -> None:
    """The stamped impedance has the same X/R at every order under XRConst=Yes."""
    _build_circuit("Yes")
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    y_pgml = _pgml_block(grid, id_map, _ORDERS)
    ratios = []
    for k in range(len(_ORDERS)):
        # Coupling entry of phase a: -y_se/tau -> its reciprocal carries X/R.
        z = 1.0 / y_pgml[k][0, 3]
        ratios.append(z.imag / z.real)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-9)
    assert ratios[0] == pytest.approx(ratios[2], rel=1e-9)
