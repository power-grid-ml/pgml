"""Oracle test: where the magnetizing (core) shunt is attached, vs live OpenDSS.

The three reference engines attach the magnetizing branch of a two-winding transformer
in three different places, which is a TOPOLOGY difference (whether the magnetizing
current sees a winding's leakage drop), not a referral difference:

- OpenDSS puts the whole branch on the LAST winding's terminal, on that winding's base.
  Measured on a live ``Yprim`` difference (branch on minus branch off): the contribution
  appears ONLY in the winding-2 diagonal, equal to ``(%noloadloss + j*(-%imag))/100 *
  S/u_wdg2^2``, with ``%imag`` taken as the susceptance directly (no Pythagorean
  subtraction of the loss component, and no coupling-block change). Swapping the winding
  order moves the contribution with winding 2, confirming it is the LAST winding.
- power-grid-model splits it half onto its ``Y_tt`` and half (through the tap) onto
  ``Y_ff``.
- pandapower keeps it on the LV base inside its pi shunt.

pgml makes the placement an explicit, documented modeling choice,
``transformer.magnetizing_placement`` (``split`` default / ``from_terminal`` /
``to_terminal``), and this test measures each against a live OpenDSS solve on a 500 kVA
20/0.4 kV unit (``XHL = 4 %``, ``%R = 0.5`` per winding, ``%noloadloss = 0.2``) at three
magnetizing currents and three loadings.

Measured max voltage deviation [pu of the bus line-to-neutral base], identical for a Dyn
and a YNyn unit, and essentially independent of loading (the branch is linear and the
terminal voltages barely move):

=========  =============  ==============  ==============  ==============
``%imag``  load           ``to_terminal``  ``split``       ``from_terminal``
=========  =============  ==============  ==============  ==============
0 (no branch) 0-500 kW    3.4e-10-1.8e-9  same            same
0.1 %      0 / 250 / 500  3.4e-10-1.8e-9  4.6e-5-4.7e-5   9.2e-5-9.5e-5
0.5 %      0 / 250 / 500  3.4e-10-1.8e-9  1.1e-4          2.2e-4
2.0 %      0 / 250 / 500  3.6e-10-1.8e-9  4.1e-4          8.2e-4
=========  =============  ==============  ==============  ==============

So ``to_terminal`` reproduces OpenDSS to the solver floor (below 1e-9 pu), and the
explicit ``from_terminal`` model has the documented deviation (~2e-4 pu at a realistic
0.5 % magnetizing current). The Vsource is made near-ideal so that pgml's
``slack="ideal"`` and the DSS source coincide (the stock 2000 MVA default would
otherwise add its own 2.5e-4 pu drop at rated load).

Tolerances: ``to_terminal`` asserted below 1e-6 pu; the other placements asserted to be
at least 10x the ``to_terminal`` residual, so a silent change of the stamp's placement
fails the suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import yaml

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml import defaults  # noqa: E402
from pgml.assembly._transformer import (  # noqa: E402
    MAGNETIZING_PLACEMENTS,
    magnetizing_placement,
)
from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.errors import ModelingError  # noqa: E402
from pgml.schemas.grid_schema import Phase, Transformer  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

pytestmark = pytest.mark.opendss

_F0 = 50.0
_KVA, _KV_HV, _KV_LV = 500.0, 20.0, 0.4
_NOLOAD_PCT = 0.2
_NODE_PHASE = {1: Phase.A, 2: Phase.B, 3: Phase.C}
_ATOL_PU = 1.0e-6


@pytest.fixture
def placement(tmp_path, monkeypatch):
    """Factory switching ``transformer.magnetizing_placement`` for one test."""
    base = yaml.safe_dump(defaults.defaults())

    def _set(value: str) -> None:
        data = yaml.safe_load(base)
        data["transformer"]["magnetizing_placement"]["value"] = value
        path = tmp_path / f"defaults_{value}.yaml"
        path.write_text(yaml.safe_dump(data))
        monkeypatch.setenv("PGML_DEFAULTS", str(path))
        defaults.reload(str(path))
        assert magnetizing_placement() == value

    yield _set
    monkeypatch.delenv("PGML_DEFAULTS", raising=False)
    defaults.reload()


def _build_circuit(conns: str, leadlag: str, imag_pct: float, load_kw: float) -> None:
    script = f"""
    Clear
    Set DefaultBaseFrequency={_F0:g}
    New Circuit.mag basekv={_KV_HV:g} pu=1.0 phases=3 bus1=hv angle=0
    Edit Vsource.source R1=1e-6 X1=1e-6 R0=1e-6 X0=1e-6
    New Transformer.t1 phases=3 windings=2 buses=[hv.1.2.3, lv.1.2.3] conns=[{conns}]
    ~ kvs=[{_KV_HV:g} {_KV_LV:g}] kvas=[{_KVA:g} {_KVA:g}] XHL=4 %Rs=[0.5 0.5]
    ~ %noloadloss={_NOLOAD_PCT:g} %imag={imag_pct:g} LeadLag={leadlag} ppm_antifloat=0
    New Load.l1 bus1=lv.1.2.3 phases=3 kv={_KV_LV:g} kw={load_kw:g} pf=1 model=1 conn=wye Vminpu=0.0001 Vmaxpu=10000
    Set VoltageBases=[{_KV_HV:g}, {_KV_LV:g}]
    CalcVoltageBases
    Set Tolerance=1e-10
    Set MaxIterations=100
    Solve
    """
    for line in script.strip().splitlines():
        dss.Text.Command(line.strip())
    assert dss.Solution.Converged()


def _deviation_pu() -> float:
    """Max |V_pgml - V_dss| over every bus phase, in pu of the bus L-N base."""
    grid, id_map = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    res = solve_power_flow(
        grid, slack="ideal", tol=1e-13, max_iter=300, dtype=torch.complex128
    )
    assert res.converged
    worst = 0.0
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v = np.array(dss.Bus.Voltages())
        per_node = dict(zip(list(dss.Bus.Nodes()), v[0::2] + 1j * v[1::2]))
        node_id = id_map["bus"][bus]
        v_base = (_KV_HV if bus == "hv" else _KV_LV) * 1000.0 / math.sqrt(3.0)
        for dss_node, v_ref in per_node.items():
            row = res.index.row(node_id, _NODE_PHASE[dss_node])
            worst = max(worst, abs(complex(res.v[row]) - v_ref) / v_base)
    return worst


@pytest.mark.parametrize(
    ("conns", "leadlag"), [("delta wye", "Lead"), ("wye wye", "Lag")]
)
@pytest.mark.parametrize("imag_pct", [0.1, 0.5, 2.0])
@pytest.mark.parametrize("load_kw", [0.0, 500.0])
def test_to_terminal_reproduces_opendss(
    placement, conns, leadlag, imag_pct, load_kw
) -> None:
    """OpenDSS's own placement: agreement at the solver floor, below 1e-6 pu."""
    _build_circuit(conns, leadlag, imag_pct, load_kw)
    placement("to_terminal")
    dev = _deviation_pu()
    assert dev < _ATOL_PU, f"{conns} i0={imag_pct}% {load_kw} kW: {dev:.3e} pu"


@pytest.mark.parametrize("imag_pct", [0.1, 0.5, 2.0])
def test_other_placements_are_the_documented_deviation(placement, imag_pct) -> None:
    """``from_terminal`` (default) and ``split`` deviate, and ``split`` sits halfway."""
    _build_circuit("delta wye", "Lead", imag_pct, 250.0)
    placement("to_terminal")
    dev_to = _deviation_pu()
    placement("from_terminal")
    dev_from = _deviation_pu()
    placement("split")
    dev_split = _deviation_pu()
    assert dev_from > 10.0 * max(dev_to, 1e-9)
    assert dev_split == pytest.approx(0.5 * dev_from, rel=0.05)


def test_magnetizing_value_matches_the_opendss_definition() -> None:
    """``%noloadloss``/``%imag`` map to G/B directly (no Pythagorean subtraction).

    The case deliberately uses ``%imag`` BELOW ``%noloadloss``, which the
    total-no-load-current reading (pandapower's ``i0_percent``) cannot represent at all:
    it would take the square root of a negative number and drop the branch.
    """
    _build_circuit("delta wye", "Lead", 0.1, 0.0)
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    xfmr = next(b for b in grid.branches if isinstance(b, Transformer))
    u_hv = _KV_HV * 1000.0
    s_rated = _KVA * 1000.0
    assert float(xfmr.magnetizing_conductance_s) == pytest.approx(
        _NOLOAD_PCT / 100.0 * s_rated / u_hv**2, rel=1e-12
    )
    b_m = 1.0 / (2.0 * math.pi * _F0 * float(xfmr.magnetizing_inductance_h))
    assert b_m == pytest.approx(0.1 / 100.0 * s_rated / u_hv**2, rel=1e-12)


def test_unknown_placement_raises(tmp_path, monkeypatch) -> None:
    """An unrecognised placement name fails loud instead of silently defaulting."""
    data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
    data["transformer"]["magnetizing_placement"]["value"] = "internal_t"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PGML_DEFAULTS", str(path))
    try:
        defaults.reload(str(path))
        with pytest.raises(ModelingError, match="magnetizing_placement"):
            magnetizing_placement()
    finally:
        monkeypatch.delenv("PGML_DEFAULTS", raising=False)
        defaults.reload()
    assert magnetizing_placement() in MAGNETIZING_PLACEMENTS
