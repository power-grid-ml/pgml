"""Phase-to-phase (two-phase delta) appliances and split-phase node ratings vs OpenDSS.

Two circuits from ``test_opendss_centre_tap_grounding``, solved by pgml and by live
OpenDSS (``opendssdirect``) at the fundamental (60 Hz) and at harmonic order 3:

1. The 50 kVA centre-tapped unit (7.967 kV phase-to-earth primary behind a
   ``10 + j2`` Ohm electrode, centre tap earthed through 5 Ohm, 440/220 V secondary).
   Added: a 440 V load across X1-X2 (OpenDSS ``Load phases=1 conn=delta`` on
   ``lv2.1.2``, pgml ``Load(phases=(A, B), connection=DELTA)``), in every load model
   (constant P/Q, Z, I and ZIP). A variant puts the two 220 V half loads into the
   constant-impedance model as well, which only agrees once the split-phase nodes carry
   ``u_rated_ln_v = 220`` (``u_rated_v/√3`` = 254 V would make their admittance 25 % too small).
2. The 400 kVA Dyn1 unit with its LV star on the neutral row (earthed through 5 Ohm)
   and a four-wire line, with an A-B delta load at 400 V added to the unbalanced wye
   loads.

The comparison metric is the largest complex deviation over every node row in pu of
the row's nominal phase-to-neutral voltage, and over every transformer terminal and
earth current in A. At order 3 the loads are pure harmonic current sources
(``NeglectLoadY=Yes``, pgml ``load_shunt="none"``); a further case keeps OpenDSS's own
harmonic load admittance (``%SeriesRL=50``, ``NeglectLoadY=No``) against pgml's
``load_shunt="opendss"`` with the same series fraction, which covers the harmonic
load shunt of the phase-to-phase element. Lines are lumped R/X with ``X ~ h`` and
``Rg=Xg=0``.

Measured (complex128, CPU, opendssdirect 0.9.4, ``Tolerance=1e-12``) over all cases:
fundamental voltages 3.4e-10 pu in every case (the fundamental iteration of the two
tools), order-3 voltages <= 9.2e-12 pu, transformer and earth currents <= 2.9e-8 A; the
harmonic-shunt case 3.4e-10 pu (h1) and 3.1e-12 pu (h3). The constant-impedance half
loads WITHOUT ``u_rated_ln_v`` deviate by 3.3e-3 pu (h1), 1.5e-3 pu (h3) and 9.0 A
(asserted only as a deviation; the reason the field exists). Bounds asserted: 1e-8 pu
(h1), 1e-9 pu (h3), 1e-6 A.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("opendssdirect")

from pgml.assembly import node_phase_index  # noqa: E402
from pgml.evaluation.oracles.opendss_scenario_oracle import (  # noqa: E402
    _scratch_datapath,
)
from pgml.schemas.grid_schema import (  # noqa: E402
    HarmonicShuntModel,
    Load,
    LoadModel,
    Phase,
    ZipCoefficients,
)
from pgml.solver import solve_harmonic_flow  # noqa: E402
from tests.reference.test_opendss_centre_tap_grounding import (  # noqa: E402
    SPECTRUM,
    _centre_tap_case,
    _compare,
    _dyn_case,
)

pytestmark = [pytest.mark.opendss, pytest.mark.usefixtures("opendss_model_defaults")]

P = Phase
_ATOL_V1_PU = 1e-8
_ATOL_V3_PU = 1e-9
_ATOL_I_A = 1e-6

#: pgml load model -> (OpenDSS ``model``, extra DSS properties, pgml ZIP block)
_MODELS = {
    LoadModel.CONST_POWER: (1, "", None),
    LoadModel.CONST_IMPEDANCE: (2, "", None),
    LoadModel.CONST_CURRENT: (5, "", None),
    LoadModel.ZIP: (
        8,
        " ZIPV=[0.3 0.3 0.4 0.5 0.2 0.3 0]",
        ZipCoefficients(z_p=0.3, i_p=0.3, p_p=0.4, z_q=0.5, i_q=0.2, p_q=0.3),
    ),
}


def _delta_load(aid, node, phases, kv, kw, kvar, model, *, name, bus, extra=""):
    """The DSS ``Load`` line and the pgml :class:`Load` of one phase-to-phase load."""
    dss_model, dss_extra, zip_c = _MODELS[model]
    line = (
        f"New Load.{name} phases=1 conn=delta bus1={bus} kv={kv} kw={kw} "
        f"kvar={kvar} model={dss_model} vminpu=0.1 vmaxpu=10 vlowpu=0.05 "
        f"spectrum=s3{dss_extra}{extra}"
    )
    load = Load(
        id=aid,
        node=node,
        phases=phases,
        connection="delta",
        p_nom_w=kw * 1e3,
        q_nom_var=kvar * 1e3,
        load_model=model,
        zip_coefficients=zip_c,
        spectrum=SPECTRUM,
    )
    return line, load


def _with_ln(grid, nodes, u_ln):
    """``grid`` with ``u_rated_ln_v = u_ln`` on ``nodes``."""
    new_nodes = [
        n.model_copy(update={"u_rated_ln_v": u_ln}) if n.id in nodes else n
        for n in grid.nodes
    ]
    return grid.model_copy(update={"nodes": new_nodes})


def _centre_tap_with_delta(model, *, halves_const_z=False, split_rating=True):
    lines, grid, rows, earth, terms = _centre_tap_case("swer", 5.0, True)
    line, load = _delta_load(
        30, 3, (P.A, P.B), 0.44, 6.0, 2.0, model, name="ll", bus="lv2.1.2"
    )
    lines = lines + [line]
    appliances = list(grid.appliances) + [load]
    if halves_const_z:
        lines = [
            ln.replace("model=1", "model=2") if ln.startswith("New Load.l") else ln
            for ln in lines
        ]
        appliances = [
            a.model_copy(update={"load_model": LoadModel.CONST_IMPEDANCE})
            if isinstance(a, Load) and len(a.phases) == 1
            else a
            for a in appliances
        ]
    grid = grid.model_copy(update={"appliances": appliances})
    if split_rating:
        grid = _with_ln(grid, {2, 3}, 220.0)
    return lines, grid, rows, earth, terms


def _dyn_with_delta(model):
    lines, grid, rows, earth, terms = _dyn_case(5.0)
    line, load = _delta_load(
        30, 3, (P.A, P.B), 0.4, 12.0, 4.0, model, name="lab", bus="lv2.1.2"
    )
    grid = grid.model_copy(update={"appliances": list(grid.appliances) + [load]})
    return lines + [line], grid, rows, earth, terms


@pytest.mark.parametrize("model", list(_MODELS))
def test_delta_load_across_split_phase_secondary_matches_opendss(model):
    worst = _compare(_centre_tap_with_delta(model), "Transformer.ct")
    assert worst["v1"] < _ATOL_V1_PU, worst
    assert worst["v3"] < _ATOL_V3_PU, worst
    assert worst["i"] < _ATOL_I_A, worst


def test_const_z_half_loads_match_opendss_with_the_split_phase_rating():
    worst = _compare(
        _centre_tap_with_delta(LoadModel.CONST_IMPEDANCE, halves_const_z=True),
        "Transformer.ct",
    )
    assert worst["v1"] < _ATOL_V1_PU, worst
    assert worst["v3"] < _ATOL_V3_PU, worst
    assert worst["i"] < _ATOL_I_A, worst


def test_const_z_half_loads_need_the_split_phase_rating():
    """Without ``u_rated_ln_v`` the halves are referred to ``440/√3`` V."""
    case = _centre_tap_with_delta(
        LoadModel.CONST_IMPEDANCE, halves_const_z=True, split_rating=False
    )
    worst = _compare(case, "Transformer.ct")
    assert worst["v1"] > 1e-4, worst


@pytest.mark.parametrize("model", list(_MODELS))
def test_ab_delta_load_on_four_wire_feeder_matches_opendss(model):
    worst = _compare(_dyn_with_delta(model), "Transformer.t1")
    assert worst["v1"] < _ATOL_V1_PU, worst
    assert worst["v3"] < _ATOL_V3_PU, worst
    assert worst["i"] < _ATOL_I_A, worst


@pytest.mark.parametrize("case", ["centre_tap", "four_wire"])
def test_delta_load_harmonic_shunt_matches_opendss(case):
    """OpenDSS's own harmonic ``Load`` admittance on the phase-to-phase element.

    The wye loads are left out: the element under test is the delta one (measured
    3e-12 pu at order 3 on both circuits, at ``%SeriesRL`` 0, 50 and 100).
    """
    import opendssdirect as dss

    if case == "centre_tap":
        lines, grid, rows, _, _ = _centre_tap_with_delta(LoadModel.CONST_POWER)
    else:
        lines, grid, rows, _, _ = _dyn_with_delta(LoadModel.CONST_POWER)
    # Only the phase-to-phase load: the shunt of the element under test.
    lines = [
        ln + " %SeriesRL=50" if ln.startswith("New Load.") else ln
        for ln in lines
        if not ln.startswith("New Load.") or "conn=delta" in ln
    ]
    shunt = HarmonicShuntModel(series_rl_fraction=0.5)
    grid = grid.model_copy(
        update={
            "appliances": [
                a.model_copy(update={"harmonic_model": shunt})
                if isinstance(a, Load)
                else a
                for a in grid.appliances
                if not isinstance(a, Load) or a.connection == "delta"
            ]
        }
    )
    for line in lines + [
        "Set NeglectLoadY=No",
        "Set Tolerance=1e-12",
        "Set MaxIterations=200",
        "Set Mode=Snap",
        "Solve",
    ]:
        dss.Text.Command(line)
    assert dss.Solution.Converged()

    def _volts():
        out = {}
        for bus in ("src", "lv", "lv2"):
            dss.Circuit.SetActiveBus(bus)
            v = np.array(dss.Bus.Voltages())
            for k, node in enumerate(dss.Bus.Nodes()):
                out[(bus, node)] = complex(v[2 * k], v[2 * k + 1])
        return out

    ref = {1: _volts()}
    with _scratch_datapath():
        for cmd in ("Set Mode=Harmonics", "Set Harmonic=3", "Solve"):
            dss.Text.Command(cmd)
        ref[3] = _volts()
        dss.Text.Command("Set Mode=Snap")

    res = solve_harmonic_flow(
        grid, [1, 3], slack="norton", load_shunt="opendss", tol=1e-12
    )
    assert res.pf.converged
    v = res.v.numpy()
    index = node_phase_index(grid)
    for k, h in enumerate((1, 3)):
        dv = max(
            abs(v[k, index.row(n, p)] - ref[h][(bus, node)]) / base
            for (n, p), (bus, node, base) in rows.items()
        )
        assert dv < (_ATOL_V1_PU if h == 1 else _ATOL_V3_PU), (h, dv)
