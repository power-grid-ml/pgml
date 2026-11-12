"""Structure of the OpenDSS reader's one-leg (phase-to-phase) elements.

A one-phase OpenDSS element has two conductors. When the second is another phase
(``bus.1.2``) it sits between the two phases whatever its connection keyword: ``conn=delta``
by definition, ``conn=wye`` because the second node of a one-phase wye element is its return
terminal. The reader returns a two-phase DELTA appliance on the two phases in OpenDSS's order;
a neutral or ground return (``bus.1``, ``bus.1.0``, ``bus.1.4``) stays a WYE element. Numerical
agreement with the live engine is in ``tests/reference/test_opendss_one_leg_delta.py``.

Covered here: the mapping and its node ratings (``u_rated_ln_v`` on a node with two phase
rows, none on a node with three, wye ratings unchanged), the cases that stay WYE, the
single-phase equivalent (unchanged), the ``kV`` warning, the controls that read the node
rating (``InvControl``, ``model=3`` regulation), the refusal of native DER harmonics, and
the round trip pgml -> OpenDSS -> pgml.
"""

from __future__ import annotations

import logging
import math

import pytest

pytest.importorskip("opendssdirect")

import opendssdirect as dss  # noqa: E402

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import from_grid, to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    Generator,
    Load,
    Phase,
    ShuntAppliance,
    Storage,
    VoltVarControl,
    WindingConnection,
)

pytestmark = pytest.mark.opendss

_F0 = 50.0
_SQRT3 = math.sqrt(3.0)
_BANDS = "vminpu=0.1 vmaxpu=10 vlowpu=0.05"


def _circuit(*elements: str) -> None:
    """A 400 V source, a three-phase line to bus ``b``, a two-phase line to bus ``c``."""
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={int(_F0)}")
    for line in (
        f"New Circuit.leg basekv=0.4 pu=1.0 phases=3 bus1=src frequency={_F0} "
        "r1=0.01 x1=0.02 r0=0.01 x0=0.02",
        "New Line.l1 phases=3 bus1=src bus2=b r1=0.3 x1=0.15 r0=0.3 x0=0.15 "
        "c1=0 c0=0 length=0.2 units=km",
        "New Line.l2 phases=2 bus1=b.1.2 bus2=c.1.2 r1=0.3 x1=0.15 r0=0.3 x0=0.15 "
        "c1=0 c0=0 length=0.2 units=km",
        f"New Load.bg phases=3 bus1=b kv=0.4 kw=12 kvar=4 model=1 {_BANDS}",
        *elements,
        "Set VoltageBases=[0.4]",
        "CalcVoltageBases",
        "Set Tolerance=1e-12",
        "Solve",
    ):
        dss.Text.Command(line)
    assert dss.Solution.Converged()


def _load(bus: str, *, conn: str = "delta", kv: float = 0.4, model: int = 1) -> str:
    return (
        f"New Load.x phases=1 conn={conn} bus1={bus} kv={kv} kw=8 kvar=3 "
        f"model={model} {_BANDS}"
    )


def _read(**kwargs):
    kwargs.setdefault("phase_mode", PhaseMode.THREE_PHASE)
    kwargs.setdefault("der_harmonics", False)
    return to_grid(dss, **kwargs)


def _by_name(grid, name):
    (item,) = [a for a in grid.appliances if a.name == name]
    return item


def _node(grid, name):
    (node,) = [n for n in grid.nodes if n.name == name]
    return node


@pytest.mark.parametrize("conn", ["delta", "wye"])
@pytest.mark.parametrize(
    "bus, phases",
    [
        ("b.1.2", (Phase.A, Phase.B)),
        ("b.2.3", (Phase.B, Phase.C)),
        ("b.3.1", (Phase.C, Phase.A)),
        ("c.1.2", (Phase.A, Phase.B)),
    ],
)
def test_one_phase_element_across_two_phases_is_a_two_phase_delta(conn, bus, phases):
    _circuit(_load(bus, conn=conn))
    grid, _ = _read()
    load = _by_name(grid, "x")
    assert isinstance(load, Load)
    assert load.connection == WindingConnection.DELTA
    assert load.phases == phases
    assert float(load.p_nom_w) == pytest.approx(8_000.0)
    assert float(load.q_nom_var) == pytest.approx(3_000.0)


def test_two_row_node_gets_the_phase_to_neutral_rating_a_delta_element_needs():
    from pgml.assembly import node_voltage_magnitude

    _circuit(_load("c.1.2"))
    grid, _ = _read()
    node = _node(grid, "c")
    assert len(node.phases) == 2
    assert float(node.u_rated_v) == pytest.approx(400.0, rel=1e-9)
    assert float(node.u_rated_ln_v) == pytest.approx(400.0 / _SQRT3, rel=1e-9)
    assert node_voltage_magnitude(node, line_to_line=True) == pytest.approx(400.0)
    assert node_voltage_magnitude(node) == pytest.approx(400.0 / _SQRT3)


def test_three_row_node_is_left_alone():
    _circuit(_load("b.1.2"))
    grid, _ = _read()
    node = _node(grid, "b")
    assert node.u_rated_ln_v is None
    assert float(node.u_rated_v) == pytest.approx(400.0, rel=1e-9)


def test_wye_rating_of_a_node_is_unchanged_by_a_delta_element_on_it():
    """The per-phase base of every row stays ``400/sqrt(3)`` on the two-row node."""
    from pgml.assembly import node_voltage_magnitude

    _circuit(_load("c.1", conn="wye", kv=0.23))
    before, _ = _read()
    _circuit(
        _load("c.1", conn="wye", kv=0.23), _load("c.1.2").replace("Load.x", "Load.y")
    )
    after, _ = _read()
    assert node_voltage_magnitude(_node(before, "c")) == pytest.approx(400.0 / _SQRT3)
    assert node_voltage_magnitude(_node(after, "c")) == node_voltage_magnitude(
        _node(before, "c")
    )
    wye = _by_name(after, "x")
    assert wye.connection == WindingConnection.WYE
    assert wye.phases == (Phase.A,)


@pytest.mark.parametrize("bus", ["b.1", "b.1.0", "c.1", "b"])
def test_neutral_or_ground_return_stays_wye(bus):
    _circuit(_load(bus, conn="wye", kv=0.23))
    grid, _ = _read()
    load = _by_name(grid, "x")
    assert load.connection == WindingConnection.WYE
    assert load.phases == (Phase.A,)


def test_one_phase_delta_on_a_bare_bus_stays_wye_to_ground():
    """``phases=1 conn=delta`` on ``bus`` has conductors 1 and 0: not a leg."""
    _circuit(_load("b", conn="delta"))
    grid, _ = _read()
    load = _by_name(grid, "x")
    assert load.connection == WindingConnection.WYE
    assert load.phases == (Phase.A,)


def test_explicit_neutral_conductor_stays_wye_on_the_neutral_row():
    for line in (
        "Clear",
        f"Set DefaultBaseFrequency={int(_F0)}",
        "New Circuit.fw basekv=0.4 pu=1.0 phases=3 bus1=src.1.2.3.4",
        "New Reactor.ng phases=1 bus1=src.4 R=0.001 X=0",
        "New Line.l1 phases=4 bus1=src.1.2.3.4 bus2=b.1.2.3.4 r1=0.3 x1=0.15 r0=0.3 "
        "x0=0.15 c1=0 c0=0 length=0.2 units=km",
        _load("b.1.4", conn="wye", kv=0.23),
        "Set VoltageBases=[0.4]",
        "CalcVoltageBases",
        "Solve",
    ):
        dss.Text.Command(line)
    grid, _ = _read()
    load = _by_name(grid, "x")
    assert load.connection == WindingConnection.WYE
    assert load.phases == (Phase.A,)
    assert load.return_path == "neutral"


def test_single_phase_equivalent_is_unchanged():
    """The single-phase equivalent has one phase row; a leg keeps its old reading."""
    _circuit(_load("b.1.2"))
    grid, _ = to_grid(dss, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV, der_harmonics=False)
    load = _by_name(grid, "x")
    assert load.phases == (Phase.A,)
    assert all(n.u_rated_ln_v is None for n in grid.nodes)


def test_open_delta_is_still_refused_and_its_one_leg_split_converts():
    _circuit("New Load.x phases=2 conn=delta bus1=b.1.2.3 kv=0.4 kw=16 kvar=6")
    with pytest.raises(ConversionError, match="open delta"):
        _read()
    _circuit(
        _load("b.1.2").replace("kw=8 kvar=3", "kw=8 kvar=3"),
        _load("b.2.3").replace("Load.x", "Load.y"),
    )
    grid, _ = _read()
    legs = {a.name: a.phases for a in grid.appliances if isinstance(a, Load)}
    assert legs["x"] == (Phase.A, Phase.B)
    assert legs["y"] == (Phase.B, Phase.C)


@pytest.mark.parametrize(
    "line",
    [
        "New Generator.x phases=1 conn=wye bus1=c.1.2 kv=0.4 kw=5 kvar=1 model=1",
        "New PVSystem.x phases=1 conn=delta bus1=c.1.2 kv=0.4 kva=10 pmpp=8 irradiance=1",
        "New Storage.x phases=1 conn=wye bus1=c.1.2 kv=0.4 kwrated=10 kwhrated=20 "
        "state=discharging kw=5 kvar=1",
    ],
)
def test_der_classes_map_like_loads(line):
    _circuit(line)
    grid, _ = _read()
    der = _by_name(grid, "x")
    assert isinstance(der, (Generator, Storage))
    assert der.connection == WindingConnection.DELTA
    assert der.phases == (Phase.A, Phase.B)
    assert float(_node(grid, "c").u_rated_ln_v) == pytest.approx(400.0 / _SQRT3)


@pytest.mark.parametrize("kind", ["Capacitor", "Reactor"])
def test_delta_shunt_across_two_phases_carries_the_one_leg_admittance(kind):
    """OpenDSS's one leg is the sum of pgml's two cyclic legs, so each leg is half."""
    _circuit(f"New {kind}.x phases=1 conn=delta bus1=c.1.2 kv=0.4 kvar=10")
    grid, _ = _read()
    shunt = _by_name(grid, "x")
    assert isinstance(shunt, ShuntAppliance)
    assert shunt.connection == WindingConnection.DELTA
    assert shunt.phases == (Phase.A, Phase.B)
    if kind == "Capacitor":
        dss.Text.Command("? Capacitor.x.Cuf")
        cuf = float(dss.Text.Result().strip().strip("[]").split()[0]) * 1e-6
        assert [float(c) for c in shunt.capacitance_f] == pytest.approx([cuf / 2] * 2)
    else:
        omega = 2 * math.pi * _F0
        dss.Text.Command("? Reactor.x.X")
        x_ohm = float(dss.Text.Result().split()[0])
        assert [float(h) for h in shunt.inductance_h] == pytest.approx(
            [2 * x_ohm / omega] * 2
        )


def test_wye_shunt_on_two_phases_stays_phase_to_ground():
    """A ``conn=wye`` Capacitor has a second terminal: ``bus.1.2`` returns to ground."""
    _circuit("New Capacitor.x phases=1 conn=wye bus1=c.1.2 kv=0.23 kvar=10")
    grid, _ = _read()
    shunt = _by_name(grid, "x")
    assert shunt.connection == WindingConnection.WYE
    assert shunt.phases == (Phase.A,)


def test_voltage_dependent_leg_with_a_different_kv_is_reported(caplog):
    _circuit(_load("b.1.2", conn="wye", kv=0.23, model=2))
    with caplog.at_level(logging.WARNING, logger="pgml"):
        _, _, report = _read(return_report=True)
    assert "approx.load.one_leg_voltage_base" in report
    assert any("phase-to-phase" in r.message for r in caplog.records)


@pytest.mark.parametrize("kv, model", [(0.4, 2), (0.23, 1)])
def test_matching_kv_or_constant_power_is_not_reported(kv, model):
    _circuit(_load("b.1.2", kv=kv, model=model))
    _, _, report = _read(return_report=True)
    assert "approx.load.one_leg_voltage_base" not in report


def test_native_der_harmonics_are_refused_across_two_phases():
    _circuit(
        "New PVSystem.x phases=1 conn=wye bus1=c.1.2 kv=0.4 kva=10 pmpp=8 irradiance=1"
    )
    with pytest.raises(ConversionError, match="two phase conductors"):
        to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, der_harmonics=True)


def test_regulating_generator_on_a_rated_node_uses_the_phase_to_neutral_base():
    """``Vpu`` is re-referred to ``u_rated_ln_v`` (not ``u_rated_v``) on a two-row node."""
    gen = "New Generator.g phases=1 bus1=c.1 kv=0.23 kw=3 model=3 Vpu=1.02 kvar=0"
    _circuit(gen)
    plain, _ = _read()
    _circuit(gen, _load("c.1.2").replace("Load.x", "Load.leg"))
    rated, _ = _read()
    v_plain = _by_name(plain, "g").voltage_regulation.v_set_pu
    v_rated = _by_name(rated, "g").voltage_regulation.v_set_pu
    assert v_rated == v_plain
    assert v_rated == pytest.approx(1.02 * 230.0 / (400.0 / _SQRT3))


def test_regulating_generator_across_two_phases_is_refused():
    _circuit("New Generator.g phases=1 bus1=c.1.2 kv=0.4 kw=3 model=3 Vpu=1.0 kvar=0")
    with pytest.raises(ConversionError, match="across two phases"):
        _read()


_VOLTVAR = (
    "New XYcurve.vvc npts=3 xarray=[0.90 1.00 1.10] yarray=[0.3 0.0 -0.3]",
    "New InvControl.{ic} PVSystemList=[{pv}] mode=VOLTVAR voltage_curvex_ref=rated "
    "vvc_curve1=vvc RefReactivePower=VARMAX",
)


def test_inv_control_reads_the_node_nameplate_on_a_rated_node():
    """A phase-to-phase PV (kV 0.4) and a phase-to-neutral one (kV 0.23) on one node."""
    _circuit(
        "New PVSystem.pvll phases=1 conn=delta bus1=c.1.2 kv=0.4 kva=10 pmpp=8 "
        "irradiance=1 %cutin=0 %cutout=0",
        "New PVSystem.pvln phases=1 bus1=c.1 kv=0.230940108 kva=10 pmpp=8 irradiance=1 "
        "%cutin=0 %cutout=0",
        _VOLTVAR[0],
        _VOLTVAR[1].format(ic="ic1", pv="pvll"),
        _VOLTVAR[1].format(ic="ic2", pv="pvln"),
    )
    grid, _, report = to_grid(
        dss,
        phase_mode=PhaseMode.THREE_PHASE,
        der_harmonics=False,
        return_report=True,
    )
    ll = _by_name(grid, "pvll")
    ln = _by_name(grid, "pvln")
    assert ll.connection == WindingConnection.DELTA
    assert ln.connection == WindingConnection.WYE
    assert isinstance(ll.control, VoltVarControl)
    assert isinstance(ln.control, VoltVarControl)
    assert "invcontrol" not in report


@pytest.mark.parametrize("bus", ["b.1.2", "c.1.2"])
def test_round_trip_two_phase_delta_load(bus):
    """pgml -> OpenDSS -> pgml keeps phases, connection, power and node ratings."""
    _circuit(_load(bus, model=2))
    first, _ = _read()
    export = from_grid(first, mode="matched", load_shunt="none")
    assert export.circuit is not None
    second, _ = _read()
    a = next(x for x in first.appliances if isinstance(x, Load) and x.name == "x")
    (b,) = [
        x
        for x in second.appliances
        if isinstance(x, Load) and x.connection == WindingConnection.DELTA
    ]
    assert b.phases == a.phases
    assert b.connection == a.connection == WindingConnection.DELTA
    assert float(b.p_nom_w) == pytest.approx(float(a.p_nom_w), rel=1e-9)
    assert float(b.q_nom_var) == pytest.approx(float(a.q_nom_var), rel=1e-9)
    assert b.load_model == a.load_model
    assert len(first.nodes) == len(second.nodes)
    for n1, n2 in zip(first.nodes, second.nodes):  # both follow YNodeOrder
        assert n2.phases == n1.phases
        assert float(n2.u_rated_v) == pytest.approx(float(n1.u_rated_v), rel=1e-6)
        if n1.u_rated_ln_v is not None:
            assert float(n2.u_rated_ln_v) == pytest.approx(
                float(n1.u_rated_ln_v), rel=1e-6
            )


def test_round_trip_two_phase_delta_shunt():
    """The exported one-leg bank reads back with the same total admittance."""
    _circuit(
        "New Capacitor.x phases=1 conn=delta bus1=c.1.2 kv=0.4 kvar=10",
        "New Reactor.r phases=1 conn=delta bus1=c.1.2 R=500 X=0",
    )
    first, _ = _read()
    from_grid(first, mode="matched", load_shunt="none")
    second, _ = _read()

    def totals(grid):
        cap = sum(
            float(c)
            for s in grid.appliances
            if isinstance(s, ShuntAppliance)
            and any(float(x) > 0 for x in s.capacitance_f)
            for c in s.capacitance_f
        )
        cond = sum(
            float(g)
            for s in grid.appliances
            if isinstance(s, ShuntAppliance)
            and any(float(x) > 0 for x in s.conductance_s)
            for g in s.conductance_s
        )
        return cap, cond

    c1, g1 = totals(first)
    c2, g2 = totals(second)
    assert c2 == pytest.approx(c1, rel=1e-9)
    assert g2 == pytest.approx(g1, rel=1e-9)
    assert g1 == pytest.approx(1.0 / 500.0, rel=1e-9)
