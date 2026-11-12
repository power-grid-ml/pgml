"""A one-phase OpenDSS element across two phase conductors vs live OpenDSS.

OpenDSS stamps a one-phase ``Load``, ``Generator``, ``PVSystem`` or ``Storage`` whose
bus string names two phases (``bus.1.2``) between those two conductors, in either
connection: a ``conn=wye`` element takes its second node as the return terminal, so
``b.1.2`` is the same phase-to-phase element as ``conn=delta`` (identical currents,
checked in :func:`test_wye_and_delta_spelling_are_the_same_opendss_element`). A
single-terminal ``conn=delta`` ``Capacitor`` or ``Reactor`` on two phases is the same
element (a ``conn=wye`` one is two-terminal and returns to ground). The reader maps all
of them to a pgml two-phase DELTA appliance on the two phases, in OpenDSS's conductor
order.

Circuit: a 400 V source with ``R1 = 0.01``, ``X1 = 0.02`` Ohm behind a three-phase
line to bus ``b`` (balanced wye background load), optionally continued by a two-phase
line to bus ``c`` that carries only two phase conductors. The element sits on
``b.1.2``, ``b.3.1`` (reversed leg), ``c.1.2`` or ``c.2.3``; ``c`` has only two phase
rows, which is where the reader sets ``u_rated_ln_v``. The circuit is read by
:func:`pgml.convert.opendss.to_grid` (nothing is hand-built on the pgml side except the
harmonic spectrum of the loads, which the reader does not import) and solved by
``solve_harmonic_flow(slack="norton")`` at the fundamental and at order 3, against
``Solve`` and ``Solve mode=harmonics`` of the live engine (``Tolerance=1e-12``).
Loads are pure harmonic current sources (``NeglectLoadY=Yes``, ``load_shunt="none"``).

Alignment: every ``(bus, node)`` of OpenDSS's ``Bus.Nodes()``/``Bus.Voltages()`` is
compared with the pgml row of ``node_phase_index`` for the bus of the same name and the
phase of the same number. ``dv`` is the largest complex difference over all rows, in V
and in per unit of the 230.94 V line-to-neutral base.

Measured (complex128, CPU, opendssdirect 0.9.4, ``Tolerance=1e-12``) over every case
below, largest difference over all rows: fundamental 7.8e-8 V (3.4e-10 pu), order 3
1.3e-9 V (5.6e-12 pu); the fundamental figure is the same in every case, the
convergence of the two tools' fixed-point iterations, not the stamp. Read as a wye
element on the first phase, the same circuits differ by 1.1 to 3.8 V (4.8e-3 to 1.7e-2
pu) at the fundamental and by up to 2.4 V at order 3; the delta ``Capacitor`` and
``Reactor`` were skipped with a warning. Bounds asserted: 1e-6 pu at the fundamental,
1e-8 pu at order 3 (three to four orders above the measurement, four to five below the
error of that reading). OpenDSS's earth-return correction is off
(``rg=xg=0``): with its default ``Rg``/``Xg`` the order-3 voltages on the two-phase lines
of this circuit differ by about 1e-4 pu for any element, including a plain phase-to-neutral
load (a line effect, not investigated here), while the fundamental is unaffected.
"""

from __future__ import annotations

import math

import pytest

opendssdirect = pytest.importorskip("opendssdirect")
dss = opendssdirect

from pgml.assembly import node_phase_index  # noqa: E402
from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.evaluation.oracles.opendss_scenario_oracle import (  # noqa: E402
    _scratch_datapath,
)
from pgml.schemas.grid_schema import (  # noqa: E402
    HarmonicComponent,
    Load,
    Phase,
    ShuntAppliance,
    SpectrumPoint,
    StaticSpectrum,
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow  # noqa: E402

pytestmark = [pytest.mark.opendss, pytest.mark.usefixtures("opendss_model_defaults")]

F0 = 50.0
V_LN = 400.0 / math.sqrt(3.0)
ATOL_V1_PU = 1e-6
ATOL_V3_PU = 1e-8
_PHASE = {1: Phase.A, 2: Phase.B, 3: Phase.C}

SPECTRUM = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=3, magnitude_pu=0.2, phase_deg=30.0),
        ]
    )
)

#: placement name -> (bus of the element, bus string, line to it is two-phase)
PLACEMENTS = {
    "3ph_b12": ("b", "b.1.2"),
    "3ph_b31": ("b", "b.3.1"),
    "2ph_c12": ("c", "c.1.2"),
    "2ph_c23": ("c", "c.2.3"),
}

_BANDS = "vminpu=0.1 vmaxpu=10"
#: load model -> (DSS model, extra DSS properties)
LOAD_MODELS = {
    "const_p": (1, ""),
    "const_z": (2, ""),
    "const_i": (5, ""),
    "zip": (8, " ZIPV=[0.3 0.3 0.4 0.5 0.2 0.3 0]"),
}


def _build(*elements: str, bus: str) -> None:
    """The source, the lines, a balanced background load and the elements at ``bus``."""
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={int(F0)}")
    for line in (
        f"New Circuit.leg basekv=0.4 pu=1.0 phases=3 bus1=src frequency={F0} "
        "r1=0.01 x1=0.02 r0=0.01 x0=0.02",
        "New Spectrum.s3 numharm=2 harmonic=[1 3] %mag=[100 20] angle=[0 30]",
        "New Line.l1 phases=3 bus1=src bus2=b r1=0.3 x1=0.15 r0=0.3 x0=0.15 "
        "c1=0 c0=0 length=0.2 units=km rg=0 xg=0",
        "New Line.l2 phases=2 bus1=b.1.2 bus2=c.1.2 r1=0.3 x1=0.15 r0=0.3 x0=0.15 "
        "c1=0 c0=0 length=0.2 units=km rg=0 xg=0",
        "New Line.l3 phases=2 bus1=b.2.3 bus2=c.2.3 r1=0.3 x1=0.15 r0=0.3 x0=0.15 "
        "c1=0 c0=0 length=0.2 units=km rg=0 xg=0",
        f"New Load.bg phases=3 bus1=b kv=0.4 kw=12 kvar=4 model=1 conn=wye spectrum=s3 {_BANDS} "
        "vlowpu=0.05",
        *[e.format(bus=bus) for e in elements],
        "Set VoltageBases=[0.4]",
        "CalcVoltageBases",
        "Set NeglectLoadY=Yes",
        "Set Tolerance=1e-12",
        "Set MaxIterations=200",
        "Set Mode=Snap",
        "Solve",
    ):
        dss.Text.Command(line)
    assert dss.Solution.Converged()


def _volts() -> dict:
    out = {}
    for bus in ("src", "b", "c"):
        dss.Circuit.SetActiveBus(bus)
        v = dss.Bus.Voltages()
        for k, node in enumerate(dss.Bus.Nodes()):
            out[(bus, int(node))] = complex(v[2 * k], v[2 * k + 1])
    return out


def _dss_reference(orders) -> dict:
    ref = {1: _volts()}
    if 3 in orders:
        with _scratch_datapath():
            for cmd in ("Set Mode=Harmonics", "Set Harmonic=3", "Solve"):
                dss.Text.Command(cmd)
            ref[3] = _volts()
            dss.Text.Command("Set Mode=Snap")
    return ref


def _with_spectrum(grid):
    """Attach the DSS ``s3`` spectrum to the converted loads (not imported)."""
    return grid.model_copy(
        update={
            "appliances": [
                a.model_copy(update={"spectrum": SPECTRUM})
                if isinstance(a, Load)
                else a
                for a in grid.appliances
            ]
        }
    )


def _pgml_deviation(ref: dict, orders, *, der_harmonics: bool = True):
    """``(grid, {order: (dv_V, dv_pu)})`` of the converted live circuit against ``ref``."""
    grid, _ = to_grid(
        dss,
        phase_mode=PhaseMode.THREE_PHASE,
        earth_return="opendss",
        der_harmonics=der_harmonics,
    )
    grid = _with_spectrum(grid)
    res = solve_harmonic_flow(
        grid, list(orders), slack="norton", load_shunt="none", tol=1e-12
    )
    assert res.pf.converged
    v = res.v.numpy()
    index = node_phase_index(grid)
    by_name = {n.name: n.id for n in grid.nodes}
    out = {}
    for k, h in enumerate(orders):
        dv = max(
            abs(v[k, index.row(by_name[b], _PHASE[ph])] - volts)
            for (b, ph), volts in ref[h].items()
        )
        out[h] = (float(dv), float(dv) / V_LN)
    return grid, out


def measure(element: str, bus: str, *, orders=(1, 3), der_harmonics: bool = True):
    """``(result, grid)``: ``{order: (dv_V, dv_pu)}`` of the live circuit vs pgml."""
    _build(element, bus=bus)
    grid, out = _pgml_deviation(
        _dss_reference(orders), orders, der_harmonics=der_harmonics
    )
    return out, grid


def _check(result):
    assert result[1][1] < ATOL_V1_PU, result
    if 3 in result:
        assert result[3][1] < ATOL_V3_PU, result


def _leg_appliance(grid, name="x"):
    (a,) = [a for a in grid.appliances if a.name == name]
    return a


@pytest.mark.parametrize("conn", ["delta", "wye"])
@pytest.mark.parametrize("placement", list(PLACEMENTS))
@pytest.mark.parametrize("model", list(LOAD_MODELS))
def test_one_leg_load_matches_opendss(model, placement, conn):
    dss_model, extra = LOAD_MODELS[model]
    bus_name, bus = PLACEMENTS[placement]
    element = (
        f"New Load.x phases=1 conn={conn} bus1={{bus}} kv=0.4 kw=8 kvar=3 "
        f"model={dss_model} {_BANDS} vlowpu=0.05 spectrum=s3{extra}"
    )
    result, grid = measure(element, bus)
    _check(result)
    load = _leg_appliance(grid)
    assert load.connection == WindingConnection.DELTA
    assert len(load.phases) == 2


_DER = {
    "Generator": "New Generator.x phases=1 conn={conn} bus1={{bus}} kv=0.4 kw=8 "
    f"kvar=2 model=1 {_BANDS}",
    "PVSystem": "New PVSystem.x phases=1 conn={conn} bus1={{bus}} kv=0.4 kva=10 "
    f"pmpp=8 irradiance=1 {_BANDS}",
    "Storage": "New Storage.x phases=1 conn={conn} bus1={{bus}} kv=0.4 kwrated=10 "
    f"kwhrated=20 state=discharging kw=5 kvar=1 {_BANDS}",
}


@pytest.mark.parametrize("conn", ["delta", "wye"])
@pytest.mark.parametrize("placement", list(PLACEMENTS))
@pytest.mark.parametrize("kind", list(_DER))
def test_one_leg_der_matches_opendss(kind, placement, conn):
    bus_name, bus = PLACEMENTS[placement]
    result, grid = measure(
        _DER[kind].format(conn=conn), bus, orders=(1,), der_harmonics=False
    )
    _check(result)
    der = _leg_appliance(grid)
    assert der.connection == WindingConnection.DELTA
    assert len(der.phases) == 2


_SHUNT = {
    "Capacitor": "New Capacitor.x phases=1 conn=delta bus1={bus} kv=0.4 kvar=10",
    "Reactor": "New Reactor.x phases=1 conn=delta bus1={bus} kv=0.4 kvar=10",
}


@pytest.mark.parametrize("placement", list(PLACEMENTS))
@pytest.mark.parametrize("kind", list(_SHUNT))
def test_one_leg_delta_shunt_matches_opendss(kind, placement):
    """A single-terminal delta ``Capacitor``/``Reactor`` on two phases, h = 1 and 3."""
    bus_name, bus = PLACEMENTS[placement]
    result, grid = measure(_SHUNT[kind], bus)
    _check(result)
    shunt = _leg_appliance(grid)
    assert isinstance(shunt, ShuntAppliance)
    assert shunt.connection == WindingConnection.DELTA
    assert len(shunt.phases) == 2


def test_wye_and_delta_spelling_are_the_same_opendss_element():
    """OpenDSS stamps ``conn=wye`` on ``b.1.2`` exactly as ``conn=delta``."""
    currents = []
    for conn in ("wye", "delta"):
        element = (
            f"New Load.x phases=1 conn={conn} bus1={{bus}} kv=0.4 kw=8 kvar=3 "
            f"model=2 {_BANDS}"
        )
        _build(element, bus="b.1.2")
        dss.Circuit.SetActiveElement("Load.x")
        currents.append(list(dss.CktElement.Currents()))
    assert currents[0] == pytest.approx(currents[1], abs=1e-9)


@pytest.mark.parametrize("model", list(LOAD_MODELS))
def test_open_delta_split_into_two_legs_matches_opendss_open_delta(model):
    """The split the open-delta error recommends, against OpenDSS's own open delta.

    OpenDSS ``Load phases=2 conn=delta bus1=b.1.2.3`` has legs 1-2 and 2-3, each with
    half of ``kW``/``kvar`` at the line-to-line ``kV``. Two one-leg elements of half the
    power on ``b.1.2`` and ``b.2.3`` are the same circuit; the open delta itself is
    refused by the reader, and the reference here is OpenDSS solving the open delta.
    """
    dss_model, extra = LOAD_MODELS[model]
    common = f"kv=0.4 model={dss_model} {_BANDS} vlowpu=0.05 spectrum=s3{extra}"
    _build(
        f"New Load.x phases=2 conn=delta bus1={{bus}} kw=16 kvar=6 {common}",
        bus="b.1.2.3",
    )
    ref = _dss_reference((1, 3))
    with pytest.raises(ConversionError, match="open delta"):
        to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    _build(
        f"New Load.x phases=1 conn=delta bus1=b.1.2 kw=8 kvar=3 {common}",
        f"New Load.y phases=1 conn=delta bus1=b.2.3 kw=8 kvar=3 {common}",
        bus="unused",
    )
    split_ref = _dss_reference((1, 3))
    # OpenDSS's split equals its open delta, and pgml reading the split equals both
    for h, atol in ((1, ATOL_V1_PU), (3, ATOL_V3_PU)):
        dv_dss = max(abs(split_ref[h][key] - volts) for key, volts in ref[h].items())
        assert dv_dss / V_LN < atol, (h, dv_dss)
    _, result = _pgml_deviation(ref, (1, 3))
    _check(result)


@pytest.mark.parametrize("kind", ["Capacitor", "Resistor", "load"])
def test_exported_two_phase_delta_circuit_matches_the_original_opendss_circuit(kind):
    """pgml -> OpenDSS: the exported one-leg element solves like the circuit it came from.

    The reader turns the leg into a two-phase DELTA appliance; the exporter writes that
    back as ``phases=1 conn=delta`` on the two conductors (a two-phase ``Capacitor`` or
    ``Reactor`` would be an open delta). Both circuits are solved by OpenDSS; node
    voltages are aligned by ``YNodeOrder`` position (the exporter renames the buses).
    """
    from pgml.convert.opendss import from_grid

    element = {
        "Capacitor": _SHUNT["Capacitor"],
        # a conductance only: the exporter writes no ShuntAppliance inductance
        "Resistor": "New Reactor.x phases=1 conn=delta bus1={bus} R=40 X=0",
        "load": (
            "New Load.x phases=1 conn=delta bus1={bus} kv=0.4 kw=8 kvar=3 model=2 "
            f"{_BANDS}"
        ),
    }[kind]
    _build(element, bus="c.1.2")
    original = [
        complex(a, b)
        for a, b in zip(*[iter(dss.Circuit.AllBusVolts())] * 2, strict=True)
    ]
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, der_harmonics=False)
    from_grid(grid, mode="matched", load_shunt="none")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged()
    exported = [
        complex(a, b)
        for a, b in zip(*[iter(dss.Circuit.AllBusVolts())] * 2, strict=True)
    ]
    assert len(original) == len(exported)
    dv = max(abs(a - b) for a, b in zip(original, exported))
    assert dv / V_LN < ATOL_V1_PU, dv
