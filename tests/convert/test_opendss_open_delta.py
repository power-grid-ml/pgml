"""The OpenDSS reader refuses an open delta (``phases=2 conn=delta``).

OpenDSS gives a two-phase delta element three conductors and stamps two phase-to-phase
legs, 1-2 and 2-3; the third conductor is node 0 when the bus string names only two
nodes (``NodeOrder() == [1, 2, 0]`` for ``bus.1.2`` and for a bare ``bus``). A pgml
two-phase DELTA appliance is ONE element between its two phases, so reading the element
as one appliance changed the circuit: for a 20 kW constant-impedance load on a 400 V
three-phase bus the bus voltages differed from OpenDSS by 1.75 V (bare bus) and 1.84 V
(``bus.1.2.3``), measured with opendssdirect 0.9.4, complex128, before the refusal.
Every appliance and shunt class with a delta connection is covered, in both phase modes.
A three-phase delta and a one-phase delta still convert.
"""

from __future__ import annotations

import pytest

pytest.importorskip("opendssdirect")

import opendssdirect as dss  # noqa: E402

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import Load, Phase, WindingConnection  # noqa: E402

pytestmark = pytest.mark.opendss

_F0 = 50.0

#: DSS class -> element definition with ``{phases}`` and ``{bus}`` placeholders
_ELEMENTS = {
    "Load": "New Load.x phases={phases} conn=delta bus1={bus} kv=0.4 kw=20 kvar=5",
    "Generator": (
        "New Generator.x phases={phases} conn=delta bus1={bus} kv=0.4 kw=10 kvar=0"
    ),
    "PVSystem": (
        "New PVSystem.x phases={phases} conn=delta bus1={bus} kv=0.4 kva=12 "
        "pmpp=10 irradiance=1"
    ),
    "Storage": (
        "New Storage.x phases={phases} conn=delta bus1={bus} kv=0.4 kwrated=10 "
        "kwhrated=20"
    ),
    "Capacitor": (
        "New Capacitor.x phases={phases} conn=delta bus1={bus} kv=0.4 kvar=10"
    ),
    "Reactor": "New Reactor.x phases={phases} conn=delta bus1={bus} kv=0.4 kvar=10",
}


def _circuit(element: str) -> None:
    """A 400 V source, one three-phase line to bus ``b`` and ``element`` at ``b``."""
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={int(_F0)}")
    dss.Text.Command(
        f"New Circuit.open_delta basekv=0.4 pu=1.0 phases=3 bus1=src frequency={_F0} "
        "r1=1e-3 x1=1e-3 r0=1e-3 x0=1e-3"
    )
    dss.Text.Command(
        "New Line.l1 phases=3 bus1=src bus2=b r1=0.05 x1=0.05 r0=0.05 x0=0.05 "
        "c1=0 c0=0 length=1 units=km"
    )
    dss.Text.Command(element)
    dss.Text.Command("Set VoltageBases=[0.4]")
    dss.Text.Command("CalcVoltageBases")
    dss.Text.Command("Solve")


@pytest.mark.parametrize("phase_mode", list(PhaseMode))
@pytest.mark.parametrize("bus", ["b", "b.1.2", "b.1.2.3"])
@pytest.mark.parametrize("kind", list(_ELEMENTS))
def test_open_delta_is_refused(kind: str, bus: str, phase_mode: PhaseMode) -> None:
    _circuit(_ELEMENTS[kind].format(phases=2, bus=bus))
    with pytest.raises(ConversionError, match=rf"{kind} 'x' is an open delta") as err:
        to_grid(dss, phase_mode=phase_mode)
    message = str(err.value)
    assert "two two-phase delta elements" in message
    assert ("b.1.2.3" if bus == "b.1.2.3" else "b.1.2.0") in message


@pytest.mark.parametrize("kind", list(_ELEMENTS))
def test_three_phase_delta_still_converts(kind: str) -> None:
    _circuit(_ELEMENTS[kind].format(phases=3, bus="b"))
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    (element,) = [a for a in grid.appliances if a.name == "x"]
    assert element.connection == WindingConnection.DELTA
    assert len(element.phases) == 3


def test_one_phase_delta_load_is_one_phase_to_phase_element() -> None:
    """``phases=1 conn=delta`` has one leg (two conductors); it is not refused and
    converts to the two-phase delta appliance between its conductors."""
    _circuit(_ELEMENTS["Load"].format(phases=1, bus="b.1.2"))
    grid, _ = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    (load,) = [a for a in grid.appliances if isinstance(a, Load) and a.name == "x"]
    assert load.connection == WindingConnection.DELTA
    assert load.phases == (Phase.A, Phase.B)
