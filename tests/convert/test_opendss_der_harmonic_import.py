"""Native DER impedance bases and spectrum normalization are retained on import."""

import math
import tempfile
import numpy as np
import pytest
from pgml.convert.opendss import PhaseMode, to_grid

pytestmark = pytest.mark.opendss
dss = pytest.importorskip("opendssdirect")


def circuit(kind, conn="wye", spectrum=""):
    dss.Text.Command("clear")
    dss.Text.Command(f"set datapath={tempfile.mkdtemp(prefix='pgml_der_import_')}")
    for command in [
        "set defaultbasefrequency=50",
        "new circuit.der basekv=.4 pu=1 phases=3",
        "new spectrum.emission numharm=3 harmonic=[1 3 5] %mag=[80 5 3] angle=[10 40 20]",
    ]:
        dss.Text.Command(command)
    args = {
        "Generator": "kw=10 kvar=2 kva=20 xdpp=.2 xrdp=7",
        "PVSystem": "pmpp=10 kva=20 %r=2 %x=8",
        "Storage": "kwrated=10 kwhrated=20 kva=20 %r=2 %x=8 state=discharging %discharge=50",
    }[kind]
    dss.Text.Command(
        f"new {kind}.der bus1=sourcebus phases=3 kv=.4 conn={conn} {args} spectrum={spectrum or 'emission'}"
    )
    dss.Text.Command("set voltagebases=[.4]")
    dss.Text.Command("calcvoltagebases")
    dss.Text.Command("set tolerance=1e-10 maxiterations=300")
    dss.Text.Command("solve")
    assert dss.Solution.Converged()


@pytest.mark.parametrize("kind", ["Generator", "PVSystem", "Storage"])
@pytest.mark.parametrize("conn", ["wye", "delta"])
def test_der_native_bases_and_spectrum(kind, conn):
    circuit(kind, conn)
    grid, mapping = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    key = {"Generator": "generator", "PVSystem": "pvsystem", "Storage": "storage"}[kind]
    device = next(a for a in grid.appliances if a.id == mapping[key]["der"])
    factor = 3 if conn == "delta" else 1
    zbase = 0.4**2 * 1000 / 20
    expected_r = 0 if kind == "Generator" else 0.02 * zbase
    expected_x = (0.2 if kind == "Generator" else 0.08) * zbase
    block = device.harmonic_impedance
    assert block.resistance_ohm == pytest.approx(factor * expected_r)
    assert block.inductance_h * 2 * math.pi * 50 == pytest.approx(factor * expected_x)
    assert block.spectrum_reference == "opendss_voltage"
    assert block.frequency_model == "opendss_admittance"
    components = {x.order: x for x in device.spectrum.spectrum.components}
    assert components[1].magnitude_pu == 1
    assert components[3].magnitude_pu == 0.05
    assert components[3].phase_deg == 10
    assert components[5].phase_deg == -30
    # Direct primitive check uses the native engine's current state, independently
    # of pgml assembly: R+jX is inverted first, then susceptance is divided by h.
    dss.Text.Command("set mode=harmonic harmonics=[3]")
    dss.Text.Command("solve")
    dss.Circuit.SetActiveElement(f"{kind}.der")
    y = np.asarray(dss.CktElement.YPrim()).reshape(-1, 2)
    y = (y[:, 0] + 1j * y[:, 1]).reshape(dss.CktElement.NumConductors(), -1)
    y1 = 1 / complex(expected_r * factor, expected_x * factor)
    expected = complex(y1.real, y1.imag / 3)
    assert y[0, 0] == pytest.approx(expected * (2 if conn == "delta" else 1), abs=1e-10)


def test_fundamental_only_import_explicitly_omits_der_harmonics():
    circuit("Generator")
    grid, mapping = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, der_harmonics=False)
    device = next(a for a in grid.appliances if a.id == mapping["generator"]["der"])
    assert device.harmonic_impedance is None
    assert device.spectrum is None


def test_native_nonstandard_phase_scope_is_explicit():
    circuit("Generator")
    dss.Text.Command("edit generator.der phases=2")
    dss.Text.Command("solve")
    from pgml.errors import ConversionError

    with pytest.raises(ConversionError, match="der_harmonics=False"):
        to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
    grid, mapping = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE, der_harmonics=False)
    assert mapping["generator"]["der"] in {a.id for a in grid.appliances}
