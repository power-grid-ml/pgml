"""Native DER harmonic voltages validate impedance AND Norton initialization."""

from __future__ import annotations

import tempfile
import numpy as np
import pytest
import torch

from pgml import defaults
from pgml.convert.opendss import PhaseMode, to_grid
from pgml.solver import solve_harmonic_flow

pytestmark = pytest.mark.opendss
dss = pytest.importorskip("opendssdirect")


def native_der_circuit(kind, conn, phases=3):
    kv = 0.4 if phases == 3 else 0.23
    commands = [
        "clear",
        "set defaultbasefrequency=50",
        f"set datapath={tempfile.mkdtemp(prefix='pgml_native_der_')}",
        f"new circuit.der basekv={kv} phases={phases} bus1=source pu=1 frequency=50 r1=.005 x1=.006 r0=.007 x0=.008",
        "new spectrum.clean numharm=1 harmonic=[1] %mag=[100] angle=[0]",
        "edit vsource.source spectrum=clean",
        "new spectrum.emission numharm=4 harmonic=[1 3 5 7] %mag=[80 5 3 2] angle=[10 40 20 0]",
        f"new line.feeder bus1=source bus2=terminal phases={phases} length=1 units=km r1=.04 x1=.03 r0=.06 x0=.05 c1=0 c0=0 rg=0 xg=0",
        f"new load.demand bus1=terminal phases={phases} conn=wye kv={kv} kw=9 kvar=2 spectrum=clean vminpu=.1 vmaxpu=2",
    ]
    args = {
        "Generator": "kw=6 kvar=1 kva=20 xdpp=.2 xrdp=7",
        "PVSystem": "pmpp=6 kva=20 %r=2 %x=8 pf=1",
        "Storage": "kwrated=10 kwhrated=20 kva=20 %r=2 %x=8 state=discharging %discharge=50",
    }[kind]
    commands += [
        f"new {kind}.der bus1=terminal phases={phases} conn={conn} kv={kv} {args} spectrum=emission",
        f"set voltagebases=[{kv}]",
        "calcvoltagebases",
        "set tolerance=1e-11 maxiterations=300",
        "solve",
    ]
    for c in commands:
        dss.Text.Command(c)
    assert dss.Solution.Converged()


def native_voltages():
    vals = np.asarray(dss.Circuit.AllBusVolts()).reshape(-1, 2)
    return dict(
        zip([x.lower() for x in dss.Circuit.YNodeOrder()], vals[:, 0] + 1j * vals[:, 1])
    )


@pytest.mark.parametrize("kind", ["Generator", "PVSystem", "Storage"])
@pytest.mark.parametrize(
    "mode,expected", [("matched", (1.0e-8, 1.0e8)), ("default", (0.9, 1.1))]
)
def test_native_der_voltage_bands_follow_oracle_mode(kind, mode, expected):
    from pgml.evaluation.oracles.opendss_scenario_oracle import export_grid_to_opendss

    native_der_circuit(kind, "wye")
    grid, _ = to_grid(
        dss, phase_mode=PhaseMode.THREE_PHASE, harmonic_line_model="naive"
    )
    exported = export_grid_to_opendss(grid, mode=mode, load_shunt="none")
    device = next(
        a for a in grid.appliances if getattr(a, "harmonic_impedance", None) is not None
    )
    name = exported.generators[device.id].elements[None]
    values = []
    for property_name in ("vminpu", "vmaxpu"):
        dss.Text.Command(f"? {kind}.{name}.{property_name}")
        values.append(float(dss.Text.Result()))
    assert values == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["Generator", "PVSystem", "Storage"])
@pytest.mark.parametrize("conn,phases", [("wye", 1), ("wye", 3), ("delta", 3)])
def test_native_der_harmonic_voltage_and_norton_initialization(kind, conn, phases):
    native_der_circuit(kind, conn, phases)
    vref = {1: native_voltages()}
    grid, mapping = to_grid(
        dss, phase_mode=PhaseMode.THREE_PHASE, harmonic_line_model="naive"
    )
    with defaults.use_preset("opendss"):
        result = solve_harmonic_flow(
            grid,
            [1, 3, 5, 7],
            slack="norton",
            load_shunt="none",
            dtype=torch.complex128,
            tol=1e-11,
            max_iter=300,
        )
    assert result.converged
    assert torch.isfinite(result.v).all()
    dss.Text.Command("set neglectloady=yes")
    dss.Text.Command("set mode=harmonic")
    for h in [3, 5, 7]:
        dss.Text.Command(f"set harmonics=[{h}]")
        dss.Text.Command("solve")
        assert dss.Solution.Converged()
        vref[h] = native_voltages()
    inverse = {n: bus for bus, n in mapping["bus"].items()}
    phase_num = {"a": 1, "b": 2, "c": 3, "n": 4}
    keys = [
        f"{inverse[result.index.node_id_of(i)]}.{phase_num[result.index.phase_of(i).value]}"
        for i in range(result.index.size)
    ]
    actual = result.v.detach().numpy().reshape(4, -1)
    for k, h in enumerate([1, 3, 5, 7]):
        reference = np.array([vref[h][key] for key in keys])
        np.testing.assert_allclose(
            actual[k], reference, atol=2e-6, rtol=2e-7, err_msg=f"{kind} {conn} h={h}"
        )


@pytest.mark.parametrize(
    "kind,power_scale",
    [
        ("Generator", 0.7),
        ("Generator", 3.0),
        ("PVSystem", 0.7),
        ("PVSystem", 3.0),
        ("Storage", 0.7),
        ("Storage", 3.0),
        ("Storage", -0.7),
    ],
)
@pytest.mark.parametrize("conn,phases", [("wye", 1), ("wye", 3), ("delta", 3)])
def test_native_der_round_trip_preserves_class_and_emission(
    kind, conn, phases, power_scale
):
    from pgml.evaluation.oracles.opendss_scenario_oracle import (
        export_grid_to_opendss,
        _extract_voltages,
        _apply_pq,
    )

    native_der_circuit(kind, conn, phases)
    grid, _ = to_grid(
        dss, phase_mode=PhaseMode.THREE_PHASE, harmonic_line_model="naive"
    )
    with defaults.use_preset("opendss"):
        ours = solve_harmonic_flow(
            grid,
            [1, 3, 5, 7],
            slack="norton",
            load_shunt="none",
            tol=1e-11,
            max_iter=300,
        )
        exported = export_grid_to_opendss(grid, load_shunt="none")
    assert ours.converged
    device = next(
        a for a in grid.appliances if getattr(a, "harmonic_impedance", None) is not None
    )
    native = exported.generators[device.id]
    assert native.dss_class == kind
    reference = [_extract_voltages(dss, exported.rowmap, ours.index.size)]
    dss.Text.Command("set mode=harmonic")
    for h in [3, 5, 7]:
        dss.Text.Command(f"set harmonics=[{h}]")
        dss.Text.Command("solve")
        assert dss.Solution.Converged()
        reference.append(_extract_voltages(dss, exported.rowmap, ours.index.size))
    np.testing.assert_allclose(
        ours.v.detach().numpy().reshape(4, -1),
        np.asarray(reference),
        atol=2e-6,
        rtol=2e-7,
    )
    # The scenario edit must target each native class's actual active-power setter.
    dss.Text.Command("set mode=snap")
    _apply_pq(
        dss,
        native,
        {"p_w": float(device.p_nom_w) * power_scale, "q_var": float(device.q_nom_var)},
        0,
    )
    dss.Text.Command("solve")
    assert dss.Solution.Converged()
    dss.Circuit.SetActiveElement(f"{kind}.{native.elements[None]}")
    power = np.asarray(dss.CktElement.Powers()).reshape(-1, 2).sum(axis=0)
    assert -power[0] * 1000 == pytest.approx(
        float(device.p_nom_w) * power_scale, rel=2e-6, abs=1e-3
    )

    # Resizing an auxiliary native rating must preserve both the impedance and
    # the emitted voltage spectrum at the changed operating point.
    reference = [_extract_voltages(dss, exported.rowmap, ours.index.size)]
    with defaults.use_preset("opendss"):
        changed = solve_harmonic_flow(
            grid,
            [1, 3, 5, 7],
            slack="norton",
            load_shunt="none",
            operating_point={
                device.id: {
                    "p_w": float(device.p_nom_w) * power_scale,
                    "q_var": float(device.q_nom_var),
                }
            },
            tol=1e-11,
            max_iter=300,
        )
    assert changed.converged
    dss.Text.Command("set mode=harmonic")
    for h in [3, 5, 7]:
        dss.Text.Command(f"set harmonics=[{h}]")
        dss.Text.Command("solve")
        assert dss.Solution.Converged()
        reference.append(_extract_voltages(dss, exported.rowmap, ours.index.size))
    np.testing.assert_allclose(
        changed.v.detach().numpy().reshape(4, -1),
        np.asarray(reference),
        atol=2e-6,
        rtol=2e-7,
    )
