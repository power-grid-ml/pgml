"""OpenDSS ``InvControl`` imports as pgml inverter control laws.

A one-phase feeder with a PVSystem under InvControl is solved by OpenDSS (its
own control sweeps) and, after import, by pgml (the control law inside the
Newton solve). The node voltages agree to the tolerance of OpenDSS's finite
control iteration, the same figure the library's reference parity tests report.
"""

from __future__ import annotations


import pytest
import torch

dss = pytest.importorskip("opendssdirect", exc_type=ImportError)
pytestmark = pytest.mark.opendss

from pgml.convert.opendss import to_grid  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    Generator,
    VoltVarControl,
    VoltVarVoltWattControl,
    VoltWattControl,
)
from pgml.solver import solve_power_flow  # noqa: E402

U_LL_KV = 0.4
ATOL_V = 0.1  # OpenDSS control-sweep residual, as in the reference parity tests


def _circuit(*, mode: str, length_km: float = 1.0, p_kw: float = 20.0):
    dss.Text.Command("Clear")
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command(
        f"New Circuit.inv basekv={U_LL_KV} phases=1 bus1=bus0.1 r1=0.001 x1=0.000001"
    )
    dss.Text.Command(
        "New Line.l1 bus1=bus0.1 bus2=bus1.1 phases=1 r1=0.3 x1=0.15 r0=0.3 x0=0.15 "
        f"c1=0 c0=0 length={length_km} units=km"
    )
    dss.Text.Command(
        f"New Load.ld1 phases=1 bus1=bus1.1 kv={U_LL_KV} kw=3 kvar=1 model=1 "
        "vminpu=0.1 vmaxpu=10"
    )
    dss.Text.Command(
        "New XYcurve.vvc npts=3 xarray=[0.90 1.00 1.10] yarray=[0.3 0.0 -0.3]"
    )
    dss.Text.Command(
        "New XYcurve.vw npts=4 xarray=[0.9 1.02 1.05 1.2] yarray=[1.0 1.0 0.2 0.2]"
    )
    dss.Text.Command(
        f"New PVSystem.pv1 phases=1 bus1=bus1.1 kv={U_LL_KV} kva=25 pmpp={p_kw} "
        "pf=1.0 irradiance=1 %cutin=0 %cutout=0"
    )
    if mode == "voltvar":
        dss.Text.Command(
            "New InvControl.ic1 PVSystemList=[pv1] mode=VOLTVAR "
            "voltage_curvex_ref=rated vvc_curve1=vvc RefReactivePower=VARMAX"
        )
    elif mode == "voltwatt":
        dss.Text.Command(
            "New InvControl.ic1 PVSystemList=[pv1] mode=VOLTWATT "
            "voltwatt_curve=vw VoltwattYAxis=PMPPPU"
        )
    elif mode == "combi":
        dss.Text.Command(
            "New InvControl.ic1 PVSystemList=[pv1] CombiMode=VV_VW "
            "voltage_curvex_ref=rated vvc_curve1=vvc voltwatt_curve=vw "
            "RefReactivePower=VARMAX VoltwattYAxis=PMPPPU"
        )
    elif mode == "avr":
        dss.Text.Command("New InvControl.ic1 PVSystemList=[pv1] mode=AVR Vsetpoint=1.0")
    dss.Text.Command(f"Set voltagebases=[{U_LL_KV}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Set MaxControlIter=100")
    dss.Text.Command("Set mode=snapshot")
    dss.Text.Command("Solve")
    assert dss.Solution.Converged()


def _dss_voltages():
    out = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v = dss.Bus.Voltages()
        out[bus.lower()] = complex(v[0], v[1])
    return out


def _pgml_voltages(grid, id_map):
    res = solve_power_flow(
        grid, slack="norton", method="newton", dtype=torch.complex128
    )
    assert res.converged
    v = res.v.detach().cpu().numpy().reshape(-1)
    return {
        bus: v[res.index.row(node, grid.nodes[0].phases[0])]
        for bus, node in id_map["bus"].items()
    }


@pytest.mark.parametrize(
    "mode, law",
    [
        ("voltvar", VoltVarControl),
        ("voltwatt", VoltWattControl),
        ("combi", VoltVarVoltWattControl),
    ],
)
def test_invcontrol_maps_to_the_control_law_and_matches_opendss(mode, law):
    _circuit(mode=mode)
    theirs = _dss_voltages()
    grid, id_map, report = to_grid(dss, harmonic_line_model="none", return_report=True)
    pv = next(a for a in grid.appliances if isinstance(a, Generator))
    assert isinstance(pv.control, law)
    assert pv.control.s_rated_va == pytest.approx(25e3)
    if mode != "voltvar":
        assert pv.p_nom_w == pytest.approx(20e3)  # the uncurtailed available power
    assert "dropped.invcontrol" not in report
    ours = _pgml_voltages(grid, id_map)
    for bus, v in theirs.items():
        assert abs(abs(ours[bus]) - abs(v)) < ATOL_V, (mode, bus, ours[bus], v)
    if mode != "voltvar":
        # the curve knee at 1.02 pu binds on this feeder
        assert abs(theirs["bus1"]) / (U_LL_KV * 1e3) > 1.02


def test_unmapped_invcontrol_mode_is_reported_as_dropped(caplog):
    _circuit(mode="avr")
    grid, id_map, report = to_grid(dss, harmonic_line_model="none", return_report=True)
    pv = next(a for a in grid.appliances if isinstance(a, Generator))
    assert pv.control is None
    (entry,) = report.get("dropped.invcontrol")
    assert entry.ids == ("ic1",)
    assert "mode=AVR" in entry.values["ic1"]
    assert any("InvControl 'ic1'" in r.getMessage() for r in caplog.records)
    assert "approx.pvsystem.snapshot" in report
