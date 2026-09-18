"""The OpenDSS importer reads each line's Rg/Xg/rho and can reproduce its law.

A three-phase R/X line is defined in OpenDSS with non-default earth-return
parameters. Imported with ``earth_return="opendss"``, pgml's assembled series
impedance at every harmonic order equals OpenDSS's own YPrim of the line; imported
with the default, the report names the parameters as a model difference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

dss = pytest.importorskip("opendssdirect", exc_type=ImportError)
pytestmark = pytest.mark.opendss

from pgml import defaults  # noqa: E402
from pgml.assembly import assemble_network_ybus  # noqa: E402
from pgml.convert.opendss import PhaseMode, to_grid  # noqa: E402
from pgml.schemas.grid_schema import Line  # noqa: E402

F0 = 50.0
RG, XG, RHO = 0.05, 0.4, 250.0  # Ohm/km, Ohm/km, Ohm m


def _circuit():
    dss.Text.Command("Clear")
    dss.Text.Command(f"Set DefaultBaseFrequency={F0:g}")
    dss.Text.Command("New Circuit.earth basekv=0.4 phases=3 bus1=b1 pu=1.0")
    dss.Text.Command(
        "New Line.l1 bus1=b1.1.2.3 bus2=b2.1.2.3 phases=3 r1=0.162 x1=0.0554 "
        f"r0=0.648 x0=0.1662 c1=0 c0=0 length=0.1 units=km rg={RG} xg={XG} rho={RHO}"
    )
    dss.Text.Command("New Load.ld bus1=b2 phases=3 kv=0.4 kw=10 kvar=2")
    dss.Text.Command("Set voltagebases=[0.4]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")


def _dss_line_z(h):
    dss.Text.Command(f"set frequency={h * F0}")
    dss.Solution.BuildYMatrix(2, 1)
    dss.Circuit.SetActiveElement("Line.l1")
    yp = np.array(dss.CktElement.YPrim())
    n = int(round((len(yp) / 2) ** 0.5))
    yy = (yp[0::2] + 1j * yp[1::2]).reshape(n, n)
    return np.linalg.inv(-yy[: n // 2, n // 2 :])


def test_opendss_earth_parameters_reproduce_the_line_at_every_order():
    _circuit()
    with defaults.use_preset("opendss"):
        grid, id_map, report = to_grid(
            dss,
            phase_mode=PhaseMode.THREE_PHASE,
            earth_return="opendss",
            return_report=True,
        )
    line_id = id_map["line"]["l1"]
    ln = next(b for b in grid.branches if b.id == line_id)
    assert ln.harmonic_line_model == "sequence_aware"
    er = ln.earth_return
    assert er.resistance_coeff_ohm_per_m_per_hz == pytest.approx(RG / 1e3 / F0)
    kxg = XG / 1e3 / math.log(658.5 * math.sqrt(RHO / F0))
    assert er.reactance_coeff_ohm_per_m_per_hz == pytest.approx(kxg / F0)
    assert er.x0_frequency == "carson_sublinear"
    assert er.x0_nonnegative is False
    assert er.r0_includes_earth_return is True
    (entry,) = report.get("model.line.earth_return_parameters")
    assert entry.matched
    assert entry.values["rg_ohm_per_m_max"] == pytest.approx(RG / 1e3)

    # the line was resolved inside the preset, so skin effect is off and the
    # two engines apply the same law
    import torch

    for h in (1, 3, 5, 7, 11, 13):
        _circuit()
        zd = _dss_line_z(h)
        y = assemble_network_ybus(grid, [h * F0], dtype=torch.complex128).Y[0].numpy()
        from pgml.assembly import node_phase_index

        idx = node_phase_index(grid)
        rf = [idx.row(ln.from_node, p) for p in ln.from_phases]
        rt = [idx.row(ln.to_node, p) for p in ln.to_phases]
        zp = np.linalg.inv(-y[np.ix_(rf, rt)])
        err = np.abs(zp - zd).max() / np.abs(zd).max()
        assert err < 1e-8, f"h={h}: relative error {err:.3e}"


def test_default_import_reports_the_parameters_as_a_model_difference():
    _circuit()
    grid, id_map, report = to_grid(
        dss, phase_mode=PhaseMode.THREE_PHASE, return_report=True
    )
    ln = next(b for b in grid.branches if isinstance(b, Line))
    assert ln.earth_return is None
    (entry,) = report.get("model.line.earth_return_parameters")
    assert not entry.matched
    assert entry.match.arguments == {"to_grid.earth_return": "opendss"}
    assert entry.values["rho_ohm_m_max"] == RHO
    assert entry.values["pgml_rg_ohm_per_m"] == pytest.approx(math.pi**2 * 1e-7 * F0)


def test_single_phase_equivalent_reports_the_term_as_unrepresented():
    _circuit()
    grid, id_map, report = to_grid(dss, return_report=True)
    (entry,) = report.get("model.line.earth_return_unrepresented")
    assert entry.ids == (id_map["line"]["l1"],)
    assert "model.line.earth_return_parameters" not in report
