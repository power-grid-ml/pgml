"""``pgml.convert.opendss.from_grid``: OpenDSS solves the exported model like pgml."""

from __future__ import annotations

import math

import pytest

dss = pytest.importorskip("opendssdirect", exc_type=ImportError)
pytestmark = pytest.mark.opendss

from pgml import defaults  # noqa: E402
from pgml.convert.opendss import UnsupportedGridError, from_grid  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

from ._solved import max_voltage_error  # noqa: E402
from .test_export_solved_fidelity import switch_grid, transformer_grid  # noqa: E402


def _dss_voltages_pu(grid, exported):
    """Phase-A voltage of every node from the live circuit, per unit."""
    out = {}
    for node in grid.nodes:
        dss.Circuit.SetActiveBus(exported.bus_of_node[node.id])
        v = dss.Bus.Voltages()
        base = float(node.u_rated_v) / (
            math.sqrt(3.0) if len(node.phases) >= 3 else 1.0
        )
        out[node.id] = complex(v[0], v[1]) / base
    return out


def _pgml_voltages_pu(grid, **kwargs):
    res = solve_power_flow(grid, tol=1e-12, **kwargs)
    assert bool(res.converged)
    v = res.v.detach().cpu().numpy().reshape(-1)
    out = {}
    for node in grid.nodes:
        base = float(node.u_rated_v) / (
            math.sqrt(3.0) if len(node.phases) >= 3 else 1.0
        )
        out[node.id] = complex(v[res.index.row(node.id, node.phases[0])]) / base
    return out


def test_single_phase_feeder_matches():
    grid = switch_grid()
    # the switch carries shunt terms the OpenDSS switch element does not
    with pytest.raises(UnsupportedGridError, match="shunt terms"):
        from_grid(grid)
    for br in grid.branches:
        if br.component == "switch":
            br.shunt_capacitance_f = 0.0
            br.shunt_conductance_s = 0.0
    with defaults.use_preset("opendss"):
        exported = from_grid(grid)
        assert exported.report.is_exact("fundamental"), exported.report.summary()
        assert "approx.source.ideal_as_finite" in exported.report
        theirs = _dss_voltages_pu(grid, exported)
        ours = _pgml_voltages_pu(grid)
    assert max_voltage_error(ours, theirs) < 1e-7


def test_three_phase_transformer_grid_matches_under_the_preset():
    grid = transformer_grid(weak_source=True)
    with defaults.use_preset("opendss"):
        exported = from_grid(grid, load_shunt="none")
        open_keys = [e.key for e in exported.report.open_entries("fundamental")]
        # OpenDSS solves behind the Vsource impedance; the report names the argument
        assert open_keys == ["model.source.impedance"]
        (entry,) = exported.report.get("model.source.impedance")
        slack = entry.match.arguments["solve_power_flow.slack"]
        theirs = _dss_voltages_pu(grid, exported)
        ours = _pgml_voltages_pu(grid, slack=slack)
    assert max_voltage_error(ours, theirs) < 1e-6
    # the magnetizing placement is closed by the preset and stays listed
    (placement,) = exported.report.get("model.transformer.magnetizing_placement")
    assert placement.matched


def test_control_laws_need_the_approximation_and_are_recorded():
    grid = transformer_grid(controlled=True)
    with pytest.raises(UnsupportedGridError, match="inverter control"):
        from_grid(grid, load_shunt="none")
    exported = from_grid(grid, allow_approximation=True, load_shunt="none")
    (entry,) = exported.report.get("approx.inverter_control")
    assert entry.ids == (22,)
    assert not exported.report.is_exact("fundamental")


def test_matched_mode_writes_the_earth_parameters_of_sequence_aware_lines():
    grid = transformer_grid()
    line = next(br for br in grid.branches if br.component == "line")
    line.harmonic_line_model = "sequence_aware"
    line.harmonic_skin_effect = False
    with defaults.use_preset("opendss"):
        exported = from_grid(grid, load_shunt="none")
        dss.Lines.Name(exported.line_of_branch[line.id])
        f0 = float(grid.base_frequency_hz)
        rc = defaults.get("line.earth_return.resistance_coeff_ohm_per_m_per_hz")
        kx = defaults.get("line.earth_return.reactance_coeff_ohm_per_m_per_hz")
        rho = defaults.get("line.earth_return.resistivity_ohm_m")
        assert dss.Lines.Rg() == pytest.approx(rc * f0)
        assert dss.Lines.Xg() == pytest.approx(
            kx * f0 * math.log(658.5 * math.sqrt(rho / f0))
        )
        assert "model.line.earth_return_clamp" not in exported.report
        (law,) = exported.report.get("model.line.earth_return_law")
        assert law.matched
        assert "model.line.skin_effect" not in exported.report
    exported = from_grid(grid, load_shunt="none")  # under the pgml defaults
    assert ("model.line.earth_return_clamp" in exported.report) == bool(
        defaults.get("line.earth_return.x0_nonnegative")
        and defaults.get("line.earth_return.x0_frequency") == "carson_sublinear"
    )
