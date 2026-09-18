"""Frequency law of the lumped zero-sequence reactance, default and options.

The shipped law is ``linear`` (``X0(h) = h*X0``): it is exact for a metallic return
path and safe for an ``X0`` derived from an ``X0/X1`` ratio. ``carson_sublinear``
subtracts the decay of a deep-earth return and is meant for overhead lines whose stored
``X0`` contains that earth term. For a cable-like ``X0`` it runs out of reactance within
ordinary harmonic orders; the guard then clamps at zero and assembly reports it.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from pgml import defaults
from pgml.assembly import assemble_network_ybus
from pgml.assembly import ybus as ybus_module
from pgml.geometry import apply_default_harmonic_model
from pgml.geometry.sequence import x0_sublinear_deficit
from pgml.schemas.grid_schema import EarthReturnModel, Grid, Line, Node, Phase, Source

RDT, CDT = torch.float64, torch.complex128
F0 = 50.0
PH = (Phase.A, Phase.B, Phase.C)
# NAYY 4x150 LV cable with ratio-derived zero sequence (R0 = 4 R1, X0 = 3 X1), ohm/m.
CABLE = dict(r1=0.208e-3, x1=0.080e-3, r0=0.832e-3, x0=0.240e-3)
# LV overhead line: X0 = 0.87 ohm/km. Still below the deep-earth term, yet large enough
# to stay positive far beyond the harmonic range.
OVERHEAD = dict(r1=0.306e-3, x1=0.29e-3, r0=1.224e-3, x0=0.87e-3)
LENGTH_M = 100.0


def _phase_matrix(z1: float, z0: float) -> list[list[float]]:
    zs, zm = (z0 + 2.0 * z1) / 3.0, (z0 - z1) / 3.0
    return [[zs if i == j else zm for j in range(3)] for i in range(3)]


def _grid(params: dict, n_lines: int = 1) -> Grid:
    w0 = 2.0 * math.pi * F0
    nodes = [Node(id=k, u_rated_v=400.0, phases=PH) for k in range(1, n_lines + 2)]
    lines = [
        Line(
            id=10 + k,
            from_node=k + 1,
            to_node=k + 2,
            from_phases=PH,
            to_phases=PH,
            length_m=LENGTH_M,
            series_resistance_ohm_per_m=_phase_matrix(params["r1"], params["r0"]),
            series_inductance_h_per_m=_phase_matrix(
                params["x1"] / w0, params["x0"] / w0
            ),
            shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
        )
        for k in range(n_lines)
    ]
    source = Source(
        id=1,
        node=1,
        phases=PH,
        u_ref_v=(230.0, 230.0, 230.0),
        u_angle_deg=(0.0, -120.0, 120.0),
        resistance_ohm=[[1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)],
        inductance_h=[[1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)],
    )
    grid = Grid(base_frequency_hz=F0, nodes=nodes, branches=lines, appliances=[source])
    return apply_default_harmonic_model(grid)


def _line_z0(grid: Grid, order: int) -> complex:
    """Zero-sequence series impedance (ohm/m) of the first line, read from Y(h)."""
    yb = assemble_network_ybus(grid, torch.tensor([order * F0], dtype=RDT), dtype=CDT)
    y = yb.Y.reshape(yb.index.size, yb.index.size)
    z_abc = torch.linalg.inv(-y[:3, 3:6]) / LENGTH_M
    return complex(z_abc.sum(-1).mean())


@pytest.fixture(autouse=True)
def _fresh_report_registry():
    ybus_module._X0_DEFICIT_REPORTED.clear()
    yield
    ybus_module._X0_DEFICIT_REPORTED.clear()


def test_default_law_is_linear_for_a_resolved_grid():
    """A grid resolved from the defaults stores no law and assembles ``X0*h``."""
    grid = _grid(CABLE)
    line = grid.branches[0]
    assert line.harmonic_line_model == "sequence_aware"
    assert line.earth_return is None
    assert defaults.get("line.earth_return.x0_frequency") == "linear"
    for order in (1, 5, 13, 25, 39):
        assert _line_z0(grid, order).imag == pytest.approx(
            order * CABLE["x0"], rel=1e-12
        )


def test_opendss_preset_selects_unguarded_sublinear_law():
    """The preset acts at assembly: same grid, Carson decay, no clamp."""
    grid = _grid(CABLE)
    expected = 13 * (CABLE["x0"] - 1.5 * 4e-7 * math.pi * F0 * math.log(13))
    assert expected < 0.0
    with defaults.use_preset("opendss"):
        assert _line_z0(grid, 13).imag == pytest.approx(expected, rel=1e-10)
    assert _line_z0(grid, 13).imag == pytest.approx(13 * CABLE["x0"], rel=1e-12)


def test_guard_clamps_and_assembly_warns_once_with_line_count(caplog):
    grid = _grid(CABLE, n_lines=3)
    for ln in grid.branches[:2]:
        ln.earth_return = EarthReturnModel(x0_frequency="carson_sublinear")
    with caplog.at_level(logging.WARNING, logger="pgml"):
        assert _line_z0(grid, 13).imag == pytest.approx(0.0, abs=1e-18)
        _line_z0(grid, 13)
    hits = [r for r in caplog.records if "carson_sublinear" in r.getMessage()]
    assert len(hits) == 1
    message = hits[0].getMessage()
    assert "2 of 3 sequence-aware line(s)" in message
    assert "clamped at zero" in message


def test_no_warning_below_the_clamp_order_or_for_an_overhead_line(caplog):
    cable = _grid(CABLE)
    cable.branches[0].earth_return = EarthReturnModel(x0_frequency="carson_sublinear")
    overhead = _grid(OVERHEAD)
    overhead.branches[0].earth_return = EarthReturnModel(
        x0_frequency="carson_sublinear"
    )
    with caplog.at_level(logging.WARNING, logger="pgml"):
        assert 0.0 < _line_z0(cable, 11).imag < 11 * CABLE["x0"]
        z0 = _line_z0(overhead, 39).imag
        _line_z0(_grid(CABLE), 39)  # linear law: nothing to report
    assert 0.0 < z0 < 39 * OVERHEAD["x0"]
    assert not [r for r in caplog.records if "carson_sublinear" in r.getMessage()]


def test_unguarded_negative_reactance_is_reported(caplog):
    grid = _grid(CABLE)
    grid.branches[0].earth_return = EarthReturnModel(
        x0_frequency="carson_sublinear", x0_nonnegative=False
    )
    with caplog.at_level(logging.WARNING, logger="pgml"):
        assert _line_z0(grid, 15).imag < 0.0
    assert any("NEGATIVE" in r.getMessage() for r in caplog.records)


def test_deficit_mask_matches_the_analytic_zero_crossing():
    """``X0(h) < 0`` iff ``ln h > X0 / (1.5*mu0*f0)``; h = 12.8 for the cable."""
    orders = torch.arange(1, 41, dtype=RDT)
    mask = x0_sublinear_deficit(
        torch.tensor([CABLE["x0"], OVERHEAD["x0"]], dtype=RDT), F0, orders * F0
    )
    crossing = math.exp(CABLE["x0"] / (1.5 * 4e-7 * math.pi * F0))
    assert 12.0 < crossing < 13.0
    assert torch.equal(mask[0], orders > crossing)
    assert not mask[1].any()
