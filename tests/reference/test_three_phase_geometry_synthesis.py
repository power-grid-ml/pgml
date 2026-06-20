"""Three-phase R/X -> Carson geometry synthesis.

The synthesized equilateral 3-conductor geometry must reproduce the target
positive-sequence impedance ``Z1`` and zero-sequence reactance ``X0`` at the
fundamental, recomputed through pgml's own Carson forward (``line_constants`` +
Fortescue). This is the pgml-side correctness check; the OpenDSS bit-exact harmonic
parity (the same geometry fed to OpenDSS) lives in the live-OpenDSS oracle tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.geometry.carson import line_constants
from pgml.geometry.sequence import sequence_impedances
from pgml.geometry.synthesis import synthesize_three_phase_geometry


def _carson_sequence(geom, f0):
    xs = [c.x_m for c in geom.conductors]
    ys = [c.y_m for c in geom.conductors]
    gmr = geom.conductors[0].gmr_m
    rdc = geom.conductors[0].r_dc_ohm_per_m
    rad = geom.conductors[0].radius_m
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    g = torch.full((3,), float(gmr), dtype=torch.float64)
    r = torch.full((3,), float(rdc), dtype=torch.float64)
    radv = torch.full((3,), float(rad), dtype=torch.float64)
    z, _c = line_constants(
        x,
        y,
        g,
        r,
        radv,
        float(geom.earth_resistivity_ohm_m),
        torch.tensor([f0], dtype=torch.float64),
        3,
    )
    z0, z1, _z2 = sequence_impedances(z[0])
    return complex(z0.item()), complex(z1.item())


# (R1, X1, X0) Ω/m at 50 Hz across overhead-ish and cable-ish ranges.
CASES = [
    (3.0e-4, 3.5e-4, 1.05e-3),  # X0/X1 = 3
    (2.0e-4, 2.5e-4, 7.5e-4),
    (5.0e-4, 1.5e-4, 6.0e-4),  # X0/X1 = 4
]


@pytest.mark.parametrize("r1,x1,x0", CASES)
def test_synthesis_reproduces_z1_x0_at_f0(r1, x1, x0):
    """pgml Carson on the synthesized geometry reproduces R1, X1, X0 at f0."""
    f0 = 50.0
    geom = synthesize_three_phase_geometry(r1, x1, x0, f0=f0)
    z0, z1 = _carson_sequence(geom, f0)
    assert math.isclose(z1.real, r1, rel_tol=1e-8), f"R1: {z1.real} vs {r1}"
    assert math.isclose(z1.imag, x1, rel_tol=1e-8), f"X1: {z1.imag} vs {x1}"
    assert math.isclose(z0.imag, x0, rel_tol=1e-7), f"X0: {z0.imag} vs {x0}"


def test_r0_follows_earth_physics():
    """R0 is the geometry's physical earth-return value (R0 > R1), recorded in provenance."""
    f0 = 50.0
    geom = synthesize_three_phase_geometry(3.0e-4, 3.5e-4, 1.05e-3, f0=f0)
    z0, z1 = _carson_sequence(geom, f0)
    assert z0.real > z1.real  # earth return adds zero-sequence resistance
    recorded = float(geom.provenance.extra["synth_r0_ohm_per_m"])
    assert math.isclose(recorded, z0.real, rel_tol=1e-6)


def test_grid_synthesis_dispatches_by_phase_count():
    """synthesize_grid_geometry builds 3-conductor geometries for 3-phase lines."""
    import numpy as np

    np.Inf = np.inf  # type: ignore[attr-defined]
    np.in1d = np.isin  # type: ignore[attr-defined]
    import warnings

    from pgml.convert.pandapower import PhaseMode
    from pgml.evaluation.references import cigre_lv_full_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry
    from pgml.schemas.grid_schema import Line

    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synthesize_grid_geometry(grid)
    lines = [b for b in grid.branches if isinstance(b, Line)]
    assert lines, "expected lines in the CIGRE LV grid"
    for ln in lines:
        if len(ln.from_phases) == 3:
            assert ln.conductor_geometry is not None
            assert len(ln.conductor_geometry.conductors) == 3
