"""OpenDSS vs pgml harmonic comparison on IEEE-33 and CIGRE LV (Carson geometry).

Each feeder is given a SYNTHESIZED single-conductor Carson geometry that reproduces
its R/X at fundamental (R/X feeders ship no conductor geometry). The SAME geometry is
fed to pgml and OpenDSS, so the harmonic comparison isolates the line model:

- the line series admittance (off-diagonal Y(h)) matches OpenDSS to ~3e-8 relative, which
  is the SI-vs-OpenDSS `mu0` constant difference (pgml uses the SI value);
- the harmonic bus voltages match (OpenDSS line model solved with pgml's converged
  injection) to ~1.4e-7 relative at every harmonic order, the same constant difference
  carried through the solve.

The orders stay below 1 kHz (20 * 50 Hz), where the two geometry models are the same; at
and above 1 kHz OpenDSS changes its conductor spacing term (GMR -> radius) and pgml does
not.

This is the end-to-end payoff of the Carson/Deri geometry path — it closes the
documented harmonic line-impedance gap on real feeders.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_network_ybus, node_phase_index
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_sources
from pgml.evaluation import oracles as ref
from pgml.geometry.synthesis import strip_grid_geometry
from pgml.schemas.grid_schema import Phase
from pgml.solver import solve_harmonic_flow

CDT = torch.complex128
ORDERS = [1, 5, 7, 11]


def _pgml_harmonic_y(grid, index, h):
    f = torch.tensor([h * float(grid.base_frequency_hz)], dtype=torch.float64)
    yb = assemble_network_ybus(
        grid, [h * float(grid.base_frequency_hz)], dtype=CDT
    ).Y.clone()
    yb = _stamp_sources(
        grid, f, yb, index, _cdtype(CDT), _rdtype(CDT), torch.device("cpu"), None
    )
    return yb[0].numpy()


@pytest.mark.parametrize(
    "builder", [ref.ieee33_geometry_grid, ref.cigre_lv_geometry_grid]
)
def test_offdiagonal_Yh_matches_opendss(builder):
    """Line series admittance Y(h) (off-diagonal) is bit-close to OpenDSS at harmonics."""
    grid, _ = builder()
    index = node_phase_index(grid)
    dssY = ref.opendss_geometry_systemy(grid, index, [1, 5, 7])
    n = index.size
    mask = ~np.eye(n, dtype=bool)
    for h in (1, 5, 7):
        yp = _pgml_harmonic_y(grid, index, h)
        yd = dssY[h]
        denom = np.abs(yd[mask]).max()
        err = np.abs(yp[mask] - yd[mask]).max() / denom
        # Measured 3.1e-8 (IEEE-33) / 1.9e-8 (CIGRE LV) relative, which is the SI-vs-
        # OpenDSS mu0 constant difference (4.9e-8) carried through Y = 1/Z; the MODEL
        # agrees to floating point. Tolerance set with headroom.
        assert err < 2e-7, f"h={h}: off-diagonal Y(h) rel error {err:.2e}"


@pytest.mark.parametrize(
    "builder", [ref.ieee33_geometry_grid, ref.cigre_lv_geometry_grid]
)
def test_harmonic_voltages_match_opendss(builder):
    """Harmonic bus voltages agree with OpenDSS's line model (same injection)."""
    grid, _ = builder()
    index = node_phase_index(grid)
    res = solve_harmonic_flow(grid, ORDERS, slack="norton", dtype=CDT)
    assert res.pf.converged

    dssY = ref.opendss_geometry_systemy(grid, index, ORDERS)
    f0 = float(grid.base_frequency_hz)
    freqs = res.frequencies_hz.numpy()
    v = res.v.numpy()
    for h in ORDERS[1:]:  # harmonics (h=1 slack node carries the tiny-shunt artifact)
        k = int(np.argmin(np.abs(freqs - h * f0)))
        vp = v[k]
        i_inj = _pgml_harmonic_y(grid, index, h) @ vp
        vd = np.linalg.solve(dssY[h], i_inj)
        rel = np.abs(vd - vp).max() / (np.abs(vp).max() + 1e-15)
        # Measured 1.4e-7 rel (IEEE-33); the bound is the SI-vs-OpenDSS mu0 constant
        # difference (4.9e-8) amplified by the voltage solve, not a model difference.
        assert rel < 1e-6, f"h={h}: |V_dss - V_pgml| rel {rel:.2e}"


def test_synthesized_geometry_is_tracked():
    """Every synthesized line carries provenance recording the synthesis (kept track of)."""
    from pgml.schemas.grid_schema import Line

    grid, _ = ref.ieee33_geometry_grid()
    geom_lines = [
        b for b in grid.branches if isinstance(b, Line) and b.conductor_geometry
    ]
    assert geom_lines
    for ln in geom_lines:
        prov = ln.conductor_geometry.provenance
        assert prov is not None and "synthes" in (prov.notes or "").lower()
        assert "synth_gmr_m" in prov.extra and "synth_rdc_ohm_per_m" in prov.extra


def test_carson_differs_from_naive_on_feeder():
    """Carson harmonic voltages differ materially from the naive R-const/X∝h model."""
    grid, _ = ref.ieee33_geometry_grid()
    index = node_phase_index(grid)
    res_geom = solve_harmonic_flow(grid, [1, 7], slack="norton", dtype=CDT)

    # Strip geometry -> falls back to the explicit R/L path (naive X∝h, R const).
    strip_grid_geometry(grid)
    res_naive = solve_harmonic_flow(grid, [1, 7], slack="norton", dtype=CDT)

    row = index.row(int(grid.nodes[-1].id), Phase.A)
    v_geom = abs(complex(res_geom.v[1, row]))
    v_naive = abs(complex(res_naive.v[1, row]))
    assert abs(v_geom - v_naive) / max(v_naive, 1e-9) > 0.01  # >1% difference at h=7
