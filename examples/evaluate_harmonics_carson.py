"""OpenDSS vs pgml harmonic evaluation with the Carson/Deri geometry model.

For IEEE-33 and the CIGRE LV residential feeder: synthesize single-conductor Carson
geometry reproducing each line's R/X, run the pgml harmonic flow, and compare to
OpenDSS's line model (same synthesized geometry, same converged injection). Generates
paper-ready / interactive figures via ``pgml.evaluation``.

Run::

    pixi run -e cpu python examples/evaluate_harmonics_carson.py [out_dir]

Per feeder (default ``evaluation_output/carson/<feeder>/``):
- ``ybus_h5.svg``          — pgml Y(5·f0) vs OpenDSS SystemY(5·f0) (|Y|, log).
- ``ybus_h5_diff.svg``     — |ΔY| of the two (near floating-point zero).
- ``harmonic_h5.svg``      — h=5 magnitude/angle profile, pgml vs OpenDSS.
- ``harmonic_3d.html``     — interactive 3D (h=5,7,11,13), pgml vs OpenDSS.
- ``carson_vs_naive.svg``  — h=7 profile: Carson vs the naive R-const/X∝h model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pgml.assembly import assemble_network_ybus, node_phase_index
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_sources
from pgml.schemas.grid_schema import Line
from pgml.solver import solve_harmonic_flow

from pgml import evaluation as ev
from pgml.evaluation import references as ref

CDT = torch.complex128
ORDERS_3D = [5, 7, 11, 13]


def _pgml_harmonic_y_labeled(grid, index, h, label):
    f = torch.tensor([h * float(grid.base_frequency_hz)], dtype=torch.float64)
    yb = assemble_network_ybus(
        grid, [h * float(grid.base_frequency_hz)], dtype=CDT
    ).Y.clone()
    yb = _stamp_sources(
        grid, f, yb, index, _cdtype(CDT), _rdtype(CDT), torch.device("cpu"), None
    )
    return ev.labeled_matrix(yb, index, label=label)


def _strip_geometry(grid):
    for b in grid.branches:
        if isinstance(b, Line):
            b.conductor_geometry = None
    return grid


def run_feeder(name: str, builder, out_dir: Path) -> None:
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    grid, _ = builder()
    index = node_phase_index(grid)
    orders = [1, *ORDERS_3D]

    res = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)

    # ---- Y(5*f0): pgml vs OpenDSS SystemY -----------------------------------
    dssY = ref.opendss_geometry_systemy(grid, index, [5])
    pgml_y5 = _pgml_harmonic_y_labeled(grid, index, 5, "pgml Y(5·f0)")
    dss_y5 = ev.LabeledMatrix(
        matrix=dssY[5], label="OpenDSS SystemY(5·f0)", row_labels=pgml_y5.row_labels
    )
    fig = ev.plot_ybus_heatmaps(
        [pgml_y5, dss_y5], part="abs", suptitle=f"{name}: harmonic Y at 5·f0 (Carson)"
    )
    ev.save_figure(fig, out / "ybus_h5.svg")
    fig = ev.plot_ybus_difference(pgml_y5, dss_y5)
    ev.save_figure(fig, out / "ybus_h5_diff.svg")

    # ---- harmonic profiles: pgml vs OpenDSS line model ----------------------
    pgml_h = ev.harmonic_profiles(res, grid, ORDERS_3D, label="pgml")
    dss_h = ref.opendss_geometry_harmonic_profiles(
        grid, res, ORDERS_3D, label="OpenDSS (Carson)"
    )

    p5 = next(p for p in pgml_h if p.order == 5)
    d5 = next(p for p in dss_h if p.order == 5)
    fig, _ = ev.plot_harmonic_profile(
        [p5, d5], grid=grid, title=f"{name}: harmonic voltage profile (h=5)"
    )
    ev.save_figure(fig, out / "harmonic_h5.svg")

    ev.plot_harmonic_profile_3d(
        [*pgml_h, *dss_h],
        grid=grid,
        reference_labels=["OpenDSS (Carson)"],
        title=f"{name}: harmonic voltage profiles (h=5,7,11,13) — pgml vs OpenDSS",
        out_html=str(out / "harmonic_3d.html"),
    )

    # ---- Carson vs naive (R const, X∝h) at h=7 ------------------------------
    res_naive = solve_harmonic_flow(
        _strip_geometry(builder()[0]), [1, 7], slack="norton", dtype=CDT
    )
    carson7 = next(p for p in pgml_h if p.order == 7)
    naive7 = ev.harmonic_profile(res_naive, grid, 7, label="naive (R const, X∝h)")
    fig, _ = ev.plot_harmonic_profile(
        [carson7, naive7], grid=grid, title=f"{name}: Carson vs naive line model (h=7)"
    )
    ev.save_figure(fig, out / "carson_vs_naive.svg")
    print(f"[{name}] wrote figures to {out.resolve()}")


def main(out_dir: str = "evaluation_output/carson") -> None:
    out = Path(out_dir)
    run_feeder("ieee33", ref.ieee33_geometry_grid, out)
    run_feeder("cigre_lv", ref.cigre_lv_geometry_grid, out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation_output/carson")
