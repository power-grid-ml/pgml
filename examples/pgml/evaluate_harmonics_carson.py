"""OpenDSS vs pgml harmonic evaluation with the Carson/Deri geometry model.

For IEEE-33 and the CIGRE LV residential feeder: synthesize single-conductor Carson
geometry reproducing each line's R/X, run the pgml harmonic flow, and compare to
OpenDSS's line model (same synthesized geometry, same converged injection). Generates
paper-ready / interactive figures via ``pgml.evaluation``.

Run::

    pixi run -e cpu python examples/pgml/evaluate_harmonics_carson.py [out_dir]

Per feeder (default ``evaluation_output/carson/<feeder>/``):
- ``ybus_h5.svg``          — pgml Y(5·f0) vs OpenDSS SystemY(5·f0) (|Y|, log).
- ``ybus_h5_diff.svg``     — |ΔY| of the two (near floating-point zero).
- ``harmonic_h5.svg``      — h=5 magnitude/angle profile, pgml vs OpenDSS.
- ``harmonic_3d.html``     — interactive 3D (h=5,7,11,13), pgml vs OpenDSS.
- ``models_h7.svg`` / ``models_h7_interactive.html`` — h=7: single-conductor Carson
  (=OpenDSS), the config-default model, positive-seq and naive overlaid (the HTML carries
  every model at once; click the legend to toggle).
- ``carson_vs_default_h7.svg`` — pairwise: Carson geometry vs the config-default model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pgml.assembly import assemble_network_ybus, node_phase_index
from pgml.assembly._stamps import _cdtype, _rdtype
from pgml.assembly.ybus import _stamp_sources
from pgml.geometry.synthesis import (
    apply_default_harmonic_model,
    apply_positive_sequence_harmonic_model,
)
from pgml.schemas.grid_schema import Line
from pgml.solver import solve_harmonic_flow

from pgml import evaluation as ev
from pgml.evaluation import oracles as ref

CDT = torch.complex128
ORDERS_3D = [5, 7, 11, 13]


# Example outputs are anchored at examples/ (not the cwd), so a run writes
# under examples/evaluation_output/ rather than the repository root.
_OUT = Path(__file__).resolve().parent.parent / "evaluation_output"


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

    # ---- line models at h=7: Carson geometry vs config default vs naive -----
    # config default for these 1-phase feeders -> positive_sequence (skin on R).
    res_default = solve_harmonic_flow(
        apply_default_harmonic_model(_strip_geometry(builder()[0])),
        [1, 7],
        slack="norton",
        dtype=CDT,
    )
    res_pos = solve_harmonic_flow(
        apply_positive_sequence_harmonic_model(_strip_geometry(builder()[0])),
        [1, 7],
        slack="norton",
        dtype=CDT,
    )
    res_naive = solve_harmonic_flow(
        _strip_geometry(builder()[0]), [1, 7], slack="norton", dtype=CDT
    )
    carson7 = next(p for p in pgml_h if p.order == 7)
    default7 = ev.harmonic_profile(
        res_default, grid, 7, label="config default (pos-seq)"
    )
    pos7 = ev.harmonic_profile(res_pos, grid, 7, label="positive-seq (skin)")
    naive7 = ev.harmonic_profile(res_naive, grid, 7, label="naive (R const, X∝h)")
    dss7 = next(
        (p for p in dss_h if p.order == 7), None
    )  # OpenDSS (Carson geometry) overlay
    models7 = [carson7, default7, pos7, naive7] + ([dss7] if dss7 else [])

    # static overlay (semi-transparent) + interactive (ALL models, toggle each).
    fig, _ = ev.plot_harmonic_profile(
        models7, grid=grid, alpha=0.55, title=f"{name}: line models compared (h=7)"
    )
    ev.save_figure(fig, out / "models_h7.svg")
    ev.plot_harmonic_profile_interactive(
        models7,
        grid=grid,
        title=f"{name}: harmonic line models (h=7) — click legend to toggle",
        out_html=str(out / "models_h7_interactive.html"),
    )
    # pairwise matplotlib: single-conductor Carson (=OpenDSS) vs the config default.
    fig_cmp, fig_diff = ev.plot_harmonic_model_comparison(
        carson7,
        default7,
        grid=grid,
        title=f"{name} h=7: Carson geometry vs config default",
    )
    ev.save_figure(fig_cmp, out / "carson_vs_default_h7.svg")
    ev.save_figure(fig_diff, out / "carson_vs_default_h7_diff.svg")
    print(f"[{name}] wrote figures to {out.resolve()}")


def main(out_dir: str = str(_OUT / "carson")) -> None:
    out = Path(out_dir)
    run_feeder("ieee33", ref.ieee33_geometry_grid, out)
    run_feeder("cigre_lv", ref.cigre_lv_geometry_grid, out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "carson"))
