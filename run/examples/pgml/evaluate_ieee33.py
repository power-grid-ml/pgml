"""Regenerate the IEEE 33-bus evaluation figure set (reference libraries vs pgml).

Run::

    pixi run -e cpu python run/examples/pgml/evaluate_ieee33.py [out_dir]

Produces (default ``data/pgml/evaluation_output/``):
- ``ybus_heatmaps.svg``       — our full Y vs pandapower network Y vs OpenDSS SystemY.
- ``ybus_difference.svg``     — |ΔY| of the two most-similar versions (our network Y
                                vs pandapower network Y) — near floating-point zero.
- ``voltage_profile.svg``     — our nonlinear power flow vs pandapower (pu vs distance).
- ``voltage_error.svg``       — per-node |Δpu| bar chart.
- ``harmonic_h5.svg``         — h=5 magnitude/angle profile, pgml vs numpy oracle.
- ``harmonic_3d.html``        — interactive 3D (h=3,5,7,9), pgml vs numpy oracle.
- ``harmonic_models_interactive.html`` — config-default vs naive line model, every
                                harmonic × model toggleable.
- ``harmonic_default_vs_naive_h5.svg`` — pairwise h=5: config default vs naive.
- ``grid_voltage_map.svg``    — topology colored by voltage pu.

The harmonic figures attach a typical 6-pulse converter spectrum to a few loads.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

import pandapower as pp
import pandapower.networks as pn

from pgml.assembly import assemble_network_ybus, assemble_ybus, node_phase_index
from pgml.convert.pandapower import to_grid
from pgml.schemas.grid_schema import (
    HarmonicComponent,
    Load,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.geometry.synthesis import apply_default_harmonic_model
from pgml.solver import solve_harmonic_flow, solve_power_flow

from pgml import evaluation as ev
from pgml.evaluation import oracles as ref

CDT = torch.complex128
# Spectrum hand-attached to a few deep-bus loads for this figure set only (mag
# relative to fundamental). Unlike the idealized 6-pulse spectrum in
# pgml.grids.CONVERTER_SPECTRUM, this adds a small order-9 component so the
# h=3,5,7,9 harmonic-profile figure (HARMONIC_ORDERS_3D below) has a nonzero
# series to plot at h=9.
EVALUATION_SPECTRUM = [
    (1, 1.0, 0.0),
    (5, 0.20, 0.0),
    (7, 0.14, 0.0),
    (9, 0.03, 0.0),
    (11, 0.09, 0.0),
    (13, 0.07, 0.0),
]
HARMONIC_ORDERS_3D = [5, 7, 9, 11, 13]


# Example outputs are anchored at the repository root (not the cwd), so a run writes
# under the untracked data root at data/pgml/evaluation_output/.
_OUT = Path(__file__).resolve().parents[3] / "data" / "pgml" / "evaluation_output"


def _build_dss_passive(net) -> None:
    """Single-phase positive-sequence IEEE-33 passive circuit in OpenDSS (no loads)."""
    import opendssdirect as dss

    f0 = float(net.f_hz)
    vn_kv = float(net.bus.at[0, "vn_kv"])
    r1_tiny = 1.0e-6
    x1_tiny = 2.0 * math.pi * f0 * 1.0e-12
    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.ieee33_passive basekv={vn_kv} pu=1.0 phases=1 "
        f"bus1=bus0.1 r1={r1_tiny} x1={x1_tiny} frequency={f0}"
    )
    for pp_idx, row in net.line[net.line["in_service"]].iterrows():
        dss.Text.Command(
            f"New Line.line{pp_idx} phases=1 bus1=bus{int(row['from_bus'])}.1 "
            f"bus2=bus{int(row['to_bus'])}.1 r1={float(row['r_ohm_per_km'])} "
            f"x1={float(row['x_ohm_per_km'])} c1={float(row.get('c_nf_per_km', 0.0) or 0.0)} "
            f"length={float(row['length_km'])} units=km"
        )
    dss.Text.Command(f"Set voltagebases=[{vn_kv}]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("Solve")


def main(out_dir: str = str(_OUT)) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- grid + reference solve --------------------------------------------
    net = pn.case33bw()
    pp.runpp(net, numba=False)
    grid, id_map = to_grid(net)
    f0 = grid.base_frequency_hz
    index = node_phase_index(grid)

    # ---- 1. Y-bus comparison ----------------------------------------------
    ours_full = ev.labeled_matrix(
        assemble_ybus(grid, [f0], dtype=CDT).Y, index, label="pgml (full)"
    )
    ours_net = ev.labeled_matrix(
        assemble_network_ybus(grid, [f0], dtype=CDT).Y, index, label="pgml (network)"
    )
    pp_net = ref.pandapower_ybus(net, grid, id_map, index, label="pandapower (network)")
    matrices = [ours_full, pp_net]
    try:
        _build_dss_passive(net)
        y_dss, node_order = ref.dss_systemy()
        matrices.append(
            ref.opendss_ybus(y_dss, node_order, id_map, index, label="OpenDSS SystemY")
        )
    except Exception as exc:  # noqa: BLE001 - OpenDSS optional in the demo
        print(f"[warn] OpenDSS step skipped: {exc}")

    fig = ev.plot_ybus_heatmaps(
        matrices, part="abs", suptitle="IEEE 33-bus Y-bus (|Y|, log scale)"
    )
    ev.save_figure(fig, out / "ybus_heatmaps.svg")
    fig = ev.plot_ybus_difference(ours_net, pp_net)
    ev.save_figure(fig, out / "ybus_difference.svg")

    # ---- 2. Voltage profile (nonlinear PF vs pandapower) -------------------
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    ours_v = ev.voltage_profile(pf, grid, label="pgml")
    pp_v = ref.pandapower_voltage_profile(net, grid, id_map, label="pandapower")
    fig, _ = ev.plot_voltage_profile(
        [ours_v, pp_v], grid=grid, alpha=0.7, title="IEEE 33-bus voltage profile"
    )
    ev.save_figure(fig, out / "voltage_profile.svg")
    fig, _ = ev.plot_profile_error(pp_v, ours_v)
    ev.save_figure(fig, out / "voltage_error.svg")

    # ---- 3. Harmonic profiles (attach converter spectra) ------------------
    comps = [
        HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a)
        for o, m, a in EVALUATION_SPECTRUM
    ]
    for app in grid.appliances:
        if isinstance(app, Load) and app.node in (17, 32, 24):  # a few deep buses
            app.spectrum = StaticSpectrum(spectrum=SpectrumPoint(components=comps))
    orders = [1, *HARMONIC_ORDERS_3D]
    hres = solve_harmonic_flow(grid, orders, slack="ideal", dtype=CDT)

    ours_h = ev.harmonic_profiles(hres, grid, HARMONIC_ORDERS_3D, label="pgml")
    oracle_h = ref.numpy_harmonic_profiles(
        grid, hres.pf.v, index, HARMONIC_ORDERS_3D, label="numpy oracle"
    )

    o5_ours = next(p for p in ours_h if p.order == 5)
    o5_ref = next(p for p in oracle_h if p.order == 5)
    fig, _ = ev.plot_harmonic_profile([o5_ours, o5_ref], grid=grid)
    ev.save_figure(fig, out / "harmonic_h5.svg")

    ev.plot_harmonic_profile_3d(
        [*ours_h, *oracle_h],
        grid=grid,
        reference_labels=["numpy oracle"],
        title="IEEE 33-bus harmonic voltage profiles (h=3,5,7,9)",
        out_html=str(out / "harmonic_3d.html"),
    )

    # ---- 3b. Line-model comparison: config default vs naive (X∝h) ----------
    # `hres` above used the raw R/L path (naive X∝h). Re-solve with the deliberate
    # config-default model (positive-seq + skin for these 1-phase lines) and compare.
    grid_default = apply_default_harmonic_model(to_grid(net)[0])
    for app in grid_default.appliances:
        if isinstance(app, Load) and app.node in (17, 32, 24):
            app.spectrum = StaticSpectrum(spectrum=SpectrumPoint(components=comps))
    hres_default = solve_harmonic_flow(grid_default, orders, slack="ideal", dtype=CDT)
    default_h = ev.harmonic_profiles(
        hres_default, grid_default, HARMONIC_ORDERS_3D, label="config default (pos-seq)"
    )
    naive_h = ev.harmonic_profiles(
        hres, grid, HARMONIC_ORDERS_3D, label="naive (R const, X∝h)"
    )
    # interactive: both models across several harmonics, each (order, model) toggleable.
    ev.plot_harmonic_profile_interactive(
        [*default_h, *naive_h],
        grid=grid,
        title="IEEE 33-bus: config default vs naive line model — click legend to toggle",
        out_html=str(out / "harmonic_models_interactive.html"),
    )
    d5 = next(p for p in default_h if p.order == 5)
    n5 = next(p for p in naive_h if p.order == 5)
    fig_cmp, fig_diff = ev.plot_harmonic_model_comparison(
        d5, n5, grid=grid, title="IEEE 33-bus h=5: config default vs naive line model"
    )
    ev.save_figure(fig_cmp, out / "harmonic_default_vs_naive_h5.svg")
    ev.save_figure(fig_diff, out / "harmonic_default_vs_naive_h5_diff.svg")

    # ---- 4. Topology colored by voltage -----------------------------------
    v_by_node = dict(zip([int(i) for i in ours_v.node_ids], ours_v.v_pu))
    fig, _ = ev.plot_grid_graph(
        grid,
        node_values=v_by_node,
        value_label="V [pu]",
        title="IEEE 33-bus — voltage magnitude",
    )
    ev.save_figure(fig, out / "grid_voltage_map.svg")

    print(f"Wrote evaluation figures to {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "references"))
