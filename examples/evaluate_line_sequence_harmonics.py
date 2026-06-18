"""Positive-sequence-aware harmonic line model — comparative figures.

Visualises WHY the single-conductor R/X->geometry synthesis is wrong for positive-
sequence feeders and what the corrected model does instead:

1. ``seq_xr_vs_harmonic.svg`` — R(h) and X(h) of a genuine 3-phase overhead geometry
   split into zero- and positive-sequence (Fortescue). The positive sequence has NO
   earth floor (X1 ∝ h, overlaid by the direct ``positive_sequence_z`` model); the zero
   sequence carries the earth-return floor (X0 sub-linear, R0 large).
2. ``gmr_floor.svg`` — sweeping X1, the single-conductor earth-return synthesis drives
   GMR PAST the conductor radius (non-physical) below the earth floor, while the
   two-conductor go/return synthesis keeps a physical GMR and spacing for every X1.
3. ``feeder_h<k>.svg`` (+ ``feeder_h<k>_interactive.html``) — IEEE-33 harmonic voltage
   profile under the corrected model vs the naive (R const, X∝h) model vs the
   single-conductor Carson model (== OpenDSS, overlaid when OpenDSS is available). The
   single-conductor earth correction inflates the harmonic voltage drop; the corrected
   positive-sequence model removes that artifact while keeping the skin-effect resistance
   growth. The static SVG draws semi-transparent lines; the interactive HTML colours one
   model per legend entry so you can click to activate/deactivate each model.

Run::

    pixi run -e cpu python examples/evaluate_line_sequence_harmonics.py [out_dir]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pgml import evaluation as ev
from pgml.evaluation import references as ref
from pgml.geometry.carson import kron_reduce, series_impedance
from pgml.geometry.sequence import (
    positive_sequence_z,
    sequence_impedances,
    two_conductor_geometry,
)
from pgml.geometry.synthesis import (
    apply_positive_sequence_harmonic_model,
    synthesize_line_geometry,
)
from pgml.schemas.grid_schema import Line
from pgml.solver import solve_harmonic_flow

RDT = torch.float64
CDT = torch.complex128
F0 = 50.0
ORDERS = [1, 5, 7, 11, 13, 17, 25]


# --- (1) sequence R/X vs harmonic ------------------------------------------
def _three_phase_geometry_z(freqs):
    """Full Carson Z(h) [H,3,3] for a 3-phase overhead line (3 phase + neutral, Kron)."""
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=RDT)
    y = torch.tensor([10.0, 10.0, 10.0, 9.0], dtype=RDT)
    gmr = torch.tensor([0.0078, 0.0078, 0.0078, 0.0050], dtype=RDT)
    rdc = torch.tensor([0.1, 0.1, 0.1, 0.3], dtype=RDT) * 1e-3
    return kron_reduce(series_impedance(x, y, gmr, rdc, 100.0, freqs), 3)


def plot_sequence_xr(out: Path) -> None:
    orders = np.array([1, 3, 5, 7, 9, 11, 13, 17, 21, 25])
    freqs = torch.tensor([h * F0 for h in orders], dtype=RDT)
    z0, z1, _ = sequence_impedances(_three_phase_geometry_z(freqs))
    # direct positive-sequence model fit to the geometry's Z1 at the fundamental.
    r1_f0, x1_f0 = float(z1[0].real), float(z1[0].imag)
    z1_dir = positive_sequence_z(r1_f0, x1_f0, F0, freqs)

    km = 1e3  # Ω/m -> Ω/km
    fig, (axr, axx) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axr.plot(orders, z0.real.numpy() * km, "o-", color="C3", label="zero seq R0")
    axr.plot(orders, z1.real.numpy() * km, "s-", color="C0", label="pos seq R1")
    axr.plot(
        orders, z1_dir.real.numpy() * km, "x--", color="C2", label="R1 model (skin)"
    )
    axr.set(xlabel="harmonic order h", ylabel="R(h) [Ω/km]", title="Resistance")
    axr.legend()
    axr.grid(alpha=0.3)

    axx.plot(orders, z0.imag.numpy() * km, "o-", color="C3", label="zero seq X0")
    axx.plot(orders, z1.imag.numpy() * km, "s-", color="C0", label="pos seq X1")
    axx.plot(orders, z1_dir.imag.numpy() * km, "x--", color="C2", label="X1 model (∝h)")
    # naive X∝h reference line through the fundamental of each sequence.
    axx.plot(orders, x1_f0 * orders * km, ":", color="0.5", label="X1·h (naive)")
    axx.set(xlabel="harmonic order h", ylabel="X(h) [Ω/km]", title="Reactance")
    axx.legend()
    axx.grid(alpha=0.3)
    fig.suptitle(
        "3-phase geometry: earth return only in the ZERO sequence "
        "(positive-seq X ∝ h, no floor)"
    )
    ev.save_figure(fig, out / "seq_xr_vs_harmonic.svg")
    plt.close(fig)
    print(
        f"[seq] X1(h)/(h·X1_f0) last = {float(z1.imag[-1] / (z1.imag[0] * orders[-1])):.4f}"
        f"  X0(h)/(h·X0_f0) last = {float(z0.imag[-1] / (z0.imag[0] * orders[-1])):.4f}"
    )


# --- (2) GMR floor: single-conductor vs two-conductor ----------------------
def plot_gmr_floor(out: Path) -> None:
    radius = 0.0102
    r1 = 3.6e-4
    x1_grid = np.linspace(0.05e-3, 0.8e-3, 40)  # Ω/m, cable -> overhead range
    single_ratio, two_ratio, two_spacing = [], [], []
    for x1 in x1_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geo = synthesize_line_geometry(r1, float(x1), f0=F0, radius_m=radius)
        single_ratio.append(float(geo.conductors[0].gmr_m) / radius)
        g = two_conductor_geometry(r1, float(x1), F0, radius_m=radius)
        two_ratio.append(g["gmr_m"] / radius)
        two_spacing.append(g["spacing_m"])

    fig, (axg, axs) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axg.semilogy(
        x1_grid * 1e3, single_ratio, "o-", color="C3", label="single-conductor synth"
    )
    axg.semilogy(
        x1_grid * 1e3, two_ratio, "s-", color="C0", label="two-conductor go/return"
    )
    axg.axhline(1.0, color="k", ls="--", lw=1, label="GMR = radius (physical limit)")
    axg.set(
        xlabel="X1 [Ω/km]",
        ylabel="GMR / radius",
        title="Synthesised GMR (log): single-conductor blows past the radius",
    )
    axg.legend()
    axg.grid(alpha=0.3, which="both")

    axs.plot(
        x1_grid * 1e3,
        np.array(two_spacing) * 1e2,
        "s-",
        color="C0",
        label="go/return spacing D",
    )
    axs.axhline(2 * radius * 1e2, color="k", ls="--", lw=1, label="2·radius (overlap)")
    axs.set(
        xlabel="X1 [Ω/km]",
        ylabel="spacing D [cm]",
        title="Two-conductor spacing stays physical for every X1",
    )
    axs.legend()
    axs.grid(alpha=0.3)
    fig.suptitle("Why single-conductor R/X->geometry synthesis fails for low-X feeders")
    ev.save_figure(fig, out / "gmr_floor.svg")
    plt.close(fig)
    n_bad = int(np.sum(np.array(single_ratio) >= 1.0))
    print(
        f"[gmr] single-conductor non-physical for {n_bad}/{len(x1_grid)} swept X1; "
        "two-conductor always physical"
    )


# --- (3) feeder harmonic voltages: corrected vs naive vs single-conductor ---
def _strip_geometry(grid):
    for b in grid.branches:
        if isinstance(b, Line):
            b.conductor_geometry = None
    return grid


def plot_feeder(out: Path, order: int = 13) -> None:
    # single-conductor Carson model (== OpenDSS line model) — geometry intact.
    grid_geom, _ = ref.ieee33_geometry_grid()
    res_geom = solve_harmonic_flow(grid_geom, ORDERS, slack="norton", dtype=CDT)

    # corrected positive-sequence model — no geometry, no earth floor.
    grid_pos = apply_positive_sequence_harmonic_model(
        _strip_geometry(ref.ieee33_geometry_grid()[0])
    )
    res_pos = solve_harmonic_flow(grid_pos, ORDERS, slack="norton", dtype=CDT)

    # naive model — R const, X ∝ h (no skin, no earth).
    grid_naive = apply_positive_sequence_harmonic_model(
        _strip_geometry(ref.ieee33_geometry_grid()[0]), skin=False
    )
    res_naive = solve_harmonic_flow(grid_naive, ORDERS, slack="norton", dtype=CDT)

    profs = [
        ev.harmonic_profile(res_pos, grid_pos, order, label="positive-seq (corrected)"),
        ev.harmonic_profile(res_naive, grid_naive, order, label="naive (R const, X∝h)"),
        ev.harmonic_profile(
            res_geom, grid_geom, order, label="single-conductor Carson"
        ),
    ]
    try:
        profs.append(
            next(
                p
                for p in ref.opendss_geometry_harmonic_profiles(
                    grid_geom, res_geom, [order], label="OpenDSS (Carson)"
                )
                if p.order == order
            )
        )
    except Exception as exc:  # OpenDSS optional
        print(f"[feeder] OpenDSS overlay skipped: {exc}")

    title = (
        f"IEEE-33 harmonic voltage profile (h={order}): "
        "corrected vs naive vs single-conductor Carson"
    )
    # Static SVG: semi-transparent lines so the heavily-overlapping models stay legible.
    fig, _ = ev.plot_harmonic_profile(profs, grid=grid_geom, alpha=0.55, title=title)
    ev.save_figure(fig, out / f"feeder_h{order}.svg")
    plt.close(fig)

    # Interactive HTML: one colour per model, click the legend to toggle models on/off.
    ev.plot_harmonic_profile_interactive(
        profs,
        grid=grid_geom,
        title=title + " — click legend to toggle",
        out_html=str(out / f"feeder_h{order}_interactive.html"),
    )
    print(
        f"[feeder] wrote h={order} profile + interactive html "
        f"({'corrected / naive / single-conductor / OpenDSS' if len(profs) == 4 else 'corrected / naive / single-conductor'})"
    )


def main(out_dir: str = "evaluation_output/sequence") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_sequence_xr(out)
    plot_gmr_floor(out)
    plot_feeder(out, order=13)
    print(f"wrote figures to {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation_output/sequence")
