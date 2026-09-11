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
3. ``feeder_h<k>.svg`` / ``feeder_h<k>_interactive.html`` / ``feeder_h<k>_default_vs_opendss.svg``
   — IEEE-33 (1-phase lines): the config-default model, the naive model and the
   single-conductor Carson model (== OpenDSS) overlaid; the interactive HTML carries every
   model at once (click the legend to toggle); the pairwise SVG diffs the config default
   against the OpenDSS line model.
4. ``unbalanced_h<k>.svg`` / ``unbalanced_interactive.html`` / ``unbalanced_h<k>_earth_effect.svg``
   — a 3-phase UNBALANCED feeder (single-phase nonlinear load → zero-sequence current):
   the config default (sequence-aware, Z0 earth-damped) vs the same model with no earth
   damping vs naive. The interactive HTML shows several harmonics × models together; the
   pairwise SVG isolates the Z0 earth-return damping. This is the config DEFAULT for a
   4-wire unbalanced harmonic study.

All model choices go through the defaults-driven entry points (`apply_default_harmonic_model`
etc.), so defaults are explicit and documented (`pgml.defaults`), never hidden.

Run::

    pixi run -e cpu python run/examples/pgml/evaluate_line_sequence_harmonics.py [out_dir]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pgml import defaults as config
from pgml import evaluation as ev
from pgml.evaluation import oracles as ref
from pgml.geometry.carson import kron_reduce, series_impedance
from pgml.geometry.sequence import (
    positive_sequence_z,
    sequence_impedances,
    two_conductor_geometry,
)
from pgml.geometry.synthesis import (
    apply_default_harmonic_model,
    apply_positive_sequence_harmonic_model,
    apply_sequence_aware_harmonic_model,
    strip_grid_geometry,
    synthesize_line_geometry,
)
from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)
from pgml.solver import solve_harmonic_flow

RDT = torch.float64
CDT = torch.complex128
F0 = 50.0
ORDERS = [1, 5, 7, 11, 13, 17, 25]


# --- (1) sequence R/X vs harmonic ------------------------------------------
# Example outputs are anchored at the repository root (not the cwd), so a run writes
# under the untracked data root at data/pgml/evaluation_output/.
_OUT = Path(__file__).resolve().parents[3] / "data" / "pgml" / "evaluation_output"


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
    return strip_grid_geometry(grid)


def _ieee33_models(orders):
    """Build every 1-phase IEEE-33 line model -> {label: (grid, harmonic result)}.

    Uses the config-driven model application so the example exercises the SAME deliberate
    entry points a user would (`apply_default_harmonic_model` etc.), not ad-hoc tweaks.
    """
    models: dict[str, tuple] = {}

    # config default for these (1-phase) lines -> positive_sequence (skin on R).
    g_def = apply_default_harmonic_model(_strip_geometry(ref.ieee33_geometry_grid()[0]))
    models["config default"] = (g_def, None)

    # naive: R const, X∝h (skin disabled).
    g_naive = apply_positive_sequence_harmonic_model(
        _strip_geometry(ref.ieee33_geometry_grid()[0]), skin=False
    )
    models["naive (R const, X∝h)"] = (g_naive, None)

    # single-conductor Carson (== OpenDSS 1-phase line) — geometry kept.
    g_geom, _ = ref.ieee33_geometry_grid()
    models["single-conductor Carson"] = (g_geom, None)

    for label, (g, _) in models.items():
        models[label] = (g, solve_harmonic_flow(g, orders, slack="norton", dtype=CDT))
    return models


def plot_feeder(out: Path, order: int = 13) -> None:
    """IEEE-33 (1-phase lines): every model overlaid + a pairwise matplotlib diff."""
    models = _ieee33_models(ORDERS)
    profs = [
        ev.harmonic_profile(res, g, order, label=label)
        for label, (g, res) in models.items()
    ]
    g_geom, res_geom = models["single-conductor Carson"]
    try:
        profs.append(
            next(
                p
                for p in ref.opendss_geometry_harmonic_profiles(
                    g_geom, res_geom, [order], label="OpenDSS (Carson)"
                )
                if p.order == order
            )
        )
    except Exception as exc:  # OpenDSS optional
        print(f"[feeder] OpenDSS overlay skipped: {exc}")

    title = f"IEEE-33 harmonic voltage profile (h={order})"
    # Static SVG: semi-transparent lines so the heavily-overlapping models stay legible.
    fig, _ = ev.plot_harmonic_profile(profs, grid=g_geom, alpha=0.55, title=title)
    ev.save_figure(fig, out / f"feeder_h{order}.svg")
    plt.close(fig)

    # Interactive HTML: ALL models at once, one colour each, click the legend to toggle.
    ev.plot_harmonic_profile_interactive(
        profs,
        grid=g_geom,
        title=title + " — click legend to toggle models",
        out_html=str(out / f"feeder_h{order}_interactive.html"),
    )

    # Pairwise matplotlib comparison: config default vs single-conductor Carson (==OpenDSS).
    g_def, res_def = models["config default"]
    a = ev.harmonic_profile(res_def, g_def, order, label="config default")
    b = ev.harmonic_profile(
        res_geom, g_geom, order, label="single-conductor Carson (=OpenDSS)"
    )
    fig_cmp, fig_diff = ev.plot_harmonic_model_comparison(
        a,
        b,
        grid=g_geom,
        title=f"IEEE-33 h={order}: config default vs OpenDSS line model",
    )
    ev.save_figure(fig_cmp, out / f"feeder_h{order}_default_vs_opendss.svg")
    ev.save_figure(fig_diff, out / f"feeder_h{order}_default_vs_opendss_diff.svg")
    plt.close("all")
    print(f"[feeder] wrote h={order} overlay + interactive + pairwise comparison")


# --- (4) unbalanced 3-phase feeder: the sequence-aware (Z1+Z0) model ---------
def _unbalanced_3phase_grid(n_bus: int = 5):
    """A small 3-phase radial feeder with a SINGLE-PHASE harmonic load (-> zero seq)."""
    z1, z0 = complex(0.32e-3, 0.30e-3), complex(0.88e-3, 1.20e-3)  # Ω/m, LV-ish
    zs, zm = (z0 + 2 * z1) / 3, (z0 - z1) / 3
    w = 2.0 * np.pi * F0
    ph = (Phase.A, Phase.B, Phase.C)
    rmat = [[zs.real if i == j else zm.real for j in range(3)] for i in range(3)]
    lmat = [[(zs.imag if i == j else zm.imag) / w for j in range(3)] for i in range(3)]
    nodes = [Node(id=i, u_rated_v=400.0, phases=ph) for i in range(1, n_bus + 1)]
    branches = [
        Line(
            id=100 + i,
            from_node=i,
            to_node=i + 1,
            from_phases=ph,
            to_phases=ph,
            length_m=120.0,
            series_resistance_ohm_per_m=rmat,
            series_inductance_h_per_m=lmat,
            shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
        )
        for i in range(1, n_bus)
    ]
    spec = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=o, magnitude_pu=m, phase_deg=0.0)
                for o, m in [(3, 0.6), (5, 0.5), (7, 0.3), (9, 0.2), (11, 0.12)]
            ]
        )
    )
    appliances = [
        Source(
            id=1,
            node=1,
            phases=ph,
            u_ref_v=(230.0, 230.0, 230.0),
            u_angle_deg=(0.0, -120.0, 120.0),
            resistance_ohm=[
                [1e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
            inductance_h=[
                [1e-5 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
        ),
        # small balanced base load + a DOMINANT single-phase nonlinear load (phase A) at
        # the end, so the harmonic injection is strongly zero-sequence (residual current).
        Load(id=2, node=n_bus, phases=ph, p_nom_w=2400.0, q_nom_var=600.0),
        Load(
            id=3,
            node=n_bus,
            phases=(Phase.A,),
            p_nom_w=8000.0,
            q_nom_var=1500.0,
            spectrum=spec,
        ),
    ]
    return Grid(
        base_frequency_hz=F0, nodes=nodes, branches=branches, appliances=appliances
    )


def plot_unbalanced_feeder(out: Path, order: int = 9) -> None:
    """3-phase unbalanced feeder: config-default sequence-aware vs no-earth vs naive.

    The single-phase nonlinear load drives a zero-sequence (residual / neutral-return)
    harmonic current. The Carson earth-return resistance is a REAL, frequency-growing
    impedance that this current must flow through, so modelling it (the config default,
    OpenDSS-consistent) RAISES the zero-sequence harmonic voltage versus neglecting it —
    i.e. the earth-free model UNDER-predicts the zero-sequence harmonics. Demonstrates the
    config DEFAULT for an unbalanced 4-wire study (`line.harmonic_model.three_phase`).
    """

    def build(apply, **kw):
        g = _unbalanced_3phase_grid()
        apply(g, **kw)
        return g, solve_harmonic_flow(g, [1, 3, 5, 7, 9, 11], slack="norton", dtype=CDT)

    default_model = config.get("line.harmonic_model.three_phase")
    models = {
        # full sequence-aware model: Z0 carries the Carson earth-return damping (default).
        f"config default ({default_model})": build(apply_default_harmonic_model),
        # same model with the earth-return coefficient zeroed -> isolates that damping.
        "no earth damping (coeff=0)": build(
            apply_sequence_aware_harmonic_model, earth_resistance_coeff=0.0
        ),
        # no skin either -> the naive R-const, X∝h reference.
        "naive (R const, X∝h)": build(
            apply_sequence_aware_harmonic_model, skin=False, earth_resistance_coeff=0.0
        ),
    }

    profs = [
        ev.harmonic_profile(res, g, order, phase=Phase.A, label=label)
        for label, (g, res) in models.items()
    ]
    g0 = next(iter(models.values()))[0]
    title = f"Unbalanced 3-phase feeder, phase A (h={order})"
    fig, _ = ev.plot_harmonic_profile(profs, grid=g0, alpha=0.6, title=title)
    ev.save_figure(fig, out / f"unbalanced_h{order}.svg")
    plt.close(fig)

    # interactive: multiple models AND several harmonics at once; each (order, model)
    # is its own toggleable legend group ("h{order} · {model}").
    interactive = [
        ev.harmonic_profile(res, g, od, phase=Phase.A, label=label)
        for od in (5, 7, 9)
        for label, (g, res) in models.items()
    ]
    ev.plot_harmonic_profile_interactive(
        interactive,
        grid=g0,
        title="Unbalanced feeder, phase A — sequence-aware vs Z1-only vs naive (toggle)",
        out_html=str(out / "unbalanced_interactive.html"),
    )

    # pairwise matplotlib: config default (earth in Z0) vs no-earth -> the earth effect.
    a, b = profs[0], profs[1]
    fig_cmp, fig_diff = ev.plot_harmonic_model_comparison(
        a,
        b,
        grid=g0,
        title=f"Phase A h={order}: Z0 earth-return effect (config default − no earth)",
    )
    ev.save_figure(fig_cmp, out / f"unbalanced_h{order}_earth_effect.svg")
    ev.save_figure(fig_diff, out / f"unbalanced_h{order}_earth_effect_diff.svg")
    plt.close("all")
    peak_a = float(a.magnitude.max())
    peak_b = float(b.magnitude.max())
    print(
        f"[unbalanced] wrote h={order} overlay + interactive + earth-effect comparison "
        f"(peak |V_h| phase A: config default {peak_a:.4g} vs no-earth {peak_b:.4g} pu)"
    )


def main(out_dir: str = str(_OUT / "sequence")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_sequence_xr(out)
    plot_gmr_floor(out)
    plot_feeder(out, order=13)
    plot_unbalanced_feeder(out, order=9)
    print(f"wrote figures to {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "sequence"))
