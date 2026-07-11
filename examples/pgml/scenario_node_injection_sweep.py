"""Scenario 1 — per-node harmonic error-source sweep on the FULL CIGRE LV grid.

WHAT THIS DEMONSTRATES
----------------------
"Inject a harmonic ERROR at one node at a time, and measure how it spreads through the
whole grid." The error is a per-node **Thévenin voltage source** carrying a voltage
spectrum (read from ``examples/spectra/VoltageSag40ms.csv``), of a user-set strength
``SOURCE_POWER_VA`` (short-circuit power), applied only at harmonics so the fundamental
power flow is untouched — the model documented in ``docs/pgml/modeling/error-injection.md``.
(Set ``KIND="current"`` for a Norton current source instead.)

For harmonic order **h=11**:

1. Load the FULL CIGRE LV benchmark (all 3 LV feeders + MV source + 3 transformers) as a
   single-phase grid with the **Carson** line model (exact vs a live OpenDSS solve).
2. Sweep the per-node source over EVERY node with
   :func:`pgml.scenarios.run_node_injection_sweep` (scenario *i* = source at node *i*).
3. Record the h=11 voltage at every measurement node *m* for every injection node *i*,
   minus a no-injection reference, **normalised per measurement node by the local
   fundamental** ``|V1|`` → an ``m x i`` "spread matrix" in PER-UNIT (``|V_h11|/|V1|``).
   The per-unit basis is essential: in raw volts the 20 kV MV nodes dwarf the 0.4 kV LV
   nodes purely from the transformer turns ratio, hiding the disturbance origin; in
   per-unit the response correctly peaks at the injecting node.
4. Save the matrix (CSV + ``.npz`` + heatmap), and compare pgml vs the **live OpenDSS**
   oracle (parity print + side-by-side CSV + overlay plot for a representative node),
   all in per-unit of the local fundamental.

RUN
---
::

    pixi run -e cpu python examples/pgml/scenario_node_injection_sweep.py [out_dir]

Outputs (default ``evaluation_output/scenario1/``): ``spread_h11.csv`` (m x i, per-unit),
``spread_h11.npz`` (per-unit magnitude + raw real/imag volts + the ``|V1|`` base),
``spread_h11.svg`` (heatmap), ``compare_h11.csv`` (pgml | opendss | Δ, per-unit, for a
representative injection), ``compare_h11.svg`` (overlay), and a printed parity + timing
summary. Tune ``SOURCE_POWER_VA`` / ``KIND`` at the top.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pgml.assembly import node_phase_index
from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.oracles import cigre_lv_full_grid
from pgml.geometry.synthesis import synthesize_grid_geometry
from pgml.scenarios import NodeInjectionSweepConfig, run_node_injection_sweep
from pgml.solver import NodeHarmonicSource, solve_harmonic_flow

CDT = torch.complex128
SPECTRUM_CSV = Path(__file__).with_name("spectra") / "VoltageSag40ms.csv"
RECORD_ORDER = 11  # the harmonic whose spread we map
SOURCE_POWER_VA = 1.0e5  # error-source strength S_sc (short-circuit power); tune freely
KIND = "voltage"  # "voltage" (Thévenin) or "current" (Norton)


# Example outputs are anchored at examples/ (not the cwd), so a run writes
# under examples/evaluation_output/ rather than the repository root.
_OUT = Path(__file__).resolve().parent.parent / "evaluation_output"


def read_spectrum_csv(path: Path) -> dict:
    """Parse ``order, magnitude_percent, phase_deg`` -> ``{order: (frac, phase_deg)}``.

    Magnitudes are normalised to a FRACTION of the fundamental (the ``order == 1`` row,
    given as 100). Only integer orders are kept; the fundamental row is dropped (it is
    the implicit reference of the source EMF).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for order_s, mag_s, phase_s in csv.reader(fh):
            rows.append((float(order_s), float(mag_s), float(phase_s)))
    fundamental = next((m for o, m, _ in rows if abs(o - 1.0) < 1e-9), 100.0)
    return {
        int(round(o)): (m / fundamental, p)
        for o, m, p in rows
        if abs(o - round(o)) < 1e-9 and round(o) >= 2
    }


def main(out_dir: str = str(_OUT / "scenario1")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. full single-phase grid (Carson line model) + the voltage-error spectrum.
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    synthesize_grid_geometry(grid)  # single-conductor Carson -> exact vs OpenDSS
    index = node_phase_index(grid)
    spectrum = read_spectrum_csv(SPECTRUM_CSV)
    print(
        f"spectrum: {len(spectrum)} integer harmonics; |h{RECORD_ORDER}| "
        f"= {spectrum[RECORD_ORDER][0]:.4f} of fundamental; source S_sc={SOURCE_POWER_VA:g} VA"
    )

    # 2-3. sweep the per-node source over EVERY node + the no-injection reference.
    cfg = NodeInjectionSweepConfig.from_spectrum(
        spectrum, source_power_va=SOURCE_POWER_VA, kind=KIND
    )
    t0 = time.perf_counter()
    res = run_node_injection_sweep(grid, cfg, dtype=CDT)
    ref = solve_harmonic_flow(grid, [1, RECORD_ORDER], slack="norton", dtype=CDT)
    dt = time.perf_counter() - t0

    injection_ids = res.sampled.samples["injection_node_id"].tolist()
    node_ids = index.node_ids.tolist()
    f_target = RECORD_ORDER * float(grid.base_frequency_hz)
    k = int((res.frequencies_hz - f_target).abs().argmin())  # h11 slice of the sweep
    kref = int((ref.frequencies_hz - f_target).abs().argmin())  # h11 slice of the ref
    v_h11 = res.v[:, k, :]  # [i_injections, m_nodes]
    v_ref = ref.v[kref, :]  # [m_nodes] (no injection ~ 0)
    spread = (v_h11 - v_ref).transpose(0, 1)  # [m_nodes, i_injections]

    # Normalise PER MEASUREMENT NODE by the local fundamental |V1| -> a per-unit
    # harmonic distortion (|V_h11| / |V1|) on each node's OWN voltage level. In raw
    # volts the 20 kV MV nodes dwarf the 0.4 kV LV nodes purely from the transformer
    # turns ratio, which hides where the disturbance actually originates; in per-unit
    # the response peaks at the injecting node, as expected. The fundamental is the
    # same for every injection (the source acts only at h > 1), so column 0 of the
    # sweep is a clean per-node voltage base.
    v_base = res.v[0, 0, :].abs().clamp_min(1e-12)  # [m_nodes] local |V1|
    spread_pu = (spread.abs() / v_base[:, None]).detach().cpu().numpy()
    mag = spread_pu  # per-unit of the local fundamental (dimensionless)

    # 4. persist the m x i spread matrix (per-unit) + heatmap.
    np.savez(
        out / "spread_h11.npz",
        magnitude_pu=mag,
        real=spread.real.detach().cpu().numpy(),
        imag=spread.imag.detach().cpu().numpy(),
        v_base_v=v_base.detach().cpu().numpy(),
        measurement_node_ids=np.array(node_ids),
        injection_node_ids=np.array(injection_ids),
    )
    with open(out / "spread_h11.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measurement_node\\injection_node", *injection_ids])
        for m, nid in enumerate(node_ids):
            w.writerow([nid, *(f"{x:.6e}" for x in mag[m])])
    _heatmap(mag, node_ids, injection_ids, out / "spread_h11.svg")
    print(
        f"pgml: swept {len(injection_ids)} nodes in {dt * 1e3:.0f} ms (loop of solves); "
        f"spread matrix {mag.shape} -> {out.resolve()}"
    )

    # 5. pgml vs live OpenDSS: parity + side-by-side CSV/plot for a representative node.
    _opendss_compare(grid, spectrum, res, k, injection_ids, node_ids, out)


def _heatmap(mag, node_ids, injection_ids, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(mag, aspect="auto", cmap="viridis")
    ax.set(
        xlabel="injection node i",
        ylabel="measurement node m",
        title=f"h{RECORD_ORDER} voltage spread |V|/|V1| [pu of local fundamental] (m x i)",
    )
    ax.set_xticks(range(len(injection_ids)))
    ax.set_xticklabels(injection_ids, rotation=90, fontsize=5)
    ax.set_yticks(range(len(node_ids)))
    ax.set_yticklabels(node_ids, fontsize=5)
    fig.colorbar(im, ax=ax, label=f"|V(h{RECORD_ORDER})| / |V1|  [pu]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _node_source(grid, node_id: int, spectrum: dict) -> NodeHarmonicSource:
    node = next(n for n in grid.nodes if int(n.id) == node_id)
    return NodeHarmonicSource(
        node_id=node_id,
        phases=tuple(node.phases),
        spectrum=spectrum,
        source_power_va=SOURCE_POWER_VA,
        kind=KIND,
    )


def _opendss_compare(
    grid, spectrum, res, k, injection_ids, node_ids, out: Path
) -> None:
    try:
        from pgml.evaluation.oracles import opendss_harmonic_voltages
    except ImportError:
        print("OpenDSS oracle unavailable — skipping comparison.")
        return

    v1 = res.v[0, 0, :].detach().cpu().numpy()  # fundamental (same for every injection)
    v_base = np.abs(v1) + 1e-12  # [m_nodes] per-node voltage base = local |V1|
    probe = [0, len(injection_ids) // 2, len(injection_ids) - 1]
    for j in probe:
        src = _node_source(grid, injection_ids[j], spectrum)
        dss = opendss_harmonic_voltages(
            grid, None, [1, RECORD_ORDER], v1=v1, node_sources=[src]
        )
        pgml = res.v[j, k, :].detach().cpu().numpy()
        rel = np.max(np.abs(pgml - dss[-1])) / (np.max(np.abs(pgml)) + 1e-30)
        print(
            f"OpenDSS parity @ injection node {injection_ids[j]}: max rel err = {rel:.2e}"
        )

    # Side-by-side for the most-affected injection node (largest per-unit spread).
    spread_pu = np.abs(res.v[:, k, :].detach().cpu().numpy()) / v_base[None, :]
    j = int(np.argmax(spread_pu.max(axis=1)))
    src = _node_source(grid, injection_ids[j], spectrum)
    dss = opendss_harmonic_voltages(
        grid, None, [1, RECORD_ORDER], v1=v1, node_sources=[src]
    )
    pgml = res.v[j, k, :].detach().cpu().numpy()
    # Express both engines in per-unit of the local fundamental (shared base) so the
    # overlay is on one scale across voltage levels, not dominated by the MV nodes.
    pgml_pu = np.abs(pgml) / v_base
    dss_pu = np.abs(dss[-1]) / v_base
    with open(out / "compare_h11.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measurement_node", "pgml_pu", "opendss_pu", "abs_diff_pu"])
        for m, nid in enumerate(node_ids):
            w.writerow(
                [
                    nid,
                    f"{pgml_pu[m]:.6e}",
                    f"{dss_pu[m]:.6e}",
                    f"{abs(pgml_pu[m] - dss_pu[m]):.3e}",
                ]
            )
    _compare_plot(
        node_ids,
        pgml_pu,
        dss_pu,
        injection_ids[j],
        out / "compare_h11.svg",
    )
    print(f"comparison (injection node {injection_ids[j]}) -> compare_h11.csv / .svg")


def _compare_plot(node_ids, pgml_pu, dss_pu, inj_node, path: Path) -> None:
    import matplotlib.pyplot as plt

    x = range(len(node_ids))
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(x, pgml_pu, "-o", ms=3, label="pgml", color="C0")
    ax.plot(x, dss_pu, "--x", ms=4, label="OpenDSS (live)", color="C3", alpha=0.8)
    ax.set(
        xlabel="measurement node",
        ylabel=f"|V(h{RECORD_ORDER})| / |V1|  [pu]",
        title=f"pgml vs OpenDSS — h{RECORD_ORDER} spread from injection node {inj_node}",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(node_ids, rotation=90, fontsize=5)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "scenario1"))
