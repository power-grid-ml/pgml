"""Scenario 2 — randomized loads + harmonic spectra on the FULL 3-phase CIGRE LV grid.

WHAT THIS DEMONSTRATES
----------------------
A reproducible randomized study on the FULL CIGRE LV benchmark in genuine 3-phase:

1. Load the whole CIGRE LV (all feeders + MV source + 3 transformers) as a THREE_PHASE
   grid.
2. Sample, per load, an INDEPENDENT-PER-PHASE active/reactive power scale drawn from
   ``Uniform(0, 1)`` (0 % .. 100 % of nominal) — so the loading is genuinely asymmetric —
   and a random per-load harmonic spectrum (orders 3,5,7,9), EN 50160-bounded.
3. Solve the harmonic flow TWICE on the same sampled batch: once SYMMETRIC (each load's
   total split equally over its phases) and once ASYMMETRIC (per-phase honoured), and
   time both on CPU.
4. Persist both result sets to parquet (+ a tidy CSV for manual checking).
5. Plot the asymmetric run (one representative scenario): a fundamental voltage profile
   for all three phases, and a 3D harmonic profile (h=3,5,7,9) where COLOR = harmonic
   order and LINE STYLE = phase (L1 solid, L2 dashed, L3 dotted).
6. Optionally compare one scenario to the OpenDSS oracle, and report timings.

RUN
---
::

    pixi run -e cpu python examples/pgml/scenario_randomized.py [out_dir]

Outputs (default ``evaluation_output/scenario2/``): ``symmetric/`` & ``asymmetric/``
parquet+CSV datasets, ``fundamental_phases.svg``, ``harmonics_3d.html``, and a printed
timing + OpenDSS parity summary.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

from pgml import evaluation as ev
from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.data import VoltageProfile, harmonic_profile
from pgml.evaluation.oracles import cigre_lv_full_grid
from pgml.geometry.synthesis import synthesize_grid_geometry
from pgml.scenarios import (
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    sample,
    run_scenarios,
    write_dataset,
)
from pgml.schemas.grid_schema import Phase

CDT = torch.complex128
ORDERS = [1, 3, 5, 7, 9]
HARM_ORDERS = [3, 5, 7, 9]
PHASES = [("L1", Phase.A, "C0"), ("L2", Phase.B, "C1"), ("L3", Phase.C, "C2")]
DASH = {"L1": "solid", "L2": "dash", "L3": "dot"}
N_SAMPLES = 16
SEED = 0


# Example outputs are anchored at examples/ (not the cwd), so a run writes
# under examples/evaluation_output/ rather than the repository root.
_OUT = Path(__file__).resolve().parent.parent / "evaluation_output"


def build_config() -> ScenarioConfig:
    """Random per-phase load scaling U(0,1) + a random EN 50160-bounded spectrum/load."""
    return ScenarioConfig(
        n_samples=N_SAMPLES,
        seed=SEED,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="pq",
                mode="scale",
                symmetry="independent",  # per-phase draws -> asymmetric is meaningful
            ),
            ParameterSpec(
                name="load_spectrum",
                selector=Selector(component="load"),
                distribution=Uniform(
                    low=0.0, high=1.0
                ),  # fraction of the EN50160 limit
                field="h_mag",
                orders=HARM_ORDERS,
                harmonic_reference="en50160",
            ),
        ],
    )


def main(out_dir: str = str(_OUT / "scenario2")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    # Synthesize a 3-conductor Carson geometry per R/X line (reproduces Z1 + X0 at f0).
    # Feeding the SAME geometry to pgml and OpenDSS makes the harmonic comparison
    # apples-to-apples Carson on every order, INCLUDING the triplen / zero-sequence
    # orders (the delta winding traps zero sequence identically on both sides) — instead
    # of pgml's analytic sequence_aware Z0 vs OpenDSS's internal Carson earth return,
    # whose mismatch otherwise inflates the triplen gap. (CIGRE LV lines are low-X
    # cables, so the synthesized GMR is non-physical — flagged by a warning — but the
    # geometry still reproduces the target impedance and matches OpenDSS bit-for-bit.)
    synthesize_grid_geometry(grid)
    sampled = sample(grid, build_config())  # ONE batch, reused for both symmetry modes

    timings = {}
    results = {}
    for mode in ("symmetric", "asymmetric"):
        t0 = time.perf_counter()
        res = run_scenarios(
            grid,
            sampled,
            calculation="harmonic",
            harmonic_orders=ORDERS,
            symmetry=mode,
            dtype=CDT,
        )
        timings[mode] = time.perf_counter() - t0
        results[mode] = res
        write_dataset(res, out / mode, layout="wide", also_csv=True)

    # how different are the two modes (sanity: asymmetric loading must differ)?
    dv = (results["asymmetric"].v - results["symmetric"].v).abs().max().item()
    print(
        f"solved {N_SAMPLES} scenarios x {len(ORDERS)} orders x {grid_n(grid)} rows; "
        f"CPU time: symmetric {timings['symmetric'] * 1e3:.0f} ms, "
        f"asymmetric {timings['asymmetric'] * 1e3:.0f} ms; "
        f"max|V_asym - V_sym| = {dv:.4g}"
    )
    print(
        f"datasets (parquet + voltages.csv) -> {(out).resolve()}/{{symmetric,asymmetric}}"
    )

    _plots(results["asymmetric"], grid, out)
    _opendss_compare(grid, sampled, results["asymmetric"], out)


def grid_n(grid) -> int:
    from pgml.assembly import node_phase_index

    return node_phase_index(grid).size


def _plots(res, grid, out: Path) -> None:
    """Fundamental all-phase profile + 3D harmonics (color=order, dash=phase)."""
    # Fundamental (order 1): one VoltageProfile per phase, colored per phase.
    vps = []
    for label, phase, _color in PHASES:
        hp = harmonic_profile(
            res, grid, 1, phase=phase, scenario=0, label=label, unit="pu"
        )
        vps.append(
            VoltageProfile(
                distances_km=hp.distances_km,
                v_pu=hp.magnitude,
                label=label,
                node_ids=hp.node_ids,
            )
        )
    fig, _ = ev.plot_voltage_profile(
        vps,
        grid=grid,
        colors=[c for _, _, c in PHASES],
        title="Fundamental voltage, all phases (asymmetric, scenario 0)",
    )
    ev.save_figure(fig, out / "fundamental_phases.svg")

    # Harmonics (3,5,7,9) x phases: color = order, line dash = phase.
    profiles = [
        harmonic_profile(res, grid, order, phase=phase, scenario=0, label=label)
        for label, phase, _ in PHASES
        for order in HARM_ORDERS
    ]
    ev.plot_harmonic_profile_3d(
        profiles,
        grid=grid,
        dash_map=DASH,
        title="Harmonic voltages h=3,5,7,9 — color=order, dash=phase (asymmetric, scenario 0)",
        out_html=str(out / "harmonics_3d.html"),
    )
    print(f"figures -> {out / 'fundamental_phases.svg'}, {out / 'harmonics_3d.html'}")


def _bake_scenario_loads(grid, operating_point, j):
    """A grid copy with scenario-``j`` loads applied as nameplate, so the OpenDSS oracle
    (which reads loads from the grid) sees the same per-phase powers pgml solved."""
    g2 = grid.model_copy(deep=True)
    by_id = {a.id: a for a in g2.appliances}
    for cid, op in operating_point.items():
        a = by_id[cid]
        if "p_per_phase_w" in op:
            pp = [float(x[j]) for x in op["p_per_phase_w"]]
            a.p_nom_w = float(sum(pp))  # total first; per-phase must sum to it
            a.p_nom_per_phase_w = pp
        elif "p_w" in op:
            a.p_nom_w = float(op["p_w"][j])
        if "q_per_phase_var" in op:
            qq = [float(x[j]) for x in op["q_per_phase_var"]]
            a.q_nom_var = float(sum(qq))
            a.q_nom_per_phase_var = qq
        elif "q_var" in op:
            a.q_nom_var = float(op["q_var"][j])
    return g2


_PNAME = {0: "L1", 1: "L2", 2: "L3", 3: "N"}


def _opendss_compare(grid, sampled, res, out: Path) -> None:
    """Live pgml-vs-OpenDSS comparison for scenario 0: per-order parity, a tidy
    ``pgml | opendss | Δ`` CSV, and a per-order overlay plot.

    Because the grid carries a synthesized 3-conductor Carson geometry on every line
    (``synthesize_grid_geometry`` in :func:`main`), the SAME geometry feeds both engines:
    ``opendss_harmonic_voltages`` builds the lines from that geometry (OpenDSS applies its
    own Carson), and the Dyn transformer is stamped with the identical winding-incidence
    vector group on both sides (its delta traps the zero sequence, so it cancels from the
    comparison). The parity is therefore at Carson precision (~1e-11 relative) on EVERY
    order, INCLUDING the triplen / zero-sequence orders (h3, h9) — the earlier triplen gap
    was the analytic ``sequence_aware`` Z0 vs OpenDSS's internal Carson earth return, which
    the shared geometry removes. The transformer vector group is independently validated
    bit-for-bit against a real OpenDSS ``Transformer`` element by
    ``references.opendss_dyn_transformer_harmonic_voltages``."""
    try:
        import numpy as np

        from pgml.evaluation.oracles import opendss_harmonic_voltages

        j = 0
        at = lambda x: float(x[j]) if hasattr(x, "__len__") else float(x)  # noqa: E731
        g2 = _bake_scenario_loads(grid, sampled.operating_point, j)
        inj = {
            cid: {o: (at(m), at(p)) for o, (m, p) in od.items()}
            for cid, od in sampled.harmonic_injection.items()
        }
        v1 = res.v[j, 0, :].detach().cpu().numpy()
        dss = opendss_harmonic_voltages(g2, inj, ORDERS, v1=v1)
        pgml = res.v[j].detach().cpu().numpy()  # [H, N]
        idx = res.index
        node_ids, pcodes = idx.node_ids.tolist(), idx.phase_codes.tolist()

        print("OpenDSS parity (scenario 0), max rel err per order:")
        for k, o in enumerate(ORDERS[1:], start=1):
            rel = np.max(np.abs(pgml[k] - dss[k])) / (np.max(np.abs(pgml[k])) + 1e-30)
            print(f"    h{o}: {rel:.2e}")

        with open(
            out / "compare_harmonics.csv", "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.writer(fh)
            w.writerow(
                ["order", "phase", "node", "pgml_mag_V", "opendss_mag_V", "abs_diff_V"]
            )
            for k, o in enumerate(ORDERS):
                if o == 1:
                    continue
                for n in range(len(node_ids)):
                    w.writerow(
                        [
                            o,
                            _PNAME.get(pcodes[n], "?"),
                            node_ids[n],
                            f"{abs(pgml[k, n]):.6e}",
                            f"{abs(dss[k, n]):.6e}",
                            f"{abs(pgml[k, n] - dss[k, n]):.3e}",
                        ]
                    )
        _compare_plot(pgml, dss, node_ids, pcodes, out / "compare_harmonics.svg")
        print("comparison (scenario 0) -> compare_harmonics.csv / .svg")
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"OpenDSS comparison skipped: {exc}")


def _compare_plot(pgml, dss, node_ids, pcodes, path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {0: "C0", 1: "C1", 2: "C2"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, o in zip(axes.flat, HARM_ORDERS):
        k = ORDERS.index(o)
        for c in (0, 1, 2):
            rows = [n for n in range(len(node_ids)) if pcodes[n] == c]
            xs = range(len(rows))
            ax.plot(
                xs,
                [abs(pgml[k, n]) for n in rows],
                "-",
                color=colors[c],
                label=f"pgml {_PNAME[c]}",
            )
            ax.plot(
                xs,
                [abs(dss[k, n]) for n in rows],
                "--x",
                ms=3,
                color=colors[c],
                alpha=0.7,
                label=f"OpenDSS {_PNAME[c]}",
            )
        ax.set(title=f"h{o}", xlabel="node (per phase)", ylabel="|V| [V]")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("pgml vs OpenDSS — harmonic |V| per phase (scenario 0, asymmetric)")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(_OUT / "scenario2"))
