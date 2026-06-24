"""Execution-speed benchmark: load flow vs harmonic flow, both solvers, CPU vs GPU.

WHAT THIS DEMONSTRATES
----------------------
A reproducible speed study of the differentiable solver on two grids of different size,
generating a large batch of randomized operating points (varying loads, PV generators,
and harmonic injections) and timing every solve path:

1. **Two grids of contrasting size.** The smaller IEEE-33 single-phase feeder and the
   larger FULL three-phase CIGRE LV benchmark (all feeders + MV source + 3 transformers)
   with **PV systems added** at a fraction of the load nodes. Both carry a synthesized
   Carson line geometry. The printed/plotted size table (node-phase rows ``N``, nodes,
   branches, loads, PV) is the x-axis of the scaling story.

2. **A randomized scenario** per grid (``pgml.scenarios``): each sample independently
   scales every load's P/Q, every PV generator's P/Q, and draws an EN 50160-bounded
   harmonic current spectrum for both loads and PV inverters. One serializable config +
   seed reproduces the whole batch.

3. **The two power-flow solvers compared** on one nominal operating point: the
   ``current_injection`` fixed point (fast, and the path that batches in a single solve)
   versus ``newton`` (quadratic, converges near the loadability nose). Reported: wall
   time, iterations to converge, and the final residual. Newton solves a single grid (its
   linear const-Z warm start does not take a batched operating point), so the large-batch
   throughput sweep below uses the current-injection path.

4. **Load flow vs harmonic flow throughput** over a sweep of batch sizes, in both
   ``complex64`` (data-generation precision) and ``complex128`` (gradcheck precision), on
   every available device. Reported per configuration: wall time, time per scenario,
   throughput (scenarios/s), fundamental iterations, and — on CUDA — peak device memory.
   The iterative solver converges at each dtype's resolvable precision (a relative floor),
   so ``complex64`` settles instead of spinning to ``max_iter``. The figures headline
   ``complex64`` (the GPU data-generation path); the CSV/JSON keep both precisions.

5. **CPU vs GPU.** The script benchmarks every device present (CPU always; CUDA when
   available) and writes one ``results_<device>.json`` per device. The plots merge every
   ``results_*.json`` found in the output directory, so a single run on a CUDA host (whose
   default environment also runs on CPU) produces the full CPU-vs-GPU comparison; a
   CPU-only host produces the CPU baseline and the same figures without the GPU series.

RUN
---
::

    # CPU-only host (this machine):
    pixi run -e cpu python examples/benchmark_speed.py [out_dir]

    # CUDA host (the GPU copy; the default pixi environment ships pytorch-gpu and also
    # runs the CPU series, so this single command yields the CPU-vs-GPU figures):
    pixi run python examples/benchmark_speed.py [out_dir]

    # Faster, smaller sweep while iterating:
    pixi run -e cpu python examples/benchmark_speed.py --quick

Outputs (default ``evaluation_output/benchmark/``): ``results_<device>.json`` (raw
numbers + host metadata), ``benchmark_summary.csv`` (flat table of every timed run),
``solver_comparison.svg`` (current-injection vs Newton), ``loadflow_vs_harmonic.svg``
(per-scenario cost at the largest batch), ``throughput_vs_batch.svg`` (the scaling
curves), and — when a CUDA series is present — ``device_speedup.svg``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import warnings
from glob import glob
from pathlib import Path
from typing import Callable, Optional

import torch

from pgml.assembly import node_phase_index
from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.oracles.grids import cigre_lv_full_grid, ieee33_geometry_grid
from pgml.geometry.synthesis import synthesize_grid_geometry
from pgml.schemas.grid_schema import Generator, Load
from pgml.scenarios import (
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    sample,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow

# Harmonic orders solved (order 1 = fundamental). The harmonics injected/varied are h>1.
ORDERS = [1, 3, 5, 7, 9, 11, 13]
HARM_ORDERS = [o for o in ORDERS if o > 1]
PV_HARM_ORDERS = [5, 7, 11, 13]  # PV-inverter dominant orders
# Both precisions: complex64 = data-generation precision (the solver converges at the
# dtype's resolvable floor), complex128 = gradcheck precision. The plots headline
# complex64 (the GPU data-gen path); the CSV/JSON keep both.
DTYPES = {"complex64": torch.complex64, "complex128": torch.complex128}
PLOT_DTYPE = "complex64"
SEED = 0


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
def add_pv_systems(grid, *, fraction: float = 0.5) -> int:
    """Attach a PV generator (unity power factor) to a fraction of the load nodes.

    Each PV unit mirrors its host load's node/phases and is rated at half the load's
    nameplate active power. Returns the number of PV systems added. The harmonic content
    of the PV inverters is supplied by the scenario (an EN 50160-bounded ``h_mag`` spec),
    so no stored spectrum is attached here.
    """
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    next_id = max((a.id for a in grid.appliances), default=0) + 1
    added = 0
    for k, ld in enumerate(loads):
        if (k % max(1, round(1.0 / fraction))) != 0:
            continue
        grid.appliances.append(
            Generator(
                id=next_id,
                name=f"pv_{ld.id}",
                node=ld.node,
                phases=ld.phases,
                p_nom_w=0.5 * float(ld.p_nom_w),
                q_nom_var=0.0,
                consumer_type="pv",
            )
        )
        next_id += 1
        added += 1
    return added


def build_ieee33():
    """IEEE-33 (single-phase equivalent) with Carson geometry + PV systems."""
    grid, _ = ieee33_geometry_grid()
    add_pv_systems(grid, fraction=0.5)
    return grid, "IEEE-33 (1-ph)"


def build_cigre_pv():
    """FULL three-phase CIGRE LV with Carson geometry + PV systems."""
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    synthesize_grid_geometry(grid)
    add_pv_systems(grid, fraction=0.5)
    return grid, "CIGRE LV +PV (3-ph)"


def grid_dims(grid) -> dict:
    """Size descriptors: node-phase rows ``N``, nodes, branches, loads, PV generators."""
    return {
        "rows": int(node_phase_index(grid).size),
        "nodes": len(grid.nodes),
        "branches": len(grid.branches),
        "loads": sum(isinstance(a, Load) for a in grid.appliances),
        "pv": sum(isinstance(a, Generator) for a in grid.appliances),
    }


# ---------------------------------------------------------------------------
# scenario (varying loads + PV + harmonic injections)
# ---------------------------------------------------------------------------
def build_scenario(grid, n_samples: int) -> ScenarioConfig:
    """Randomized batch: per-sample load P/Q, PV P/Q, and load+PV harmonic spectra.

    Generator (PV) specs are only included when the grid actually carries generators, so
    the same builder serves a load-only grid and a load+PV grid.
    """
    has_gen = any(isinstance(a, Generator) for a in grid.appliances)
    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.3, high=1.0),
            field="pq",
            mode="scale",
        ),
        ParameterSpec(
            name="load_spectrum",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.0, high=1.0),  # fraction of the EN 50160 limit
            field="h_mag",
            orders=HARM_ORDERS,
            harmonic_reference="en50160",
        ),
    ]
    if has_gen:
        params += [
            ParameterSpec(
                name="pv_scale",
                selector=Selector(component="generator"),
                distribution=Uniform(low=0.0, high=1.0),  # 0..100 % of rated PV power
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="pv_spectrum",
                selector=Selector(component="generator"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                orders=PV_HARM_ORDERS,
                harmonic_reference="en50160",
            ),
        ]
    return ScenarioConfig(n_samples=n_samples, seed=SEED, parameters=params)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn: Callable, device: str, repeats: int) -> tuple[float, object]:
    """Median wall time of ``fn`` over ``repeats`` runs after one warm-up (CUDA-synced)."""
    out = fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), out


def _peak_vram_mb(device: str) -> Optional[float]:
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def available_devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------
def bench_solver_comparison(grid, grid_label, device, repeats) -> list[dict]:
    """current_injection vs newton on ONE nominal operating point (single grid).

    Newton's linear const-Z warm start does not take a batched operating point, so the
    two solvers are compared at batch size 1; the throughput sweep uses the batchable
    current-injection path.
    """
    rows = []
    for method in ("current_injection", "newton"):
        dev = torch.device(device)
        dt, res = timed(
            lambda m=method: solve_power_flow(
                grid,
                method=m,
                dtype=torch.complex128,
                device=dev,
                criticality="never",
            ),
            device,
            repeats,
        )
        rows.append(
            {
                "kind": "solver_comparison",
                "grid": grid_label,
                "device": device,
                "method": method,
                "dtype": "complex128",
                "batch": 1,
                "time_s": dt,
                "ms_per_scenario": dt * 1e3,
                "iterations": int(res.iterations),
                "converged": bool(res.converged),
                "residual": float(res.residual),
            }
        )
    return rows


def bench_throughput(grid, grid_label, device, batch_sizes, repeats) -> list[dict]:
    """Load-flow and harmonic-flow throughput over a batch-size sweep, both dtypes."""
    rows = []
    dev = torch.device(device)
    for dtype_name, dtype in DTYPES.items():
        for batch in batch_sizes:
            sampled = sample(grid, build_scenario(grid, batch))
            op = sampled.operating_point
            inj = sampled.harmonic_injection or None

            for calc in ("power_flow", "harmonic"):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                def run_power_flow():
                    return solve_power_flow(
                        grid,
                        operating_point=op,
                        method="current_injection",
                        dtype=dtype,
                        device=dev,
                        criticality="never",
                    )

                def run_harmonic():
                    return solve_harmonic_flow(
                        grid,
                        ORDERS,
                        operating_point=op,
                        harmonic_injection=inj,
                        dtype=dtype,
                        device=dev,
                    )

                fn = run_power_flow if calc == "power_flow" else run_harmonic
                dt, res = timed(fn, device, repeats)
                pf = res if calc == "power_flow" else res.pf
                rows.append(
                    {
                        "kind": "throughput",
                        "grid": grid_label,
                        "device": device,
                        "calculation": calc,
                        "dtype": dtype_name,
                        "batch": int(batch),
                        "time_s": dt,
                        "ms_per_scenario": dt * 1e3 / batch,
                        "scenarios_per_s": batch / dt,
                        "iterations": int(pf.iterations),
                        "converged": bool(pf.converged),
                        "peak_vram_mb": _peak_vram_mb(device),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# persistence + plotting
# ---------------------------------------------------------------------------
def _host_meta(device: str) -> dict:
    meta = {
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
    }
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def write_results(out: Path, device: str, dims: dict, rows: list[dict]) -> None:
    payload = {"meta": _host_meta(device), "grid_dims": dims, "rows": rows}
    (out / f"results_{device}.json").write_text(json.dumps(payload, indent=2))


def load_all_results(out: Path) -> tuple[list[dict], dict, dict]:
    """Merge every ``results_<device>.json`` in ``out`` -> (rows, dims, host meta)."""
    rows, dims, meta = [], {}, {}
    for path in sorted(glob(str(out / "results_*.json"))):
        payload = json.loads(Path(path).read_text())
        rows.extend(payload["rows"])
        dims.update(payload.get("grid_dims", {}))
        meta[payload["meta"]["device"]] = payload["meta"]
    return rows, dims, meta


def write_summary_csv(out: Path, rows: list[dict]) -> None:
    import csv

    cols = [
        "kind",
        "grid",
        "device",
        "calculation",
        "method",
        "dtype",
        "batch",
        "time_s",
        "ms_per_scenario",
        "scenarios_per_s",
        "iterations",
        "converged",
        "residual",
        "peak_vram_mb",
    ]
    with open(out / "benchmark_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plots(out: Path, rows: list[dict], dims: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted({r["grid"] for r in rows})
    devices = sorted({r["device"] for r in rows})
    dev_color = {"cpu": "C0", "cuda": "C3"}
    calc_marker = {"power_flow": "o", "harmonic": "s"}

    # 1. Solver comparison: current_injection vs newton (time + iterations), per grid.
    sc = [r for r in rows if r["kind"] == "solver_comparison"]
    if sc:
        methods = ["current_injection", "newton"]
        combos = [(dev, m) for dev in devices for m in methods]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        x = range(len(grids))
        width = 0.8 / max(1, len(combos))
        for ax, metric, ylabel in (
            (axes[0], "ms_per_scenario", "wall time [ms]"),
            (axes[1], "iterations", "iterations to converge"),
        ):
            for ci, (dev, method) in enumerate(combos):
                vals = [
                    next(
                        (
                            r[metric]
                            for r in sc
                            if r["grid"] == g
                            and r["device"] == dev
                            and r["method"] == method
                        ),
                        0,
                    )
                    for g in grids
                ]
                offs = [xi + (ci - (len(combos) - 1) / 2) * width for xi in x]
                ax.bar(
                    offs,
                    vals,
                    width=width,
                    color=dev_color.get(dev, "C2"),
                    alpha=0.6 if method == "newton" else 1.0,
                    hatch="//" if method == "newton" else None,
                    edgecolor="k",
                    linewidth=0.4,
                    label=f"{dev}/{method}",
                )
            ax.set_xticks(list(x))
            ax.set_xticklabels(grids, fontsize=8)
            ax.set_ylabel(ylabel)
        axes[0].set_title("Power-flow solver wall time (nominal, batch=1)")
        axes[1].set_title("Iterations to converge")
        axes[0].legend(fontsize=7)
        fig.suptitle("Current-injection vs Newton power flow")
        fig.savefig(out / "solver_comparison.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2. Load flow vs harmonic flow: per-scenario cost at the largest batch. The
    # throughput plots headline PLOT_DTYPE (the data-generation precision); both dtypes
    # remain in the CSV/JSON for the precision comparison.
    tp = [r for r in rows if r["kind"] == "throughput" and r["dtype"] == PLOT_DTYPE]
    if tp:
        bmax = max(r["batch"] for r in tp)
        big = [r for r in tp if r["batch"] == bmax]
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        calcs = ["power_flow", "harmonic"]
        x = range(len(grids))
        width = 0.8 / max(1, len(devices) * len(calcs))
        i = 0
        for dev in devices:
            for calc in calcs:
                vals = [
                    next(
                        (
                            r["ms_per_scenario"]
                            for r in big
                            if r["grid"] == g
                            and r["device"] == dev
                            and r["calculation"] == calc
                        ),
                        0,
                    )
                    for g in grids
                ]
                offs = [
                    xi + (i - (len(devices) * len(calcs) - 1) / 2) * width for xi in x
                ]
                ax.bar(
                    offs,
                    vals,
                    width=width,
                    color=dev_color.get(dev, "C2"),
                    alpha=1.0 if calc == "power_flow" else 0.6,
                    hatch=None if calc == "power_flow" else "//",
                    edgecolor="k",
                    linewidth=0.4,
                    label=f"{dev}/{calc}",
                )
                i += 1
        ax.set_xticks(list(x))
        ax.set_xticklabels(grids)
        ax.set_ylabel("time per scenario [ms]")
        ax.set_title(f"Load flow vs harmonic flow ({PLOT_DTYPE}, batch={bmax})")
        ax.legend(fontsize=8)
        fig.savefig(out / "loadflow_vs_harmonic.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3. Throughput vs batch size (the scaling curves), per grid subplot.
    if tp:
        fig, axes = plt.subplots(
            1,
            len(grids),
            figsize=(6.2 * len(grids), 4.6),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, g in zip(axes[0], grids):
            for dev in devices:
                for calc in ("power_flow", "harmonic"):
                    series = sorted(
                        [
                            r
                            for r in tp
                            if r["grid"] == g
                            and r["device"] == dev
                            and r["calculation"] == calc
                        ],
                        key=lambda r: r["batch"],
                    )
                    if not series:
                        continue
                    ax.plot(
                        [r["batch"] for r in series],
                        [r["scenarios_per_s"] for r in series],
                        marker=calc_marker[calc],
                        color=dev_color.get(dev, "C2"),
                        linestyle="-" if calc == "power_flow" else "--",
                        label=f"{dev}/{calc}",
                    )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("batch size (scenarios)")
            ax.set_ylabel("throughput [scenarios/s]")
            ax.set_title(g)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Throughput vs batch size ({PLOT_DTYPE})")
        fig.savefig(out / "throughput_vs_batch.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 4. CPU vs GPU speedup at the largest batch (only if a CUDA series is present).
    if "cuda" in devices and "cpu" in devices and tp:
        bmax = max(r["batch"] for r in tp)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        labels, speedups = [], []
        for g in grids:
            for calc in ("power_flow", "harmonic"):
                cpu = next(
                    (
                        r["time_s"]
                        for r in tp
                        if r["grid"] == g
                        and r["device"] == "cpu"
                        and r["calculation"] == calc
                        and r["batch"] == bmax
                    ),
                    None,
                )
                gpu = next(
                    (
                        r["time_s"]
                        for r in tp
                        if r["grid"] == g
                        and r["device"] == "cuda"
                        and r["calculation"] == calc
                        and r["batch"] == bmax
                    ),
                    None,
                )
                if cpu and gpu:
                    labels.append(f"{g}\n{calc}")
                    speedups.append(cpu / gpu)
        ax.bar(range(len(labels)), speedups, color="C2", edgecolor="k")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("CPU time / GPU time")
        ax.set_title(f"GPU speedup over CPU ({PLOT_DTYPE}, batch={bmax})")
        for i, s in enumerate(speedups):
            ax.text(i, s, f"{s:.1f}x", ha="center", va="bottom", fontsize=8)
        fig.savefig(out / "device_speedup.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def batch_sizes_for(max_batch: int, quick: bool) -> list[int]:
    """Batch-size sweep up to ``max_batch``.

    Uses the predefined base points, then EXTENDS the geometric progression (x4) up to
    ``max_batch`` so a large ``--max-batch`` is actually exercised instead of being
    silently capped at the base's last point. ``max_batch`` itself is always included.
    """
    base = [1, 16, 64] if quick else [1, 8, 32, 128, 512]
    sizes = {b for b in base if b <= max_batch} | {max_batch}
    b = base[-1]
    while b < max_batch:
        b *= 4
        sizes.add(min(b, max_batch))
    return sorted(s for s in sizes if s >= 1)


def print_summary(rows: list[dict], dims: dict) -> None:
    print("\n=== Grid sizes ===")
    print(
        f"{'grid':<22} {'rows(N)':>8} {'nodes':>6} {'branches':>9} {'loads':>6} {'PV':>4}"
    )
    for g, d in dims.items():
        print(
            f"{g:<22} {d['rows']:>8} {d['nodes']:>6} {d['branches']:>9} "
            f"{d['loads']:>6} {d['pv']:>4}"
        )

    sc = [r for r in rows if r["kind"] == "solver_comparison"]
    if sc:
        print("\n=== Power-flow solvers (nominal, batch=1) ===")
        print(
            f"{'grid':<22} {'device':<6} {'method':<18} {'time[ms]':>9} {'iters':>6} {'conv':>5}"
        )
        for r in sc:
            print(
                f"{r['grid']:<22} {r['device']:<6} {r['method']:<18} "
                f"{r['ms_per_scenario']:>9.2f} {r['iterations']:>6} "
                f"{str(r['converged']):>5}"
            )

    tp = [r for r in rows if r["kind"] == "throughput"]
    if tp:
        print("\n=== Throughput (load flow vs harmonic flow) ===")
        print(
            f"{'grid':<22} {'device':<6} {'calc':<11} {'dtype':<11} {'B':>5} "
            f"{'time[ms]':>9} {'ms/scn':>8} {'scn/s':>9} {'iters':>6} {'VRAM[MB]':>9}"
        )
        for r in sorted(
            tp,
            key=lambda r: (
                r["grid"],
                r["device"],
                r["calculation"],
                r["dtype"],
                r["batch"],
            ),
        ):
            vram = f"{r['peak_vram_mb']:.0f}" if r.get("peak_vram_mb") else "-"
            print(
                f"{r['grid']:<22} {r['device']:<6} {r['calculation']:<11} "
                f"{r['dtype']:<11} {r['batch']:>5} {r['time_s'] * 1e3:>9.1f} "
                f"{r['ms_per_scenario']:>8.3f} {r['scenarios_per_s']:>9.1f} "
                f"{r['iterations']:>6} {vram:>9}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", nargs="?", default="evaluation_output/benchmark")
    ap.add_argument("--max-batch", type=int, default=512, help="Largest batch size.")
    ap.add_argument("--repeats", type=int, default=3, help="Timed repeats (median).")
    ap.add_argument("--quick", action="store_true", help="Smaller, faster sweep.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")  # the synthesized-GMR notice is expected here
    torch.manual_seed(SEED)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    repeats = 1 if args.quick else args.repeats
    batches = batch_sizes_for(args.max_batch, args.quick)

    builders = [build_ieee33, build_cigre_pv]
    grids = [b() for b in builders]
    dims = {label: grid_dims(g) for g, label in grids}

    devices = available_devices()
    print(f"Devices: {devices}  |  batch sizes: {batches}  |  repeats: {repeats}")

    for device in devices:
        rows: list[dict] = []
        for grid, label in grids:
            print(f"[{device}] {label} ...", flush=True)
            rows += bench_solver_comparison(grid, label, device, repeats)
            rows += bench_throughput(grid, label, device, batches, repeats)
        write_results(out, device, dims, rows)

    all_rows, all_dims, meta = load_all_results(out)
    write_summary_csv(out, all_rows)
    make_plots(out, all_rows, all_dims)

    print_summary(all_rows, all_dims)
    print("\nHost(s):")
    for dev, m in meta.items():
        extra = f" | {m.get('gpu')}" if m.get("gpu") else ""
        print(f"  {dev}: torch {m['torch']} | {m['cpu']}{extra}")
    print(f"\nResults + figures -> {out.resolve()}")
    print(
        "  results_<device>.json, benchmark_summary.csv, solver_comparison.svg, "
        "loadflow_vs_harmonic.svg, throughput_vs_batch.svg"
        + (", device_speedup.svg" if "cuda" in meta else "")
    )


if __name__ == "__main__":
    main()
