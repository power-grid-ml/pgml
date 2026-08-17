"""Switch-state sweeps: low-rank update-solve vs assembling every state.

WHAT THIS DEMONSTRATES
----------------------
A switch-state sweep (``branch_states`` with a batched state) solves ONE network
under ``S`` per-branch admittance scalings. Two strategies exist
(``solve_power_flow(..., branch_states_method=...)``):

- ``"assemble"`` (default) — assemble and factor the admittance of every state:
  ``O(S·N³)`` work and an ``[S, N, N]`` matrix in memory;
- ``"woodbury"`` — assemble and factor the BASE network once and reach each state
  through a Sherman-Morrison-Woodbury low-rank update of that factorization
  (:mod:`pgml.solver.lowrank`). A switched ``P``-phase branch contributes a
  rank-``2P`` term, so with ``k = Σ 2P`` update columns a state costs
  ``O(N²k + k³)``.

The win therefore grows with ``S`` and shrinks with ``k``: this script sweeps the
system size ``N`` (rows), the number of switched 3-phase branches (``k = 6`` each)
and the state count ``S``, and reports the end-to-end ``solve_power_flow`` wall
time of both strategies plus the crossover in ``k``.

The two paths are also compared numerically (max relative voltage difference) so a
speedup is never reported without its accuracy.

Run (CPU): ``pixi run -e cpu python run/examples/pgml/benchmark_woodbury.py``
Run (GPU): ``pixi run python run/examples/pgml/benchmark_woodbury.py``
Optional args: ``--sizes 200 400 700`` (nodes), ``--switched 1 4 16 64``
(3-phase branches, ``k = 6·switched``), ``--states 8``, ``--json out.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

from pgml.grids import synthetic_feeder
from pgml.solver import solve_power_flow

logging.getLogger("pgml").setLevel(logging.WARNING)

TIE0 = 30000  # first tie-switch id of synthetic_feeder
BYTES_PER_ENTRY = 16  # complex128


def _time(fn, *, repeat: int = 2, sync: bool = False) -> float:
    """Best-of-``repeat`` wall time of ``fn()`` in seconds (1 warmup call)."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def switch_states(grid, n_switched: int, states: int, seed: int = 0) -> dict:
    """A batch of ``states`` configurations over ``n_switched`` 3-phase branches.

    The grid's tie switches come first (open/closed, the switch-sweep case); beyond
    them the feeder's own line segments are scaled continuously in ``[0.5, 1]`` (a
    differentiable topology parameter — they are bridges, so they may not open) to
    push the update rank ``k = 6·n_switched`` past what the ties alone reach.
    """
    g = torch.Generator().manual_seed(seed)
    ties = [b.id for b in grid.branches if b.id >= TIE0][:n_switched]
    lines = [b.id for b in grid.branches if b.id < TIE0][: n_switched - len(ties)]
    st = {
        bid: torch.randint(0, 2, (states,), generator=g).to(torch.float64)
        for bid in ties
    }
    st.update(
        {
            bid: 0.5 + 0.5 * torch.rand(states, generator=g, dtype=torch.float64)
            for bid in lines
        }
    )
    return st


def bench_case(nodes: int, n_switched: int, states: int, device) -> dict:
    """Both strategies on one (size, switched-branch-count, state-count) case."""
    grid = synthetic_feeder(nodes, n_feeders=8, tie_switches=min(7, n_switched))
    st = switch_states(grid, n_switched, states)
    rows = 3 * nodes
    k = 6 * len(st)
    sync = device.type == "cuda"

    def run(method):
        return solve_power_flow(
            grid, branch_states=st, branch_states_method=method, device=device
        )

    dense_gib = states * rows * rows * BYTES_PER_ENTRY / 2**30
    out = {
        "nodes": nodes,
        "rows": rows,
        "switched": len(st),
        "k": k,
        "states": states,
        "device": device.type,
        "assemble_gib": dense_gib,
    }
    t_wb = _time(lambda: run("woodbury"), sync=sync)
    out["woodbury_s"] = t_wb
    if dense_gib > 4.0:
        out["assemble_s"] = None  # the [S, N, N] stack would not fit
        return out
    t_as = _time(lambda: run("assemble"), sync=sync)
    v_as, v_wb = run("assemble").v, run("woodbury").v
    out["assemble_s"] = t_as
    out["speedup"] = t_as / t_wb
    out["max_rel_diff"] = float((v_as - v_wb).abs().max() / v_as.abs().max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 700, 1000])
    ap.add_argument("--switched", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []
    for device in devices:
        for nodes in args.sizes:
            print(
                f"\n=== {nodes} nodes -> {3 * nodes} rows, S={args.states} states, "
                f"{device.type} ==="
            )
            for n_switched in args.switched:
                r = bench_case(nodes, n_switched, args.states, device)
                rows.append(r)
                if r["assemble_s"] is None:
                    print(
                        f"  k={r['k']:3d} | woodbury {r['woodbury_s'] * 1e3:9.1f} ms | "
                        f"assemble skipped ({r['assemble_gib']:.1f} GiB of Y)"
                    )
                    continue
                print(
                    f"  k={r['k']:3d} | woodbury {r['woodbury_s'] * 1e3:9.1f} ms | "
                    f"assemble {r['assemble_s'] * 1e3:9.1f} ms | "
                    f"speedup {r['speedup']:5.2f}x | rel diff {r['max_rel_diff']:.1e}"
                )

    # The crossover: the largest k at which the update still wins, per size.
    print("\ncrossover (largest k with speedup > 1):")
    for device in devices:
        for nodes in args.sizes:
            wins = [
                r["k"]
                for r in rows
                if r["nodes"] == nodes
                and r["device"] == device.type
                and r.get("speedup", 0.0) > 1.0
            ]
            best = max(wins) if wins else None
            print(
                f"  {device.type:4s} {3 * nodes:5d} rows: "
                + (f"k <= {best}" if best else "never (assemble wins at every k)")
            )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
