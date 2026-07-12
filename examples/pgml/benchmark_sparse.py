"""Sparse-vs-dense factorization benchmark: where does each backend win?

WHAT THIS DEMONSTRATES
----------------------
The nonlinear power flow factors the constant ``Y_eff`` once and back-substitutes
every fixed-point iteration and scenario (:func:`pgml.solver.harmonic.lu_factor_system`
/ :func:`solve_factored`). Two backends exist:

- ``dense``  — batched ``torch.linalg.lu_factor`` / ``lu_solve`` (CPU + CUDA);
- ``sparse`` — scipy SuperLU (CPU only). A power-grid ``Y`` has O(N) nonzeros, so
  the sparse factorization scales ~O(N) where dense LU is O(N³).

This script sweeps synthetic radial MV feeders (:func:`pgml.grids.synthetic_feeder`)
across system sizes and reports, per backend:

1. factorization time,
2. back-substitution time for a single RHS and a ``B``-column scenario batch,
3. the end-to-end ``solve_power_flow`` wall time (fixed point, one factorization,
   ~5-15 back-substitutions) with a batched operating point.

On a CUDA host the dense rows are additionally measured on the GPU, answering the
practical question "is CPU-sparse or GPU-dense faster at this size?" — GPUs excel
at batched dense linear algebra, so the sparse advantage must always be verified
against the dense-GPU baseline, never assumed.

The crossover observed on CPU calibrates ``_SPARSE_MIN_ROWS`` (the ``"auto"``
backend threshold in ``pgml/solver/harmonic.py``).

Run (CPU): ``pixi run -e cpu python examples/pgml/benchmark_sparse.py``
Run (GPU): ``pixi run python examples/pgml/benchmark_sparse.py``
Optional args: ``--sizes 33 100 300 1000`` (nodes), ``--batch 256``, ``--json out.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

from pgml.grids import synthetic_feeder
from pgml.solver import solve_power_flow
from pgml.solver.harmonic import lu_factor_system, solve_factored
from pgml.assembly import assemble_network_ybus

logging.getLogger("pgml").setLevel(logging.WARNING)


def _time(fn, *, repeat: int = 3, sync: bool = False) -> float:
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


def batched_operating_point(grid, b: int, seed: int = 0) -> dict:
    """Uniform ±50 % scaling of every load's nameplate P/Q, ``b`` scenarios."""
    g = torch.Generator().manual_seed(seed)
    op = {}
    for a in grid.appliances:
        if getattr(a, "p_nom_w", None) is not None and a.id >= 20000:
            scale = 0.5 + torch.rand(b, generator=g)
            op[a.id] = {
                "p_w": float(a.p_nom_w) * scale,
                "q_var": float(a.q_nom_var) * scale,
            }
    return op


def bench_backend(y, backend: str, b: int, device) -> dict:
    """Factor + back-substitution timings for one backend on one system."""
    y = y.to(device)
    n = y.shape[-1]
    rhs1 = torch.randn(1, n, dtype=y.dtype, device=device)
    rhsb = torch.randn(b, 1, n, dtype=y.dtype, device=device)
    sync = device.type == "cuda"
    t_factor = _time(lambda: lu_factor_system(y, backend=backend), sync=sync)
    fac = lu_factor_system(y, backend=backend)
    t_solve1 = _time(lambda: solve_factored(fac, rhs1), sync=sync)
    t_solveb = _time(lambda: solve_factored(fac, rhsb), sync=sync)
    return {
        "backend": backend,
        "device": device.type,
        "factor_s": t_factor,
        "solve1_s": t_solve1,
        f"solve_b{b}_s": t_solveb,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[33, 100, 200, 400, 800, 1600]
    )
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []
    for n_nodes in args.sizes:
        grid = synthetic_feeder(n_nodes)
        yb = assemble_network_ybus(grid, [50.0])
        n = yb.Y.shape[-1]
        print(f"\n=== feeder {n_nodes} nodes -> {n} node-phase rows ===")

        for device in devices:
            backends = ["dense", "sparse"] if device.type == "cpu" else ["dense"]
            for backend in backends:
                r = bench_backend(yb.Y, backend, args.batch, device)
                r.update(nodes=n_nodes, rows=n)
                rows.append(r)
                print(
                    f"  {device.type:4s} {backend:6s} | factor {r['factor_s'] * 1e3:9.2f} ms"
                    f" | solve x1 {r['solve1_s'] * 1e3:8.2f} ms"
                    f" | solve x{args.batch} {r[f'solve_b{args.batch}_s'] * 1e3:8.2f} ms"
                )

        # End-to-end nonlinear solve (fixed point), batched operating point.
        op = batched_operating_point(grid, args.batch)
        for device in devices:
            backends = ["dense", "sparse"] if device.type == "cpu" else ["dense"]
            for backend in backends:
                t = _time(
                    lambda: solve_power_flow(
                        grid,
                        operating_point=op,
                        linear_solver=backend,
                        device=device,
                    ),
                    repeat=1,
                    sync=device.type == "cuda",
                )
                res = solve_power_flow(
                    grid, operating_point=op, linear_solver=backend, device=device
                )
                rows.append(
                    {
                        "backend": backend,
                        "device": device.type,
                        "nodes": n_nodes,
                        "rows": n,
                        "solve_power_flow_s": t,
                        "iterations": res.iterations,
                        "converged": bool(res.converged),
                        "scenarios_per_s": args.batch / t,
                    }
                )
                print(
                    f"  {device.type:4s} {backend:6s} | solve_power_flow B={args.batch}: "
                    f"{t * 1e3:9.1f} ms ({res.iterations} it, conv={res.converged}) "
                    f"-> {args.batch / t:9.0f} scen/s"
                )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
