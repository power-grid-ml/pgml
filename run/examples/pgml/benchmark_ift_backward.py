"""Cost of the gradient: backward time and peak memory against the forward solve.

WHAT THIS MEASURES
------------------
A differentiable power flow is only useful if its BACKWARD pass is affordable. The
gradient comes from the implicit function theorem: the backward builds the real
block-diagonal state Jacobian ``J = dR/dx`` of the converged residual, solves one adjoint
system ``J^T λ = grad_x`` against its factorization, and forms the parameter gradients
with a single vector-Jacobian product of the residual. The Jacobian build is the expensive
part and is chosen by a memory budget (``solver.ift.jacobian_budget_mb``):

- the whole batch in ONE vectorized call — fastest, but its intermediate grows with
  ``B² · 2N³`` because the residual's ``[B, N, N]`` admittance is replicated once per
  output row;
- as many scenarios at a time as the budget allows (the same build per chunk);
- column by column, ``2N`` batched Jacobian-vector products, ``O(B·(2N)²)`` memory.

The adjoint factorization is cached on the autograd node, so a caller that needs SEVERAL
vector-Jacobian products of one solve (a full output Jacobian built row by row, or any
second backward under ``retain_graph``) pays one back-substitution per further product
instead of rebuilding the Jacobian.

This script reports, per grid and batch size: forward time, backward time, their ratio,
the peak memory of the backward, the extra cost of a SECOND backward (the factorization
cache), and the build the budget selected. Memory is ``torch.cuda.max_memory_allocated``
on CUDA and the high-water mark of the process' resident set (``ru_maxrss``) on CPU, where
torch allocates outside the python allocator; the CPU figure is a LOWER bound, because a
larger earlier peak hides a smaller later one, so the batch sizes are measured in
increasing order and a zero means "no new high-water mark". The absolute high-water mark of
the process is reported next to it, which is the number to read when the script is run with
ONE configuration per process (the only way to get a clean CPU memory figure).

Run (CPU): ``pixi run -e cpu python run/examples/pgml/benchmark_ift_backward.py``
Run (GPU): ``pixi run python run/examples/pgml/benchmark_ift_backward.py --device cuda``
Optional args: ``--grids ieee33 kerber4``, ``--batch 1 4 16 64``,
``--budget-mb 1024`` (the Jacobian budget to measure with; repeatable),
``--linear-solver auto``, ``--json out.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import time
from pathlib import Path

import torch

from pgml.schemas.grid_schema import Grid, InjectionAppliance
from pgml.solver import solve_power_flow
from pgml.solver.power_flow import _jacobian_chunk

CDT = torch.complex128


# --------------------------------------------------------------------------- grids
def _ieee33() -> Grid:
    from pgml.grids import ieee33_geometry_grid

    return ieee33_geometry_grid()[0]


def _kerber(copies: int = 1) -> Grid:
    import pandapower as pp
    import pandapower.networks as pn

    from pgml.convert.pandapower import to_grid
    from pgml.multigrid import merge_grids

    net = pn.create_kerber_vorstadtnetz_kabel_1()
    pp.runpp(net, numba=False)
    grid, _ = to_grid(net)
    if copies == 1:
        return grid
    return merge_grids([grid] * copies).grid


GRIDS = {
    "ieee33": _ieee33,
    "kerber": lambda: _kerber(1),
    "kerber4": lambda: _kerber(4),
}


def _batched_operating_point(grid: Grid, batch: int) -> tuple[dict, torch.Tensor]:
    """Scale every load's active power per scenario, through ONE leaf tensor.

    One leaf keeps the measurement about the backward of the solve rather than about the
    number of parameters; the per-scenario factors make the batch a genuine scenario
    batch (the operating point varies, the network does not).
    """
    scale = torch.linspace(0.8, 1.2, batch, dtype=torch.float64, requires_grad=True)
    op: dict = {}
    for appliance in grid.appliances:
        if not isinstance(appliance, InjectionAppliance) or not appliance.in_service:
            continue
        p = appliance.p_nom_w
        q = appliance.q_nom_var
        if p is None:
            continue
        op[int(appliance.id)] = {
            "p_w": float(p) * scale,
            "q_var": float(q or 0.0) * scale,
        }
    return op, scale


def _sync(device) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()


def _host_peak_bytes() -> int:
    """Process resident-set high-water mark [bytes] (monotone over the process)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


#: Set once the process has paid torch's lazy initialisation (see ``_warm_up``).
_WARMED = False


def _warm_up(grid: Grid, device, linear_solver: str) -> None:
    """One untimed solve + backward, once per process.

    The first call pays torch's lazy initialisation and the assembly's first-touch
    allocations, which would otherwise land on the first measured point. It is done on the
    FIRST (smallest) configuration only — repeating it per point would double the cost of
    the expensive ones.
    """
    global _WARMED
    if _WARMED:
        return
    op, _ = _batched_operating_point(grid, 1)
    res = solve_power_flow(
        grid, operating_point=op, dtype=CDT, device=device, linear_solver=linear_solver
    )
    res.v.abs().sum().backward()
    del res, op
    gc.collect()
    _WARMED = True


def _measure(grid: Grid, batch: int, device, linear_solver: str) -> dict:
    _warm_up(grid, device, linear_solver)
    op, scale = _batched_operating_point(grid, batch)
    cuda = device is not None and device.type == "cuda"

    _sync(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = solve_power_flow(
        grid,
        operating_point=op,
        dtype=CDT,
        device=device,
        linear_solver=linear_solver,
    )
    _sync(device)
    t_forward = time.perf_counter() - t0
    forward_peak = torch.cuda.max_memory_allocated() if cuda else 0
    loss = res.v.abs().sum()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    host_before = _host_peak_bytes()
    t0 = time.perf_counter()
    loss.backward(retain_graph=True)
    _sync(device)
    t_backward = time.perf_counter() - t0
    backward_peak = (
        torch.cuda.max_memory_allocated()
        if cuda
        else max(0, _host_peak_bytes() - host_before)
    )

    # A second product of the SAME solve: the cached adjoint factorization should make it
    # a back-substitution instead of another Jacobian build.
    scale.grad = None
    t0 = time.perf_counter()
    res.v.abs().sum().backward()
    _sync(device)
    t_backward_again = time.perf_counter() - t0

    n = res.index.size
    budget = None
    from pgml.solver.power_flow import _ift_jacobian_budget_bytes

    budget = _ift_jacobian_budget_bytes()
    chunk = _jacobian_chunk(batch, n, CDT, budget)
    build = (
        "vectorized-whole-batch"
        if chunk >= batch
        else ("vectorized-chunked" if chunk >= 1 else "column-by-column")
    )
    out = {
        "rows": n,
        "batch": batch,
        "converged": bool(res.converged),
        "iterations": int(res.iterations),
        "forward_s": t_forward,
        "backward_s": t_backward,
        "backward_again_s": t_backward_again,
        "ratio": t_backward / max(t_forward, 1e-12),
        "backward_peak_mib": backward_peak / 1024**2,
        "process_peak_mib": _host_peak_bytes() / 1024**2,
        "forward_peak_mib": forward_peak / 1024**2,
        "jacobian_build": build,
        "jacobian_chunk": chunk,
        "budget_mib": budget / 1024**2,
        "grad_finite": bool(torch.isfinite(scale.grad).all()),
    }
    del res, loss, op, scale
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", default=["ieee33", "kerber4"])
    ap.add_argument("--batch", nargs="*", type=int, default=[1, 4, 16, 64])
    ap.add_argument("--budget-mb", nargs="*", type=float, default=[None])
    ap.add_argument("--linear-solver", default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    device = torch.device(args.device) if args.device else None
    report: dict = {
        "device": str(device or "cpu"),
        "torch": torch.__version__,
        "linear_solver": args.linear_solver,
        "runs": [],
    }
    for budget_mb in args.budget_mb:
        if budget_mb is not None:
            import pgml.solver.power_flow as pf_mod

            pf_mod._ift_jacobian_budget_bytes = lambda *a, _b=budget_mb, **k: int(
                _b * 1024 * 1024
            )
        for name in args.grids:
            grid = GRIDS[name]()
            for batch in args.batch:
                try:
                    rec = _measure(grid, batch, device, args.linear_solver)
                except (RuntimeError, torch.OutOfMemoryError) as exc:
                    rec = {
                        "batch": batch,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                rec["grid"] = name
                rec["budget_mb_requested"] = budget_mb
                report["runs"].append(rec)
                if "error" in rec:
                    print(f"{name:10s} B={batch:5d}  {rec['error']}", flush=True)
                else:
                    print(
                        f"{name:10s} N={rec['rows']:5d} B={batch:5d}  "
                        f"fwd {rec['forward_s'] * 1e3:9.1f} ms  "
                        f"bwd {rec['backward_s'] * 1e3:10.1f} ms  "
                        f"({rec['ratio']:7.1f}x)  "
                        f"bwd again {rec['backward_again_s'] * 1e3:8.1f} ms  "
                        f"peak {rec['backward_peak_mib']:8.1f} MiB  "
                        f"rss {rec['process_peak_mib']:8.1f} MiB  "
                        f"{rec['jacobian_build']} (chunk {rec['jacobian_chunk']})",
                        flush=True,
                    )
            del grid
            gc.collect()

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
