"""Compare linear-algebra strategies and nonlinear methods of the pgml solvers.

Two independent parts, selected with ``--part``:

``refactor``
    One harmonic order of a synthetic feeder, solved for a batch of scenarios under
    up to eight strategies: keep one factorization of the shared network and
    back-substitute every scenario; factor every scenario's own matrix (through the
    solver wrapper, through its sparse backend, and through the bare torch LU); call
    ``torch.linalg.solve`` on the assembled matrices (batched and one scenario at a
    time); correct one shared factorization onto each scenario's EXACT matrix with a
    Woodbury update; and an iterative solve that never factors anything. The
    scenario-dependent matrices come from the device harmonic shunt on the
    operating-point basis, so every scenario really does carry its own network.

``method``
    The current-injection fixed point against Newton on the same grids, loadings and
    batch sizes: iterations, forward wall time, the agreement of the two solutions,
    and the wall time of one gradient through each.

Example::

    PYTHONPATH=src python run/examples/pgml/benchmark_solver_strategy.py \
        --part refactor --out /tmp/refactor.json

Uses pgml's synthetic grid builders and installed dependencies only. Timings describe
the machine they were measured on and do not transfer to other hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import platform
import statistics
import time

import torch

from pgml.assembly import node_phase_index
from pgml.grids import (
    CONVERTER_SPECTRUM,
    cigre_lv_geometry_grid,
    ieee33_geometry_grid,
    synthetic_feeder,
)
from pgml.schemas import HarmonicComponent, Load, SpectrumPoint, StaticSpectrum
from pgml.schemas.grid_schema import Grid
from pgml.solver import (
    HarmonicFlowSystem,
    assemble_harmonic_system,
    lu_factor_system,
    solve_factored,
    solve_harmonic_flow,
    solve_power_flow,
)
from pgml.solver.lowrank import low_rank_update, solve_factored_updated

_ORDER = 5  # one harmonic order keeps every strategy on one matrix per scenario


def _spectrum() -> StaticSpectrum:
    return StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=h, magnitude_pu=m, phase_deg=a)
                for h, m, a in CONVERTER_SPECTRUM
            ]
        )
    )


def _attach_spectra(grid, every: int = 4) -> None:
    spec = _spectrum()
    for i, load in enumerate(a for a in grid.appliances if isinstance(a, Load)):
        if i % every == 0:
            load.spectrum = spec


def _thin_loads(grid: Grid, keep_every: int) -> Grid:
    """Keep a load on every ``keep_every``-th load bus, redistributing the total power.

    A feeder with a load on every bus has a device harmonic shunt on every row, so no
    low-rank strategy applies to it. Thinning the population is what makes the
    structured-change regime (a shunt on a few buses) measurable next to the dense one.
    """
    if keep_every <= 1:
        return grid
    data = grid.model_dump()
    kept, dropped = [], 0
    load_i = 0
    for appliance in data["appliances"]:
        if appliance.get("component") == "load":
            keep = load_i % keep_every == 0
            load_i += 1
            if not keep:
                dropped += 1
                continue
            appliance["p_nom_w"] = appliance["p_nom_w"] * keep_every
            appliance["q_nom_var"] = appliance["q_nom_var"] * keep_every
        kept.append(appliance)
    data["appliances"] = kept
    return Grid(**data)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, *, repeats: int, device: torch.device) -> tuple[float, object]:
    """Median wall time of ``repeats`` calls after one warm-up, plus the last result."""
    out = fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), out


def _load_operating_point(grid, batch, device, rdt, seed: int = 0) -> dict:
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    rng = torch.Generator().manual_seed(seed)
    scale = 0.6 + 0.8 * torch.rand((batch, len(loads)), generator=rng, dtype=rdt)
    scale = scale.to(device)
    return {
        a.id: {"p_w": a.p_nom_w * scale[:, i], "q_var": a.q_nom_var * scale[:, i]}
        for i, a in enumerate(loads)
    }


def _bicgstab(matvec, rhs, diag, *, rtol: float, max_iter: int):
    """Batched Jacobi-preconditioned BiCGStab; returns ``(x, iterations, residual)``.

    Stands in for "solve without ever factoring". Batched over the leading dims of
    ``rhs``, with one shared iteration count and no per-system early exit, which is what
    a vectorised matrix-free solve on an accelerator does.
    """
    minv = 1.0 / diag
    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    r0 = r.clone()
    rho = torch.ones(rhs.shape[:-1], dtype=rhs.dtype, device=rhs.device)
    alpha = torch.ones_like(rho)
    omega = torch.ones_like(rho)
    v = torch.zeros_like(rhs)
    p = torch.zeros_like(rhs)
    b_norm = torch.linalg.vector_norm(rhs, dim=-1)
    tiny = torch.finfo(rhs.real.dtype).tiny

    def safe(t):
        return torch.where(t.abs() < tiny, torch.full_like(t, tiny), t)

    it = 0
    for it in range(1, max_iter + 1):
        rho_new = (r0.conj() * r).sum(-1)
        beta = (rho_new / safe(rho)) * (alpha / safe(omega))
        p = r + beta.unsqueeze(-1) * (p - omega.unsqueeze(-1) * v)
        v = matvec(minv * p)
        alpha = rho_new / safe((r0.conj() * v).sum(-1))
        s = r - alpha.unsqueeze(-1) * v
        t = matvec(minv * s)
        omega = (t.conj() * s).sum(-1) / safe((t.conj() * t).sum(-1))
        x = x + alpha.unsqueeze(-1) * (minv * p) + omega.unsqueeze(-1) * (minv * s)
        r = s - omega.unsqueeze(-1) * t
        rho = rho_new
        if bool((torch.linalg.vector_norm(r, dim=-1) <= rtol * b_norm).all()):
            break
    residual = float((torch.linalg.vector_norm(rhs - matvec(x), dim=-1) / b_norm).max())
    return x, it, residual


def _harmonic_systems(grid, batch, device, cdt):
    """Shared, shunt-free and per-scenario matrices plus the right-hand side."""
    rdt = torch.float64 if cdt == torch.complex128 else torch.float32
    op = _load_operating_point(grid, batch, device, rdt)
    pf = solve_power_flow(
        grid,
        slack="norton",
        operating_point=op,
        dtype=cdt,
        device=device,
        criticality="never",
    )
    if not pf.converged:
        raise RuntimeError("Nonconverged fundamental makes the benchmark invalid")
    common = dict(
        operating_point=pf.resolved_operating_point(op),
        dtype=cdt,
        device=device,
    )

    def square(y, b):
        n = y.shape[-1]
        return y.reshape(-1, n, n)[:b] if b > 1 else y.reshape(-1, n, n)[:1]

    y_scen, i_rhs, _ = assemble_harmonic_system(
        grid,
        [_ORDER],
        pf.v,
        load_shunt="opendss",
        load_shunt_basis="operating_point",
        **common,
    )
    y_share, _, _ = assemble_harmonic_system(
        grid,
        [_ORDER],
        pf.v,
        load_shunt="opendss",
        load_shunt_basis="nameplate",
        **common,
    )
    y_free, _, _ = assemble_harmonic_system(
        grid, [_ORDER], pf.v, load_shunt="none", **common
    )
    n = y_scen.shape[-1]
    return (
        square(y_share, 1)[0],
        square(y_free, 1)[0],
        y_scen.reshape(-1, n, n),
        i_rhs.reshape(-1, n),
    )


def _lowrank_terms(y_scen, y_free, limit_ratio: float):
    """Exact ``(U, C)`` with ``y_scen = y_free + U C U^H`` on the touched rows.

    The device shunt stamps a block on the rows its terminals occupy, so the
    scenario-to-scenario difference is supported on those rows only. Selecting them
    gives an exact low-rank factorization without assuming anything about the stamp.
    Returns ``None`` when the support is too large for a low-rank strategy to pay.
    """
    diff = y_scen - y_free
    touched = (diff.abs().amax(dim=0).amax(dim=0) > 0) | (
        diff.abs().amax(dim=0).amax(dim=1) > 0
    )
    rows = torch.nonzero(touched, as_tuple=False).reshape(-1)
    k = int(rows.numel())
    if k == 0 or 3 * k >= y_scen.shape[-1] * limit_ratio:
        return None, k
    n = y_scen.shape[-1]
    u = torch.zeros(n, k, dtype=y_scen.dtype, device=y_scen.device)
    u[rows, torch.arange(k, device=y_scen.device)] = 1.0
    c = diff.index_select(-2, rows).index_select(-1, rows)
    return (u, c), k


def _refactor_case(grid, name, batch, device, cdt, args, use_sparse):
    y_share, y_free, y_scen, i_rhs = _harmonic_systems(grid, batch, device, cdt)
    n = y_scen.shape[-1]
    diag = torch.diagonal(y_scen, dim1=-2, dim2=-1)
    # Some LAPACK builds refuse the row interchange of a batched complex LU on
    # matrices that need pivoting. That takes a strategy off the table on that
    # machine without saying anything about the others, so each one is measured on
    # its own and a refusal is recorded next to the strategy it belongs to.
    try:
        reference = torch.linalg.solve(y_scen, i_rhs.unsqueeze(-1)).squeeze(-1)
        batched_dense = True
    except RuntimeError:
        reference = torch.stack(
            [torch.linalg.solve(y_scen[b], i_rhs[b]) for b in range(batch)]
        )
        batched_dense = False
    ref_scale = reference.abs().amax()
    terms, k_support = _lowrank_terms(y_scen, y_free, args.lowrank_limit)

    def error(v) -> float:
        return float((v - reference).abs().amax() / ref_scale)

    def matvec(x):
        return torch.einsum("bij,bj->bi", y_scen, x)

    def factor_reuse():
        return solve_factored(lu_factor_system(y_share, backend="dense"), i_rhs)

    def factor_per_scenario(backend="dense"):
        return solve_factored(lu_factor_system(y_scen, backend=backend), i_rhs)

    def torch_lu_per_scenario():
        lu, piv = torch.linalg.lu_factor(y_scen)
        return torch.linalg.lu_solve(lu, piv, i_rhs.unsqueeze(-1)).squeeze(-1)

    def direct_batched():
        return torch.linalg.solve(y_scen, i_rhs.unsqueeze(-1)).squeeze(-1)

    def direct_looped():
        return torch.stack(
            [torch.linalg.solve(y_scen[b], i_rhs[b]) for b in range(batch)]
        )

    def woodbury():
        fac = lu_factor_system(y_free, backend="dense")
        upd = low_rank_update(fac, terms[0], terms[1], estimate_amplification=False)
        return solve_factored_updated(upd, i_rhs)

    def iterative():
        return _bicgstab(matvec, i_rhs, diag, rtol=1e-10, max_iter=args.bicg_iter)

    plan = [
        ("factor_reuse_shared", factor_reuse),
        ("factor_per_scenario", factor_per_scenario),
        ("torch_lu_per_scenario", torch_lu_per_scenario),
        ("direct_batched", direct_batched),
        ("direct_looped", direct_looped),
    ]
    if use_sparse:
        plan.append(
            ("factor_per_scenario_sparse", lambda: factor_per_scenario("sparse"))
        )
    if terms is not None:
        plan.append(("woodbury_shared_factor", woodbury))
    plan.append(("iterative_no_factor", iterative))

    modes = {}
    for label, fn in plan:
        try:
            seconds, out = _time(fn, repeats=args.repeats, device=device)
        except RuntimeError as exc:
            modes[label] = dict(unavailable=str(exc).splitlines()[0])
            continue
        extra = {}
        if label == "iterative_no_factor":
            v, iters, residual = out
            extra = dict(iterations=iters, relative_backward_error=residual)
        else:
            v = out
        finite = bool(torch.isfinite(v).all())
        modes[label] = dict(
            seconds=seconds,
            scenarios_per_s=batch / seconds,
            max_rel_difference=error(v) if finite else float("nan"),
            finite=finite,
            **extra,
        )
    # The shared-network strategy answers a DIFFERENT model (the nameplate-basis
    # shunt), so its difference from the reference is a modeling error, not round-off.
    if "seconds" in modes["factor_reuse_shared"]:
        modes["factor_reuse_shared"]["difference_is_model_error"] = True
    result = dict(
        grid=name,
        rows=n,
        batch=batch,
        dtype=str(cdt).replace("torch.", ""),
        device=device.type,
        order=_ORDER,
        lowrank_support_rows=k_support,
        batched_dense_lu_available=batched_dense,
        modes=modes,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_refactor(args, device) -> list[dict]:
    rows = []
    for n_nodes in args.nodes:
        grid = _thin_loads(synthetic_feeder(n_nodes), args.keep_every_load)
        _attach_spectra(grid)
        name = f"synthetic{n_nodes}"
        n_rows = node_phase_index(grid).size
        use_sparse = device.type == "cpu" and n_rows >= 512 and args.sparse
        for cdt in args.dtypes:
            item = 16 if cdt == torch.complex128 else 8
            for batch in args.batches:
                if batch * n_rows * n_rows * item > args.max_matrix_bytes:
                    continue
                row = _refactor_case(grid, name, batch, device, cdt, args, use_sparse)
                rows.append(row)
                timed = {k: v for k, v in row["modes"].items() if "seconds" in v}
                best = min(timed.items(), key=lambda kv: kv[1]["seconds"])
                print(
                    f"{name} N={row['rows']} B={batch} {row['dtype']} {device.type} "
                    f"k={row['lowrank_support_rows']}: "
                    + "  ".join(
                        f"{k}={v['seconds'] * 1e3:.2f}ms" for k, v in timed.items()
                    )
                    + f"  -> best {best[0]}",
                    flush=True,
                )
    return rows


# ---------------------------------------------------------------------- preparation


def _preparation_case(grid, name, batch, shunt, basis, device, args):
    """One harmonic study run three ways: prepared and replayed, prepared with a new
    operating point every call, and without any preparation."""
    orders = args.orders
    rdt = torch.float64
    stream = [
        _load_operating_point(grid, batch, device, rdt, seed=s)
        for s in range(args.stream)
    ]
    kwargs = dict(
        slack="norton",
        load_shunt=shunt,
        load_shunt_basis=basis,
        criticality="never",
        dtype=torch.complex128,
        device=device,
    )

    def call(op, system):
        res = solve_harmonic_flow(
            grid, orders, operating_point=op, system=system, **kwargs
        )
        if not res.converged or not bool(torch.isfinite(res.v).all()):
            raise RuntimeError(
                "Nonconverged/nonfinite study is not a valid measurement"
            )
        return res

    reference = [call(op, None).v for op in stream]

    def check(values):
        return max(float((v - r).abs().amax()) for v, r in zip(values, reference))

    # Uncached: every call rebuilds assembly and factors from scratch.
    uncached = []
    for op in stream:
        seconds, _ = _time(lambda o=op: call(o, None), repeats=1, device=device)
        uncached.append(seconds)

    # First prepared call pays assembly, factorization AND the key snapshots.
    system = HarmonicFlowSystem()
    _sync(device)
    t0 = time.perf_counter()
    call(stream[0], system)
    _sync(device)
    first_prepared = time.perf_counter() - t0

    replay, replay_v = [], []
    for _ in range(args.stream):
        _sync(device)
        t0 = time.perf_counter()
        res = call(stream[0], system)
        _sync(device)
        replay.append(time.perf_counter() - t0)
        replay_v.append(res.v)
    stats_replay = system.stats
    bytes_replay = system.nbytes()

    changing, changing_v = [], []
    for op in stream:
        _sync(device)
        t0 = time.perf_counter()
        res = call(op, system)
        _sync(device)
        changing.append(time.perf_counter() - t0)
        changing_v.append(res.v)

    streaming = HarmonicFlowSystem(cache_batched_factors=False)
    call(stream[0], streaming)
    stream_times = []
    for op in stream:
        _sync(device)
        t0 = time.perf_counter()
        call(op, streaming)
        _sync(device)
        stream_times.append(time.perf_counter() - t0)

    med_uncached = statistics.median(uncached)
    med_replay = statistics.median(replay)
    med_changing = statistics.median(changing)

    def break_even(per_call):
        saving = med_uncached - per_call
        if saving <= 0:
            return None
        return 1.0 + max(0.0, first_prepared - med_uncached) / saving

    return dict(
        grid=name,
        rows=node_phase_index(grid).size,
        batch=batch,
        shunt=shunt,
        basis=basis,
        orders=orders,
        device=device.type,
        uncached_s=med_uncached,
        first_prepared_s=first_prepared,
        replay_s=med_replay,
        changing_s=med_changing,
        streaming_s=statistics.median(stream_times),
        speedup_replay=med_uncached / med_replay,
        speedup_changing=med_uncached / med_changing,
        break_even_calls_replay=break_even(med_replay),
        break_even_calls_changing=break_even(med_changing),
        retained_bytes_replay=bytes_replay,
        retained_bytes_changing=system.nbytes(),
        retained_bytes_streaming=streaming.nbytes(),
        stats_replay=stats_replay,
        stats_changing=system.stats,
        stats_streaming=streaming.stats,
        max_abs_difference_v=max(check(replay_v[:1]), check(changing_v)),
    )


def run_preparation(args, device) -> list[dict]:
    rows = []
    for n_nodes in args.nodes:
        grid = synthetic_feeder(n_nodes)
        _attach_spectra(grid)
        name = f"synthetic{n_nodes}"
        for shunt, basis in (
            ("none", "nameplate"),
            ("opendss", "nameplate"),
            ("opendss", "operating_point"),
        ):
            for batch in args.batches:
                row = _preparation_case(grid, name, batch, shunt, basis, device, args)
                rows.append(row)
                print(
                    f"{name} N={row['rows']} B={batch} {shunt}/{basis}: "
                    f"uncached {row['uncached_s'] * 1e3:.1f}ms  "
                    f"replay {row['replay_s'] * 1e3:.1f}ms ({row['speedup_replay']:.2f}x)  "
                    f"changing {row['changing_s'] * 1e3:.1f}ms "
                    f"({row['speedup_changing']:.2f}x)  "
                    f"retained {row['retained_bytes_changing'] / 1024**2:.1f} MiB  "
                    f"dV={row['max_abs_difference_v']:.1e} V",
                    flush=True,
                )
    return rows


# --------------------------------------------------------------------------- method


def _scaled_operating_point(grid, factor, batch, device, spread: float = 0.1):
    loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    if batch == 1:
        return {
            a.id: {"p_w": a.p_nom_w * factor, "q_var": a.q_nom_var * factor}
            for a in loads
        }
    rng = torch.Generator().manual_seed(1)
    jitter = (1.0 - 0.5 * spread) + spread * torch.rand(
        (batch, len(loads)), generator=rng, dtype=torch.float64
    )
    jitter = (jitter * factor).to(device)
    return {
        a.id: {"p_w": a.p_nom_w * jitter[:, i], "q_var": a.q_nom_var * jitter[:, i]}
        for i, a in enumerate(loads)
    }


def _solve(grid, method, op, device, max_iter):
    return solve_power_flow(
        grid,
        slack="ideal",
        method=method,
        operating_point=op,
        device=device,
        criticality="never",
        max_iter=max_iter,
    )


def _fixed_point_limit(grid, device, max_iter, hi=400.0) -> float:
    """Largest load multiple (to 1 %) at which the fixed point still converges."""
    lo = 1.0
    base = _scaled_operating_point(grid, lo, 1, device)
    if not _solve(grid, "current_injection", base, device, max_iter).converged:
        return lo
    while hi - lo > 0.01 * lo:
        mid = 0.5 * (lo + hi)
        op = _scaled_operating_point(grid, mid, 1, device)
        ok = _solve(grid, "current_injection", op, device, max_iter).converged
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def _gradient_seconds(grid, method, op, device, repeats, max_iter):
    """Wall time of one forward plus backward w.r.t. a line resistance."""
    line = next(b for b in grid.branches if hasattr(b, "series_resistance_ohm_per_m"))
    base = line.series_resistance_ohm_per_m
    leaf = (
        torch.as_tensor(base, dtype=torch.float64)
        .clone()
        .to(device)
        .requires_grad_(True)
    )
    line.series_resistance_ohm_per_m = leaf

    def once():
        if leaf.grad is not None:
            leaf.grad = None
        res = _solve(grid, method, op, device, max_iter)
        res.v.abs().sum().backward()
        return leaf.grad.clone()

    try:
        return _time(once, repeats=repeats, device=device)
    finally:
        line.series_resistance_ohm_per_m = base


def run_method(args, device) -> list[dict]:
    cases = [
        ("ieee33", ieee33_geometry_grid()[0]),
        ("cigre_lv", cigre_lv_geometry_grid()[0]),
    ]
    cases += [(f"synthetic{n}", synthetic_feeder(n)) for n in args.nodes]
    rows = []
    for name, grid in cases:
        n_rows = node_phase_index(grid).size
        limit = _fixed_point_limit(grid, device, args.max_iter)
        for label, factor in (("nominal", 1.0), ("near_limit", 0.95 * limit)):
            # Near the limit a wide scenario spread pushes part of the batch past the
            # loadability nose, which measures infeasibility rather than the method.
            spread = 0.1 if label == "nominal" else 0.02
            for batch in args.batches:
                op = _scaled_operating_point(grid, factor, batch, device, spread)
                entry = dict(
                    grid=name,
                    rows=n_rows,
                    batch=batch,
                    loading=label,
                    load_factor=factor,
                    fixed_point_limit=limit,
                    device=device.type,
                    methods={},
                )
                voltages = {}
                for method in ("current_injection", "newton"):
                    seconds, res = _time(
                        lambda m=method: _solve(grid, m, op, device, args.max_iter),
                        repeats=args.repeats,
                        device=device,
                    )
                    voltages[method] = res.v
                    entry["methods"][method] = dict(
                        seconds=seconds,
                        scenarios_per_s=batch / seconds,
                        iterations=res.iterations,
                        converged=bool(res.converged),
                        n_failed=len(res.failed_states),
                        mismatch_pu=float(res.diagnostics.mismatch_max_pu),
                        likely_cause=res.diagnostics.likely_cause,
                    )
                delta = voltages["current_injection"] - voltages["newton"]
                entry["max_abs_difference_v"] = float(delta.abs().amax())
                if batch == args.batches[0] and label == "nominal":
                    grads = {}
                    for method in ("current_injection", "newton"):
                        seconds, grad = _gradient_seconds(
                            grid, method, op, device, args.grad_repeats, args.max_iter
                        )
                        entry["methods"][method]["forward_backward_seconds"] = seconds
                        grads[method] = grad
                    denom = grads["current_injection"].abs().amax().clamp_min(1e-300)
                    entry["max_rel_gradient_difference"] = float(
                        (grads["current_injection"] - grads["newton"]).abs().amax()
                        / denom
                    )
                rows.append(entry)
                ci = entry["methods"]["current_injection"]
                nt = entry["methods"]["newton"]
                print(
                    f"{name} N={n_rows} B={batch} {label}(x{factor:.2f}): "
                    f"CI {ci['iterations']}it {ci['seconds'] * 1e3:.1f}ms "
                    f"conv={ci['converged']} | NT {nt['iterations']}it "
                    f"{nt['seconds'] * 1e3:.1f}ms conv={nt['converged']} | "
                    f"dV={entry['max_abs_difference_v']:.2e} V",
                    flush=True,
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=["refactor", "method", "preparation"], required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--grad-repeats", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--nodes", type=int, nargs="+", default=[33, 100, 200])
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--bicg-iter", type=int, default=400)
    parser.add_argument("--keep-every-load", type=int, default=6)
    parser.add_argument("--lowrank-limit", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--dtype", nargs="+", default=["complex128"])
    parser.add_argument("--no-sparse", dest="sparse", action="store_false")
    parser.add_argument("--max-matrix-bytes", type=float, default=3.0e9)
    parser.add_argument("--stream", type=int, default=7)
    parser.add_argument(
        "--orders", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 13]
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    logging.getLogger("pgml").setLevel(logging.ERROR)
    device = torch.device(args.device)
    args.dtypes = [getattr(torch, d) for d in args.dtype]

    if args.part == "refactor":
        with torch.no_grad():
            rows = run_refactor(args, device)
    elif args.part == "preparation":
        with torch.no_grad():
            rows = run_preparation(args, device)
    else:
        rows = run_method(args, device)

    payload = dict(
        part=args.part,
        platform=platform.platform(),
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        device=args.device,
        device_name=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        repeats=args.repeats,
        keep_every_load=args.keep_every_load,
        rows=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
