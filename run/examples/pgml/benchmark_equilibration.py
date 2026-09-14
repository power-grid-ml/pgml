"""Conditioning and single-precision accuracy, with and without equilibration.

WHAT THIS MEASURES
------------------
The engine solves in SI units, so its nodal matrices are badly SCALED: a stiff source row
carries an admittance near ``1e5 S`` while a low-voltage cable row carries ``1e-2 S``, and
at harmonic order ``h`` the series reactances grow with ``h`` while the shunt terms on the
diagonal do not. The solvers therefore factor the equilibrated matrix ``D_r A D_c`` and
undo the scaling on the solution (:mod:`pgml.solver.equilibration`, on by default, mode
``solver.equilibration.mode``).

This script reports, per grid and per harmonic order:

1. the condition number of the matrix the solver factors — the fundamental free block
   ``Y_ff`` and each requested order's ``Y(h)`` — as assembled and under each
   equilibration mode. A 1-norm estimate (Hager's power method over the existing
   factorization, :func:`pgml.solver.harmonic.estimate_condition`) is taken at every size;
   the exact 2-norm condition number is added below ``--exact-max-rows``;
2. the accuracy of a single-precision solve of the SAME system, against a complex128
   reference: plain ``complex64`` and ``precision="mixed"`` (complex64 factors refined
   against complex128 residuals), each with and without equilibration.

Both answer one question: how much of the ill conditioning of an SI-unit power-flow
system is an artefact of its units, and does removing it make single precision usable.

Run (CPU): ``pixi run -e cpu python run/examples/pgml/benchmark_equilibration.py``
Run (GPU): ``pixi run python run/examples/pgml/benchmark_equilibration.py --device cuda``
Optional args: ``--grids ieee33 cigre_lv_3ph``, ``--grid-json path.json[:label]`` (a
persisted :class:`~pgml.schemas.grid_schema.Grid`, repeatable), ``--orders 1 13``,
``--modes off symmetric``, ``--exact-max-rows 3200``, ``--skip-accuracy``,
``--json out.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch

from pgml.assembly import node_phase_index
from pgml.assembly._params import phase_voltage_magnitude
from pgml.schemas.grid_schema import Grid
from pgml.solver import prepare_power_flow, solve_harmonic_flow
from pgml.solver.equilibration import equilibrate_matrix
from pgml.solver.harmonic import estimate_condition, lu_factor_system
from pgml.solver.harmonic_flow import assemble_harmonic_ybus

CDT = torch.complex128
CF = torch.complex64


# --------------------------------------------------------------------------- grids
def _ieee33():
    from pgml.grids import ieee33_geometry_grid

    return ieee33_geometry_grid()[0]


def _cigre_lv(phase_mode: str):
    from pgml.convert.pandapower import PhaseMode
    from pgml.grids import cigre_lv_full_grid

    mode = (
        PhaseMode.THREE_PHASE
        if phase_mode == "three_phase"
        else PhaseMode.SINGLE_PHASE_EQUIV
    )
    return cigre_lv_full_grid(phase_mode=mode)[0]


def _pandapower_net(builder: str):
    import pandapower as pp
    import pandapower.networks as pn

    from pgml.convert.pandapower import to_grid

    net = getattr(pn, builder)()
    pp.runpp(net, numba=False)
    return to_grid(net)[0]


def _synthetic(nodes: int):
    from pgml.grids import synthetic_feeder

    return synthetic_feeder(nodes)


GRIDS = {
    "ieee33": _ieee33,
    "cigre_lv_1ph": lambda: _cigre_lv("single_phase"),
    "cigre_lv_3ph": lambda: _cigre_lv("three_phase"),
    "mv_oberrhein": lambda: _pandapower_net("mv_oberrhein"),
    "kerber": lambda: _pandapower_net("create_kerber_vorstadtnetz_kabel_1"),
    "synthetic_1200": lambda: _synthetic(400),
}


# --------------------------------------------------------------------- the matrices
def _fundamental_matrix(grid: Grid, device) -> torch.Tensor:
    """The matrix the fundamental solve factors (the free block under ideal slack)."""
    fac = prepare_power_flow(
        grid, dtype=CDT, device=device, linear_solver="dense", equilibrate="off"
    )
    return fac.factorization.y_mat.reshape(fac.factorization.y_mat.shape[-2:])


def _harmonic_matrix(grid: Grid, order: int, device) -> torch.Tensor:
    y, _ = assemble_harmonic_ybus(grid, [order], dtype=CDT, device=device)
    return y.reshape(y.shape[-1], y.shape[-1])


def _conditioning(a: torch.Tensor, modes, exact_max_rows: int) -> list[dict]:
    m = int(a.shape[-1])
    backend = "sparse" if (a.device.type == "cpu" and m >= 400) else "dense"
    out = []
    for mode in modes:
        a_hat, _, _ = equilibrate_matrix(a, mode=mode)
        t0 = time.perf_counter()
        fac = lu_factor_system(a_hat, backend=backend, equilibrate="off")
        rec = {
            "scaling": mode,
            "rows": m,
            "backend": backend,
            "cond_1_estimate": estimate_condition(fac, iters=8),
            "estimate_seconds": time.perf_counter() - t0,
        }
        if m <= exact_max_rows:
            rec["cond_2_exact"] = float(torch.linalg.cond(a_hat))
        out.append(rec)
        del fac, a_hat
        gc.collect()
    return out


# ------------------------------------------------------------------------- accuracy
def _voltage_bases(grid: Grid, device) -> torch.Tensor:
    index = node_phase_index(grid)
    by_id = {int(n.id): n for n in grid.nodes}
    return torch.tensor(
        [
            phase_voltage_magnitude(
                float(by_id[int(i)].u_rated_v), len(by_id[int(i)].phases)
            )
            for i in index.node_ids.tolist()
        ],
        dtype=torch.float64,
        device=device,
    )


def _accuracy(grid: Grid, orders, modes, device) -> list[dict]:
    """Max ``|ΔV|`` in per unit against the complex128 reference, per order."""
    bases = _voltage_bases(grid, device)
    ref = solve_harmonic_flow(
        grid, orders, dtype=CDT, device=device, equilibrate="symmetric"
    )
    out = []
    runs = [("complex64", CF, "full"), ("mixed", CDT, "mixed")]
    for label, dtype, precision in runs:
        for mode in modes:
            t0 = time.perf_counter()
            try:
                got = solve_harmonic_flow(
                    grid,
                    orders,
                    dtype=dtype,
                    precision=precision,
                    device=device,
                    equilibrate=mode,
                )
            except Exception as exc:  # a single-precision solve may fail outright
                out.append(
                    {
                        "run": label,
                        "scaling": mode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            err = ((got.v.to(CDT) - ref.v).abs() / bases).amax(dim=-1)  # [H]
            out.append(
                {
                    "run": label,
                    "scaling": mode,
                    "seconds": time.perf_counter() - t0,
                    "converged": bool(got.pf.converged),
                    "iterations": int(got.pf.iterations),
                    "max_dv_pu": {
                        str(h): float(err.reshape(-1)[k]) for k, h in enumerate(orders)
                    },
                }
            )
            del got
            gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", default=["ieee33", "cigre_lv_3ph"])
    ap.add_argument(
        "--grid-json",
        nargs="*",
        default=[],
        help="persisted Grid JSON, optionally 'path:label'",
    )
    ap.add_argument("--orders", nargs="*", type=int, default=[1, 13])
    ap.add_argument("--modes", nargs="*", default=["off", "symmetric"])
    ap.add_argument("--exact-max-rows", type=int, default=3200)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    device = torch.device(args.device) if args.device else None
    jobs: list[tuple[str, callable]] = [(name, GRIDS[name]) for name in args.grids]
    for spec in args.grid_json:
        path, _, label = spec.partition(":")
        jobs.append(
            (
                label or Path(path).stem,
                lambda p=path: Grid.model_validate_json(Path(p).read_text()),
            )
        )

    report: dict = {
        "device": str(device or "cpu"),
        "torch": torch.__version__,
        "orders": args.orders,
        "grids": {},
    }
    for label, build in jobs:
        grid = build()
        entry: dict = {"nodes": len(grid.nodes), "conditioning": {}}
        for order in args.orders:
            a = (
                _fundamental_matrix(grid, device)
                if order == 1
                else _harmonic_matrix(grid, order, device)
            )
            rows = _conditioning(a, args.modes, args.exact_max_rows)
            entry["conditioning"][str(order)] = rows
            entry["rows"] = rows[0]["rows"]
            for rec in rows:
                print(
                    f"{label:16s} N={rec['rows']:6d} h={order:3d} {rec['scaling']:11s} "
                    f"cond1_est={rec['cond_1_estimate']:.3e} "
                    f"cond2={rec.get('cond_2_exact', float('nan')):.3e}",
                    flush=True,
                )
            del a
            gc.collect()
        if not args.skip_accuracy:
            entry["accuracy"] = _accuracy(grid, args.orders, args.modes, device)
            for rec in entry["accuracy"]:
                if "error" in rec:
                    print(
                        f"{label:16s} {rec['run']:10s} {rec['scaling']:11s} {rec['error']}"
                    )
                    continue
                worst = max(rec["max_dv_pu"].values())
                print(
                    f"{label:16s} {rec['run']:10s} {rec['scaling']:11s} "
                    f"worst max|dV| = {worst:.3e} pu  "
                    f"converged={rec['converged']} its={rec['iterations']} "
                    f"({rec['seconds']:.2f}s)",
                    flush=True,
                )
        report["grids"][label] = entry
        del grid
        gc.collect()

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
