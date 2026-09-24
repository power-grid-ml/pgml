"""Measure uncached, first-prepared and warm harmonic calls with equality guards.

Example::

    PYTHONPATH=src python run/examples/pgml/benchmark_harmonic_preparation.py \
        --out /tmp/harmonic-preparation.json

Uses pgml's synthetic feeder builder and installed dependencies only. No network
access, reference solver, Git checkout or external dataset is needed. Timings
describe the local CPU and are not transferable to other hardware.
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
from pgml.grids import CONVERTER_SPECTRUM, synthetic_feeder
from pgml.schemas import HarmonicComponent, Load, SpectrumPoint, StaticSpectrum
from pgml.solver import HarmonicFlowSystem, solve_harmonic_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--nodes", type=int, nargs="+", default=[33, 100])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["none", "nameplate", "operating_point"],
        default=["none", "nameplate", "operating_point"],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.getLogger("pgml").setLevel(logging.ERROR)
    orders = [1, 3, 5, 7, 9, 11, 13]
    grids = [(f"synthetic{n}", synthetic_feeder(n)) for n in args.nodes]
    spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=h, magnitude_pu=m, phase_deg=a)
                for h, m, a in CONVERTER_SPECTRUM
            ]
        )
    )
    for _, grid in grids:
        for i, load in enumerate(a for a in grid.appliances if isinstance(a, Load)):
            if i % 4 == 0:
                load.spectrum = spectrum

    rows = []
    with torch.no_grad():
        for name, grid in grids:
            loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
            for batch in args.batches:
                rng = torch.Generator().manual_seed(0)
                scale = 0.8 + 0.4 * torch.rand(
                    (batch, len(loads)), generator=rng, dtype=torch.float64
                )
                op = {
                    a.id: {
                        "p_w": a.p_nom_w * scale[:, i],
                        "q_var": a.q_nom_var * scale[:, i],
                    }
                    for i, a in enumerate(loads)
                }
                for shunt, basis in [
                    ("none", "operating_point"),
                    ("opendss", "nameplate"),
                    ("opendss", "operating_point"),
                ]:
                    if ("none" if shunt == "none" else basis) not in args.models:
                        continue
                    kwargs = dict(
                        slack="norton",
                        operating_point={
                            k: {f: v.clone() for f, v in entry.items()}
                            for k, entry in op.items()
                        },
                        load_shunt=shunt,
                        load_shunt_basis=basis,
                        criticality="never",
                        tol=1e-8,
                    )
                    cache = HarmonicFlowSystem()

                    def call(system):
                        t0 = time.perf_counter()
                        result = solve_harmonic_flow(
                            grid, orders, system=system, **kwargs
                        )
                        elapsed = time.perf_counter() - t0
                        if not result.converged or not torch.isfinite(result.v).all():
                            raise RuntimeError(
                                "Nonconverged/nonfinite benchmark is invalid"
                            )
                        return elapsed, result

                    _, reference = call(
                        None
                    )  # import/backend warmup, not a prepared solve
                    first_s, first = call(cache)
                    torch.testing.assert_close(
                        first.v, reference.v, rtol=1e-10, atol=1e-8
                    )
                    cold, warm = [], []
                    for repeat in range(args.repeats):
                        # Alternating order limits systematic timing drift.
                        for prepared in (
                            [False, True] if repeat % 2 == 0 else [True, False]
                        ):
                            elapsed, result = call(cache if prepared else None)
                            torch.testing.assert_close(
                                result.v, reference.v, rtol=1e-10, atol=1e-8
                            )
                            (warm if prepared else cold).append(elapsed)
                    stats_before = cache.stats
                    streaming = HarmonicFlowSystem(cache_batched_factors=False)
                    call(streaming)
                    changed_times = {"uncached": [], "prepared": [], "streaming": []}
                    error = 0.0
                    for repeat in range(args.repeats):
                        # In-place overrides model NEW scenarios, not replay.
                        for entry in kwargs["operating_point"].values():
                            entry["p_w"].mul_(0.997)
                            entry["q_var"].mul_(0.997)
                        outputs = {}
                        modes = [
                            ("uncached", None),
                            ("prepared", cache),
                            ("streaming", streaming),
                        ]
                        # Rotate timing order without changing the input scenario.
                        modes = modes[repeat % 3 :] + modes[: repeat % 3]
                        for label, preparation in modes:
                            elapsed, result = call(preparation)
                            changed_times[label].append(elapsed)
                            outputs[label] = result.v
                        for label in ("prepared", "streaming"):
                            error = max(
                                error,
                                float(
                                    (outputs[label] - outputs["uncached"]).abs().max()
                                ),
                            )
                            torch.testing.assert_close(
                                outputs[label],
                                outputs["uncached"],
                                rtol=1e-10,
                                atol=1e-8,
                            )
                    row = dict(
                        grid=name,
                        rows=node_phase_index(grid).size,
                        batch=batch,
                        shunt=shunt,
                        basis=basis,
                        first_prepared_s=first_s,
                        uncached_s=cold,
                        warm_s=warm,
                        speedup=statistics.median(cold) / statistics.median(warm),
                        changed_times_s=changed_times,
                        changed_max_complex_error_v=error,
                        stats_before_change=stats_before,
                        stats_after_change=cache.stats,
                        streaming_stats=streaming.stats,
                    )
                    rows.append(row)
                    print(
                        f"{name} B={batch} {shunt}/{basis}: "
                        f"{statistics.median(cold) * 1e3:.2f} -> {statistics.median(warm) * 1e3:.2f} ms "
                        f"({row['speedup']:.2f}x), changed error {error:.2g} V",
                        flush=True,
                    )

    payload = dict(
        platform=platform.platform(),
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        cuda_available=torch.cuda.is_available(),
        orders=orders,
        repeats=args.repeats,
        rows=rows,
    )
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
