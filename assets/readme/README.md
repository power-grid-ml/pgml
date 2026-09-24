# README figures

These small artifacts contain the README's measured evidence. They are produced
by the [pgml paper repository](https://github.com/power-grid-ml/pgml-paper) (to be
published soon). `provenance.json` identifies source revisions and SHA-256 hashes
of the original inputs. No experiment is run by
the figure renderer.

- `batch_throughput.json`: complex128 throughput against batch size on four
  grids, for pgml on one GPU, pgml on eight CPU worker processes, pgml in one
  batched CPU call, pandapower, power-grid-model and OpenDSS, with the measured
  environment and the CPU allocation each run had. The driver is
  `scripts/bench/bench_batch.py` in pgml-paper, stage `fair_batch` of
  `scripts/cluster/run_all.sh`. Every engine on the CPU side gets the same eight
  physical cores and the same eight workers or threads.
- `harmonic_throughput.json`: the same sweep for whole harmonic studies, where
  OpenDSS is the only reference tool with a harmonic power flow. Its points are
  accepted at a looser tolerance than the fundamental ones, and each record
  carries the tolerance it was judged at.
- `harmonic_shunt_basis.json`: what the harmonic device-shunt basis costs. The
  same studies with the shunt built per scenario, which is the model OpenDSS
  solves, and with one shunt shared by the batch, against OpenDSS on both
  (`scripts/bench/bench_harmonic_basis.py`, drawn by
  `scripts/fig_harmonic_basis.py` in pgml-paper, not by the renderer here).
- `size_scaling.json`: the same tools over a family of radial feeders with 16 to
  4096 buses at four batch sizes (`scripts/bench/bench_size.py`, stage
  `fair_size`).
- `footprint.json`, `footprint_harmonic.json`: peak host memory per tool and peak
  device memory for pgml, one fresh interpreter per measured point, the process
  tree sampled during the solve (`scripts/bench/bench_footprint.py`).
- `cost.json`: cost per million solved scenarios from the measured throughput and
  published list prices (`scripts/bench/bench_cost.py`, `scripts/bench/prices.py`).
  A price model, not a measurement.
- `crossovers.json`: the batch and grid sizes at which each pair of tools crosses
  (`scripts/bench/fairness_tables.py`).
- The renderer rejects invalid comparison points, including a pandapower point
  measured without active numba, a point whose deviation exceeds the tolerance its
  record carries, and a point validated on fewer scenarios than the run declares.
- All files are assembled by `scripts/bench/readme_assets.py --fair` in
  pgml-paper, which also writes the jobs, versions and source hashes into
  `provenance.json`. `../PERFORMANCE.md` describes the measurement protocol.
- `solverconf_*` and `modelconf_*`: figures of the two conformance checks, copied
  unchanged from `figures/solverconf/` and `figures/modelconf/` of pgml-paper, each
  with the JSON of its numbers and its generated caption (`*.caption.txt`).
  `provenance.json` records the paper commit, the pgml source commit
  and the SHA-256 of every copied file under `conformance_source`. The README shows
  `solverconf_factor_dtype_vs_tools.svg`; `assets/CONFORMANCE.md` explains both
  checks and shows the others. The figures are produced by
  `scripts/conformance/solver_conformance/figures.py` and
  `python -m model_conformance.figures` in pgml-paper, not by the renderer here.
- `app_digital_twin.json`: recorded full experiment, including true parameters,
  measurement locations/noise, estimates, and four-draw standard deviations.
  `resistance_recovery.svg` redraws only the resistance panel in a wide format.
  The 30%/20% generation spreads are Gaussian standard deviations, not bounds.

From the repository root:

```bash
pixi run -e cpu python run/readme/render.py
```

The renderer writes the throughput, size-scaling, harmonic, memory, cost and
recovery SVGs here and preview PNGs to `/tmp`. A figure whose source file is not
published is skipped; the README's own figure and the recovery panel are
required. Refresh the result JSONs, source hashes, environment and the README
measurement caption together after checking provenance, coverage and validity.
