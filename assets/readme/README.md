# README figures

These small artifacts contain the README's measured evidence. `provenance.json`
identifies source revisions, the benchmark job and SHA-256 hashes of the original
inputs. Measurement dates remain in the raw metadata. No experiment is run by
the figure renderer.

- `batch_throughput.json`: IEEE 33-bus and Kerber complex128 series of pgml on
  CPU and GPU with the pandapower, power-grid-model and OpenDSS baselines and the measured environment. All seven batch sizes (1 to 4096) are retained
  for every curve. Every scenario is checked, not just a prefix, and the renderer
  rejects invalid comparison points, including a pandapower point measured
  without active numba. The driver is `scripts/bench/bench_readme.py` in
  pgml-paper, one grid per process, submitted as the stages `readme_ieee33` and
  `readme_kerber` of `scripts/cluster/run_all.sh`. Eight pandapower and eight
  OpenDSS workers have one thread each and power-grid-model uses eight threads.
- `size_scaling.json`: the same tools over a family of radial feeders with 16 to
  4096 buses at 1 and 256 scenarios per batch (`scripts/bench/bench_size.py`,
  stage `readme_size`), with the same every-scenario check.
- Both files are assembled by `scripts/bench/readme_assets.py` in pgml-paper,
  which also writes the jobs, versions and source hashes into `provenance.json`.
  `../PERFORMANCE.md` describes the measurement protocol.
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

The renderer writes the throughput, size-scaling and recovery SVGs here and preview PNGs to
`/tmp`. Refresh batch JSON, source hashes, environment and README measurement
caption together after checking provenance, coverage and validity.
