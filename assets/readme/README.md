# README figures

These small artifacts contain the README's measured evidence. `provenance.json`
identifies source revisions, the benchmark job and SHA-256 hashes of the original
inputs. Measurement dates remain in the raw metadata. No experiment is run by
the figure renderer.

- `batch_throughput.json`: complete IEEE33 and Kerber complex128 CPU/GPU series
  and pandapower/power-grid-model baselines, with the measured environment.
  All seven batch sizes (1–4096) are retained for all five curves. Every scenario
  is checked, not just a prefix; the renderer rejects invalid comparison points.
  The focused driver is `scripts/bench/bench_readme.py` in pgml-paper, submitted
  through `scripts/cluster/run_readme.sbatch` with the suite's target settings.
  Each grid runs in a fresh process. Eight pandapower workers each have one
  internal thread; power-grid-model uses eight native threads.
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

The renderer writes the throughput and recovery SVGs here and preview PNGs to
`/tmp`. Refresh batch JSON, source hashes, environment and README measurement
caption together after checking provenance, coverage and validity.
