# README figures

These small artifacts contain the README's measured evidence. `provenance.json`
identifies source revisions, the benchmark job and SHA-256 hashes of the original
inputs. Measurement dates remain in the raw metadata. No experiment is run by
the figure renderer.

- `batch_throughput.json`: complete IEEE33 and Kerber complex128 CPU/GPU series
  and pandapower/power-grid-model baselines, with the measured thki-lev environment.
  All seven batch sizes (1–4096) are retained for all five curves. Every scenario
  is checked, not just a prefix; the renderer rejects invalid comparison points.
  The focused driver is `scripts/bench/bench_readme.py` in pgml-paper, submitted
  through `scripts/cluster/run_readme.sbatch` with the suite's target settings.
  Each grid runs in a fresh process. Eight pandapower workers each have one
  internal thread; power-grid-model uses eight native threads.
- `conformance_worst.svg`: the paper's Figure 5, copied without changing its
  data or layout. `conformance_worst.json` retains the selected grid records
  and valid-pair summary. Its bars compare three reference engines; they do
  not show a before/after intervention.
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
