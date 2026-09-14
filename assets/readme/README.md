# README figures

These small artifacts are self-contained copies/selections of the paper's
recorded evidence. `provenance.json` identifies the paper commit and SHA-256
hashes of the original inputs. No experiment is run by the figure renderer.

- `batch_throughput.json`: IEEE 33-bus CPU/GPU complex128 series from the
  September 12 batch benchmark, with its original environment. The renderer
  excludes nonconverged, nonfinite or over-tolerance records. This is a dated
  snapshot, awaiting the current solver's job-320 result; it is not labeled as
  a measurement of the current review build.
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
`/tmp`. Refresh the curated batch JSON, original-source hash, environment,
README measurement caption and `performance_current_review` flag together only
after checking the replacement stage's provenance, coverage and validity.
