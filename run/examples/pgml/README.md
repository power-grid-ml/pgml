# `pgml` examples — differentiable harmonic power flow

Runnable, documented studies on real benchmark grids for the **power-grid-ml** (`pgml`)
core — the differentiable, GPU-ready harmonic power-flow engine. Each script's module
docstring is the authoritative "what / how / outputs" reference; this file is the map.

Run from the repository root. Outputs are written under `data/pgml/evaluation_output/<name>/`
(the untracked data root, anchored to the repository root so they never land in the source
tree):

```bash
pixi run -e cpu python run/examples/pgml/<script>.py [out_dir]
```

## The full-CIGRE-LV scenario studies (Carson line model, validated vs live OpenDSS)

- **`scenario_node_injection_sweep.py`** — inject a harmonic error source at each node in
  turn (voltage spectrum from `spectra/VoltageSag40ms.csv`) and record the per-unit `h=11`
  voltage spread `|V_h11|/|V1|` across the grid; parity vs live OpenDSS (bit-exact ~1e-13).
- **`scenario_randomized.py`** — a randomized symmetric-vs-asymmetric study: per-load
  independent-per-phase power scale + EN 50160-bounded harmonic spectra, solved both ways,
  timed, plotted (fundamental profile + 3D harmonic), parity vs OpenDSS (bit-exact per order).

## Validation / diagnostics

- **`evaluate_ieee33.py`** — load-flow evaluation vs pandapower (IEEE-33).
- **`evaluate_harmonics_carson.py`** — OpenDSS-vs-`pgml` harmonic comparison (Carson geometry).
- **`evaluate_line_sequence_harmonics.py`** — positive-sequence vs naive vs OpenDSS line models.
- **`current_injection_convergence.py`** — why the current-injection power flow oscillates near
  the loadability nose (a 2-bus radial with a closed-form P-V nose).
- **`loadability_continuation.py`** — `loadability_limit` on CIGRE LV: the breaking λ*/margin,
  which bus collapses, and which load most limits the margin.
- **`benchmark_speed.py`** — CPU/GPU execution-speed study across every solve path and a
  batch-size sweep (auto-detects CUDA; writes one `results_<device>.json` per device).

## The scenario system these are built on

Everything is `pgml.scenarios` — a serializable config (+ seed) deterministically defines a
batch; one batched solve produces aligned results; persist reproducibly with `write_dataset`
/ `read_dataset`. Full interface ledger: `src/pgml/scenarios/CONTEXT.md`.
