# Examples

Runnable, documented studies, organised per package:

- **`pgml` examples** — the differentiable harmonic power-flow core — live in **this
  directory** (documented below).
- **`pgl` examples** — harmonic state estimation — live in **[`pgl/`](pgl/)** (see
  [`pgl/README.md`](pgl/README.md)).
- `pgg` and later packages get their own subdirectory as they land.

Run any `pgml` example from the repository root:

```bash
pixi run -e cpu python examples/<script>.py [out_dir]
```

Outputs default to `examples/evaluation_output/<name>/` (anchored to the `examples/` tree,
not the current directory), or `$PGML_EXPERIMENTS/<name>/` when that variable is set. Each
script's module docstring is the authoritative "what / how / outputs" reference; this file is
the map + how to control them.

## `pgml` examples

## The two full-CIGRE-LV scenario studies

Both use the **full** CIGRE LV benchmark (all 3 LV feeders + MV source + 3 transformers)
via `pgml.evaluation.references.cigre_lv_full_grid(phase_mode=...)`, modeled with the
**Carson earth-return** harmonic line model, and compared to a **live OpenDSS** solve
(`pgml.evaluation.references.opendss_harmonic_voltages`).

### `scenario_node_injection_sweep.py` — per-node error-source sweep (Scenario 1)
Single-phase, **Carson** line model. Injects a harmonic **error source at each node**
(a Thévenin voltage source, or `KIND="current"` for a Norton source) carrying the
voltage spectrum from `spectra/VoltageSag40ms.csv`, of strength `SOURCE_POWER_VA`
(short-circuit power) — applied only at h>1 so the fundamental is exact (the model in
`docs/pgml/modeling/error-injection.md`). Sweeps every node with
`pgml.scenarios.run_node_injection_sweep`, records the **h=11 voltage at every node for
every injection node** minus a no-injection reference, **normalised per node by the local
fundamental `|V1|`** → an `m × i` **per-unit spread matrix** `|V_h11|/|V1|`
(`spread_h11.csv`/`.npz`/heatmap). The per-unit basis matters: in raw volts the 20 kV MV
nodes dwarf the 0.4 kV LV nodes from the transformer ratio alone and hide the disturbance
origin; in per-unit the response peaks at the injecting node. Then **pgml vs live
OpenDSS**: parity print (**bit-exact ~1e-13**, Carson on both) + `compare_h11.csv`
(pgml | opendss | Δ, per-unit) + `compare_h11.svg` (overlay) for the most-affected
injection node.

Control knobs (top of file): `RECORD_ORDER`, `SOURCE_POWER_VA` (source strength),
`KIND` ("voltage"/"current"); plus `source_impedance_ohm` on `cigre_lv_full_grid`.

### `scenario_randomized.py` — randomized symmetric vs asymmetric study (Scenario 2)
Three-phase, **Carson geometry** line model (`synthesize_grid_geometry` gives every R/X
line a 3-conductor geometry reproducing `Z1 + X0` at f0; the same geometry feeds OpenDSS).
Per load, an **independent-per-phase** `U(0,1)` power scale + a random EN 50160-bounded
harmonic spectrum (h=3,5,7,9). Solves the SAME sampled batch **symmetric and asymmetric**,
times both (CPU), writes both to parquet + CSV, and plots the asymmetric run: a fundamental
all-phase profile (now correctly ~1.0 pu on the line-to-neutral base) and a 3D harmonic plot
(**color = order, line style = phase** L1 solid / L2 dashed / L3 dotted).

OpenDSS parity (per-order, printed): **bit-exact ~1e-11 on every order, including the
triplen h3/h9**. Because the same Carson geometry feeds both engines, the line zero-sequence
model is identical, and the Dyn vector group (stamped identically on both sides) traps the
zero sequence — so the earlier triplen gap (pgml's analytic `sequence_aware` Z0 vs OpenDSS's
internal Carson earth return) is gone. The transformer vector group is independently
validated bit-for-bit against a real OpenDSS `Transformer` element via
`opendss_dyn_transformer_harmonic_voltages`. (CIGRE LV lines are low-X cables, so the
synthesized GMR is non-physical — flagged by a warning — but still reproduces the target
impedance and matches OpenDSS exactly; for physically-representative magnitudes on R/X
feeders use `apply_positive_sequence_harmonic_model` / `sequence_aware` instead.)

Control knobs: `N_SAMPLES`, `SEED`, `ORDERS`, the two `ParameterSpec`s in `build_config`
(distributions, `symmetry`), `source_impedance_ohm`.

## The flexible scenario system (what these are built on)

Everything is `pgml.scenarios` — a serializable config (+ seed) deterministically defines a
batch; one batched solve produces aligned results; persist reproducibly.

| Need | Use |
|---|---|
| Random / QMC sampling of load P/Q (per-phase symmetry, correlated fleets) | `ScenarioConfig` + `ParameterSpec` (`field`, `mode`, `per`, `correlation`, `symmetry`) |
| Random per-device harmonic spectra (EN 50160-bounded) | `ParameterSpec(field="h_mag"|"h_phase", orders=..., harmonic_reference="en50160")` |
| Node-coherent harmonic "fingerprints" over a time sequence | `CoherentSpectrumConfig` → `sample_coherent_spectra` |
| Inject a device (load) harmonic current at each node, one at a time | `SpectrumSweepConfig.from_spectrum(...)` → `spectrum_sweep` |
| Inject a per-NODE harmonic error SOURCE (Thévenin/Norton, any node) at each node | `NodeInjectionSweepConfig.from_spectrum(spectrum, source_power_va=…, kind=…)` → `run_node_injection_sweep`; single source: `solve_harmonic_flow(node_sources=[NodeHarmonicSource(...)])` (see `docs/pgml/modeling/error-injection.md`) |
| Inject one P/Q error at each node, one at a time | `Perturbation` → `perturbation_sweep` |
| Deterministic grid-sweep (cartesian product) | `CartesianConfig` / `CartesianAxis` |
| Solve a batch | `run_scenarios(grid, spec, calculation=..., harmonic_orders=..., symmetry=...)` |
| Persist as ML training data | `write_dataset(result, dir, layout="wide"|"long", also_csv=True)` / `read_dataset` |

`run_scenarios` accepts a config OR a pre-built `SampledScenarios`; results are
`[B, N]` (power flow), `[B, H, N]` (harmonic), or `[B, T, H, N]` (coherent).

See `src/pgml/scenarios/CONTEXT.md` for the full interface ledger.

## Other examples
- `benchmark_speed.py` — execution-speed study (CPU and GPU). Generates a large batch of
  randomized operating points (varying loads, added PV systems, and harmonic injections)
  on two grids of contrasting size — the smaller IEEE-33 vs the full 3-phase CIGRE LV +PV
  — and times every solve path: the two power-flow solvers (`current_injection` vs
  `newton`, with iteration counts), load flow vs harmonic flow, over a batch-size sweep,
  on every device present. Auto-detects CUDA; writes one `results_<device>.json` per
  device and merges them, so a single run on a GPU host (whose default environment also
  runs on CPU) yields the CPU-vs-GPU figures. Plots: `solver_comparison.svg`,
  `loadflow_vs_harmonic.svg`, `throughput_vs_batch.svg`, `device_speedup.svg` (when a CUDA
  series is present), plus `benchmark_summary.csv`. CPU host:
  `pixi run -e cpu python examples/benchmark_speed.py`; GPU host:
  `pixi run python examples/benchmark_speed.py` (default environment ships `pytorch-gpu`).
- `evaluate_ieee33.py` — load-flow evaluation vs pandapower (IEEE-33).
- `evaluate_harmonics_carson.py` — OpenDSS-vs-pgml harmonic comparison (Carson geometry).
- `evaluate_line_sequence_harmonics.py` — positive-sequence vs naive vs OpenDSS line models.
- `current_injection_convergence.py` — why the current-injection power flow oscillates near
  the loadability nose. A 2-bus radial with a closed-form P-V nose; plots `||ΔV||` and the
  load-bus `|V|/E` per iteration at several load levels (smooth convergence → slowdown →
  sustained oscillation past the nose) and the P-V curve showing the method's convergence
  region sitting inside the feasible region. Motivates the Newton/continuation tools and
  the `ConvergenceDiagnostics` "did not settle" verdict.
- `loadability_continuation.py` — `pgml.solver.loadability_limit` on CIGRE LV: how much load
  until voltage collapse (the breaking λ\*/margin), **which bus** collapses (the
  voltage-collapse mode), and **which load** most limits the margin. Plots the P-V nose
  curve at the critical bus, the voltage profile at the nose with the critical bus(es)
  highlighted, and a bar chart ranking the limiting loads.
