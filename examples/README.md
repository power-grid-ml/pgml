# pgml examples

Runnable, documented studies on real benchmark grids. Run any with:

```bash
pixi run -e cpu python examples/<script>.py [out_dir]
```

Each script's module docstring is the authoritative "what / how / outputs" reference;
this file is the map + how to control them.

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
`references/error_injection.md`). Sweeps every node with
`pgml.scenarios.run_node_injection_sweep`, records the **h=11 voltage at every node for
every injection node** minus a no-injection reference → an `m × i` **spread matrix**
(`spread_h11.csv`/`.npz`/heatmap). Then **pgml vs live OpenDSS**: parity print
(**bit-exact ~1e-13**, Carson on both) + `compare_h11.csv` (pgml | opendss | Δ) +
`compare_h11.svg` (overlay) for the most-affected injection node.

Control knobs (top of file): `RECORD_ORDER`, `SOURCE_POWER_VA` (source strength),
`KIND` ("voltage"/"current"); plus `source_impedance_ohm` on `cigre_lv_full_grid`.

### `scenario_randomized.py` — randomized symmetric vs asymmetric study (Scenario 2)
Three-phase. Per load, an **independent-per-phase** `U(0,1)` power scale + a random
EN 50160-bounded harmonic spectrum (h=3,5,7,9). Solves the SAME sampled batch **symmetric
and asymmetric**, times both (CPU), writes both to parquet + CSV, and plots the asymmetric
run: a fundamental all-phase profile and a 3D harmonic plot (**color = order, line style =
phase** L1 solid / L2 dashed / L3 dotted).

OpenDSS parity (per-order, printed): non-triplen **h5/h7 ~1%**, triplen **h3/h9 diverge**
(zero-sequence — pgml's `sequence_aware` Z0 + simplified non-Dyn transformer vs OpenDSS's
Carson Z0 + vector group; deferred — see `TODO.md` item 4). Single-phase Scenario 1 is the
bit-exact comparison; the 3-phase zero-sequence model is approximate.

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
| Inject a per-NODE harmonic error SOURCE (Thévenin/Norton, any node) at each node | `NodeInjectionSweepConfig.from_spectrum(spectrum, source_power_va=…, kind=…)` → `run_node_injection_sweep`; single source: `solve_harmonic_flow(node_sources=[NodeHarmonicSource(...)])` (see `references/error_injection.md`) |
| Inject one P/Q error at each node, one at a time | `Perturbation` → `perturbation_sweep` |
| Deterministic grid-sweep (cartesian product) | `CartesianConfig` / `CartesianAxis` |
| Solve a batch | `run_scenarios(grid, spec, calculation=..., harmonic_orders=..., symmetry=...)` |
| Persist as ML training data | `write_dataset(result, dir, layout="wide"|"long", also_csv=True)` / `read_dataset` |

`run_scenarios` accepts a config OR a pre-built `SampledScenarios`; results are
`[B, N]` (power flow), `[B, H, N]` (harmonic), or `[B, T, H, N]` (coherent).

See `src/pgml/scenarios/CONTEXT.md` for the full interface ledger.

## Other examples
- `evaluate_ieee33.py` — load-flow evaluation vs pandapower (IEEE-33).
- `evaluate_harmonics_carson.py` — OpenDSS-vs-pgml harmonic comparison (Carson geometry).
- `evaluate_line_sequence_harmonics.py` — positive-sequence vs naive vs OpenDSS line models.
