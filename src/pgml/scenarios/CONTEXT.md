# scenarios — the batch contract

A scenario batch is a set of per-component DELTAS on one grid: what a spec names varies,
every other component and field keeps the grid's nominal value. This subpackage owns that
contract end to end (how a batch is declared, drawn, solved and persisted) and nothing
about which variations a particular study should draw.

Scope rule: a construct belongs here if a user who has never heard of harmonic state
estimation would still reach for it when sweeping a grid. Device populations, calibrated
emission ranges and training-task designs live in the package that calibrates them and
reach this one through the `ScenarioSpec` protocol. The former recipe names
(`_RECIPE_NAMES` in `__init__.py`) raise an explanatory relocation error.

Signatures: docstrings / API_INDEX.md.

## Codemap

- `sampler.py` — `SampledScenarios` (the frozen batch object every path returns),
  `sample`, `cartesian_sample`, `unit_samples` (the shared unit-cube draw),
  `nominal_power`, the realized-injection audit records.
- `config.py` — the serializable declarations: distributions (closed-form `icdf`),
  `Selector`, `ParameterSpec`, `Correlation` / `LatentFactor`, `ScenarioConfig`,
  `CartesianAxis` / `CartesianConfig`, `BackgroundHarmonicConfig`, the sweep configs.
  `python -m pgml.scenarios.config --json-schema | --example` prints the contract.
- `batch.py` — `batch_from_values` (a batch from tensors the caller already has),
  `broadcast_operating_point`.
- `perturbation.py`, `harmonics.py`, `node_injection.py` — excitation primitives:
  `perturbation_sweep`, `spectrum_sweep`, `build_background_sources`,
  `run_node_injection_sweep`.
- `random.py` — `ar1_noise` (stationary AR(1) along the last axis).
- `en50160.py`, `iec61000_3_2.py` — the standards tables (packaged YAML under
  `pgml/data/standards/`, `PGML_EN50160` / `PGML_IEC61000_3_2` overrides, `*_provenance`).
- `run.py` — `ScenarioSpec` protocol, `run_scenarios`, `ScenarioResult`.
- `persistence.py` — `write_dataset`, `read_dataset`, `LoadedDataset`,
  `SCENARIO_CONFIG_TYPES`, `config_hash`, `generation_provenance`.
- Storage dispatch is NOT here: it is device time coupling and lives in `pgml.dispatch`.

## The batch object

- `n_steps` (`T`) and the per-scenario vs batch-shared record split (`samples` vs
  `shared_samples`) are DECLARED, never inferred from shapes: `[B, T, N]` and `[B, H, N]`
  have the same rank, and a record of length `B` is indistinguishable from a per-scenario
  column. The run and the persistence layer both read the declarations.
- `validate()` runs on construction (shape-only broadcast check against `batch_shape`;
  a record may not be both per-scenario and shared). `run_scenarios` re-validates against
  the grid it solves, so a batch built for another grid fails before the solver.
- Reserved record keys a consumer may rely on: `time_s` `[T]` (relative seconds),
  `time_unix_s` `[T]` (epoch seconds, one anchor for the whole batch), and
  `<key>_mag` / `<key>_phase` / `<key>_device_ids` (realized-injection audit, below).
- `batch_from_values`: an absent component or field keeps the nominal (no fill value);
  ids resolve against the grid (a typo raises); values pass through untouched, so
  gradients, device and dtype are the caller's.

## Declaring and drawing

- Unit-cube layout `U[B, D]`: one column per declared `LatentFactor`, then per spec a
  BASE block (component-level draws: #matched for `each`/correlated, 1 for `shared`, 0
  for `independent`) followed by a PER-PHASE block. Each column goes through
  `distribution.icdf` (or the copula score path). Same config + seed → identical output.
  Correlation and per-phase noise are torch ops on `U` (autograd-safe).
- `per="fixed"` / `per="class"` (harmonic fields) consume NO cube column: they draw from
  a stream seeded by `config.seed` + spec name (`fixed`) or the spec name alone (`class`,
  identical for every seed), so adding them leaves every other draw unchanged.
- `mode="scale"` multiplies the nominal (`nominal_power`, differentiable w.r.t. a
  tensor nameplate); `"absolute"` sets W/var (or pu for harmonics) directly.
- Per-phase `symmetry` (`independent`, `small_imbalance`) writes per-phase overrides,
  which auto-promote the solve to asymmetric.
- Correlation is what keeps an aggregate varying: with `rho=0` the mean over `N` matched
  components concentrates as `1/sqrt(N)`. It preserves the marginal and composes with
  every `symmetry` (per phase under `independent`).
- `harmonic_reference`: `"iec61000-3-2"` is the per-device CURRENT-emission reference
  (the physically meaningful one, default for fingerprints); `"en50160"` is a supply
  VOLTAGE compatibility level, a background shape, not an emission model.
- `field="h_param"` is a free parameter: drawn in the cube, recorded, written nowhere —
  the hook for an emission model defined outside this package. Exempt from the
  overlapping-writer check.
- Two specs writing the same `(device, field, order)` or the same power field of one
  component raise `InputError`: last-writer-wins would desync the record from the
  realized operating point.
- `BackgroundHarmonicConfig` is the one way to express supply-side distortion: a
  Thevenin source at every in-service `Source` node carrying the same realized spectrum;
  `source_power_va` is batch-constant, so `Y(h)` stays shared and only the Norton current
  varies. The AR(1) drift walks the step axis; a snapshot batch draws from its stationary
  distribution.
- `ScenarioSpec`: an object with `sample(grid)` and an optional `harmonic_orders` hint;
  a non-empty hint switches `run_scenarios` to a harmonic calculation. This is the seam
  downstream generators plug into.

## Running

- `run_scenarios` returns `v` `[B, N]` (power flow), `[B, H, N]` (harmonic) or
  `[B, T, H, N]` (sequence). It never raises on a failed scenario: `converged` is the
  AND over scenarios, `failed_states` lists failed scenario indices (a sequence fails if
  any step fails) and their `v` is a best-effort iterate.
- Power flow prepares the operating-point-independent state once (`prepare_power_flow`)
  and reuses it across the batch and every chunk.
- `chunk_size` bounds the per-solve workspace, not the collected output; `output_device`
  moves each chunk's result off the GPU (non-differentiable data generation only).
- A harmonic `load_shunt` derived from a per-scenario operating point makes `Y(h)`
  scenario-dependent (one factorization per scenario); `"none"` keeps one per order.

## Persistence (a cross-package file format)

- A dataset is a DIRECTORY: `voltages.parquet` (`layout="long"` tidy rows or
  `layout="wide"` array columns over `[H*N]`; both read back the identical `v`),
  `samples.parquet` (dtype-preserving array columns), `meta.json`, optional `voltages.csv`.
  Detached result I/O, not part of the differentiable core.
- The on-disk layout comes from the declarations (`n_samples`, `n_steps`, result index,
  frequencies), checked only against the element count.
- `meta.json` records the serialized config + seed, `config_hash`, the bare
  `config_type` / `config_module` / `config_class`, the time axis (`n_steps`,
  `step_size_s`, `t0_unix_s`), `schema_version`, `generation_provenance()` (code
  provenance + active standards tables with content hash) and the caller's
  `extra_provenance`. Readers tolerate missing keys (older datasets).
- `read_dataset` reconstructs a config only from `SCENARIO_CONFIG_TYPES` plus explicit
  `config_types=` (reading must not import a module the file names); otherwise the raw
  dict, with a warning when the class no longer accepts it. A major `schema_version`
  mismatch raises; minor drift warns. `failed_scenarios` marks best-effort rows a
  training-data consumer must drop or keep explicitly.
- `config_hash` covers the fields only (no module/class name), so moving a config class
  between packages does not invalidate recorded hashes.

## Realized-injection audit columns

One convention across every generating path, so a dataset records WHAT WAS INJECTED and
not only the draw (a draw is a fraction of a per-device reference; a composed aggregate
has no draw at all). A block is `<key>_mag` / `<key>_phase` — magnitude in per unit of
the device's own fundamental current, phase in degrees, exactly the `harmonic_injection`
the solver received — over the batch-shared device axis `<key>_device_ids`:

| path | `<key>` | shape |
|---|---|---|
| randomized `h_mag` spec (`sampler`) | `<spec.name>` | `[B, n_dev, n_ord]` |
| a sequence generator | its own `<name>` | `[B, n_dev, n_ord, T]` |

The raw draw stays under `<spec.name>`. The columns are additive parquet samples:
`SCHEMA_VERSION` (which versions `pgml.schemas`) is unaffected.
