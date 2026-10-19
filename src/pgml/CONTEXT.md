# pgml package map

The subpackages and top-level modules of `pgml`, with what each owns. Rules that hold for the
whole repository (the two hard constraints, the frozen schemas, the conventions) are in the
root `CONTEXT.md`; each subpackage has its own `CONTEXT.md`.

Signatures: docstrings / `API_INDEX.md`.

## Subpackages

| Owns | Where |
|---|---|
| Input and output contracts (`Grid`, nodes, branches, appliances, results, scenario records), frozen | `schemas/` |
| Shipped modelling defaults, standards tables and reference presets | `data/` (loaded by `defaults.py`) |
| Y-bus assembly: per-phase, per-harmonic, batched stamps; device injections; node-phase index; branch currents; vector-group transformers; inverter control laws; exact bus fusion of zero-impedance branches | `assembly/` |
| Solves: batched complex linear; nonlinear constant-power/ZIP by current injection or Newton with IFT gradients; harmonic flow; diagnostics and loadability; low-rank switch sweeps | `solver/` |
| Geometry to impedance: differentiable Carson/Deri with skin effect; R/X to geometry synthesis; sequence-aware harmonic line models | `geometry/` |
| Converters from pandapower, OpenDSS and power-grid-model to `Grid`, and back | `convert/` |
| The batch contract: declared batches of solver-input deltas, sampling, sweeps, background distortion, batched solve, parquet persistence | `scenarios/` |
| Plots and the optional reference oracles | `evaluation/` |
| Grid-planning studies on top of assembly and solver: impedance scan (driving-point / transfer impedance vs frequency, resonances) | `studies/` |

## Top-level modules

- `simulation.py` — the public entry points `simulate`, `simulate_serializable`,
  `SimulationConfig`, `SolvedState` (eager voltages, lazy currents, flows, spectra and THD),
  `ResultBundle`. User-facing description: `docs/pgml/public-api.md`.
- `defaults.py` — `pgml.defaults`: the modelling-default loader (`get`, `describe`, `units`,
  `resolve`) and `use_preset(name)`, a context manager that selects a reference tool's
  modelling choices. Precedence: explicit value > modelling default > converter value.
- `paths.py` — `experiments_root()`: where run outputs and config instances live
  (`PGML_EXPERIMENTS`, default `./data`).
- `provenance.py` — `code_provenance()` / `git_state()`: the commit, dirty flag and versions
  every persisted artifact records. Reads git, or a `.git-provenance.json` written next to a
  deployed copy without a `.git` directory.
- `errors.py` — `PgmlError` → `InputError` (`ConfigurationError`, `ConversionError`,
  `ConnectivityError`, `ModelingError`) / `ComputationError` (`ConvergenceError`), with HTTP
  status hints. `PgmError` is a deprecated alias.
- `topology.py` — slack anchor, branch edges, distance from the slack, connectivity report,
  energized subgrid, layout and network fingerprints. Imports only the standard library and
  the schemas (no torch), so a lean process can import it.
- `multigrid.py` — `merge_grids`: disjoint-union merge of a grid ensemble into one solvable
  `Grid` (block-diagonal `Y`), with per-member id maps, `operating_point` and `branch_states`
  translation, `MergedGrid.split()` row views and `block_rows()` for the block-diagonal
  factorization backend.
- `dispatch.py` — storage state-of-charge integration and the realized power sequence of a
  `Storage` element. Off the autograd tape.
- `grids.py` — benchmark grids (IEEE 33-bus and CIGRE LV, built from the pandapower
  networks, with conductor geometry and PV placement) and `synthetic_feeder` (no optional
  dependency) for scaling studies.

## Dependency direction

`schemas` imports nothing else of pgml. `geometry` builds on `schemas` and `defaults`;
`assembly` on those and `geometry`; `solver` on `assembly` and `topology`; `simulation` and
`scenarios` on `solver`; `studies` on `simulation`, `solver` and `assembly` (nothing in pgml imports
`studies`).
`convert`, `evaluation` and `grids` (whose benchmark nets come from pandapower) sit on top
and are the only places that import the optional reference tools (pandapower, OpenDSS,
power-grid-model), always inside a function or behind an extra.
pgml imports no other package of the suite.

## Tests

Oracle comparisons (`tests/reference`), the differentiability gate (`tests/differentiability`)
and the device/dtype gate (`tests/gpu`); see `tests/CONTEXT.md`.
