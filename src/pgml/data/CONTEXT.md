# data — shipped reference data

Package data: physical and standards tables and the documented modeling defaults. They ship
inside the wheel and are read via `importlib.resources` (never a filesystem walk-up, never a
required environment variable). They are not run configuration; the serializable run
schema is `pgml.scenarios.config`.

Signatures: docstrings / `API_INDEX.md`.

## Codemap

- `defaults.yaml` — modeling default values and default model choices, ordered by
  component. Every leaf is `{value, units, description}`, so a default is a deliberate,
  documented decision. Loaded by `pgml.defaults`.
- `presets/{pgml,opendss,pandapower,power-grid-model}.yaml` — overlays that select a
  reference tool's modeling choices; applied by `pgml.defaults.use_preset(name)`.
- `standards/en50160.yaml` — EN 50160 per-order harmonic voltage limits (relative to the
  fundamental, orders 1..49). Loaded by `pgml.scenarios.en50160`.
- `standards/iec61000_3_2.yaml` — IEC 61000-3-2 per-class harmonic current limits, the
  default emission table of the scenario sampler. Loaded by `pgml.scenarios.iec61000_3_2`.

Top-level sections of `defaults.yaml`: `converter`, `calculation`, `appliance`, `line`,
`branch`, `transformer`, `source`, `solver`, `studies`. The meaning of each key is its `description`
field; read it in the file or with `pgml.defaults.describe(key)` (`units(key)`, `get(key)`).
`docs/pgml/modeling/presets.md` is the published view.

## Loading and overrides

- `pgml.defaults` — `get / resolve / describe / units / defaults / reload / use_preset`.
  `use_preset(name)` overlays a preset within a thread/task-local context and restores the
  previous selection on exit; explicit component fields win, unlisted settings keep the
  active defaults.
- Optional environment overrides (never required): `PGML_DEFAULTS` (or `reload(path)`),
  `PGML_EN50160`, `PGML_IEC61000_3_2`; the table loaders also take an explicit `path=`.

## Invariants

- Resolution precedence, highest first (`resolve(key, explicit, converted)`):
  1. explicit — a value set on the component or grid, always wins;
  2. the active defaults (`defaults.yaml`, or the selected preset);
  3. converter — a value inferred from a source library, last resort.
- Consumers read defaults from `pgml.defaults`; no module re-hard-codes a value from
  `defaults.yaml`. The module-level constants that mirror a default
  (`geometry.sequence._DEFAULT_GMR_OVER_RADIUS`, `CARSON_EARTH_R_PER_HZ`,
  `geometry.synthesis._DEFAULT_HEIGHT` / `_DEFAULT_RADIUS` / `_DEFAULT_EARTH_RHO`) are read
  from it at import.
- Assembly never resolves the line harmonic-model default; the converters (`to_grid`, which
  logs it) and `apply_default_harmonic_model(grid)` write `Line.harmonic_line_model`.
- Defaults that select a model raise on an unknown value (`transformer.harmonic_resistance`,
  `transformer.magnetizing_placement`, `branch.zero_impedance`, ...).
- A converter fallback to a default zero-sequence ratio (`line.zero_sequence`,
  `source.zero_sequence`) warns once per grid.

## Where each section is consumed

- `calculation.symmetry`, `appliance.load.*` — `assembly._symmetry`
  (`resolve_asymmetric`, `resolve_connection`).
- `appliance.harmonic_shunt.*` — `assembly._load_shunt` (`resolve_shunt_model_name`,
  `resolve_harmonic_shunt`, `resolve_shunt_basis`, `generation_shunt_is_neglected`);
  per-device `harmonic_model`, per-run `load_shunt=` / `load_shunt_basis=`.
- `line.*` — `geometry.sequence`, `geometry.synthesis`, `convert._common` (zero-sequence
  ratios); per-line overrides on `Line` (`harmonic_line_model`, `harmonic_skin_effect`,
  `earth_return`, `LineGeometry.internal_inductance`).
- `source.*` — `convert._common.source_zero_sequence_ratios`,
  `evaluation.oracles.grids` (source impedance synthesis).
- `transformer.*` — `assembly._transformer` (`resolve_vector_group`,
  `harmonic_resistance_law`, `magnetizing_placement`, `zero_sequence_leakage`).
- `branch.zero_impedance` — `assembly._fusion.resolve_fusion` (every assembler and solve
  entry point). `branch.near_ideal_series_resistance_ohm` — the stand-in for an ideal
  branch that must stay stamped (`zero_impedance: error`, a swept switch), named by
  `check_branch_impedances`; a converter writes it for a closed zero-impedance switch only
  under `branch.switch_model: near_ideal`. `branch.switch_model` — read by
  `convert.pandapower.converter._closed_switch_resistance_ohm`.
- `solver.convergence`, `solver.precision`, `solver.ift` — `solver.power_flow`
  (`_resolve_tolerances`, `_warn_complex64_conditioning`, `_ift_jacobian_budget_bytes`,
  `_ift_adjoint_cache_bytes`) and `solver.harmonic.lu_factor_system`.
- `solver.equilibration` — `solver.equilibration` (`resolve_equilibration`,
  `equilibration_scales`); entry points take `equilibrate=`. `solve_anchored` /
  `AnchoredSystem` do not apply it.
- `solver.harmonic.system_budget_mb` — `solver.harmonic_flow._harmonic_system_budget_bytes`
  (dense charge `_harmonic_chunk`, sparse-backend charge `_pattern_harmonic_chunk`).
- `solver.harmonic.preparation_min_rows` — `scenarios.run._preparation_pays`.
- `solver.loadability.ramp` — `solver.loadability_limit`.

## Decisions

- The default X0 frequency law is `linear` (exact for a metallic return, safe for a
  ratio-derived X0); `carson_sublinear` is the option for overhead lines whose X0 contains
  the deep-earth term, guarded by `x0_nonnegative`.
- `transformer.magnetizing_placement` ships `split`; the `opendss` preset selects the
  OpenDSS choices (unguarded sub-linear X0 law, power-frequency GMR internal inductance, no
  skin correction, `inductive` load-shunt reactance, magnetizing on the last winding's
  terminal).
- `branch.zero_impedance: fuse` and `branch.switch_model: ideal` ship as defaults: exact
  bus fusion reproduces the reference tools' treatment of ideal switches.
</content>
</invoke>
