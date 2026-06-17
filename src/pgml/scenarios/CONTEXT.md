# Interface ledger: scenarios (batched, reproducible sampling)

Config-driven generation of batched operating points for ML training data. A
`ScenarioConfig` + `seed` deterministically defines the batch — saving the config
reproduces the dataset (reproducibility is paramount). Reuses the already-batched
solver (`solve_power_flow` / `solve_harmonic_flow` broadcast a leading scenario dim;
verified `batched == loop-of-individual`).

## Public API (IMPLEMENTED — increment 1)
`from pgml.scenarios import ...`
- Distributions (closed-form `icdf(u)` for QMC; `u in [0,1]`): `Uniform(low,high)`,
  `Normal(loc,scale)`, `LogNormal(loc,scale)`, `LogUniform(low,high)`, `Constant(value)`.
  Discriminated union `Distribution` (field `kind`).
- `Selector(component="load"|"generator", ids=None, consumer_type=None)` ->
  `.resolve(grid) -> [ids]` (None+None = all of that kind; filters AND).
- `ParameterSpec(name, selector, distribution, field="p"|"q"|"pq", mode="scale"|"absolute",
  per="each"|"shared")`. `pq` varies P&Q by the same factor (scale only).
- `ScenarioConfig(n_samples, seed=0, method="sobol"|"lhs"|"independent", parameters=[...])`.
- `sample(grid, config) -> SampledScenarios(operating_point, samples, n_samples, config)`
  where `operating_point = {id: {"p_w": Tensor[B], "q_var": Tensor[B]}}` and
  `samples = {param_name: Tensor[B, d]}` (raw realized values; ML input record).
- CARTESIAN sweep (pgm-style, deterministic): `CartesianAxis(name, selector, values=[...],
  field, mode)`, `CartesianConfig(axes=[...])` -> `cartesian_sample(grid, config) ->
  SampledScenarios` (B = prod(len(axis.values)); each axis level applied to all matched
  comps; list id-selector axes for per-component sweeps).
- `run_scenarios(grid, spec, *, calculation="power_flow"|"harmonic", harmonic_orders=None,
  slack="ideal", dtype, device) -> ScenarioResult(v, index, sampled, frequencies_hz)`.
  `spec` = `ScenarioConfig` | `CartesianConfig` | a pre-built `SampledScenarios`.
  `v` is `[B,N]` (power_flow) or `[B,H,N]` (harmonic).

## Conventions
- Sampling dim `D` = Σ over params of (#matched comps if `per="each"` else 1).
  `U[B,D]` in `[0,1)` from Sobol (QMC, recommended) / LHS / independent, then each
  column -> `distribution.icdf`. Same config+seed -> identical output (deterministic).
- `mode="scale"` multiplies the component's nominal `p_nom_w`/`q_nom_var`; `"absolute"`
  sets W/var directly. Unset of P or Q -> solver keeps the nominal for that one.

## Deferred (next increments — see memory `batching-scenarios-design`)
- Structured "one perturbation per node" sweep (use case: inject an error at each
  node, measure spread) — an enumeration over selector targets (not a cartesian of
  levels); needs a small dedicated builder.
- Correlated / "shared-by-physics" sampling (e.g. all PV ~correlated) beyond
  `per="shared"`; per-phase SYMMETRY control.
- HARMONIC SPECTRUM distributions (vary injection mag/phase; EN50160-bounded) ->
  feed `solve_harmonic_flow(harmonic_injection=...)`.
- Network-parameter & TOPOLOGY (switch-state) batching; MULTI-GRID batching.
- Parquet persistence of (scenario, component, step, frequency) -> result_schema.
- Beta / scipy-backed distributions (no closed-form icdf).
